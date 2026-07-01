// @etzhayyim/kami-nv-compat/utsushimi
//
// Clean-room synthetic-data-generation engine (utsushimi 写し身) — the
// canonical KAMI implementation behind `nv-compat/omni-replicator-core`.
// Bit-reproducible DR sampler + distributions + create/modify/randomize ops +
// COCO/Kitti writers + a render bridge that grounds annotations in real
// kami-rt camera projection and (optionally) kami-rt RGB frames.
//
// ADR-2605261800 §D6 / D10.4 utsushimi.

export { Sampler, seedGlobal, globalSampler } from "./sampler.js";
export { type Dist, type Sampled, distribution, sample } from "./distribution.js";
export {
  type Semantic,
  type PrimSpec,
  type ModifyOp,
  type RandomizeOp,
  type ResolvedOp,
  type ScatterPose,
  create,
  modify,
  randomize,
  resolve,
} from "./randomize.js";
export {
  type AnnotatedPrim,
  type FrameSample,
  type Writer,
  type WriterInit,
  type CocoDataset,
  type CocoImage,
  type CocoAnnotation,
  type CocoCategory,
  BasicWriter,
  CocoWriter,
  KittiWriter,
  WriterRegistry,
} from "./writers.js";
export {
  type ProjCamera,
  makeProjCamera,
  projectPoint,
  projectAabb,
  annotateFrame,
  primsToScene,
  renderFrameCPU,
} from "./render-bridge.js";

import { Sampler } from "./sampler.js";
import { type PrimSpec, type RandomizeOp, type ResolvedOp, resolve } from "./randomize.js";
import { type AnnotatedPrim, type FrameSample, type Writer } from "./writers.js";
import { type Vec3 } from "../kami-rt/index.js";
import { type ProjCamera, annotateFrame, makeProjCamera, renderFrameCPU } from "./render-bridge.js";

// ── orchestrator: DR → scene → annotate → write ─────────────────────────────

export interface CameraSpec {
  eye: Vec3;
  target: Vec3;
  up?: Vec3;
  vfovDeg?: number;
}

export interface GenerateOptions {
  /** Base scene primitives (scatter ops mutate clones of these per frame). */
  prims: PrimSpec[];
  /** Domain-randomization ops applied (in order) each frame. */
  randomizers: RandomizeOp[];
  camera: CameraSpec;
  numFrames: number;
  imageWidth?: number;
  imageHeight?: number;
  seed?: number;
  writers: Writer[];
  /** When true, also ray-trace an RGB frame per frame (returned, not written). */
  render?: boolean;
}

export interface GenerateResult {
  /** Per-frame annotated primitives (with real 2D bboxes). */
  frames: AnnotatedPrim[][];
  /** Finalized writer outputs, in the order of `writers`. */
  outputs: unknown[];
  /** RGBA-float framebuffers when `render` is set. */
  rgb?: Float32Array[];
}

function clonePrim(p: PrimSpec): PrimSpec {
  return { ...p, position: p.position ? [...p.position] : undefined, semantics: p.semantics ? [...p.semantics] : undefined };
}

function applyResolved(op: RandomizeOp, res: ResolvedOp, working: PrimSpec[]): void {
  if (res.kind === "scatter_2d" || res.kind === "scatter_3d") {
    const opPrims = (op as { prims: PrimSpec[] }).prims;
    res.poses.forEach((pose, i) => {
      const target = opPrims[i];
      const idx = working.indexOf(target);
      const placed: PrimSpec = {
        ...clonePrim(target),
        position: [...pose.position],
        rotation_y: pose.rotation_z ?? (pose.rotation ? pose.rotation[1] : 0),
      };
      if (idx >= 0) working[idx] = placed;
      else working.push(placed);
    });
  } else if (res.kind === "randomize_materials" && op._kind === "randomize_materials") {
    for (const prim of op.prims) {
      const idx = working.indexOf(prim);
      const tinted = clonePrim(idx >= 0 ? working[idx] : prim);
      tinted.semantics = [...(tinted.semantics ?? []), ["color", String(res.material)]];
      if (idx >= 0) working[idx] = tinted;
    }
  }
  // randomize_lights / randomize_physics resolve for reproducibility but do not
  // alter 2D annotation geometry.
}

/** Run a full synthetic-data generation pass: for each frame, advance the DR
 *  sampler, randomize the scene, project ground-truth 2D boxes, and write to
 *  every writer. Deterministic given `seed`. */
export function generateDataset(opts: GenerateOptions): GenerateResult {
  const W = opts.imageWidth ?? 640;
  const H = opts.imageHeight ?? 480;
  const up: Vec3 = opts.camera.up ?? [0, 1, 0];
  const vfov = opts.camera.vfovDeg ?? 45;
  const cam: ProjCamera = makeProjCamera(opts.camera.eye, opts.camera.target, up, vfov, W / H);
  const sampler = new Sampler(opts.seed ?? 0);

  const frames: AnnotatedPrim[][] = [];
  const rgb: Float32Array[] = [];

  for (let f = 0; f < opts.numFrames; f++) {
    const working = opts.prims.map(clonePrim);
    // Re-link scatter/material op prim references to the working clones by
    // index position (base prims preserve order in `working`).
    for (const op of opts.randomizers) {
      const res = resolve(op, sampler);
      applyResolved(remapOp(op, opts.prims, working), res, working);
    }
    const annotated = annotateFrame(cam, working, W, H);
    const sample: FrameSample = { frame: f, primitives: annotated };
    for (const w of opts.writers) w.writeFrame(f, sample);
    frames.push(annotated);
    if (opts.render) {
      rgb.push(renderFrameCPU(opts.camera.eye, opts.camera.target, up, vfov, working, W, H));
    }
  }

  const outputs = opts.writers.map((w) => w.finalize());
  return opts.render ? { frames, outputs, rgb } : { frames, outputs };
}

/** Rebind an op's prim references from the base list to the per-frame working
 *  clones (matched by index in the base list). */
function remapOp(op: RandomizeOp, base: PrimSpec[], working: PrimSpec[]): RandomizeOp {
  const remap = (p: PrimSpec): PrimSpec => {
    const i = base.indexOf(p);
    return i >= 0 ? working[i] : p;
  };
  switch (op._kind) {
    case "randomize_materials":
    case "scatter_2d":
    case "scatter_3d":
      return { ...op, prims: op.prims.map(remap) };
    case "randomize_physics":
      return { ...op, prim: remap(op.prim) };
    default:
      return op;
  }
}
