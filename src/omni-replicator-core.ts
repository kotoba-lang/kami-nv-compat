// @etzhayyim/kami-nv-compat/omni-replicator-core
//
// Drop-in `omni.replicator.core` API-compat facade — synthetic-data generation
// + domain randomization. Mirrors the documented Replicator surface
// (`rep.distribution.*`, `rep.create/modify/randomize.*`, `rep.WriterRegistry`,
// `rep.new_layer()` / `rep.trigger.on_frame()` / `rep.orchestrator.run()`) so
// Isaac Sim Replicator scripts port to KAMI via import-path-only changes —
// e.g.
//
//     import * as rep from "@etzhayyim/kami-nv-compat/omni-replicator-core";
//
//     rep.seedGlobal(42);
//     const cubes = [rep.create.cube([0,0,0], [["class","cube"]])];
//     const writer = rep.WriterRegistry.get("CocoWriter");
//     writer.initialize({ outputDir: "out", imageWidth: 640, imageHeight: 480 });
//     const ds = rep.generateDataset({
//       prims: cubes,
//       randomizers: [rep.randomize.scatter_2d(cubes, "xy", [[-2,-2],[2,2]])],
//       camera: { eye: [0,3,6], target: [0,0,0] },
//       numFrames: 8, writers: [writer], render: false,
//     });
//
// Backed by the clean-room utsushimi engine (bit-reproducible DR sampler +
// kami-rt projection for real ground-truth boxes). No Replicator source or
// binaries; from-spec reproduction of the public API + on-disk schema
// (Google v. Oracle, 593 U.S. ___ (2021)). The canonical engine is `utsushimi`.
//
// Trademark: NVIDIA® / Omniverse® / Replicator are trademarks of NVIDIA
// Corporation; API-compat identifiers only.
//
// ADR-2605261800 §D1/D6, R1.3 omni-replicator-core surface.

export {
  // sampler + distributions
  Sampler,
  seedGlobal,
  globalSampler,
  distribution,
  sample,
  // scene ops
  create,
  modify,
  randomize,
  resolve,
  // writers
  BasicWriter,
  CocoWriter,
  KittiWriter,
  WriterRegistry,
  // render bridge
  makeProjCamera,
  projectPoint,
  projectAabb,
  annotateFrame,
  primsToScene,
  renderFrameCPU,
  // orchestrator
  generateDataset,
} from "./utsushimi/index.js";

export type {
  Dist,
  Sampled,
  Semantic,
  PrimSpec,
  ModifyOp,
  RandomizeOp,
  ResolvedOp,
  ScatterPose,
  AnnotatedPrim,
  FrameSample,
  Writer,
  WriterInit,
  CocoDataset,
  ProjCamera,
  CameraSpec,
  GenerateOptions,
  GenerateResult,
} from "./utsushimi/index.js";

import { type PrimSpec, type RandomizeOp, type Writer } from "./utsushimi/index.js";

// ── new_layer / trigger / orchestrator (Replicator script-graph shape) ───────

/** A Replicator authoring layer: captured primitives, triggers, and writers.
 *  Mirrors the `with rep.new_layer():` context (TS has no `with`, so the layer
 *  is an explicit object). */
export class Layer {
  readonly primitives: PrimSpec[] = [];
  readonly randomizers: RandomizeOp[] = [];
  readonly writers: Writer[] = [];
  numFrames = 1;

  /** `with rep.trigger.on_frame(n):` mirror — sets the frame count. */
  onFrame(numFrames: number): this {
    this.numFrames = numFrames;
    return this;
  }
  addPrimitives(...prims: PrimSpec[]): this {
    this.primitives.push(...prims);
    return this;
  }
  addRandomizer(...ops: RandomizeOp[]): this {
    this.randomizers.push(...ops);
    return this;
  }
  addWriter(...writers: Writer[]): this {
    this.writers.push(...writers);
    return this;
  }
}

export function newLayer(): Layer {
  return new Layer();
}

/** Canonical KAMI engine name behind this facade. */
export const KAMI_ENGINE = "utsushimi";
export const ADR = "ADR-2605261800";
