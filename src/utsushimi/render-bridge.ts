// utsushimi — render bridge (synthetic-data ground truth via kami-rt/kami-rtx).
//
// Turns DR-randomized scene primitives into (a) real 2D bounding boxes by
// projecting each semantic prim's AABB through the camera, and (b) optional
// RGB frames by tessellating prims into triangles and ray-tracing them with
// kami-rt. This upgrades the upstream Replicator placeholder (full-image bbox /
// no pixels) to genuine annotated images — the payoff of pairing Replicator DR
// with the kami renderers (R1.2 + R1.4).
//
// Projection reuses kami-rt's pinhole basis so the projected boxes align with
// a kami-rt render from the same camera parameters.
//
// ADR-2605261800 §D6 / D10.4 utsushimi.

import {
  type Scene,
  type Vec3,
  buildScene,
  lookAt,
  traceImageCPU,
} from "../kami-rt/index.js";
import { type AnnotatedPrim } from "./writers.js";
import { type PrimSpec } from "./randomize.js";

// ── projection camera (kami-rt pinhole basis) ────────────────────────────────

export interface ProjCamera {
  eye: Vec3;
  /** kami-rt basis: w = norm(eye-target), u = norm(up×w), v = w×u. */
  w: Vec3;
  u: Vec3;
  v: Vec3;
  tanHalf: number;
  aspect: number;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export function makeProjCamera(
  eye: Vec3,
  target: Vec3,
  up: Vec3,
  vfovDeg: number,
  aspect: number,
): ProjCamera {
  const w = norm(sub(eye, target));
  const u = norm(cross(up, w));
  const v = cross(w, u);
  return { eye, w, u, v, tanHalf: Math.tan((vfovDeg * Math.PI) / 180 / 2), aspect };
}

/** Project a world point to pixel coords (top-left origin, y down). Returns
 *  null when the point is behind the camera. */
export function projectPoint(
  cam: ProjCamera,
  p: Vec3,
  width: number,
  height: number,
): [number, number] | null {
  const d = sub(p, cam.eye);
  const depth = -dot(d, cam.w); // forward = -w
  if (depth <= 1e-4) return null;
  const cx = dot(d, cam.u);
  const cy = dot(d, cam.v);
  const ndcX = cx / depth / (cam.tanHalf * cam.aspect);
  const ndcY = cy / depth / cam.tanHalf;
  const px = (ndcX * 0.5 + 0.5) * width;
  const py = (1 - (ndcY * 0.5 + 0.5)) * height;
  return [px, py];
}

// ── prim AABB ────────────────────────────────────────────────────────────────

function asVec3(a: readonly number[] | undefined): Vec3 {
  return a ? [a[0], a[1], a[2]] : [0, 0, 0];
}

function primAabb(prim: PrimSpec): { min: Vec3; max: Vec3 } | null {
  const p = asVec3(prim.position);
  if (prim._kind === "cube") {
    return { min: [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5], max: [p[0] + 0.5, p[1] + 0.5, p[2] + 0.5] };
  }
  if (prim._kind === "sphere") {
    const r = prim.radius ?? 1;
    return { min: [p[0] - r, p[1] - r, p[2] - r], max: [p[0] + r, p[1] + r, p[2] + r] };
  }
  return null; // cameras / lights have no annotatable extent
}

/** Project an AABB to a 2D bbox `[x, y, w, h]` (clamped to the image), or null
 *  if entirely behind the camera. */
export function projectAabb(
  cam: ProjCamera,
  min: Vec3,
  max: Vec3,
  width: number,
  height: number,
): [number, number, number, number] | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let any = false;
  for (let i = 0; i < 8; i++) {
    const corner: Vec3 = [
      i & 1 ? max[0] : min[0],
      i & 2 ? max[1] : min[1],
      i & 4 ? max[2] : min[2],
    ];
    const pr = projectPoint(cam, corner, width, height);
    if (!pr) continue;
    any = true;
    minX = Math.min(minX, pr[0]);
    minY = Math.min(minY, pr[1]);
    maxX = Math.max(maxX, pr[0]);
    maxY = Math.max(maxY, pr[1]);
  }
  if (!any) return null;
  const x0 = Math.max(0, Math.min(width, minX));
  const y0 = Math.max(0, Math.min(height, minY));
  const x1 = Math.max(0, Math.min(width, maxX));
  const y1 = Math.max(0, Math.min(height, maxY));
  if (x1 <= x0 || y1 <= y0) return null;
  return [x0, y0, x1 - x0, y1 - y0];
}

/** Annotate every semantic prim in `prims` with a real 2D bbox. Prims with no
 *  class semantic or no on-screen extent are returned without `bbox2d`. */
export function annotateFrame(
  cam: ProjCamera,
  prims: readonly PrimSpec[],
  width: number,
  height: number,
): AnnotatedPrim[] {
  return prims.map((prim) => {
    const aabb = primAabb(prim);
    if (!aabb) return { ...prim };
    const bbox = projectAabb(cam, aabb.min, aabb.max, width, height);
    return bbox ? { ...prim, bbox2d: bbox } : { ...prim };
  });
}

// ── tessellation + RGB render (optional bonus ground truth) ──────────────────

function cubeTris(c: Vec3, h: number): Vec3[][] {
  const v: Vec3[] = [
    [c[0] - h, c[1] - h, c[2] - h], [c[0] + h, c[1] - h, c[2] - h],
    [c[0] + h, c[1] + h, c[2] - h], [c[0] - h, c[1] + h, c[2] - h],
    [c[0] - h, c[1] - h, c[2] + h], [c[0] + h, c[1] - h, c[2] + h],
    [c[0] + h, c[1] + h, c[2] + h], [c[0] - h, c[1] + h, c[2] + h],
  ];
  const q = (a: number, b: number, cc: number, d: number): Vec3[][] => [
    [v[a], v[b], v[cc]], [v[a], v[cc], v[d]],
  ];
  return [
    ...q(0, 1, 2, 3), ...q(5, 4, 7, 6), ...q(4, 0, 3, 7),
    ...q(1, 5, 6, 2), ...q(4, 5, 1, 0), ...q(3, 2, 6, 7),
  ];
}

function sphereTris(c: Vec3, r: number, seg = 8): Vec3[][] {
  const pt = (i: number, j: number): Vec3 => {
    const theta = (i / seg) * Math.PI;
    const phi = (j / seg) * 2 * Math.PI;
    return [
      c[0] + r * Math.sin(theta) * Math.cos(phi),
      c[1] + r * Math.cos(theta),
      c[2] + r * Math.sin(theta) * Math.sin(phi),
    ];
  };
  const tris: Vec3[][] = [];
  for (let i = 0; i < seg; i++) {
    for (let j = 0; j < seg; j++) {
      const a = pt(i, j), b = pt(i + 1, j), cc = pt(i + 1, j + 1), d = pt(i, j + 1);
      tris.push([a, b, cc], [a, cc, d]);
    }
  }
  return tris;
}

/** Tessellate the renderable prims (cubes + spheres) into a kami-rt scene. */
export function primsToScene(prims: readonly PrimSpec[]): Scene {
  const tris: Vec3[][] = [];
  for (const prim of prims) {
    const p = asVec3(prim.position);
    if (prim._kind === "cube") tris.push(...cubeTris(p, 0.5));
    else if (prim._kind === "sphere") tris.push(...sphereTris(p, prim.radius ?? 1));
  }
  return buildScene(tris);
}

/** Render an RGB frame (RGBA float framebuffer) of the prims via kami-rt's CPU
 *  ray tracer, from a camera matching {@link makeProjCamera}'s parameters. */
export function renderFrameCPU(
  eye: Vec3,
  target: Vec3,
  up: Vec3,
  vfovDeg: number,
  prims: readonly PrimSpec[],
  width: number,
  height: number,
): Float32Array {
  const scene = primsToScene(prims);
  const cam = lookAt(eye, target, up, vfovDeg, width / height);
  return traceImageCPU(scene, cam, width, height).framebuffer;
}
