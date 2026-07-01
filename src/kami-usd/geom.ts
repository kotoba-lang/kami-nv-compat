// kami-usd geometry bridge — USDA prim tree → kami-rt triangles + materials.
//
// Triangulates UsdGeomMesh prims (fan triangulation over faceVertexCounts),
// composes the hierarchical Xform stack into a world matrix, and extracts a
// Lambertian material from displayColor / emissive primvars. The output feeds
// directly into kami-rt `buildScene` (ray) and kami-rtx `buildPathScene`
// (path trace).
//
// ADR-2605261800 §D6 / D10.4 kami-usd.

import { type Material, type Vec3, material } from "../kami-rt/index.js";
import { type UsdPrimNode, type UsdValue } from "./usda.js";

// ── row-major 4×4 matrix (point' = M · [x y z 1]) ─────────────────────────

export type Mat4 = number[]; // length 16, row-major

export function identity4(): Mat4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

export function mul4(a: Mat4, b: Mat4): Mat4 {
  const out = new Array<number>(16).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = s;
    }
  }
  return out;
}

export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  const x = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3];
  const y = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7];
  const z = m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11];
  return [x, y, z];
}

function translate4(t: Vec3): Mat4 {
  return [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1];
}
function scale4(s: Vec3): Mat4 {
  return [s[0], 0, 0, 0, 0, s[1], 0, 0, 0, 0, s[2], 0, 0, 0, 0, 1];
}
function rotX(deg: number): Mat4 {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
}
function rotY(deg: number): Mat4 {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
}
function rotZ(deg: number): Mat4 {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

// ── value coercion helpers ─────────────────────────────────────────────────

function asVec3(v: UsdValue | null | undefined, fallback: Vec3): Vec3 {
  if (Array.isArray(v) && v.length >= 3) {
    return [Number(v[0]), Number(v[1]), Number(v[2])];
  }
  return fallback;
}

/** Read an attribute value by name (returns null when absent). */
function attr(prim: UsdPrimNode, name: string): UsdValue | null {
  return prim.attributes.get(name)?.value ?? null;
}

// ── Xform op composition ───────────────────────────────────────────────────

function opMatrix(prim: UsdPrimNode, op: string): Mat4 {
  const v = attr(prim, op);
  if (op === "xformOp:translate") return translate4(asVec3(v, [0, 0, 0]));
  if (op === "xformOp:scale") return scale4(asVec3(v, [1, 1, 1]));
  if (op === "xformOp:rotateXYZ") {
    const e = asVec3(v, [0, 0, 0]);
    return mul4(rotX(e[0]), mul4(rotY(e[1]), rotZ(e[2])));
  }
  if (op === "xformOp:rotateX") return rotX(typeof v === "number" ? v : 0);
  if (op === "xformOp:rotateY") return rotY(typeof v === "number" ? v : 0);
  if (op === "xformOp:rotateZ") return rotZ(typeof v === "number" ? v : 0);
  if (op === "xformOp:transform" && Array.isArray(v)) {
    // matrix4d: 16 numbers, or 4 rows of 4. USD stores row-major rows.
    const flat: number[] = [];
    for (const e of v) {
      if (Array.isArray(e)) for (const x of e) flat.push(Number(x));
      else flat.push(Number(e));
    }
    if (flat.length === 16) return flat;
  }
  return identity4();
}

/** Local transform of a prim from its xformOp stack. `xformOpOrder` lists ops
 *  outermost-first; the matrix is their product in that order. With no order
 *  attribute, present ops apply as translate · rotate · scale (SRT). */
export function localTransform(prim: UsdPrimNode): Mat4 {
  const orderRaw = attr(prim, "xformOpOrder");
  let order: string[];
  if (Array.isArray(orderRaw)) {
    order = orderRaw.map((t) => String(t));
  } else {
    order = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"].filter((op) =>
      prim.attributes.has(op),
    );
  }
  let m = identity4();
  for (const op of order) m = mul4(m, opMatrix(prim, op));
  return m;
}

// ── material extraction ────────────────────────────────────────────────────

const DEFAULT_ALBEDO: Vec3 = [0.8, 0.8, 0.8];

function firstColor(v: UsdValue | null): Vec3 | null {
  if (!Array.isArray(v)) return null;
  // color3f[] primvars → [[r,g,b], ...]; take element 0. color3f scalar →
  // [r,g,b] directly.
  if (Array.isArray(v[0])) return asVec3(v[0], DEFAULT_ALBEDO);
  if (v.length >= 3 && typeof v[0] === "number") return [Number(v[0]), Number(v[1]), Number(v[2])];
  return null;
}

export function meshMaterial(prim: UsdPrimNode): Material {
  const albedo =
    firstColor(attr(prim, "primvars:displayColor")) ??
    firstColor(attr(prim, "inputs:diffuseColor")) ??
    DEFAULT_ALBEDO;
  const emission =
    firstColor(attr(prim, "primvars:emissiveColor")) ??
    firstColor(attr(prim, "inputs:emissiveColor")) ??
    ([0, 0, 0] as Vec3);
  return material(albedo, emission);
}

// ── mesh triangulation ─────────────────────────────────────────────────────

function readPoints(prim: UsdPrimNode): Vec3[] {
  const v = attr(prim, "points");
  if (!Array.isArray(v)) return [];
  return v.map((e) => asVec3(e, [0, 0, 0]));
}

function readIntArray(prim: UsdPrimNode, name: string): number[] {
  const v = attr(prim, name);
  if (!Array.isArray(v)) return [];
  return v.map((e) => Number(e));
}

/** Triangulate one UsdGeomMesh prim into world-space triangles. Polygons are
 *  fan-triangulated; if `faceVertexCounts` is absent the index stream is taken
 *  as consecutive triangles. */
export function triangulateMesh(prim: UsdPrimNode, world: Mat4): Vec3[][] {
  const points = readPoints(prim).map((p) => transformPoint(world, p));
  const indices = readIntArray(prim, "faceVertexIndices");
  let counts = readIntArray(prim, "faceVertexCounts");
  if (indices.length === 0 || points.length === 0) return [];
  if (counts.length === 0) {
    counts = new Array(Math.floor(indices.length / 3)).fill(3);
  }
  const tris: Vec3[][] = [];
  let cursor = 0;
  for (const n of counts) {
    if (n >= 3 && cursor + n <= indices.length) {
      const v0 = points[indices[cursor]];
      for (let k = 1; k < n - 1; k++) {
        const v1 = points[indices[cursor + k]];
        const v2 = points[indices[cursor + k + 1]];
        if (v0 && v1 && v2) tris.push([v0, v1, v2]);
      }
    }
    cursor += n;
  }
  return tris;
}

// ── stage flatten ──────────────────────────────────────────────────────────

export interface FlatScene {
  triangles: Vec3[][];
  materials: Material[];
}

/** Walk a USDA prim tree, accumulating world transforms, and collect every
 *  UsdGeomMesh as world-space triangles + a per-triangle material. */
export function flattenStage(roots: readonly UsdPrimNode[]): FlatScene {
  const triangles: Vec3[][] = [];
  const materials: Material[] = [];

  const walk = (prim: UsdPrimNode, parentWorld: Mat4): void => {
    const world = mul4(parentWorld, localTransform(prim));
    if (prim.typeName === "Mesh") {
      const mat = meshMaterial(prim);
      for (const t of triangulateMesh(prim, world)) {
        triangles.push(t);
        materials.push(mat);
      }
    }
    for (const child of prim.children) walk(child, world);
  };

  for (const r of roots) walk(r, identity4());
  return { triangles, materials };
}
