/**
 * nv-compat kami-rtx path-tracer primitives.
 *
 * Direct coverage for the path-tracer building blocks: the branchless
 * orthonormal basis (both sign branches), the xorshift RNG + seed hash, and the
 * material/scene assembly. These underpin the GPU/CPU radiance integrator.
 *
 *     pnpm exec vitest run test/nv-compat-kami-rtx-primitives.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 kami-rtx.
 */

import { describe, it, expect } from "vitest";
import {
  type Vec3,
  buildPathScene,
  material,
  materialSoup,
  nextFloat,
  onb,
  seedHash,
} from "../src/kami-rt/index.js";

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

describe("orthonormal basis (onb)", () => {
  // onb assumes a UNIT normal — normalize before passing.
  const norm = (v: Vec3): Vec3 => {
    const l = len(v) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  const normals: Vec3[] = [
    [0, 0, 1],
    [0, 0, -1], // exercises the sign < 0 branch
    [0, 0.6, 0.8],
    [0, 0.6, -0.8],
    [1, 0, 0],
    [1, 1, 1],
  ].map((v) => norm(v as Vec3));

  it.each(normals)("builds an orthonormal frame for n=%j", (...n) => {
    const nv = n as unknown as Vec3;
    const [t, bt] = onb(nv);
    expect(dot(t, nv)).toBeCloseTo(0, 5); // tangent ⟂ normal
    expect(dot(bt, nv)).toBeCloseTo(0, 5); // bitangent ⟂ normal
    expect(dot(t, bt)).toBeCloseTo(0, 5); // tangent ⟂ bitangent
    expect(len(t)).toBeCloseTo(1, 5);
    expect(len(bt)).toBeCloseTo(1, 5);
  });
});

describe("RNG primitives", () => {
  it("seedHash is deterministic, nonzero, and input-sensitive", () => {
    expect(seedHash(3, 7, 1)).toBe(seedHash(3, 7, 1));
    expect(seedHash(0, 0, 0)).not.toBe(0);
    expect(seedHash(3, 7, 1)).not.toBe(seedHash(3, 7, 2));
  });

  it("nextFloat advances a reproducible stream in [0,1)", () => {
    const a = { v: seedHash(1, 2, 3) };
    const b = { v: seedHash(1, 2, 3) };
    for (let i = 0; i < 64; i++) {
      const x = nextFloat(a);
      expect(x).toBe(nextFloat(b));
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});

describe("materials + path scene assembly", () => {
  it("material defaults emission to zero", () => {
    const m = material([0.2, 0.3, 0.4]);
    expect(m.albedo).toEqual([0.2, 0.3, 0.4]);
    expect(m.emission).toEqual([0, 0, 0]);
  });

  it("materialSoup flattens albedo + emission into parallel Float32Arrays", () => {
    const soup = materialSoup([material([1, 0, 0]), material([0, 0, 0], [5, 6, 7])]);
    expect(Array.from(soup.albedo)).toEqual([1, 0, 0, 0, 0, 0]);
    expect(Array.from(soup.emission)).toEqual([0, 0, 0, 5, 6, 7]);
  });

  it("buildPathScene assembles soup + bvh + per-triangle materials", () => {
    const tris: Vec3[][] = [
      [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
      [[-1, -1, 1], [1, -1, 1], [0, 1, 1]],
    ];
    const scene = buildPathScene(tris, [material([1, 0, 0]), material([0, 1, 0])]);
    expect(scene.soup.count).toBe(2);
    expect(scene.bvh.nodeCount).toBeGreaterThanOrEqual(1);
    expect(scene.mats.albedo).toHaveLength(2 * 3);
    expect(scene.mats.albedo[0]).toBe(1); // tri 0 red
    expect(scene.mats.albedo[4]).toBe(1); // tri 1 green (index 3+1)
  });
});
