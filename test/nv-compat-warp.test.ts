/**
 * nv-compat warp (NVIDIA Warp port) math + kernel primitives.
 *
 * Direct coverage for the Warp-parity helpers that the cross-validation suite
 * only exercises indirectly: Vec3 algebra, the sequential launch grid + tid,
 * vector/quaternion/transform math, atomics, and WpArray containers.
 *
 *     pnpm exec vitest run test/nv-compat-warp.test.ts
 *
 * ADR-2605261800 §D6 (Warp → kami-warp).
 */

import { describe, it, expect } from "vitest";
import * as wp from "../src/warp/index.js";

const close = (a: readonly number[], b: readonly number[], p = 5) =>
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i], p));

describe("Vec3 algebra", () => {
  it("add / sub / mul / neg", () => {
    const a = new wp.Vec3(1, 2, 3);
    expect(a.add(new wp.Vec3(1, 1, 1)).toArray()).toEqual([2, 3, 4]);
    expect(a.sub([1, 0, 1]).toArray()).toEqual([0, 2, 2]);
    expect(a.mul(2).toArray()).toEqual([2, 4, 6]);
    expect(a.neg().toArray()).toEqual([-1, -2, -3]);
  });
});

describe("launch grid + tid", () => {
  it("runs the kernel prod(dim) times with linear tid", () => {
    const arr = wp.zeros<number>(5);
    const k = wp.kernel((a: wp.WpArray<number>) => a.set(wp.tid(), wp.tid()));
    wp.launch({ kernel: k, dim: 5, inputs: [arr] });
    expect(arr.toArray()).toEqual([0, 1, 2, 3, 4]);
  });

  it("tid() outside a launch throws", () => {
    expect(() => wp.tid()).toThrow(/tid/);
  });

  it("multi-dim launch runs prod(dim) iterations", () => {
    let count = 0;
    const k = wp.kernel(() => {
      count++;
    });
    wp.launch({ kernel: k, dim: [2, 3], inputs: [] });
    expect(count).toBe(6);
  });
});

describe("vector math", () => {
  it("dot / cross / length / normalize", () => {
    expect(wp.dot([1, 2, 3], [4, 5, 6])).toBe(32);
    close(wp.cross(new wp.Vec3(1, 0, 0), new wp.Vec3(0, 1, 0)).toArray(), [0, 0, 1]);
    expect(wp.length([3, 4, 0])).toBe(5);
    close(wp.normalize(new wp.Vec3(0, 3, 0)).toArray(), [0, 1, 0]);
  });

  it("scalar + vector min/max and clamp", () => {
    expect(wp.min(3, 5)).toBe(3);
    expect(wp.max(3, 5)).toBe(5);
    close((wp.min(new wp.Vec3(1, 5, 2), new wp.Vec3(3, 1, 4)) as wp.Vec3).toArray(), [1, 1, 2]);
    expect(wp.clamp(10, 0, 5)).toBe(5);
    expect(wp.clamp(-1, 0, 5)).toBe(0);
  });
});

describe("quaternion math", () => {
  it("axis-angle rotation, multiply, inverse round-trip", () => {
    const q = wp.quatFromAxisAngle([0, 0, 1], Math.PI / 2);
    close(wp.quatRotate(q, [1, 0, 0]).toArray(), [0, 1, 0]); // 90° about z
    close(wp.quatRotateInv(q, wp.quatRotate(q, [1, 0, 0])).toArray(), [1, 0, 0]); // inverse undoes it
    const id = wp.quatMul(q, wp.quatInverse(q));
    close(id.toArray(), wp.quatIdentity().toArray());
  });
});

describe("transform math", () => {
  it("transformPoint applies rotation + translation; multiply composes", () => {
    const t = new wp.Transform(new wp.Vec3(5, 0, 0), wp.quatFromAxisAngle([0, 0, 1], Math.PI / 2));
    close(wp.transformPoint(t, [1, 0, 0]).toArray(), [5, 1, 0]); // rotate (1,0,0)→(0,1,0), +translate
    const composed = wp.transformMultiply(wp.transformIdentity(), t);
    close(wp.transformGetTranslation(composed).toArray(), [5, 0, 0]);
  });
});

describe("atomics + WpArray containers", () => {
  it("atomicAdd / Sub / Max / Min mutate and return the old value", () => {
    const a = wp.fromTypedArray<number>([10, 10, 10, 10]);
    expect(wp.atomicAdd(a, 0, 5)).toBe(10);
    expect(a.get(0)).toBe(15);
    wp.atomicSub(a, 1, 3);
    expect(a.get(1)).toBe(7);
    wp.atomicMax(a, 2, 20);
    expect(a.get(2)).toBe(20);
    wp.atomicMin(a, 3, 4);
    expect(a.get(3)).toBe(4);
  });

  it("zeros / fill / assign / indexOf", () => {
    const z = wp.zeros<number>(3);
    expect(z.toArray()).toEqual([0, 0, 0]);
    z.fill(9);
    expect(z.toArray()).toEqual([9, 9, 9]);
    z.assign([1, 2, 3]);
    expect(wp.indexOf(z, 1)).toBe(2);
    expect(() => z.assign([1, 2])).toThrow(/length mismatch/);
  });
});
