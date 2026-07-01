/**
 * nv-compat warp value-type + dtype edge cases.
 *
 * Coverage for the Warp-parity value types: Mat33 construction variants +
 * bounds, degenerate quaternion/normalize, dtype sentinel coercion, and the
 * rotation-only transformVector path.
 *
 *     pnpm exec vitest run test/nv-compat-warp-types.test.ts
 *
 * ADR-2605261800 §D6 (Warp → kami-warp).
 */

import { describe, it, expect } from "vitest";
import * as wp from "../src/warp/index.js";

const close = (a: readonly number[], b: readonly number[], p = 5) =>
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i], p));

describe("Mat33 construction variants", () => {
  it("zero-arg → all zeros", () => {
    expect(new wp.Mat33().rows).toEqual([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
  });

  it("9 scalar args and a flat-9 array both fill row-major", () => {
    const a = new wp.Mat33(1, 2, 3, 4, 5, 6, 7, 8, 9);
    const b = new wp.Mat33([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(a.rows).toEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(b.rows).toEqual(a.rows);
  });

  it("nested 3×3 array is copied", () => {
    const m = new wp.Mat33([[1, 0, 0], [0, 2, 0], [0, 0, 3]]);
    expect(m.get(1, 1)).toBe(2);
    expect(m.get(0)).toEqual([1, 0, 0]); // row accessor
  });

  it("set mutates an element", () => {
    const m = new wp.Mat33();
    m.set(2, 1, 7);
    expect(m.get(2, 1)).toBe(7);
  });

  it("throws on an unsupported argument count", () => {
    expect(() => new (wp.Mat33 as unknown as new (...a: number[]) => unknown)(1, 2)).toThrow(/expected 0 \/ 1 \/ 9/);
  });
});

describe("degenerate quaternion + normalize", () => {
  it("quatFromAxisAngle with a zero axis returns identity", () => {
    expect(wp.quatFromAxisAngle([0, 0, 0], 1).toArray()).toEqual([0, 0, 0, 1]);
  });

  it("normalizing a zero vector is safe", () => {
    expect(wp.normalize(new wp.Vec3(0, 0, 0)).toArray()).toEqual([0, 0, 0]);
    expect(wp.normalize(new wp.Vec4(0, 0, 0, 0)).toArray()).toEqual([0, 0, 0, 0]);
    expect(wp.normalize(new wp.Quat(0, 0, 0, 0)).toArray()).toEqual([0, 0, 0, 1]);
  });

  it("lengthSq is the squared magnitude", () => {
    expect(wp.lengthSq([3, 4, 0])).toBeCloseTo(25, 6);
  });
});

describe("dtype sentinels", () => {
  it("float keeps fractional, int truncates, bool coerces", () => {
    expect(wp.float32(2.7)).toBe(2.7);
    expect(wp.float64(-1.25)).toBe(-1.25);
    expect(wp.int32(2.9)).toBe(2);
    expect(wp.int32(-2.9)).toBe(-2);
    expect(wp.uint32(5.8)).toBe(5);
    expect(wp.boolDtype(1)).toBe(true);
    expect(wp.boolDtype(0)).toBe(false);
  });
});

describe("transform vector vs point", () => {
  it("transformVector rotates but ignores translation", () => {
    const t = new wp.Transform(new wp.Vec3(5, 0, 0), wp.quatFromAxisAngle([0, 0, 1], Math.PI / 2));
    close(wp.transformVector(t, [1, 0, 0]).toArray(), [0, 1, 0]); // no +5 translation
    close(wp.transformGetRotation(t).toArray(), wp.quatFromAxisAngle([0, 0, 1], Math.PI / 2).toArray());
  });
});

describe("kernel helpers are no-ops / pass-through", () => {
  it("func returns the same function; init + config do not throw", () => {
    const f = (x: number) => x * 2;
    expect(wp.func(f)(3)).toBe(6);
    expect(() => wp.init()).not.toThrow();
    expect(wp.config.mode).toBeDefined();
  });
});
