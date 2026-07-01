/**
 * nv-compat utsushimi distribution statistical validation.
 *
 * Over many deterministic draws, the DR distributions must hit their declared
 * statistics: uniform bounds + mean, normal mean + std, truncated-normal
 * range, choice coverage, sequence cycling, multi-dim + combine.
 *
 *     pnpm exec vitest run test/nv-compat-distribution-stats.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 utsushimi.
 */

import { describe, it, expect } from "vitest";
import { Sampler, distribution, sample } from "../src/utsushimi/index.js";

function stats(xs: number[]): { mean: number; std: number; min: number; max: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const std = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length);
  return { mean, std, min: Math.min(...xs), max: Math.max(...xs) };
}

describe("uniform", () => {
  it("stays in bounds with a centered mean", () => {
    const s = new Sampler(1);
    const xs = Array.from({ length: 4000 }, () => (sample(distribution.uniform([0], [10]), s) as number[])[0]);
    const { mean, min, max } = stats(xs);
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThan(10);
    expect(mean).toBeCloseTo(5, 0); // ≈ midpoint
  });

  it("handles multi-dimensional ranges per component", () => {
    const s = new Sampler(2);
    for (let i = 0; i < 100; i++) {
      const v = sample(distribution.uniform([0, 10, -5], [1, 20, 5]), s) as number[];
      expect(v[0]).toBeGreaterThanOrEqual(0);
      expect(v[0]).toBeLessThan(1);
      expect(v[1]).toBeGreaterThanOrEqual(10);
      expect(v[1]).toBeLessThan(20);
      expect(v[2]).toBeGreaterThanOrEqual(-5);
      expect(v[2]).toBeLessThan(5);
    }
  });
});

describe("normal", () => {
  it("approaches the requested mean and std over many draws", () => {
    const s = new Sampler(7);
    const xs = Array.from({ length: 6000 }, () => (sample(distribution.normal([5], [2]), s) as number[])[0]);
    const { mean, std } = stats(xs);
    expect(mean).toBeCloseTo(5, 1);
    expect(std).toBeCloseTo(2, 1);
  });
});

describe("truncated_normal", () => {
  it("never leaves [low, high] and stays roughly centered", () => {
    const s = new Sampler(9);
    const xs = Array.from(
      { length: 3000 },
      () => (sample(distribution.truncated_normal([0], [5], [-1], [1]), s) as number[])[0],
    );
    const { mean, min, max } = stats(xs);
    expect(min).toBeGreaterThanOrEqual(-1);
    expect(max).toBeLessThanOrEqual(1);
    expect(Math.abs(mean)).toBeLessThan(0.1); // symmetric around 0
  });
});

describe("choice + sequence", () => {
  it("choice eventually covers every option", () => {
    const s = new Sampler(11);
    const seen = new Set<unknown>();
    for (let i = 0; i < 500; i++) seen.add(sample(distribution.choice(["a", "b", "c", "d"]), s));
    expect([...seen].sort()).toEqual(["a", "b", "c", "d"]);
  });

  it("sequence cycles deterministically modulo length", () => {
    const s = new Sampler(0);
    const seq = distribution.sequence([1, 2, 3]);
    expect(Array.from({ length: 7 }, () => sample(seq, s))).toEqual([1, 2, 3, 1, 2, 3, 1]);
  });
});

describe("combine", () => {
  it("concatenates independent draws (uniform + tight normal)", () => {
    const s = new Sampler(3);
    const v = sample(distribution.combine([distribution.uniform([0], [1]), distribution.normal([100], [0.001])]), s) as number[];
    expect(v).toHaveLength(2);
    expect(v[0]).toBeGreaterThanOrEqual(0);
    expect(v[0]).toBeLessThan(1);
    expect(v[1]).toBeCloseTo(100, 1); // tight std → very near the mean
  });
});
