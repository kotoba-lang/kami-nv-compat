/**
 * nv-compat warp WebGPU-backend CPU fallback.
 *
 * Coverage for the dual-path WebGPU dispatch under a GPU-less runtime: feature
 * detection, device acquisition returning null, the wgpuKernel builder, and
 * wgpuLaunch transparently falling back to the synchronous JS launch (same
 * semantics) so callers never have to branch.
 *
 *     pnpm exec vitest run test/nv-compat-warp-wgpu-fallback.test.ts
 *
 * ADR-2605261800 §D6 (Warp WebGPU backend).
 */

import { describe, it, expect } from "vitest";
import {
  acquireWebGPUDevice,
  fromTypedArray,
  hasWebGPU,
  tid,
  wgpuKernel,
  wgpuLaunch,
  type WpArray,
} from "../src/warp/index.js";

describe("feature detection without a GPU (Node)", () => {
  it("hasWebGPU() is false when navigator.gpu is absent", () => {
    expect(hasWebGPU()).toBe(false);
  });

  it("acquireWebGPUDevice() resolves to null", async () => {
    expect(await acquireWebGPUDevice()).toBeNull();
  });
});

describe("wgpuKernel builder", () => {
  it("carries the WGSL, bindings, and a default workgroup size", () => {
    const k = wgpuKernel({
      js: () => {},
      wgsl: "@compute @workgroup_size(64) fn main() {}",
      bindings: [{ binding: 0, kind: "storage", inputIndex: 0 }],
    });
    expect(k.wgsl).toContain("@compute");
    expect(k.bindings).toHaveLength(1);
    expect(k.workgroupSize).toBe(64);
  });

  it("honors a custom workgroup size", () => {
    const k = wgpuKernel({ js: () => {}, wgsl: "x", bindings: [], workgroupSize: 32 });
    expect(k.workgroupSize).toBe(32);
  });
});

describe("wgpuLaunch falls back to the JS path", () => {
  it("runs the JS kernel over the grid and writes back the storage buffer", async () => {
    const arr = fromTypedArray<number>([1, 2, 3, 4]);
    const k = wgpuKernel({
      js: (a: WpArray<number>) => a.set(tid(), a.get(tid()) * 2),
      wgsl: "@compute @workgroup_size(64) fn main() {}",
      bindings: [{ binding: 0, kind: "storage", inputIndex: 0 }],
    });
    await wgpuLaunch({ kernel: k, dim: 4, inputs: [arr] });
    expect(arr.toArray()).toEqual([2, 4, 6, 8]); // doubled via the sync fallback
  });

  it("passes scalar inputs through to the JS kernel on the fallback path", async () => {
    const arr = fromTypedArray<number>([10, 20, 30]);
    const k = wgpuKernel({
      js: (a: WpArray<number>, k: number) => a.set(tid(), a.get(tid()) + k),
      wgsl: "x",
      bindings: [
        { binding: 0, kind: "storage", inputIndex: 0 },
        { binding: 1, kind: "uniform", inputIndex: 1 },
      ],
    });
    await wgpuLaunch({ kernel: k, dim: 3, inputs: [arr, 5] });
    expect(arr.toArray()).toEqual([15, 25, 35]);
  });
});
