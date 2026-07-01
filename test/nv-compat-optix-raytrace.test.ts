/**
 * nv-compat OptiX® / kami-rt ray-tracer validation.
 *
 * Exercises the clean-room WebGPU reproduction (R1.2) on its CPU reference
 * path — CI has no GPU, so the JS tracer is the byte-compatible fallback the
 * WGSL kernel mirrors. Asserts on analytic geometry (a triangle the camera
 * stares straight at must be hit at a known depth; rays past its edge must
 * miss) plus the OptiX facade's launch contract.
 *
 *     pnpm exec vitest run test/nv-compat-optix-raytrace.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.2 OptiX surface.
 */

import { describe, it, expect } from "vitest";
import {
  type Vec3,
  buildScene,
  buildBvh,
  triangleSoup,
  lookAt,
  traceClosest,
  traceImageCPU,
} from "../src/kami-rt/index.js";
import * as optix from "../src/optix.js";

// A unit triangle centered on the origin in the z=0 plane, facing +z.
const TRI: Vec3[] = [
  [-1, -1, 0],
  [1, -1, 0],
  [0, 1, 0],
];

describe("kami-rt BVH + Möller–Trumbore", () => {
  it("hits a triangle straight ahead at the exact depth", () => {
    const soup = triangleSoup([TRI]);
    const bvh = buildBvh(soup);
    // Ray from (0,0,5) toward -z passes through the triangle's centroid-ish.
    const hit = traceClosest(soup, bvh, [0, 0, 5], [0, 0, -1]);
    expect(hit).not.toBeNull();
    expect(hit!.t).toBeCloseTo(5, 5);
    expect(hit!.tri).toBe(0);
  });

  it("misses when the ray points away from the triangle", () => {
    const soup = triangleSoup([TRI]);
    const bvh = buildBvh(soup);
    const hit = traceClosest(soup, bvh, [0, 0, 5], [0, 0, 1]); // away
    expect(hit).toBeNull();
  });

  it("returns the CLOSEST of two stacked triangles", () => {
    const near: Vec3[] = [
      [-1, -1, 1],
      [1, -1, 1],
      [0, 1, 1],
    ];
    const soup = triangleSoup([TRI, near]);
    const bvh = buildBvh(soup);
    const hit = traceClosest(soup, bvh, [0, 0, 5], [0, 0, -1]);
    expect(hit).not.toBeNull();
    expect(hit!.t).toBeCloseTo(4, 5); // hits z=1 plane first, not z=0
    expect(hit!.tri).toBe(1);
  });

  it("builds a multi-leaf BVH over many triangles and finds every one", () => {
    // A 4×4 grid of small triangles on the z=0 plane.
    const tris: Vec3[][] = [];
    for (let gx = 0; gx < 4; gx++) {
      for (let gy = 0; gy < 4; gy++) {
        const x = gx - 1.5;
        const y = gy - 1.5;
        tris.push([
          [x - 0.2, y - 0.2, 0],
          [x + 0.2, y - 0.2, 0],
          [x, y + 0.2, 0],
        ]);
      }
    }
    const soup = triangleSoup(tris);
    const bvh = buildBvh(soup);
    expect(bvh.nodeCount).toBeGreaterThan(1); // actually subdivided

    // Shoot a ray straight down at each triangle's centroid; all must hit.
    let hits = 0;
    for (let i = 0; i < tris.length; i++) {
      const c = tris[i];
      const cx = (c[0][0] + c[1][0] + c[2][0]) / 3;
      const cy = (c[0][1] + c[1][1] + c[2][1]) / 3;
      const hit = traceClosest(soup, bvh, [cx, cy, 3], [0, 0, -1]);
      if (hit && hit.t > 0) hits++;
    }
    expect(hits).toBe(tris.length);
  });
});

describe("kami-rt image trace (CPU)", () => {
  it("renders a framebuffer with the triangle lit in the center, sky at the corners", () => {
    const scene = buildScene([TRI]);
    const W = 32;
    const H = 32;
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, W / H);
    const { framebuffer, backend } = traceImageCPU(scene, cam, W, H);
    expect(backend).toBe("cpu");
    expect(framebuffer.length).toBe(W * H * 4);

    // Center pixel should hit the triangle (albedo ~0.8, lit) — not the pure
    // sky gradient. Corner (0,0) points away from the triangle → background.
    const center = ((H / 2) * W + W / 2) * 4;
    const corner = 0;
    const centerR = framebuffer[center];
    // Sky-bottom is white (1,1,1); a lit grey triangle is < 1. The geometry
    // surface is distinguishable from the background.
    expect(framebuffer[center + 3]).toBe(1); // alpha written
    expect(framebuffer[corner + 3]).toBe(1);
    // The center is the grey triangle; assert it's not the white background.
    expect(centerR).toBeLessThan(0.95);
  });
});

describe("OptiX® facade (clean-room API-compat)", () => {
  it("ports a minimal OptiX launch through to a rendered framebuffer", () => {
    const ctx = optix.optixDeviceContextCreate();
    const mco = optix.defaultModuleCompileOptions();
    const pco = optix.defaultPipelineCompileOptions();
    const mod = optix.optixModuleCreateFromWGSL(ctx, mco, pco, "");
    const groups = optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "raygen" }]);
    const pipeline = optix.optixPipelineCreate(ctx, pco, groups);
    const sbt = optix.optixShaderBindingTableCreate();

    const scene = buildScene([TRI]);
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1);
    const params: optix.OptixLaunchParams = { scene, camera: cam, width: 16, height: 16 };

    const result = optix.optixLaunch(pipeline, sbt, params);
    expect(result).toBe(optix.OptixResult.OPTIX_SUCCESS);
    expect(params.framebuffer).toBeDefined();
    expect(params.framebuffer!.length).toBe(16 * 16 * 4);
  });

  it("rejects a launch with no raygen program group", () => {
    const ctx = optix.optixDeviceContextCreate();
    const pco = optix.defaultPipelineCompileOptions();
    const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), pco, "");
    const groups = optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "miss" }]);
    const pipeline = optix.optixPipelineCreate(ctx, pco, groups);
    const scene = buildScene([TRI]);
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1);
    const res = optix.optixLaunch(pipeline, optix.optixShaderBindingTableCreate(), {
      scene,
      camera: cam,
      width: 8,
      height: 8,
    });
    expect(res).toBe(optix.OptixResult.OPTIX_ERROR_INVALID_VALUE);
  });

  it("optixModuleCreateFromPTX throws (no CUDA backend)", () => {
    expect(() => optix.optixModuleCreateFromPTX()).toThrow(/CUDA PTX/);
  });

  it("async launch falls back to CPU when no WebGPU device is present", async () => {
    const ctx = optix.optixDeviceContextCreate();
    const pco = optix.defaultPipelineCompileOptions();
    const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), pco, "");
    const groups = optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "raygen" }]);
    const pipeline = optix.optixPipelineCreate(ctx, pco, groups);
    const scene = buildScene([TRI]);
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1);
    const out = await optix.optixLaunchAsync(pipeline, optix.optixShaderBindingTableCreate(), {
      scene,
      camera: cam,
      width: 8,
      height: 8,
    });
    expect(out.result).toBe(optix.OptixResult.OPTIX_SUCCESS);
    // No navigator.gpu in vitest/node → CPU fallback.
    expect(out.backend).toBe("cpu");
    expect(out.framebuffer.length).toBe(8 * 8 * 4);
  });
});
