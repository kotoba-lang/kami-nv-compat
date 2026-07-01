/**
 * nv-compat OptiX / RTX facade launch + WGSL param-packing edge cases.
 *
 * Coverage for the OptiX launch result codes (invalid dims, OOM framebuffer,
 * pre-allocated write), the option/SBT constructors, the RTX renderer defaults,
 * and the WebGPU uniform packers (pure functions, no GPU needed).
 *
 *     pnpm exec vitest run test/nv-compat-optix-rtx-facade.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.2.
 */

import { describe, it, expect } from "vitest";
import * as optix from "../src/optix.js";
import { RtxRenderMode, createRenderer, defaultRenderSettings } from "../src/rtx-renderer.js";
import {
  DEFAULT_SHADE,
  buildScene,
  buildPathScene,
  lookAt,
  material,
  packParams,
  packPathParams,
} from "../src/kami-rt/index.js";

const TRI: [number, number, number][][] = [[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]];

function pipeline() {
  const ctx = optix.optixDeviceContextCreate();
  const pco = optix.defaultPipelineCompileOptions();
  const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), pco, "");
  return optix.optixPipelineCreate(ctx, pco, optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "raygen" }]));
}

describe("OptiX launch result codes", () => {
  const scene = buildScene(TRI);
  const cam = lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1);

  it("rejects non-positive launch dimensions", () => {
    const r = optix.optixLaunch(pipeline(), optix.optixShaderBindingTableCreate(), { scene, camera: cam, width: 0, height: 8 });
    expect(r).toBe(optix.OptixResult.OPTIX_ERROR_INVALID_VALUE);
  });

  it("reports OOM when a pre-allocated framebuffer is too small", () => {
    const params: optix.OptixLaunchParams = { scene, camera: cam, width: 16, height: 16, framebuffer: new Float32Array(10) };
    expect(optix.optixLaunch(pipeline(), optix.optixShaderBindingTableCreate(), params)).toBe(
      optix.OptixResult.OPTIX_ERROR_HOST_OUT_OF_MEMORY,
    );
  });

  it("writes into a correctly-sized pre-allocated framebuffer", () => {
    const fb = new Float32Array(16 * 16 * 4);
    const params: optix.OptixLaunchParams = { scene, camera: cam, width: 16, height: 16, framebuffer: fb };
    expect(optix.optixLaunch(pipeline(), optix.optixShaderBindingTableCreate(), params)).toBe(optix.OptixResult.OPTIX_SUCCESS);
    expect(fb[3]).toBe(1); // alpha written into the caller's buffer
  });

  it("OptixResult.OPTIX_SUCCESS is 0", () => {
    expect(optix.OptixResult.OPTIX_SUCCESS).toBe(0);
  });
});

describe("OptiX option / SBT constructors", () => {
  it("default compile options have sane fields", () => {
    expect(optix.defaultModuleCompileOptions().optLevel).toBe(3);
    expect(optix.defaultPipelineCompileOptions().numPayloadValues).toBe(2);
  });

  it("shader binding table defaults + partial override", () => {
    expect(optix.optixShaderBindingTableCreate()).toEqual({ raygenRecord: 0, missRecordBase: 0, hitgroupRecordBase: 0 });
    expect(optix.optixShaderBindingTableCreate({ raygenRecord: 5 }).raygenRecord).toBe(5);
  });
});

describe("RTX renderer defaults", () => {
  it("default settings + REAL_TIME clamping", () => {
    const d = defaultRenderSettings();
    expect(d.mode).toBe(RtxRenderMode.PATH_TRACED);
    const rt = createRenderer({ mode: RtxRenderMode.REAL_TIME, samplesPerPixel: 256, maxBounces: 16 });
    const scene = rt.createScene(TRI, [material([0.8, 0.8, 0.8])]);
    const out = rt.renderSync(scene, lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1), 8, 8);
    expect(out.samplesPerPixel).toBe(4); // REAL_TIME caps spp at 4
  });

  it("async render falls back to CPU without a GPU", async () => {
    const r = createRenderer({ samplesPerPixel: 4 });
    const scene = r.createScene(TRI, [material([0.8, 0.8, 0.8])]);
    const out = await r.render(scene, lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1), 8, 8);
    expect(out.backend).toBe("cpu");
    expect(out.framebuffer.length).toBe(8 * 8 * 4);
  });
});

describe("WebGPU uniform packers", () => {
  const cam = lookAt([1, 2, 3], [0, 0, 0], [0, 1, 0], 45, 1);

  it("packParams lays out the ray-trace uniform block (32 floats)", () => {
    const p = packParams(cam, 320, 240, 7, DEFAULT_SHADE);
    expect(p.length).toBe(32);
    expect(p[0]).toBeCloseTo(1, 6); // origin.x
    expect(p[3]).toBe(320); // width in origin.w
    expect(p[7]).toBe(240); // height in lowerLeft.w
    expect(p[11]).toBe(7); // numTris in horizontal.w
  });

  it("packPathParams lays out the path-trace uniform block (20 floats)", () => {
    const p = packPathParams(cam, 320, 240, { samplesPerPixel: 8, maxBounces: 5, background: [0, 0, 0] });
    expect(p.length).toBe(20);
    expect(p[3]).toBe(320); // width
    expect(p[7]).toBe(240); // height
    expect(p[11]).toBe(8); // spp in horizontal.w
    expect(p[15]).toBe(5); // maxBounces in vertical.w
  });

  it("packPathParams clamps samplesPerPixel to ≥ 1", () => {
    const p = packPathParams(cam, 8, 8, { samplesPerPixel: 0, maxBounces: 2, background: [0, 0, 0] });
    expect(p[11]).toBe(1);
  });
});
