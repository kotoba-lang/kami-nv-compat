/**
 * nv-compat OptiX facade internals.
 *
 * Coverage for the OptiX C-style bookkeeping not hit by the render tests:
 * device-context module/pipeline registration, the PTX-unsupported throw,
 * program-group kind mapping, log-callback options, and optixLaunchAsync
 * (success + no-raygen error).
 *
 *     pnpm exec vitest run test/nv-compat-optix-internals.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.2 OptiX surface.
 */

import { describe, it, expect } from "vitest";
import * as optix from "../src/optix.js";
import { type Vec3, buildScene, lookAt } from "../src/kami-rt/index.js";

const scene = buildScene([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]] as Vec3[]]);
const cam = lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1);

describe("device context bookkeeping", () => {
  it("starts with empty module/pipeline registries", () => {
    const ctx = optix.optixDeviceContextCreate();
    expect(ctx._modules).toEqual([]);
    expect(ctx._pipelines).toEqual([]);
    expect(ctx.logCallbackLevel).toBe(0);
  });

  it("registers modules and pipelines on the context", () => {
    const ctx = optix.optixDeviceContextCreate();
    const pco = optix.defaultPipelineCompileOptions();
    const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), pco, "");
    expect(ctx._modules).toHaveLength(1);
    optix.optixPipelineCreate(ctx, pco, optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "raygen" }]));
    expect(ctx._pipelines).toHaveLength(1);
  });

  it("stores a log callback + level from options", () => {
    const calls: string[] = [];
    const ctx = optix.optixDeviceContextCreate(undefined, {
      logCallback: (_lvl, _tag, msg) => calls.push(msg),
      logCallbackLevel: 3,
    });
    expect(ctx.logCallbackLevel).toBe(3);
    ctx.logCallback?.(3, "tag", "hello");
    expect(calls).toEqual(["hello"]);
  });
});

describe("module + program group construction", () => {
  it("optixModuleCreateFromPTX is unsupported (no CUDA backend)", () => {
    expect(() => optix.optixModuleCreateFromPTX()).toThrow(/CUDA PTX|WGSL/);
  });

  it("optixProgramGroupCreate maps each kind through", () => {
    const ctx = optix.optixDeviceContextCreate();
    const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), optix.defaultPipelineCompileOptions(), "");
    const groups = optix.optixProgramGroupCreate(ctx, [
      { module: mod, kind: "raygen" },
      { module: mod, kind: "miss" },
      { module: mod, kind: "hitgroup" },
    ]);
    expect(groups.map((g) => g.kind)).toEqual(["raygen", "miss", "hitgroup"]);
    expect(groups[0].module).toBe(mod);
  });
});

describe("optixLaunchAsync", () => {
  function pipeline(kind: optix.OptixProgramKind) {
    const ctx = optix.optixDeviceContextCreate();
    const pco = optix.defaultPipelineCompileOptions();
    const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), pco, "");
    return optix.optixPipelineCreate(ctx, pco, optix.optixProgramGroupCreate(ctx, [{ module: mod, kind }]));
  }

  it("resolves SUCCESS on the CPU backend with a framebuffer", async () => {
    const out = await optix.optixLaunchAsync(pipeline("raygen"), optix.optixShaderBindingTableCreate(), {
      scene,
      camera: cam,
      width: 8,
      height: 8,
    });
    expect(out.result).toBe(optix.OptixResult.OPTIX_SUCCESS);
    expect(out.backend).toBe("cpu");
    expect(out.framebuffer.length).toBe(8 * 8 * 4);
  });

  it("rejects a pipeline without a raygen program", async () => {
    const out = await optix.optixLaunchAsync(pipeline("miss"), optix.optixShaderBindingTableCreate(), {
      scene,
      camera: cam,
      width: 8,
      height: 8,
    });
    expect(out.result).toBe(optix.OptixResult.OPTIX_ERROR_INVALID_VALUE);
    expect(out.framebuffer.length).toBe(0);
  });
});

describe("compat-map metadata", () => {
  it("reports the canonical hikari-rt engine + ADR", () => {
    expect(optix.KAMI_ENGINE).toBe("hikari-rt");
    expect(optix.ADR).toBe("ADR-2605261800");
  });
});
