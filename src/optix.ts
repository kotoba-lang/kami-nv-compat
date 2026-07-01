// @etzhayyim/kami-nv-compat/optix
//
// Drop-in NVIDIA OptiX® C-style API-compat facade. Mirrors the documented
// public surface of optix.h (device context / module / program group /
// pipeline / shader binding table / launch) so existing OptiX host code
// ports to KAMI via import-path-only changes — e.g.
//
//     import * as optix from "@etzhayyim/kami-nv-compat/optix";
//
//     const ctx = optix.optixDeviceContextCreate();
//     const mod = optix.optixModuleCreateFromWGSL(ctx, mco, pco, wgsl);
//     const pg  = optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "raygen" }]);
//     const pl  = optix.optixPipelineCreate(ctx, pco, pg);
//     optix.optixLaunch(pl, sbt, { scene, camera, width, height, framebuffer });
//
// TypeScript port of the Python reference at
//   40-engine/kotoba/.../kotodama/nv_compat/optix.py
// promoted from the R1.0 no-op stub to the R1.2 backed path: `optixLaunch`
// now dispatches to kami-rt (WebGPU ray-query class WGSL traversal + CPU
// fallback) instead of returning a no-op success.
//
// Clean-room: this is a from-spec re-implementation of the OptiX *public
// API names* (Google v. Oracle, 593 U.S. ___ (2021)). No OptiX source,
// headers, PTX, or SDK binaries are used. The canonical engine has a
// distinct name — `hikari-rt` / `kami-rt` (see ./kami-rt).
//
// Trademark: NVIDIA® and OptiX® are trademarks of NVIDIA Corporation; this
// project is not affiliated with or endorsed by NVIDIA.
//
// ADR-2605261800 §D1/D6 (nv-compat facade), R1.2 OptiX surface.

import {
  type Camera,
  type Scene,
  type ShadeParams,
  traceImageCPU,
  traceImage,
} from "./kami-rt/index.js";

// ── result codes (optix.h OptixResult subset) ─────────────────────────────

export enum OptixResult {
  OPTIX_SUCCESS = 0,
  OPTIX_ERROR_INVALID_VALUE = 7001,
  OPTIX_ERROR_HOST_OUT_OF_MEMORY = 7002,
  OPTIX_ERROR_LAUNCH_FAILURE = 7050,
}

// ── compile / pipeline option records ─────────────────────────────────────

export interface OptixModuleCompileOptions {
  maxRegisterCount: number;
  optLevel: number;
  debugLevel: number;
}

export function defaultModuleCompileOptions(): OptixModuleCompileOptions {
  return { maxRegisterCount: 0, optLevel: 3, debugLevel: 0 };
}

export interface OptixPipelineCompileOptions {
  usesMotionBlur: boolean;
  traversableGraphFlags: number;
  numPayloadValues: number;
  numAttributeValues: number;
  exceptionFlags: number;
}

export function defaultPipelineCompileOptions(): OptixPipelineCompileOptions {
  return {
    usesMotionBlur: false,
    traversableGraphFlags: 0,
    numPayloadValues: 2,
    numAttributeValues: 2,
    exceptionFlags: 0,
  };
}

// ── opaque handles ────────────────────────────────────────────────────────

export type OptixProgramKind = "raygen" | "miss" | "hitgroup" | "callable";

export interface OptixDeviceContext {
  readonly logCallback?: (level: number, tag: string, message: string) => void;
  readonly logCallbackLevel: number;
  /** Internal bookkeeping (mirrors OptixDeviceContext._modules/_pipelines). */
  readonly _modules: OptixModule[];
  readonly _pipelines: OptixPipeline[];
}

export interface OptixModule {
  readonly context: OptixDeviceContext;
  readonly sourceWGSL: string;
  readonly compileOptions: OptixModuleCompileOptions;
}

export interface OptixProgramGroup {
  readonly module: OptixModule;
  readonly kind: OptixProgramKind;
}

export interface OptixPipeline {
  readonly context: OptixDeviceContext;
  readonly programGroups: OptixProgramGroup[];
  readonly compileOptions: OptixPipelineCompileOptions;
}

export interface OptixShaderBindingTable {
  raygenRecord: number;
  missRecordBase: number;
  hitgroupRecordBase: number;
}

// ── constructors (C-style, mirrors optix.py) ──────────────────────────────

/** C-style device context constructor. `cudaContext` is accepted for API
 *  parity and ignored — kami-rt targets a WebGPU device. */
export function optixDeviceContextCreate(
  _cudaContext?: unknown,
  options?: { logCallback?: OptixDeviceContext["logCallback"]; logCallbackLevel?: number },
): OptixDeviceContext {
  return {
    logCallback: options?.logCallback,
    logCallbackLevel: options?.logCallbackLevel ?? 0,
    _modules: [],
    _pipelines: [],
  };
}

/** Upstream OptiX builds modules from CUDA PTX/OptiX-IR. kami-rt has no CUDA
 *  backend; use {@link optixModuleCreateFromWGSL} instead. */
export function optixModuleCreateFromPTX(): never {
  throw new Error(
    "optixModuleCreateFromPTX requires CUDA PTX; the kami-rt backend has no " +
      "CUDA path. Use optixModuleCreateFromWGSL (KAMI-native) instead.",
  );
}

/** KAMI-native extension (not in upstream OptiX): build a module from a WGSL
 *  string. The default kami-rt raytrace kernel is used when `wgslSource` is
 *  empty. */
export function optixModuleCreateFromWGSL(
  context: OptixDeviceContext,
  moduleCompileOptions: OptixModuleCompileOptions,
  _pipelineCompileOptions: OptixPipelineCompileOptions,
  wgslSource: string,
): OptixModule {
  const mod: OptixModule = { context, sourceWGSL: wgslSource, compileOptions: moduleCompileOptions };
  context._modules.push(mod);
  return mod;
}

export function optixProgramGroupCreate(
  _context: OptixDeviceContext,
  groups: ReadonlyArray<{ module: OptixModule; kind: OptixProgramKind }>,
): OptixProgramGroup[] {
  return groups.map((g) => ({ module: g.module, kind: g.kind }));
}

export function optixPipelineCreate(
  context: OptixDeviceContext,
  pipelineCompileOptions: OptixPipelineCompileOptions,
  programGroups: ReadonlyArray<OptixProgramGroup>,
): OptixPipeline {
  const pl: OptixPipeline = {
    context,
    programGroups: [...programGroups],
    compileOptions: pipelineCompileOptions,
  };
  context._pipelines.push(pl);
  return pl;
}

export function optixShaderBindingTableCreate(
  init: Partial<OptixShaderBindingTable> = {},
): OptixShaderBindingTable {
  return {
    raygenRecord: init.raygenRecord ?? 0,
    missRecordBase: init.missRecordBase ?? 0,
    hitgroupRecordBase: init.hitgroupRecordBase ?? 0,
  };
}

// ── launch ────────────────────────────────────────────────────────────────

/** KAMI launch parameters. Upstream OptiX passes a device pointer
 *  (`launchParamsPtr`) to an opaque struct; kami-rt takes the structured
 *  scene + camera + output buffer directly. */
export interface OptixLaunchParams {
  scene: Scene;
  camera: Camera;
  width: number;
  height: number;
  /** Optional pre-allocated RGBA-float framebuffer (`width*height*4`). When
   *  omitted, {@link optixLaunchAsync} allocates and returns one. */
  framebuffer?: Float32Array;
  shade?: ShadeParams;
  /** Optional WebGPU device override (else the cached one is used). */
  device?: unknown;
}

function validateLaunch(pipeline: OptixPipeline, params: OptixLaunchParams): OptixResult | null {
  if (pipeline.programGroups.length === 0) return OptixResult.OPTIX_ERROR_INVALID_VALUE;
  if (!pipeline.programGroups.some((g) => g.kind === "raygen")) {
    return OptixResult.OPTIX_ERROR_INVALID_VALUE;
  }
  if (params.width <= 0 || params.height <= 0) return OptixResult.OPTIX_ERROR_INVALID_VALUE;
  return null;
}

/** Synchronous launch — dispatches the CPU kami-rt tracer. Returns a result
 *  code; the rendered framebuffer is written into `params.framebuffer` (and
 *  also available via {@link optixLaunchResult}). Mirrors `optixLaunch` from
 *  optix.py but actually traces (R1.2) rather than no-op (R1.0). */
export function optixLaunch(
  pipeline: OptixPipeline,
  _sbt: OptixShaderBindingTable,
  params: OptixLaunchParams,
): OptixResult {
  const bad = validateLaunch(pipeline, params);
  if (bad !== null) return bad;
  try {
    const res = traceImageCPU(params.scene, params.camera, params.width, params.height, params.shade);
    if (params.framebuffer) {
      if (params.framebuffer.length < res.framebuffer.length) {
        return OptixResult.OPTIX_ERROR_HOST_OUT_OF_MEMORY;
      }
      params.framebuffer.set(res.framebuffer);
    } else {
      (params as { framebuffer?: Float32Array }).framebuffer = res.framebuffer;
    }
    return OptixResult.OPTIX_SUCCESS;
  } catch {
    return OptixResult.OPTIX_ERROR_LAUNCH_FAILURE;
  }
}

/** Async launch — dispatches the WebGPU kami-rt tracer when a device is
 *  available, transparently falling back to CPU. Resolves to the rendered
 *  framebuffer plus the backend actually used. This is the recommended entry
 *  point in browsers. */
export async function optixLaunchAsync(
  pipeline: OptixPipeline,
  _sbt: OptixShaderBindingTable,
  params: OptixLaunchParams,
): Promise<{ result: OptixResult; framebuffer: Float32Array; backend: "webgpu" | "cpu" }> {
  const bad = validateLaunch(pipeline, params);
  if (bad !== null) return { result: bad, framebuffer: new Float32Array(0), backend: "cpu" };
  try {
    const res = await traceImage(params.scene, params.camera, params.width, params.height, {
      shade: params.shade,
      device: params.device,
    });
    if (params.framebuffer && params.framebuffer.length >= res.framebuffer.length) {
      params.framebuffer.set(res.framebuffer);
    }
    return { result: OptixResult.OPTIX_SUCCESS, framebuffer: res.framebuffer, backend: res.backend };
  } catch {
    return { result: OptixResult.OPTIX_ERROR_LAUNCH_FAILURE, framebuffer: new Float32Array(0), backend: "cpu" };
  }
}

// ── compat-map metadata ───────────────────────────────────────────────────

/** Canonical KAMI engine name behind this facade (per NV_COMPAT_MAP). */
export const KAMI_ENGINE = "hikari-rt";
export const ADR = "ADR-2605261800";
