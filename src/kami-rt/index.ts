// @etzhayyim/kami-nv-compat/kami-rt
//
// Clean-room ray-tracing engine — the canonical KAMI implementation that
// backs the `nv-compat/optix` API-compat facade. Reproduces an OptiX-style
// launch (raygen → BVH → closest-hit / miss → framebuffer) on WebGPU + WGSL
// with a byte-compatible CPU fallback.
//
// Two entry points, both producing an identical RGBA-float framebuffer:
//   - traceImageSync : CPU, runs everywhere (Node + browsers).
//   - traceImage     : WebGPU when available, transparently falls back to CPU.
//
// `Scene` bundles the triangle soup + its BVH so callers build the
// acceleration structure once and trace many frames.
//
// ADR-2605261800 §D6 nv-compat namespace localization; D10.4 kami-rt
// (canonical name for the OptiX-compat path "hikari-rt").

import {
  type Bvh,
  type Camera,
  type ShadeParams,
  type TriangleSoup,
  type Vec3,
  buildBvh,
  triangleSoup,
} from "./bvh.js";
import { traceImageGPU } from "./wgpu-raytrace.js";
import { DEFAULT_SHADE, traceImageSync } from "./bvh.js";

export {
  type Vec3,
  type Camera,
  type ShadeParams,
  type TriangleSoup,
  type Bvh,
  type Hit,
  buildBvh,
  triangleSoup,
  lookAt,
  traceClosest,
  traceImageSync,
  triNormal,
  DEFAULT_SHADE,
  NODE_STRIDE,
} from "./bvh.js";
export { traceImageGPU, packParams } from "./wgpu-raytrace.js";
export { RAYTRACE_WGSL } from "./raytrace-wgsl.js";

// kami-rtx path tracer (the rtx-renderer.ts backend).
export {
  type Material,
  type MaterialSoup,
  type PathScene,
  type PathSettings,
  type Rng,
  material,
  materialSoup,
  buildPathScene,
  pathTraceSync,
  seedHash,
  nextFloat,
  onb,
  DEFAULT_PATH_SETTINGS,
} from "./pathtrace.js";
export { pathTraceGPU, packPathParams } from "./wgpu-pathtrace.js";
export { PATHTRACE_WGSL } from "./pathtrace-wgsl.js";

import {
  type PathScene,
  type PathSettings,
  DEFAULT_PATH_SETTINGS,
  pathTraceSync,
} from "./pathtrace.js";
import { pathTraceGPU } from "./wgpu-pathtrace.js";

/** Progressive path-trace `scene` from `cam`. WebGPU when available (or a
 *  `device` is passed), else CPU. Returns the RGBA-float framebuffer. */
export async function pathTrace(
  scene: PathScene,
  cam: Camera,
  width: number,
  height: number,
  opts: { settings?: PathSettings; device?: unknown } = {},
): Promise<TraceResult> {
  const { framebuffer, backend } = await pathTraceGPU(
    scene,
    cam,
    width,
    height,
    opts.settings ?? DEFAULT_PATH_SETTINGS,
    opts.device,
  );
  return { framebuffer, backend, width, height };
}

/** Synchronous CPU path trace. */
export function pathTraceCPU(
  scene: PathScene,
  cam: Camera,
  width: number,
  height: number,
  settings: PathSettings = DEFAULT_PATH_SETTINGS,
): TraceResult {
  return {
    framebuffer: pathTraceSync(scene, cam, width, height, settings),
    backend: "cpu",
    width,
    height,
  };
}

/** A traceable scene: triangle soup + its acceleration structure. */
export interface Scene {
  soup: TriangleSoup;
  bvh: Bvh;
}

/** Build a {@link Scene} from triangles (each a `[v0, v1, v2]` of `Vec3`). */
export function buildScene(triangles: readonly Vec3[][]): Scene {
  const soup = triangleSoup(triangles);
  return { soup, bvh: buildBvh(soup) };
}

export interface TraceResult {
  framebuffer: Float32Array;
  backend: "webgpu" | "cpu";
  width: number;
  height: number;
}

/** Render `scene` from `cam` into a `width × height` RGBA-float framebuffer.
 *  Uses WebGPU when a device is available (or one is passed via `device`),
 *  otherwise the CPU tracer. */
export async function traceImage(
  scene: Scene,
  cam: Camera,
  width: number,
  height: number,
  opts: { shade?: ShadeParams; device?: unknown } = {},
): Promise<TraceResult> {
  const shade = opts.shade ?? DEFAULT_SHADE;
  const { framebuffer, backend } = await traceImageGPU(
    scene.soup,
    scene.bvh,
    cam,
    width,
    height,
    shade,
    opts.device,
  );
  return { framebuffer, backend, width, height };
}

/** Synchronous CPU render — convenience wrapper over {@link traceImageSync}. */
export function traceImageCPU(
  scene: Scene,
  cam: Camera,
  width: number,
  height: number,
  shade: ShadeParams = DEFAULT_SHADE,
): TraceResult {
  return {
    framebuffer: traceImageSync(scene.soup, scene.bvh, cam, width, height, shade),
    backend: "cpu",
    width,
    height,
  };
}
