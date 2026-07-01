// kami-rtx WebGPU dispatch for the Monte-Carlo path tracer.
//
// Uploads geometry + BVH + per-triangle albedo/emission as storage buffers,
// camera + settings as a uniform, dispatches one thread per pixel running
// `spp` paths, reads the RGBA-float framebuffer back. Falls back transparently
// to the CPU tracer (`pathTraceSync`) when no WebGPU device is available.
//
// Device acquisition + structural GPU typings reuse the warp wgpu-backend so
// there is one cached device per runtime and no @webgpu/types dependency.
//
// ADR-2605261800 §D6 / D10.4 kami-rtx.

import { acquireWebGPUDevice } from "../warp/wgpu-backend.js";
import { type Camera } from "./bvh.js";
import {
  type PathScene,
  type PathSettings,
  DEFAULT_PATH_SETTINGS,
  pathTraceSync,
} from "./pathtrace.js";
import { PATHTRACE_WGSL } from "./pathtrace-wgsl.js";

interface DeviceMin {
  createBuffer(opts: { size: number; usage: number }): { getMappedRange(): ArrayBuffer; unmap(): void };
  createShaderModule(opts: { code: string }): unknown;
  createComputePipeline(opts: {
    layout: "auto";
    compute: { module: unknown; entryPoint: string };
  }): { getBindGroupLayout(idx: number): unknown };
  createBindGroup(opts: {
    layout: unknown;
    entries: { binding: number; resource: { buffer: unknown } }[];
  }): unknown;
  createCommandEncoder(): {
    beginComputePass(): {
      setPipeline(p: unknown): void;
      setBindGroup(idx: number, g: unknown): void;
      dispatchWorkgroups(x: number, y?: number, z?: number): void;
      end(): void;
    };
    copyBufferToBuffer(s: unknown, so: number, d: unknown, dof: number, n: number): void;
    finish(): unknown;
  };
  queue: {
    writeBuffer(buf: unknown, offset: number, data: ArrayBufferView): void;
    submit(buffers: unknown[]): void;
  };
}

const STORAGE = 0x80;
const UNIFORM = 0x40;
const COPY_SRC = 0x04;
const COPY_DST = 0x08;
const MAP_READ = 0x01;

/** Pack the 5×vec4 uniform block consumed by PATHTRACE_WGSL's `Params`. */
export function packPathParams(
  cam: Camera,
  width: number,
  height: number,
  settings: PathSettings,
): Float32Array {
  // prettier-ignore
  return new Float32Array([
    cam.origin[0],     cam.origin[1],     cam.origin[2],     width,
    cam.lowerLeft[0],  cam.lowerLeft[1],  cam.lowerLeft[2],  height,
    cam.horizontal[0], cam.horizontal[1], cam.horizontal[2], Math.max(1, settings.samplesPerPixel),
    cam.vertical[0],   cam.vertical[1],   cam.vertical[2],   settings.maxBounces,
    settings.background[0], settings.background[1], settings.background[2], 0,
  ]);
}

let _pipelineCache: WeakMap<object, unknown> | null = null;

/** Path-trace `width × height` on the GPU via WebGPU. Resolves to the CPU
 *  result when no device is available. */
export async function pathTraceGPU(
  scene: PathScene,
  cam: Camera,
  width: number,
  height: number,
  settings: PathSettings = DEFAULT_PATH_SETTINGS,
  deviceOverride?: unknown,
): Promise<{ framebuffer: Float32Array; backend: "webgpu" | "cpu" }> {
  const device = (deviceOverride ?? (await acquireWebGPUDevice())) as DeviceMin | null;
  if (device === null) {
    return { framebuffer: pathTraceSync(scene, cam, width, height, settings), backend: "cpu" };
  }

  const fbBytes = width * height * 4 * 4;
  const mk = (data: ArrayBufferView, usage: number): unknown => {
    const buf = device.createBuffer({ size: Math.max(data.byteLength, 4), usage });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  };

  const triBuf = mk(scene.soup.verts, STORAGE | COPY_DST);
  const nodeBuf = mk(scene.bvh.nodes, STORAGE | COPY_DST);
  const idxBuf = mk(scene.bvh.triIndex, STORAGE | COPY_DST);
  const albBuf = mk(scene.mats.albedo, STORAGE | COPY_DST);
  const emiBuf = mk(scene.mats.emission, STORAGE | COPY_DST);
  const fbBuf = device.createBuffer({ size: fbBytes, usage: STORAGE | COPY_SRC });
  const uniBuf = mk(packPathParams(cam, width, height, settings), UNIFORM | COPY_DST);

  if (_pipelineCache === null) _pipelineCache = new WeakMap();
  let pipeline = _pipelineCache.get(device as object) as
    | { getBindGroupLayout(i: number): unknown }
    | undefined;
  if (!pipeline) {
    const module = device.createShaderModule({ code: PATHTRACE_WGSL });
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    _pipelineCache.set(device as object, pipeline);
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: triBuf } },
      { binding: 1, resource: { buffer: nodeBuf } },
      { binding: 2, resource: { buffer: idxBuf } },
      { binding: 3, resource: { buffer: albBuf } },
      { binding: 4, resource: { buffer: emiBuf } },
      { binding: 5, resource: { buffer: fbBuf } },
      { binding: 6, resource: { buffer: uniBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();

  const staging = device.createBuffer({ size: fbBytes, usage: MAP_READ | COPY_DST });
  encoder.copyBufferToBuffer(fbBuf, 0, staging, 0, fbBytes);
  device.queue.submit([encoder.finish()]);

  await (staging as unknown as { mapAsync(m: number): Promise<void> }).mapAsync(MAP_READ);
  const framebuffer = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();

  return { framebuffer, backend: "webgpu" };
}
