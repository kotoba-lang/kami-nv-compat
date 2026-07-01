// kami-rt WebGPU dispatch path.
//
// Compiles RAYTRACE_WGSL, uploads the scene (tris / BVH nodes / triIdx) as
// storage buffers + camera/shade as a uniform, dispatches one thread per
// pixel, reads the RGBA float framebuffer back. Falls back transparently to
// the CPU tracer (`traceImageSync`) when no WebGPU device is available, so
// callers never branch — same contract the warp wgpu-backend established.
//
// Device acquisition + the structural GPU typings are reused from
// nv-compat/warp/wgpu-backend so there is one cached device per runtime and
// no @webgpu/types dependency.
//
// ADR-2605261800 §D6 / D10.4 kami-rt.

import { acquireWebGPUDevice } from "../warp/wgpu-backend.js";
import {
  type Bvh,
  type Camera,
  type ShadeParams,
  type TriangleSoup,
  DEFAULT_SHADE,
  traceImageSync,
} from "./bvh.js";
import { RAYTRACE_WGSL } from "./raytrace-wgsl.js";

// ── structural GPU surface (subset; mirrors warp/wgpu-backend) ─────────────

interface DeviceMin {
  createBuffer(opts: { size: number; usage: number; mappedAtCreation?: boolean }): {
    getMappedRange(): ArrayBuffer;
    unmap(): void;
  };
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

// GPUBufferUsage flags (spec constants; redeclared to avoid a typings dep).
const STORAGE = 0x80;
const UNIFORM = 0x40;
const COPY_SRC = 0x04;
const COPY_DST = 0x08;
const MAP_READ = 0x01;

/** Pack the 8×vec4 uniform block consumed by RAYTRACE_WGSL's `Params`. */
export function packParams(
  cam: Camera,
  width: number,
  height: number,
  numTris: number,
  shade: ShadeParams,
): Float32Array {
  // prettier-ignore
  return new Float32Array([
    cam.origin[0],     cam.origin[1],     cam.origin[2],     width,
    cam.lowerLeft[0],  cam.lowerLeft[1],  cam.lowerLeft[2],  height,
    cam.horizontal[0], cam.horizontal[1], cam.horizontal[2], numTris,
    cam.vertical[0],   cam.vertical[1],   cam.vertical[2],   0,
    shade.lightDir[0], shade.lightDir[1], shade.lightDir[2], shade.ambient,
    shade.albedo[0],   shade.albedo[1],   shade.albedo[2],   0,
    shade.bgTop[0],    shade.bgTop[1],    shade.bgTop[2],    0,
    shade.bgBottom[0], shade.bgBottom[1], shade.bgBottom[2], 0,
  ]);
}

let _pipelineCache: WeakMap<object, unknown> | null = null;

/** Render `width × height` RGBA-float framebuffer on the GPU via WebGPU.
 *  Resolves to the CPU result when no device is available. */
export async function traceImageGPU(
  soup: TriangleSoup,
  bvh: Bvh,
  cam: Camera,
  width: number,
  height: number,
  shade: ShadeParams = DEFAULT_SHADE,
  deviceOverride?: unknown,
): Promise<{ framebuffer: Float32Array; backend: "webgpu" | "cpu" }> {
  const device = (deviceOverride ?? (await acquireWebGPUDevice())) as DeviceMin | null;
  if (device === null) {
    return { framebuffer: traceImageSync(soup, bvh, cam, width, height, shade), backend: "cpu" };
  }

  const fbBytes = width * height * 4 * 4;

  const triBuf = device.createBuffer({ size: Math.max(soup.verts.byteLength, 4), usage: STORAGE | COPY_DST });
  device.queue.writeBuffer(triBuf, 0, soup.verts);

  const nodeBuf = device.createBuffer({ size: Math.max(bvh.nodes.byteLength, 4), usage: STORAGE | COPY_DST });
  device.queue.writeBuffer(nodeBuf, 0, bvh.nodes);

  const idxBuf = device.createBuffer({ size: Math.max(bvh.triIndex.byteLength, 4), usage: STORAGE | COPY_DST });
  device.queue.writeBuffer(idxBuf, 0, bvh.triIndex);

  const fbBuf = device.createBuffer({ size: fbBytes, usage: STORAGE | COPY_SRC });

  const params = packParams(cam, width, height, soup.count, shade);
  const uniBuf = device.createBuffer({ size: params.byteLength, usage: UNIFORM | COPY_DST });
  device.queue.writeBuffer(uniBuf, 0, params);

  // Cache the compiled pipeline per device (WGSL source is constant).
  if (_pipelineCache === null) _pipelineCache = new WeakMap();
  let pipeline = _pipelineCache.get(device as object) as
    | { getBindGroupLayout(i: number): unknown }
    | undefined;
  if (!pipeline) {
    const module = device.createShaderModule({ code: RAYTRACE_WGSL });
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    _pipelineCache.set(device as object, pipeline);
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: triBuf } },
      { binding: 1, resource: { buffer: nodeBuf } },
      { binding: 2, resource: { buffer: idxBuf } },
      { binding: 3, resource: { buffer: fbBuf } },
      { binding: 4, resource: { buffer: uniBuf } },
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
