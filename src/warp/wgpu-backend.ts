// WebGPU compute-shader backend for nv-compat/warp.
//
// Implements the actual GPU execution path the user has been asking
// for ("webgpu, wasm で実装"). Sibling of the sync `launch` in iter 76:
// `wgpuLaunch` takes a kernel function (JS, identical semantics to
// iter 76) AND a WGSL source. When `navigator.gpu` is available the
// runtime compiles WGSL, marshals WpArray data into GPU storage
// buffers, dispatches a compute pipeline, reads back. When WebGPU
// isn't available it falls back to the sync JS path so callers don't
// have to branch.
//
// Architecture:
//
//   wgpuLaunch is async (WebGPU APIs are Promise-based: requestAdapter,
//   requestDevice, createShaderModule, mapAsync). The sync `launch`
//   from iter 76 remains the default; this path is opt-in for kernels
//   that have a WGSL counterpart.
//
//   Bindings are described declaratively via `bindings: BindingSpec[]`
//   so the runtime knows how to wire each kernel arg to a GPU buffer.
//   The current spec covers the dominant case — WpArray<number> args
//   passed to the kernel get a `storage,read_write` buffer binding;
//   scalar uniforms get a `uniform` binding.
//
// Lifecycle (per launch):
//   1. Lazy-initialise: request adapter + device once, cache on module.
//   2. For each WpArray input, allocate a STORAGE | COPY_SRC | COPY_DST
//      buffer; for each scalar input, build a small uniform buffer.
//   3. writeBuffer all inputs.
//   4. Compile WGSL into a shader module (cached per WGSL source string).
//   5. Build a compute pipeline + bind group.
//   6. Dispatch with `Math.ceil(dim / workgroupSize)` workgroups.
//   7. Copy storage buffers to readback buffers; mapAsync + read back.
//   8. Write results back into the original WpArray.data fields.
//
// ADR-2605261800 §D6 nv-compat namespace localization. Trademark:
// "NVIDIA®" and "Warp®" are trademarks of NVIDIA Corporation.

import { type Kernel, type KernelFn, launch, WpArray } from "./warp.js";

// ── Feature detection ─────────────────────────────────────────────────────

/** Returns true when WebGPU is available in the current runtime
 *  (browser / Deno / Node 22+ with --experimental-webgpu / Node 26+
 *  via opt-in / wgpu-bun et al.).
 *
 *  Real WebGPU detection requires `navigator.gpu` AND a successful
 *  `requestAdapter()`. This helper only checks the synchronous surface
 *  — `acquireWebGPUDevice` does the async probe.
 */
export function hasWebGPU(): boolean {
  return (
    typeof globalThis !== "undefined" &&
    typeof (globalThis as { navigator?: { gpu?: unknown } }).navigator !== "undefined" &&
    (globalThis as { navigator: { gpu?: unknown } }).navigator.gpu !== undefined
  );
}

// ── Binding declarations ──────────────────────────────────────────────────

export type BindingKind = "storage" | "uniform";

export interface BindingSpec {
  /** Binding index in the @group(0) of the WGSL shader. */
  binding: number;
  /** Type — `storage` for WpArray inputs, `uniform` for scalar uniforms. */
  kind: BindingKind;
  /** Index into the `inputs` array this binding draws from. */
  inputIndex: number;
  /** When true, results are written back into the WpArray.data field
   *  after the launch completes. Only meaningful for storage bindings.
   *  Default: true. */
  writeback?: boolean;
}

// ── Device caching ────────────────────────────────────────────────────────

interface WebGPUMin {
  navigator: {
    gpu: {
      requestAdapter(opts?: object): Promise<{
        requestDevice(): Promise<unknown>;
      } | null>;
    };
  };
}

let _cachedDevice: unknown | null = null;

/** Request and cache a WebGPU device. Returns null if WebGPU is not
 *  available or adapter request fails. Caches on success — subsequent
 *  calls return the same device. */
export async function acquireWebGPUDevice(): Promise<unknown | null> {
  if (_cachedDevice !== null) return _cachedDevice;
  if (!hasWebGPU()) return null;
  try {
    const nav = (globalThis as unknown as WebGPUMin).navigator;
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    _cachedDevice = device;
    return device;
  } catch {
    return null;
  }
}

// ── Shader-module cache ───────────────────────────────────────────────────

const _shaderCache = new WeakMap<object, Map<string, unknown>>();

function getCachedShader(device: unknown, wgsl: string): unknown | null {
  const inner = _shaderCache.get(device as object);
  if (!inner) return null;
  return inner.get(wgsl) ?? null;
}

function setCachedShader(device: unknown, wgsl: string, mod: unknown): void {
  let inner = _shaderCache.get(device as object);
  if (!inner) {
    inner = new Map();
    _shaderCache.set(device as object, inner);
  }
  inner.set(wgsl, mod);
}

// ── wgpuKernel: ergonomic builder for dual JS+WGSL kernels ────────────────

export interface WgpuKernel extends Kernel {
  wgsl: string;
  bindings: BindingSpec[];
  workgroupSize: number;
}

/** Build a kernel with both a JS implementation and a WGSL source. The
 *  WGSL source is dispatched via wgpuLaunch; the JS impl is used by
 *  the sync `launch` path from iter 76 (and by wgpuLaunch as a fallback
 *  when no GPU is available).
 *
 *  `entryPoint` defaults to "main"; `workgroupSize` defaults to 64.
 */
export function wgpuKernel(opts: {
  js: KernelFn;
  wgsl: string;
  bindings: BindingSpec[];
  workgroupSize?: number;
}): WgpuKernel {
  const wrapper = ((...args: unknown[]) => opts.js(...args)) as WgpuKernel;
  Object.defineProperty(wrapper, "name", { value: opts.js.name || "wgpuKernel" });
  (wrapper as { fn: KernelFn }).fn = opts.js;
  wrapper.wgsl = opts.wgsl;
  wrapper.bindings = opts.bindings;
  wrapper.workgroupSize = opts.workgroupSize ?? 64;
  return wrapper;
}

// ── wgpuLaunch ────────────────────────────────────────────────────────────

interface ComputePipelineMin {
  getBindGroupLayout(idx: number): unknown;
}
interface DeviceMin {
  createBuffer(opts: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): unknown;
  createShaderModule(opts: { code: string }): unknown;
  createComputePipeline(opts: {
    layout: "auto";
    compute: { module: unknown; entryPoint: string };
  }): ComputePipelineMin;
  createBindGroup(opts: {
    layout: unknown;
    entries: { binding: number; resource: { buffer: unknown } }[];
  }): unknown;
  createCommandEncoder(): unknown;
  queue: {
    writeBuffer(buf: unknown, offset: number, data: ArrayBufferView): void;
    submit(buffers: unknown[]): void;
  };
}

/** Dispatch a kernel with WGSL compute-shader execution when WebGPU is
 *  available; otherwise fall back to the sync JS path from iter 76.
 *
 *  The WGSL source must declare bindings matching `kernel.bindings`,
 *  with `@workgroup_size(kernel.workgroupSize)` on the entry point.
 *  WpArray<number> inputs are uploaded as storage buffers; scalar
 *  number inputs are uploaded as 4-byte uniform buffers (little-endian
 *  f32).
 *
 *  After dispatch + queue.onSubmittedWorkDone, storage buffers with
 *  `writeback: true` (default) are copied back into the corresponding
 *  WpArray.data fields.
 */
export async function wgpuLaunch(opts: {
  kernel: WgpuKernel;
  dim: number;
  inputs: unknown[];
  device?: unknown;
}): Promise<void> {
  const device = (opts.device ?? (await acquireWebGPUDevice())) as DeviceMin | null;
  if (device === null) {
    // No GPU — synchronous JS fallback. Same semantics as iter 76 launch.
    launch({ kernel: opts.kernel, dim: opts.dim, inputs: opts.inputs });
    return;
  }
  // WebGPU constants (GPUBufferUsage flags from spec; redefined here so
  // we don't need a TypeScript GPU typings dep).
  const STORAGE = 0x80;
  const UNIFORM = 0x40;
  const COPY_SRC = 0x04;
  const COPY_DST = 0x08;
  const MAP_READ = 0x01;

  // Allocate GPU buffers per binding.
  interface BufferRecord {
    buffer: unknown;
    isStorage: boolean;
    bytes: number;
    inputIndex: number;
    writeback: boolean;
    arr?: WpArray<number>;
  }
  const buffers: BufferRecord[] = [];
  for (const bind of opts.kernel.bindings) {
    const input = opts.inputs[bind.inputIndex];
    if (bind.kind === "storage") {
      if (!(input instanceof WpArray)) {
        throw new Error(
          `wgpuLaunch: binding ${bind.binding} expects storage (WpArray); got ${typeof input}`,
        );
      }
      const data = new Float32Array(input.data as readonly number[]);
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: STORAGE | COPY_SRC | COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, data);
      buffers.push({
        buffer: buf,
        isStorage: true,
        bytes: data.byteLength,
        inputIndex: bind.inputIndex,
        writeback: bind.writeback ?? true,
        arr: input as WpArray<number>,
      });
    } else {
      // uniform: 4-byte float32 scalar
      if (typeof input !== "number") {
        throw new Error(
          `wgpuLaunch: binding ${bind.binding} expects uniform scalar; got ${typeof input}`,
        );
      }
      const data = new Float32Array([input]);
      const buf = device.createBuffer({
        size: 16, // uniform buffer minimum-bind size
        usage: UNIFORM | COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, data);
      buffers.push({
        buffer: buf,
        isStorage: false,
        bytes: 16,
        inputIndex: bind.inputIndex,
        writeback: false,
      });
    }
  }

  // Compile + cache shader module.
  let shader = getCachedShader(device, opts.kernel.wgsl);
  if (shader === null) {
    shader = device.createShaderModule({ code: opts.kernel.wgsl });
    setCachedShader(device, opts.kernel.wgsl, shader);
  }

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });

  // Build bind group.
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: opts.kernel.bindings.map((bind, idx) => ({
      binding: bind.binding,
      resource: { buffer: buffers[idx].buffer },
    })),
  });

  // Dispatch.
  const encoder = device.createCommandEncoder() as {
    beginComputePass(): {
      setPipeline(p: unknown): void;
      setBindGroup(idx: number, g: unknown): void;
      dispatchWorkgroups(x: number, y?: number, z?: number): void;
      end(): void;
    };
    copyBufferToBuffer(
      src: unknown,
      srcOffset: number,
      dst: unknown,
      dstOffset: number,
      size: number,
    ): void;
    finish(): unknown;
  };
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const wg = opts.kernel.workgroupSize;
  pass.dispatchWorkgroups(Math.ceil(opts.dim / wg));
  pass.end();

  // Copy writeback storage buffers to staging.
  const stagings: Array<{ staging: unknown; rec: BufferRecord }> = [];
  for (const rec of buffers) {
    if (rec.isStorage && rec.writeback) {
      const staging = device.createBuffer({
        size: rec.bytes,
        usage: MAP_READ | COPY_DST,
      });
      encoder.copyBufferToBuffer(rec.buffer, 0, staging, 0, rec.bytes);
      stagings.push({ staging, rec });
    }
  }
  device.queue.submit([encoder.finish()]);

  // Read back.
  for (const { staging, rec } of stagings) {
    // mapAsync is a Promise-returning method on GPUBuffer
    await (staging as { mapAsync(mode: number): Promise<void> }).mapAsync(MAP_READ);
    const range = (staging as { getMappedRange(): ArrayBuffer }).getMappedRange();
    const view = new Float32Array(range.slice(0));
    (staging as { unmap(): void }).unmap();
    if (rec.arr) {
      for (let i = 0; i < view.length; i++) rec.arr.data[i] = view[i];
    }
  }
}
