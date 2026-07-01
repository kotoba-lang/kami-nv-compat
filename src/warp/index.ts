// @etzhayyim/kami-nv-compat/warp
//
// TypeScript port of nv_compat.isaaclab.utils.warp — NVIDIA Warp®
// kernel-API parity. Two execution paths:
//
//   - `launch` (sync, iter 76): sequential JS — semantics match Python
//     iter 66 exactly; runs everywhere (Node + browsers).
//   - `wgpuLaunch` (async, iter 77): WebGPU compute-shader dispatch
//     when navigator.gpu is available; falls back to the sync JS path
//     transparently otherwise.
//
// ADR-2605261800 §D6 nv-compat namespace localization.

export * from "./warp.js";
export {
  type BindingKind,
  type BindingSpec,
  type WgpuKernel,
  hasWebGPU,
  acquireWebGPUDevice,
  wgpuKernel,
  wgpuLaunch,
} from "./wgpu-backend.js";
export * as examples from "./examples.js";
