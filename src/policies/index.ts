// @etzhayyim/kami-nv-compat/policies
//
// Loaders + helpers for in-browser inference of trained RL policies.
// Bridges externally-trained PyTorch / TensorFlow / JAX policies into
// the WGSL kernels under nv-compat/warp/.
//
// ADR-2605261800 §D6.

export {
  type MlpPolicySpec,
  loadMlpFromJson,
  serializeMlpToJson,
  makeRandomMlpSpec,
  runMlpPolicy,
} from "./mlp.js";
