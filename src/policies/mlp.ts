// MLP policy-checkpoint loader.
//
// Bridges the gap between externally-trained policies (stable-baselines3,
// rsl_rl, Isaac Lab) and the in-browser mlpPolicyForwardKernel from
// iter 106. Users export their trained policy via a small Python script:
//
//     {
//       "type": "mlp_policy",
//       "version": 1,
//       "obs_dim": 14, "hidden_dim": 64, "action_dim": 7,
//       "W1_flat": [...],   // hidden_dim × obs_dim    row-major
//       "b1": [...],        // hidden_dim
//       "W2_flat": [...],   // action_dim × hidden_dim row-major
//       "b2": [...]         // action_dim
//     }
//
// and serve it as a static asset alongside the demo.
//
// ADR-2605261800 §D6.

import {
  mlpPolicyForwardInline,
  mulberry32,
} from "../warp/examples.js";

export interface MlpPolicySpec {
  type: "mlp_policy";
  version: 1;
  obsDim: number;
  hiddenDim: number;
  actionDim: number;
  /** Row-major hidden_dim × obs_dim. */
  W1Flat: number[];
  /** hidden_dim. */
  b1: number[];
  /** Row-major action_dim × hidden_dim. */
  W2Flat: number[];
  /** action_dim. */
  b2: number[];
}

/** Parse + validate a JSON-encoded MLP policy spec.
 *  Accepts either a string (raw JSON) or an already-parsed object. */
export function loadMlpFromJson(input: string | object): MlpPolicySpec {
  const raw = (typeof input === "string" ? JSON.parse(input) : input) as Record<string, unknown>;

  if (raw.type !== "mlp_policy") {
    throw new Error(`loadMlpFromJson: expected type='mlp_policy', got ${JSON.stringify(raw.type)}`);
  }
  if (raw.version !== 1) {
    throw new Error(`loadMlpFromJson: unsupported version ${raw.version}; this loader handles version 1`);
  }
  const obsDim = numField(raw, "obs_dim");
  const hiddenDim = numField(raw, "hidden_dim");
  const actionDim = numField(raw, "action_dim");
  if (obsDim < 1 || hiddenDim < 1 || actionDim < 1) {
    throw new Error(`loadMlpFromJson: dims must be ≥1; got obs=${obsDim} hidden=${hiddenDim} action=${actionDim}`);
  }
  if (hiddenDim > 128) {
    throw new Error(`loadMlpFromJson: hidden_dim=${hiddenDim} exceeds kernel compile-bound MAX_HIDDEN=128`);
  }
  if (obsDim > 64) {
    throw new Error(`loadMlpFromJson: obs_dim=${obsDim} exceeds kernel compile-bound MAX_OBS=64`);
  }

  const W1Flat = floatArrayField(raw, "W1_flat", hiddenDim * obsDim);
  const b1 = floatArrayField(raw, "b1", hiddenDim);
  const W2Flat = floatArrayField(raw, "W2_flat", actionDim * hiddenDim);
  const b2 = floatArrayField(raw, "b2", actionDim);

  return { type: "mlp_policy", version: 1, obsDim, hiddenDim, actionDim, W1Flat, b1, W2Flat, b2 };
}

/** Round-trip serialiser (used by tests and host code that wants to save a spec). */
export function serializeMlpToJson(spec: MlpPolicySpec, pretty = false): string {
  return JSON.stringify({
    type: spec.type,
    version: spec.version,
    obs_dim: spec.obsDim,
    hidden_dim: spec.hiddenDim,
    action_dim: spec.actionDim,
    W1_flat: spec.W1Flat,
    b1: spec.b1,
    W2_flat: spec.W2Flat,
    b2: spec.b2,
  }, null, pretty ? 2 : 0);
}

/** Deterministic random MLP fixture (mulberry32 → He-ish init).
 *  Useful for tests and for sanity-checking the in-browser path
 *  before a real checkpoint is available. */
export function makeRandomMlpSpec(
  obsDim: number, hiddenDim: number, actionDim: number, seed: number,
): MlpPolicySpec {
  const rng = mulberry32(seed);
  const heScale1 = Math.sqrt(2 / obsDim);
  const heScale2 = Math.sqrt(2 / hiddenDim);
  const W1Flat = new Array(hiddenDim * obsDim);
  for (let i = 0; i < W1Flat.length; i++) W1Flat[i] = (rng() * 2 - 1) * heScale1;
  const b1 = new Array(hiddenDim).fill(0);
  const W2Flat = new Array(actionDim * hiddenDim);
  for (let i = 0; i < W2Flat.length; i++) W2Flat[i] = (rng() * 2 - 1) * heScale2;
  const b2 = new Array(actionDim).fill(0);
  return { type: "mlp_policy", version: 1, obsDim, hiddenDim, actionDim, W1Flat, b1, W2Flat, b2 };
}

/** Run the spec on a flat observation buffer (envs × obsDim).
 *  Convenience wrapper around mlpPolicyForwardInline. */
export function runMlpPolicy(spec: MlpPolicySpec, obs: readonly number[]): number[] {
  if (obs.length % spec.obsDim !== 0) {
    throw new Error(`runMlpPolicy: obs.length=${obs.length} not a multiple of obs_dim=${spec.obsDim}`);
  }
  return mlpPolicyForwardInline(
    obs, spec.W1Flat, spec.b1, spec.W2Flat, spec.b2,
    spec.obsDim, spec.hiddenDim, spec.actionDim);
}

function numField(raw: Record<string, unknown>, key: string): number {
  const v = raw[key];
  if (typeof v !== "number" || !Number.isInteger(v)) {
    throw new Error(`loadMlpFromJson: '${key}' must be integer, got ${JSON.stringify(v)}`);
  }
  return v;
}

function floatArrayField(raw: Record<string, unknown>, key: string, expectedLen: number): number[] {
  const v = raw[key];
  if (!Array.isArray(v)) {
    throw new Error(`loadMlpFromJson: '${key}' must be array, got ${typeof v}`);
  }
  if (v.length !== expectedLen) {
    throw new Error(`loadMlpFromJson: '${key}' length=${v.length}, expected ${expectedLen}`);
  }
  const out = new Array(expectedLen);
  for (let i = 0; i < expectedLen; i++) {
    if (typeof v[i] !== "number") {
      throw new Error(`loadMlpFromJson: '${key}'[${i}] not a number: ${JSON.stringify(v[i])}`);
    }
    out[i] = v[i];
  }
  return out;
}
