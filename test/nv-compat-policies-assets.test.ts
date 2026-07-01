/**
 * nv-compat policies (MLP) + assets (URDF builder) coverage.
 *
 * Direct coverage for the RL-policy + robot-asset support layer: MLP forward
 * (ReLU hidden, tanh output), serialize/load round-trip + validation,
 * deterministic random init, and the URDF builder helpers (serial / branched /
 * countJoints / jointNames) including the Franka asset.
 *
 *     pnpm exec vitest run test/nv-compat-policies-assets.test.ts
 *
 * ADR-2605261800 §D6.
 */

import { describe, it, expect } from "vitest";
import {
  type MlpPolicySpec,
  loadMlpFromJson,
  makeRandomMlpSpec,
  runMlpPolicy,
  serializeMlpToJson,
} from "../src/policies/index.js";
import {
  buildBranchedUrdf,
  buildSerialChainUrdf,
  countJoints,
  jointNames,
  makeFrankaPanda,
} from "../src/assets/index.js";

// obsDim 2, hiddenDim 2 (identity W1), actionDim 1 (sum W2): out = tanh(relu(obs)·1).
const spec: MlpPolicySpec = {
  type: "mlp_policy",
  version: 1,
  obsDim: 2,
  hiddenDim: 2,
  actionDim: 1,
  W1Flat: [1, 0, 0, 1],
  b1: [0, 0],
  W2Flat: [1, 1],
  b2: [0],
};

describe("MLP forward (ReLU hidden, tanh output)", () => {
  it("computes tanh(sum) for a positive observation", () => {
    expect(runMlpPolicy(spec, [0.3, 0.4])[0]).toBeCloseTo(Math.tanh(0.7), 6);
  });

  it("ReLU zeroes a negative hidden unit", () => {
    expect(runMlpPolicy(spec, [-0.3, 0.4])[0]).toBeCloseTo(Math.tanh(0.4), 6);
  });

  it("processes a multi-env batch (obs.length = envs × obsDim)", () => {
    const out = runMlpPolicy(spec, [0.3, 0.4, -0.3, 0.4]);
    expect(out).toHaveLength(2);
    expect(out[0]).toBeCloseTo(Math.tanh(0.7), 6);
    expect(out[1]).toBeCloseTo(Math.tanh(0.4), 6);
  });

  it("throws when obs length is not a multiple of obsDim", () => {
    expect(() => runMlpPolicy(spec, [0.3])).toThrow(/multiple/);
  });
});

describe("MLP serialize / load / random init", () => {
  it("serialize → load round-trips the spec", () => {
    const json = serializeMlpToJson(spec);
    expect(loadMlpFromJson(json)).toEqual(spec);
    expect(loadMlpFromJson(JSON.parse(json))).toEqual(spec); // accepts a parsed object too
  });

  it("makeRandomMlpSpec yields the right dims and is deterministic per seed", () => {
    const a = makeRandomMlpSpec(4, 8, 2, 123);
    expect(a.obsDim).toBe(4);
    expect(a.W1Flat).toHaveLength(8 * 4);
    expect(a.W2Flat).toHaveLength(2 * 8);
    expect(makeRandomMlpSpec(4, 8, 2, 123).W1Flat).toEqual(a.W1Flat); // same seed
    expect(makeRandomMlpSpec(4, 8, 2, 999).W1Flat).not.toEqual(a.W1Flat); // different seed
  });

  it("loadMlpFromJson validates the (snake_case) JSON fields", () => {
    // The on-disk schema is snake_case; corrupt a parsed serialization.
    const json = JSON.parse(serializeMlpToJson(spec)) as Record<string, unknown>;
    expect(() => loadMlpFromJson({ ...json, W1_flat: [1, 2, 3] })).toThrow(/W1_flat/); // wrong length
    const missing = { ...json };
    delete missing.obs_dim;
    expect(() => loadMlpFromJson(missing)).toThrow(/obs_dim/);
    expect(() => loadMlpFromJson({ ...json, type: "nope" })).toThrow(/mlp_policy/);
  });
});

describe("URDF builder helpers", () => {
  it("serial chain → countJoints + jointNames", () => {
    const urdf = buildSerialChainUrdf("arm", [
      { name: "j0", type: "revolute" },
      { name: "j1", type: "prismatic" },
      { name: "j2", type: "revolute" },
    ]);
    expect(countJoints(urdf)).toBe(3);
    expect(jointNames(urdf)).toEqual(["j0", "j1", "j2"]);
  });

  it("branched URDF sums joints across branches", () => {
    const urdf = buildBranchedUrdf("robot", "base", [
      [{ name: "a0", type: "revolute" }, { name: "a1", type: "revolute" }],
      [{ name: "b0", type: "revolute" }],
    ]);
    expect(countJoints(urdf)).toBe(3);
  });

  it("the Franka asset's joint count is internally consistent", () => {
    const franka = makeFrankaPanda();
    expect(countJoints(franka.urdfText)).toBe(jointNames(franka.urdfText).length);
    expect(countJoints(franka.urdfText)).toBe(franka.jointNames.length);
  });
});
