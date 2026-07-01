/**
 * nv-compat e7m-shugyo manager + env edge cases.
 *
 * Branch coverage for the Isaac Lab managers: EventManager mode separation +
 * interval gating + resetIntervals, ObservationManager multi-group, the
 * RewardManager episode log, and ManagerBasedRLEnv wiring (pre-built vs raw
 * managers, vectorized auto-reset, re-seeding).
 *
 *     pnpm exec vitest run test/nv-compat-e7m-shugyo-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 e7m-shugyo.
 */

import { describe, it, expect } from "vitest";
import {
  EventManager,
  ManagerBasedRLEnv,
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
} from "../src/e7m-shugyo/index.js";

describe("EventManager mode separation + interval gating", () => {
  type E = { fired: string[] };
  it("applies only terms matching the requested mode", () => {
    const env: E = { fired: [] };
    const mgr = new EventManager<E>({
      boot: { func: (e) => e.fired.push("boot"), mode: "startup" },
      respawn: { func: (e) => e.fired.push("respawn"), mode: "reset" },
    });
    mgr.apply(env, "startup");
    expect(env.fired).toEqual(["boot"]);
    mgr.apply(env, "reset");
    expect(env.fired).toEqual(["boot", "respawn"]);
  });

  it("interval terms fire only after intervalS of sim time has elapsed", () => {
    const env: E = { fired: [] };
    const mgr = new EventManager<E>({
      push: { func: (e) => e.fired.push("push"), mode: "interval", intervalS: 1.0 },
    });
    mgr.apply(env, "interval", 0); // first fire (lastFired = -inf)
    mgr.apply(env, "interval", 0.5); // too soon → skipped
    mgr.apply(env, "interval", 1.0); // 1.0 - 0 ≥ 1 → fires
    expect(env.fired).toEqual(["push", "push"]);
    mgr.resetIntervals();
    mgr.apply(env, "interval", 1.2); // after reset, fires again
    expect(env.fired).toHaveLength(3);
  });
});

describe("ObservationManager + RewardManager", () => {
  type E = { v: number };
  it("computes multiple named groups + reports group names", () => {
    const mgr = new ObservationManager<E>({
      policy: new ObsGroup<E>({ a: { func: (e) => e.v } }),
      critic: new ObsGroup<E>({ b: { func: (e) => [e.v, e.v] } }),
    });
    const out = mgr.compute({ v: 2 });
    expect(out.policy).toEqual([2]);
    expect(out.critic).toEqual([2, 2]);
    expect(mgr.groupNames().sort()).toEqual(["critic", "policy"]);
  });

  it("accumulates an episode log and resets it", () => {
    const mgr = new RewardManager<E>(new RewGroup<E>({ r: { func: (e) => e.v, weight: 1 } }));
    mgr.compute({ v: 3 });
    mgr.compute({ v: 4 });
    expect(mgr.logEpisodeReward()).toMatchObject({ total: 7, steps: 2, r: 7 });
    expect(mgr.getBreakdown({ v: 5 })).toEqual({ r: 5 });
    mgr.resetEpisodeLog();
    expect(mgr.logEpisodeReward()).toMatchObject({ total: 0, steps: 0 });
  });
});

describe("ManagerBasedRLEnv wiring", () => {
  it("accepts a pre-built RewardManager and exposes it via the getter", () => {
    const rm = new RewardManager(new RewGroup({ alive: { func: () => 1, weight: 2 } }));
    const env = new ManagerBasedRLEnv({}, { rewards: rm });
    expect(env.rewardManager).toBe(rm);
    env.reset(0);
    const out = env.step([0]);
    expect(out.reward).toBeCloseTo(2, 9); // weight 2 × alive 1
  });

  it("accepts a raw ObsGroup map and a pre-built ObservationManager", () => {
    const raw = new ManagerBasedRLEnv({}, { observations: { policy: new ObsGroup({ x: { func: (e) => e.state.x } }) } });
    expect(raw.reset(0).policy).toHaveLength(1);
    const om = new ObservationManager({ policy: new ObsGroup({ x: { func: (e) => [e.state.x, e.state.theta] } }) });
    const built = new ManagerBasedRLEnv({}, { observations: om });
    expect(built.observationManager).toBe(om);
    expect(built.reset(0).policy).toHaveLength(2);
  });

  it("re-seeding produces a different initial observation", () => {
    const env = new ManagerBasedRLEnv();
    const a = env.reset(1).policy;
    const b = env.reset(2).policy;
    expect(a).not.toEqual(b);
  });

  it("vectorized stepAll auto-resets a done env and keeps all rewards finite", () => {
    const env = new ManagerBasedRLEnv({ numEnvs: 3, poleBound: 0.2 });
    env.resetAll(0);
    let sawDone = false;
    for (let i = 0; i < 300 && !sawDone; i++) {
      const results = env.stepAll([[1], [1], [1]]); // hard push all three
      for (const r of results) {
        expect(Number.isFinite(r.reward)).toBe(true);
        expect(r.observations.policy).toHaveLength(5);
        if (r.terminated || r.truncated) sawDone = true;
      }
    }
    expect(sawDone).toBe(true);
  });
});
