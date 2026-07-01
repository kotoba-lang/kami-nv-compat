/**
 * nv-compat isaaclab.envs / e7m-shugyo validation.
 *
 * Exercises the clean-room manager-based RL environment: the manager framework
 * (observation / reward / termination / event), the cart-pole dynamics + MDP
 * terms, the ManagerBasedRLEnv Gym loop (single + vectorized), and a closed-
 * loop PD controller that demonstrably balances the pole. Deterministic.
 *
 *     pnpm exec vitest run test/nv-compat-isaaclab-envs.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.5 isaaclab/envs surface.
 */

import { describe, it, expect } from "vitest";
import {
  ManagerBasedRLEnv,
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
  TerminationManager,
  cartpoleStep,
  defaultCartpoleCfg,
  mdp,
} from "../src/e7m-shugyo/index.js";
import { managers as managersNs, envs } from "../src/isaaclab-envs.js";
import { Sampler } from "../src/utsushimi/index.js";

describe("e7m-shugyo manager framework", () => {
  type E = { v: number };
  it("ObsGroup concatenates + scales terms", () => {
    const g = new ObsGroup<E>({
      a: { func: (e) => [e.v, e.v * 2] },
      b: { func: (e) => e.v, scale: 10 },
    });
    expect(g.evaluate({ v: 3 })).toEqual([3, 6, 30]);
  });

  it("RewardManager weights, sums, and logs an episode", () => {
    const mgr = new RewardManager<E>(
      new RewGroup<E>({ pos: { func: (e) => e.v, weight: 2 }, neg: { func: () => 1, weight: -0.5 } }),
    );
    expect(mgr.compute({ v: 4 })).toBeCloseTo(2 * 4 - 0.5, 9);
    mgr.compute({ v: 1 });
    const log = mgr.logEpisodeReward();
    expect(log.steps).toBe(2);
    expect(log.total).toBeCloseTo(7.5 + 1.5, 9);
  });

  it("TerminationManager separates hard termination from time-out truncation", () => {
    const mgr = new TerminationManager<E>({
      fail: { func: (e) => e.v > 10 },
      timeout: { func: (e) => e.v < 0, timeOut: true },
    });
    expect(mgr.compute({ v: 11 })).toMatchObject({ terminated: true, truncated: false });
    expect(mgr.compute({ v: -1 })).toMatchObject({ terminated: false, truncated: true });
    expect(mgr.compute({ v: 5 })).toMatchObject({ terminated: false, truncated: false });
  });
});

describe("cart-pole dynamics + MDP terms", () => {
  it("an unforced upright pole falls (theta grows)", () => {
    const cfg = defaultCartpoleCfg();
    let s = { x: 0, xDot: 0, theta: 0.05, thetaDot: 0 };
    for (let i = 0; i < 60; i++) s = cartpoleStep(s, 0, cfg);
    expect(Math.abs(s.theta)).toBeGreaterThan(0.05); // diverges from upright
  });

  it("MDP observation/termination terms read the state", () => {
    const view = {
      state: { x: 1, xDot: 2, theta: 0.1, thetaDot: -0.3 },
      lastAction: [0.5],
      terminated: false,
      stepCount: 0,
      maxSteps: 300,
      cfg: defaultCartpoleCfg(),
      rng: new Sampler(0),
    };
    expect(mdp.jointPosRel(view)).toEqual([1, 0.1]);
    expect(mdp.jointVelRel(view)).toEqual([2, -0.3]);
    expect(mdp.poleOutOfBounds(view)).toBe(false);
    view.state.theta = 1.0;
    expect(mdp.poleOutOfBounds(view)).toBe(true);
  });
});

describe("ManagerBasedRLEnv (Gym loop)", () => {
  it("reset returns the policy observation group; step returns the standard tuple", () => {
    const env = new ManagerBasedRLEnv();
    const obs = env.reset(0);
    expect(obs.policy).toBeDefined();
    expect(obs.policy).toHaveLength(5); // [x, theta] + [xDot, thetaDot] + [0.1*lastAction]
    const out = env.step([0]);
    expect(out).toHaveProperty("observations");
    expect(out).toHaveProperty("reward");
    expect(out).toHaveProperty("terminated");
    expect(out).toHaveProperty("truncated");
    expect(typeof out.reward).toBe("number");
  });

  it("terminates + auto-resets when the pole exceeds bounds", () => {
    const env = new ManagerBasedRLEnv({ poleBound: 0.2 });
    env.reset(0);
    // Drive the pole over with a constant push; expect a termination within the episode.
    let sawTermination = false;
    for (let i = 0; i < 300; i++) {
      const out = env.step([1]);
      if (out.terminated || out.truncated) {
        sawTermination = true;
        break;
      }
    }
    expect(sawTermination).toBe(true);
  });

  it("is reproducible given the same seed", () => {
    const run = () => {
      const env = new ManagerBasedRLEnv();
      env.reset(123);
      let acc = 0;
      for (let i = 0; i < 30; i++) acc += env.step([Math.sin(i)]).reward;
      return acc;
    };
    expect(run()).toBe(run());
  });

  it("vectorizes over numEnvs with per-env seeds", () => {
    const env = new ManagerBasedRLEnv({ numEnvs: 4 });
    const obs = env.resetAll(0);
    expect(obs).toHaveLength(4);
    // Different per-env seeds → different initial states.
    expect(obs[0].policy).not.toEqual(obs[1].policy);
    const results = env.stepAll([[0], [0.5], [-0.5], [1]]);
    expect(results).toHaveLength(4);
    for (const r of results) expect(typeof r.reward).toBe("number");
  });

  it("a PD controller keeps the pole within bounds far longer than no control", () => {
    const cfg = { maxEpisodeLengthS: 4.0 };
    // Uncontrolled: how long until termination?
    const survive = (control: (obs: number[]) => number): number => {
      const env = new ManagerBasedRLEnv(cfg);
      let obs = env.reset(7).policy;
      for (let i = 0; i < 240; i++) {
        const out = env.step([control(obs)]);
        if (out.terminated) return i;
        obs = out.observations.policy;
      }
      return 240;
    };
    const none = survive(() => 0);
    // obs = [x, theta, xDot, thetaDot, 0.1*lastAction]; PD on pole angle + rate.
    const pd = survive((o) => Math.max(-1, Math.min(1, 8 * o[1] + 1.5 * o[3] + 0.2 * o[0])));
    expect(pd).toBeGreaterThan(none);
    expect(pd).toBeGreaterThan(120); // balances at least ~2 s
  });
});

describe("isaaclab-envs facade namespaces", () => {
  it("exposes isaaclab.envs / managers", () => {
    expect(envs.ManagerBasedRLEnv).toBe(ManagerBasedRLEnv);
    expect(managersNs.ObservationManager).toBe(ObservationManager);
    const env = new envs.ManagerBasedRLEnv({ numEnvs: 1 });
    expect(env.reset(0).policy).toHaveLength(5);
  });
});
