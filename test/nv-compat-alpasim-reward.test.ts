/**
 * nv-compat AlpaSim reward + rollout-metrics coverage.
 *
 * Coverage for the closed-loop scoring: reward is bounded [0,1], a clear fast
 * road scores near the progress ceiling, a collision zeroes the reward, and the
 * RolloutMetrics / RolloutStep records carry the documented fields.
 *
 *     pnpm exec vitest run test/nv-compat-alpasim-reward.test.ts
 *
 * AV scope per ADR-2606010600 (kami-autodrive).
 */

import { describe, it, expect } from "vitest";
import { type Scenario, type WorldAgent, runClosedLoop } from "../src/alpasim.js";
import { AlpamayoR1 } from "../src/alpamayo.js";

const model = AlpamayoR1.fromPretrained();

function scenario(over: Partial<Scenario> = {}): Scenario {
  return {
    ego: { x: 0, y: 0, yaw: 0, speed: 8, radius: 1.5 },
    agents: [],
    command: "keep_lane",
    speedLimit: 13,
    durationS: 4,
    hz: 10,
    ...over,
  };
}

describe("reward bounds", () => {
  it("stays within [0,1] across scenarios", () => {
    for (const sc of [scenario(), scenario({ ego: { x: 0, y: 0, yaw: 0, speed: 0, radius: 1.5 } }), scenario({ command: "stop" })]) {
      const r = runClosedLoop(model, sc).reward;
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(1);
    }
  });

  it("a clear road cruising at the limit scores near the progress ceiling", () => {
    const res = runClosedLoop(model, scenario({ ego: { x: 0, y: 0, yaw: 0, speed: 13, radius: 1.5 } }));
    expect(res.metrics.collision).toBe(false);
    expect(res.reward).toBeGreaterThan(0.85); // ~full progress, low jerk
  });

  it("a slow start scores lower than cruising (less progress)", () => {
    const slow = runClosedLoop(model, scenario({ ego: { x: 0, y: 0, yaw: 0, speed: 3, radius: 1.5 } })).reward;
    const fast = runClosedLoop(model, scenario({ ego: { x: 0, y: 0, yaw: 0, speed: 13, radius: 1.5 } })).reward;
    expect(slow).toBeLessThan(fast);
    expect(slow).toBeGreaterThan(0);
  });
});

describe("collision zeroes the reward", () => {
  it("an unavoidable close obstacle → collision true, reward 0", () => {
    const close: WorldAgent = { id: "wall", kind: "vehicle", x: 3, y: 0, vx: 0, vy: 0, radius: 1 };
    const res = runClosedLoop(model, scenario({ ego: { x: 0, y: 0, yaw: 0, speed: 14, radius: 1.5 }, agents: [close], durationS: 3 }));
    expect(res.metrics.collision).toBe(true);
    expect(res.reward).toBe(0);
  });
});

describe("rollout metrics + steps", () => {
  it("metrics carry progress / collision / minClearance / jerkRms", () => {
    const res = runClosedLoop(model, scenario());
    expect(res.metrics.progress).toBeGreaterThan(0);
    expect(res.metrics.collision).toBe(false);
    expect(res.metrics.minClearance).toBe(Infinity); // no agents
    expect(res.metrics.jerkRms).toBeGreaterThanOrEqual(0);
  });

  it("each step records t / ego / action / minClearance; count = duration × hz", () => {
    const res = runClosedLoop(model, scenario({ durationS: 2, hz: 10 }));
    expect(res.steps).toHaveLength(20);
    const s0 = res.steps[0];
    expect(s0.t).toBe(0);
    expect(s0.ego).toBeDefined();
    expect(typeof s0.action.accel).toBe("number");
    expect(typeof s0.action.curvature).toBe("number");
    expect(res.steps[19].t).toBeCloseTo(1.9, 6);
  });

  it("tracks minClearance to a passing agent (finite, > 0 when avoided)", () => {
    const lead: WorldAgent = { id: "lead", kind: "vehicle", x: 45, y: 0, vx: 0, vy: 0, radius: 1.5 };
    const res = runClosedLoop(model, scenario({ agents: [lead], durationS: 6, ego: { x: 0, y: 0, yaw: 0, speed: 8, radius: 1.5 } }));
    expect(res.metrics.collision).toBe(false);
    expect(res.metrics.minClearance).toBeGreaterThan(0);
    expect(Number.isFinite(res.metrics.minClearance)).toBe(true);
  });
});
