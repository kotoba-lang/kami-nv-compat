/**
 * nv-compat Alpamayo VLA facade input handling.
 *
 * Coverage for the raw-input path: egomotion-history speed estimation (empty /
 * single / zero-dt / moving), the predictFromInput command-from-instruction
 * routing, agent pass-through, and the model-card constants.
 *
 *     pnpm exec vitest run test/nv-compat-alpamayo-input.test.ts
 *
 * AV scope per ADR-2606010600 (kami-autodrive).
 */

import { describe, it, expect } from "vitest";
import {
  type AlpamayoInput,
  type EgomotionWaypoint,
  AlpamayoR1,
  CAMERA_NAMES,
  HISTORY_S,
  TRAJECTORY_HZ,
} from "../src/alpamayo.js";

const IDENT9: [number, number, number, number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const wp = (x: number, t: number): EgomotionWaypoint => ({ translation: [x, 0, 0], rotation: IDENT9, timestamp: t });
const input = (history: EgomotionWaypoint[], command = "keep lane"): AlpamayoInput => ({ images: [], command, egomotionHistory: history });

const model = AlpamayoR1.fromPretrained();
const baseObs = { command: "keep_lane" as const, agents: [], speedLimit: 13 };

describe("egomotion → ego speed estimation", () => {
  it("empty history → speed 0", () => {
    expect(model.predictFromInput(input([]), baseObs).trajectory[0].speed).toBe(0);
  });

  it("single waypoint → speed 0", () => {
    expect(model.predictFromInput(input([wp(0, 0)]), baseObs).trajectory[0].speed).toBe(0);
  });

  it("zero time delta → speed 0 (no divide-by-zero)", () => {
    expect(model.predictFromInput(input([wp(0, 0.5), wp(1, 0.5)]), baseObs).trajectory[0].speed).toBe(0);
  });

  it("1 m over 0.1 s → 10 m/s", () => {
    expect(model.predictFromInput(input([wp(0, 0), wp(1, 0.1)]), baseObs).trajectory[0].speed).toBeCloseTo(10, 6);
  });
});

describe("predictFromInput command routing + agents", () => {
  it("routes the input command string through the planner (turn → intersection)", () => {
    const out = model.predictFromInput(input([wp(0, 0), wp(1, 0.1)], "turn left"), baseObs);
    expect(out.reasoning.eventCluster).toBe("intersection");
  });

  it("passes agents through to the planner (pedestrian → yield)", () => {
    const out = model.predictFromInput(input([wp(0, 0), wp(1, 0.1)]), {
      command: "keep_lane",
      agents: [{ id: "p", kind: "pedestrian", x: 10, y: 0, vx: 0, vy: 0 }],
      speedLimit: 13,
    });
    expect(out.reasoning.eventCluster).toBe("vru_interaction");
  });
});

describe("predict on a structured observation", () => {
  it("a zero-speed ego on a clear road produces a full 6.4 s trajectory and accelerates", () => {
    const out = model.predict({ ego: { x: 0, y: 0, yaw: 0, speed: 0 }, command: "keep_lane", agents: [], speedLimit: 13 });
    expect(out.trajectory).toHaveLength(65); // t0 + 64
    expect(out.trajectory[64].speed).toBeGreaterThan(0);
    expect(out.explanation).toBe(out.reasoning.narrative);
  });
});

describe("model-card constants", () => {
  it("exposes the documented camera names + history window + rate", () => {
    expect(CAMERA_NAMES).toEqual(["front_wide", "front_tele", "cross_left", "cross_right"]);
    expect(HISTORY_S).toBe(0.4);
    expect(TRAJECTORY_HZ).toBe(10);
  });
});
