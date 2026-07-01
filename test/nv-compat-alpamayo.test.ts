/**
 * nv-compat Alpamayo (AV VLA) / kami-drive validation.
 *
 * Exercises the clean-room reasoning planner (michibiki), the Alpamayo VLA
 * facade, the Chain-of-Causation schema, and the AlpaSim closed-loop harness —
 * all deterministic and CPU-only. Asserts on driving behaviour (yields to a
 * pedestrian, turns on command, avoids a stopped lead vehicle) plus the
 * facade + Charter invariants (64-waypoint horizon, SAE-L4 ceiling, Murakumo
 * narration injection).
 *
 *     pnpm exec vitest run test/nv-compat-alpamayo.test.ts
 *
 * AV scope per ADR-2605242000 (wadachi) / ADR-2606010600 (kami-autodrive).
 */

import { describe, it, expect } from "vitest";
import {
  type DrivingObservation,
  CausationBuilder,
  parseReasoningRecord,
  recordFromTrace,
  recordToDatoms,
  commandFromInstruction,
  plan,
  rolloutTrajectory,
  stepUnicycle,
  trajectoryLength,
} from "../src/kami-drive/index.js";
import {
  AlpamayoR1,
  CAMERA_NAMES,
  SAE_CEILING,
  TRAJECTORY_HORIZON_S,
  TRAJECTORY_HZ,
  TRAJECTORY_WAYPOINTS,
} from "../src/alpamayo.js";
import { type Scenario, runClosedLoop } from "../src/alpasim.js";

const yawOf = (wp: { rotation: number[] }) => Math.atan2(wp.rotation[3], wp.rotation[0]);

describe("kami-drive BEV unicycle", () => {
  it("integrates a straight line at constant speed", () => {
    const wps = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: 10 }, Array(10).fill({ accel: 0, curvature: 0 }), 0.1);
    expect(wps).toHaveLength(11);
    expect(wps[10].translation[0]).toBeCloseTo(10, 6); // 10 m/s × 1 s
    expect(wps[10].translation[1]).toBeCloseTo(0, 6);
    expect(yawOf(wps[10])).toBeCloseTo(0, 6);
  });

  it("turns under positive curvature (yaw increases)", () => {
    const wps = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: 10 }, Array(20).fill({ accel: 0, curvature: 0.1 }), 0.1);
    expect(yawOf(wps[20])).toBeGreaterThan(0);
    expect(trajectoryLength(wps)).toBeGreaterThan(0);
  });

  it("clamps speed to ≥ 0 under braking", () => {
    const s = stepUnicycle({ x: 0, y: 0, yaw: 0, speed: 1 }, { accel: -100, curvature: 0 }, 0.1);
    expect(s.speed).toBe(0);
  });
});

describe("Chain-of-Causation schema", () => {
  it("builds a trace and renders a narrative", () => {
    const coc = new CausationBuilder("yield")
      .add("Vehicle 12.0 m ahead", "must keep a safe gap", "reduce speed", 7)
      .build();
    expect(coc.eventCluster).toBe("yield");
    expect(coc.steps).toHaveLength(1);
    expect(coc.narrative.toLowerCase()).toContain("because");
  });

  it("parses a dataset record from snake_case or camelCase", () => {
    const snake = parseReasoningRecord({
      clip_uuid: "abc-123",
      event_cluster: "vru_interaction",
      chain_of_causation: "Yielded to a pedestrian.",
      keyframe_indices: [3, 9, 14],
    });
    expect(snake.clipUuid).toBe("abc-123");
    expect(snake.eventCluster).toBe("vru_interaction");
    expect(snake.keyframeIndices).toEqual([3, 9, 14]);
    expect(() => parseReasoningRecord({ event_cluster: "nominal" })).toThrow(/clipUuid/);
    expect(() => parseReasoningRecord({ clip_uuid: "x", event_cluster: "bogus" })).toThrow(/eventCluster/);
  });

  it("projects a record to kotoba :coc/* datoms", () => {
    const coc = new CausationBuilder("stop").add("Stop sign", "must halt", "decelerate", 5).build();
    const datoms = recordToDatoms(recordFromTrace("clip-9", coc));
    expect(datoms.some((d) => d.a === ":coc/event-cluster" && d.v === "stop")).toBe(true);
    expect(datoms.some((d) => d.a === ":coc.step/observation" && d.v === "Stop sign")).toBe(true);
  });
});

describe("kami-drive reasoning planner", () => {
  const ego = { x: 0, y: 0, yaw: 0, speed: 8 };

  it("cruises on a clear road and emits a 6.4 s / 64-step trajectory", () => {
    const obs: DrivingObservation = { ego, command: "keep_lane", agents: [], speedLimit: 13 };
    const { trajectory, reasoning } = plan(obs);
    expect(trajectory).toHaveLength(65); // t0 + 64 future waypoints
    expect(reasoning.eventCluster).toBe("nominal");
    // Accelerates toward the limit.
    expect(trajectory[64].speed).toBeGreaterThan(trajectory[0].speed);
    expect(trajectory[64].speed).toBeLessThanOrEqual(13 + 1e-6);
  });

  it("yields to a pedestrian in the ego corridor (decelerates to stop)", () => {
    const obs: DrivingObservation = {
      ego,
      command: "keep_lane",
      agents: [{ id: "p1", kind: "pedestrian", x: 10, y: 0, vx: 0, vy: 0 }],
    };
    const { trajectory, reasoning } = plan(obs);
    expect(reasoning.eventCluster).toBe("vru_interaction");
    expect(reasoning.narrative.toLowerCase()).toContain("yield");
    expect(trajectory[64].speed).toBeLessThan(trajectory[0].speed); // braking
  });

  it("turns left on command (trajectory yaw becomes positive)", () => {
    const obs: DrivingObservation = { ego, command: "turn_left", agents: [] };
    const { trajectory, reasoning } = plan(obs);
    expect(reasoning.eventCluster).toBe("intersection");
    // Check ~1 s in, before a multi-second constant-curvature arc wraps past π.
    expect(yawOf(trajectory[10])).toBeGreaterThan(0);
  });

  it("ignores agents outside the corridor", () => {
    const obs: DrivingObservation = {
      ego,
      command: "keep_lane",
      agents: [{ id: "v1", kind: "vehicle", x: 10, y: 6, vx: 0, vy: 0 }], // 6 m to the side
      speedLimit: 13,
    };
    const { reasoning } = plan(obs);
    expect(reasoning.eventCluster).toBe("nominal");
  });

  it("maps free-text instructions to commands", () => {
    expect(commandFromInstruction("please turn left at the light")).toBe("turn_left");
    expect(commandFromInstruction("STOP now")).toBe("stop");
    expect(commandFromInstruction("右に曲がって")).toBe("turn_right");
    expect(commandFromInstruction("continue ahead")).toBeNull();
  });
});

describe("Alpamayo® VLA facade (clean-room)", () => {
  it("fromPretrained + predict returns trajectory + reasoning; reports michibiki engine", () => {
    const model = AlpamayoR1.fromPretrained("nvidia/Alpamayo-R1-10B");
    expect(model.engine).toBe("michibiki");
    expect(model.pretrainedName).toBe("nvidia/Alpamayo-R1-10B");
    const out = model.predict({ ego: { x: 0, y: 0, yaw: 0, speed: 9 }, command: "keep_lane", agents: [], speedLimit: 13 });
    expect(out.trajectory).toHaveLength(65);
    expect(out.explanation).toBe(out.reasoning.narrative);
  });

  it("exposes the model-card constants and the SAE-L4 ceiling", () => {
    expect(TRAJECTORY_HORIZON_S).toBe(6.4);
    expect(TRAJECTORY_HZ).toBe(10);
    expect(TRAJECTORY_WAYPOINTS).toBe(64);
    expect(SAE_CEILING).toBe(4);
    expect(CAMERA_NAMES).toContain("front_wide");
  });

  it("predictAsync uses an injected Murakumo narrator, falling back on error", async () => {
    const model = AlpamayoR1.fromPretrained();
    const obs: DrivingObservation = {
      ego: { x: 0, y: 0, yaw: 0, speed: 5 },
      command: "keep_lane",
      agents: [{ id: "p", kind: "pedestrian", x: 8, y: 0, vx: 0, vy: 0 }],
    };
    const narrated = await model.predictAsync(obs, async () => "ペデストリアンに譲るため減速します。");
    expect(narrated.explanation).toBe("ペデストリアンに譲るため減速します。");
    expect(narrated.reasoning.narrative).toBe("ペデストリアンに譲るため減速します。");
    // Narrator throws → deterministic template retained.
    const fallback = await model.predictAsync(obs, async () => {
      throw new Error("murakumo offline");
    });
    expect(fallback.explanation.toLowerCase()).toContain("yield");
  });

  it("predictFromInput estimates ego speed from egomotion history", () => {
    const model = AlpamayoR1.fromPretrained();
    const out = model.predictFromInput(
      {
        images: [],
        command: "keep lane",
        egomotionHistory: [
          { translation: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], timestamp: 0 },
          { translation: [1, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], timestamp: 0.1 },
        ],
      },
      { command: "keep_lane", agents: [], speedLimit: 13 },
    );
    // 1 m in 0.1 s ⇒ 10 m/s initial speed.
    expect(out.trajectory[0].speed).toBeCloseTo(10, 6);
  });
});

describe("AlpaSim closed-loop harness (clean-room)", () => {
  const model = AlpamayoR1.fromPretrained();

  function emptyRoad(): Scenario {
    return {
      ego: { x: 0, y: 0, yaw: 0, speed: 8, radius: 1.5 },
      agents: [],
      command: "keep_lane",
      speedLimit: 13,
      durationS: 4,
      hz: 10,
    };
  }

  it("makes forward progress on an empty road with positive reward", () => {
    const res = runClosedLoop(model, emptyRoad());
    expect(res.metrics.collision).toBe(false);
    expect(res.metrics.progress).toBeGreaterThan(20); // ≥ 8 m/s × 4 s minus accel ramp
    expect(res.reward).toBeGreaterThan(0);
  });

  it("avoids a stopped lead vehicle far enough to brake (no collision)", () => {
    const sc = emptyRoad();
    sc.ego.speed = 8;
    sc.agents = [{ id: "lead", kind: "vehicle", x: 40, y: 0, vx: 0, vy: 0, radius: 1.5 }];
    sc.durationS = 6;
    const res = runClosedLoop(model, sc);
    expect(res.metrics.collision).toBe(false);
    expect(res.metrics.minClearance).toBeGreaterThan(0);
  });

  it("detects a collision when a stopped vehicle is too close to avoid", () => {
    const sc = emptyRoad();
    sc.ego.speed = 14; // fast, agent within braking distance
    sc.agents = [{ id: "close", kind: "vehicle", x: 3, y: 0, vx: 0, vy: 0, radius: 1.0 }];
    sc.durationS = 3;
    const res = runClosedLoop(model, sc);
    expect(res.metrics.collision).toBe(true);
    expect(res.reward).toBe(0);
  });

  it("is deterministic: identical scenario → identical reward + step count", () => {
    const a = runClosedLoop(model, emptyRoad());
    const b = runClosedLoop(model, emptyRoad());
    expect(a.reward).toBe(b.reward);
    expect(a.steps.length).toBe(b.steps.length);
    expect(a.metrics.progress).toBe(b.metrics.progress);
  });
});
