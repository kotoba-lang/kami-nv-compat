/**
 * nv-compat kami-drive planner / unicycle / CoC edge cases.
 *
 * Branch coverage for the Alpamayo backend planner: instruction→command
 * mapping, the yield corridor logic (behind / beyond look-ahead / crossing
 * VRU / lead vehicle), turn/stop/cruise clusters, instruction override,
 * unicycle clamp + rollout shape, and Chain-of-Causation rendering/datoms.
 *
 *     pnpm exec vitest run test/nv-compat-kami-drive-edge.test.ts
 *
 * AV scope per ADR-2605242000 / ADR-2606010600.
 */

import { describe, it, expect } from "vitest";
import {
  type DrivingObservation,
  type PerceivedAgent,
  CausationBuilder,
  DEFAULT_PLANNER,
  commandFromInstruction,
  plan,
  recordFromTrace,
  recordToDatoms,
  renderNarrative,
  rolloutTrajectory,
  stepUnicycle,
  trajectoryLength,
  yawToMat3,
} from "../src/kami-drive/index.js";

const ego = { x: 0, y: 0, yaw: 0, speed: 10 };
const obs = (over: Partial<DrivingObservation>): DrivingObservation => ({
  ego,
  command: "keep_lane",
  agents: [],
  speedLimit: 13,
  ...over,
});

describe("commandFromInstruction mapping", () => {
  it.each([
    ["please turn left now", "turn_left"],
    ["turn right at the light", "turn_right"],
    ["go left", "turn_left"],
    ["右に寄って", "turn_right"],
    ["STOP immediately", "stop"],
    ["halt the vehicle", "stop"],
    ["please brake", "stop"],
    ["止まって", "stop"],
  ])("maps %s → %s", (text, expected) => {
    expect(commandFromInstruction(text)).toBe(expected);
  });

  it("returns null for instructions with no directional/stop keyword", () => {
    expect(commandFromInstruction("continue straight ahead")).toBeNull();
    expect(commandFromInstruction("maintain speed")).toBeNull();
  });
});

describe("planner clusters + corridor logic", () => {
  it("cruises (nominal) on a clear road", () => {
    const { reasoning, trajectory } = plan(obs({}));
    expect(reasoning.eventCluster).toBe("nominal");
    expect(trajectory).toHaveLength(65);
  });

  it("turn_right yields negative curvature (yaw turns clockwise early)", () => {
    const { trajectory, reasoning } = plan(obs({ command: "turn_right" }));
    expect(reasoning.eventCluster).toBe("intersection");
    const yaw = Math.atan2(trajectory[10].rotation[3], trajectory[10].rotation[0]);
    expect(yaw).toBeLessThan(0);
  });

  it("stop command decelerates and reports the stop cluster", () => {
    const { trajectory, reasoning } = plan(obs({ command: "stop" }));
    expect(reasoning.eventCluster).toBe("stop");
    expect(trajectory[64].speed).toBeLessThan(trajectory[0].speed);
    expect(reasoning.narrative.toLowerCase()).toMatch(/halt|zero|stop/);
  });

  it("ignores an agent behind the ego (x ≤ 0)", () => {
    const behind: PerceivedAgent = { id: "b", kind: "vehicle", x: -5, y: 0, vx: 0, vy: 0 };
    expect(plan(obs({ agents: [behind] })).reasoning.eventCluster).toBe("nominal");
  });

  it("ignores an agent beyond the look-ahead", () => {
    const far: PerceivedAgent = { id: "f", kind: "vehicle", x: DEFAULT_PLANNER.lookAhead + 20, y: 0, vx: 0, vy: 0 };
    expect(plan(obs({ agents: [far] })).reasoning.eventCluster).toBe("nominal");
  });

  it("yields to a crossing pedestrian whose path enters the corridor", () => {
    // Pedestrian starts off-lane (y=-3) but crosses the centerline within ~1 s.
    const ped: PerceivedAgent = { id: "p", kind: "pedestrian", x: 12, y: -3, vx: 0, vy: 5 };
    const { reasoning, trajectory } = plan(obs({ agents: [ped] }));
    expect(reasoning.eventCluster).toBe("vru_interaction");
    expect(trajectory[64].speed).toBeLessThan(trajectory[0].speed);
  });

  it("keeps a safe following speed behind a lead vehicle (yield, not full stop)", () => {
    const lead: PerceivedAgent = { id: "v", kind: "vehicle", x: 30, y: 0, vx: 0, vy: 0 };
    const { reasoning } = plan(obs({ ego: { ...ego, speed: 12 }, agents: [lead] }));
    expect(reasoning.eventCluster).toBe("yield");
    expect(reasoning.narrative.toLowerCase()).toMatch(/gap|speed/);
  });

  it("a free-text instruction overrides the explicit command", () => {
    const { reasoning } = plan(obs({ command: "keep_lane", instruction: "turn left here" }));
    expect(reasoning.eventCluster).toBe("intersection");
  });
});

describe("unicycle kinematics", () => {
  it("clamps integrated speed to maxSpeed", () => {
    const s = stepUnicycle({ x: 0, y: 0, yaw: 0, speed: 29 }, { accel: 100, curvature: 0 }, 1, 30);
    expect(s.speed).toBe(30);
  });

  it("rolloutTrajectory emits actions.length + 1 waypoints; length tracks distance", () => {
    const wps = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: 5 }, new Array(8).fill({ accel: 0, curvature: 0 }), 0.1);
    expect(wps).toHaveLength(9);
    expect(trajectoryLength(wps)).toBeCloseTo(5 * 0.8, 6); // 5 m/s × 0.8 s
  });

  it("yawToMat3 encodes a rotation about +z", () => {
    const m = yawToMat3(Math.PI / 2);
    expect(m[0]).toBeCloseTo(0, 6); // cos(90°)
    expect(m[3]).toBeCloseTo(1, 6); // sin(90°)
    expect(m[8]).toBe(1);
  });
});

describe("Chain-of-Causation rendering + datoms", () => {
  it("renders the empty-trace fallback narrative", () => {
    expect(renderNarrative([])).toMatch(/nominal conditions/i);
  });

  it("recordFromTrace + recordToDatoms project steps to :coc.step/* datoms", () => {
    const coc = new CausationBuilder("yield")
      .add("Vehicle ahead", "keep a gap", "reduce speed", 5)
      .build();
    const rec = recordFromTrace("clip-1", coc);
    expect(rec.eventCluster).toBe("yield");
    expect(rec.keyframeIndices).toEqual([5]);
    const datoms = recordToDatoms(rec);
    expect(datoms.some((d) => d.a === ":coc/clip" && d.v === "clip-1")).toBe(true);
    expect(datoms.some((d) => d.a === ":coc.step/action" && d.v === "reduce speed")).toBe(true);
  });
});
