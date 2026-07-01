/**
 * nv-compat DriveSim integration paths.
 *
 * Coverage for the DriveSim class glue: the world→ego observation transform
 * (under ego yaw + relative velocity), full-rig sense(), open- vs closed-loop
 * step(), and reset(). Complements the per-sensor tests in
 * nv-compat-drive-sim.test.ts.
 *
 *     pnpm exec vitest run test/nv-compat-drive-sim-integration.test.ts
 *
 * ADR-2605261800 §D6 (DriveSim → wadachi-sim).
 */

import { describe, it, expect } from "vitest";
import {
  type Actor,
  DriveSim,
  createCamera,
  createLidar,
  createRadar,
  createScenario,
} from "../src/drive-sim.js";
import { AlpamayoR1 } from "../src/alpamayo.js";

const car = (over: Partial<Actor>): Actor => ({
  id: "c", kind: "vehicle", x: 0, y: 0, yaw: 0, vx: 0, vy: 0, extent: [2, 1, 0.75], ...over,
});

describe("observation() world→ego transform", () => {
  it("maps an actor directly ahead to +x in the ego frame (yaw 0)", () => {
    const sim = new DriveSim({ scenario: createScenario({ ego: { speed: 0 }, actors: [car({ x: 20, y: 0 })] }), rig: {}, hz: 10 });
    const a = sim.observation().agents[0];
    expect(a.x).toBeCloseTo(20, 6);
    expect(a.y).toBeCloseTo(0, 6);
  });

  it("rotates world coordinates into the ego frame under ego yaw 90°", () => {
    // Ego facing +y; an actor at world (0, 20) is directly ahead → ego-frame +x.
    const sim = new DriveSim({
      scenario: createScenario({ ego: { yaw: Math.PI / 2, speed: 0 }, actors: [car({ x: 0, y: 20 })] }),
      rig: {},
      hz: 10,
    });
    const a = sim.observation().agents[0];
    expect(a.x).toBeCloseTo(20, 5);
    expect(a.y).toBeCloseTo(0, 5);
  });

  it("reports relative velocity: a static actor ahead closes as the ego drives forward", () => {
    const sim = new DriveSim({ scenario: createScenario({ ego: { speed: 10 }, actors: [car({ x: 30, vx: 0 })] }), rig: {}, hz: 10 });
    const a = sim.observation().agents[0];
    expect(a.vx).toBeCloseTo(-10, 6); // ego closes at 10 m/s
  });
});

describe("sense() sensor frame", () => {
  it("populates camera + lidar + radar + groundTruth for a full rig", () => {
    const sim = new DriveSim({
      scenario: createScenario({ ego: { speed: 0 }, actors: [car({ x: 12, y: 0 })] }),
      rig: { camera: createCamera({ width: 32, height: 18 }), lidar: createLidar({ azimuthSteps: 30, elevationSteps: 2, maxRange: 60 }), radar: createRadar() },
      hz: 10,
    });
    const frame = sim.sense();
    expect(frame.camera!.rgb.length).toBe(32 * 18 * 4);
    expect(frame.lidar!.returns.length).toBeGreaterThan(0);
    expect(frame.radar!.length).toBe(1);
    expect(frame.groundTruth).toHaveLength(1);
    expect(frame.tick).toBe(0);
  });

  it("returns only groundTruth for an empty rig", () => {
    const sim = new DriveSim({ scenario: createScenario({ actors: [car({ x: 5, y: 0 })] }), rig: {}, hz: 10 });
    const frame = sim.sense();
    expect(frame.camera).toBeUndefined();
    expect(frame.lidar).toBeUndefined();
    expect(frame.radar).toBeUndefined();
    expect(frame.groundTruth).toHaveLength(1);
  });
});

describe("step() open- vs closed-loop", () => {
  it("open-loop explicit action advances the ego", () => {
    const sim = new DriveSim({ scenario: createScenario({ ego: { speed: 8 } }), rig: {}, hz: 10 });
    const x0 = sim.world.ego.x;
    for (let i = 0; i < 10; i++) sim.step({ action: { accel: 0, curvature: 0 } });
    expect(sim.world.ego.x).toBeGreaterThan(x0);
    expect(sim.tick).toBe(10);
  });

  it("closed-loop model brakes for a pedestrian ahead", () => {
    const sim = new DriveSim({
      scenario: createScenario({ ego: { speed: 8 }, actors: [car({ id: "p", kind: "pedestrian", x: 14, extent: [0.3, 0.3, 0.9] })] }),
      rig: {},
      hz: 10,
      command: "keep_lane",
    });
    const v0 = sim.world.ego.speed;
    const model = AlpamayoR1.fromPretrained();
    for (let i = 0; i < 20; i++) sim.step({ model });
    expect(sim.world.ego.speed).toBeLessThan(v0);
  });

  it("reset restores the initial scenario and zeroes the clock", () => {
    const sim = new DriveSim({ scenario: createScenario({ ego: { speed: 9 } }), rig: {}, hz: 10 });
    for (let i = 0; i < 5; i++) sim.step();
    expect(sim.tick).toBe(5);
    sim.reset();
    expect(sim.tick).toBe(0);
    expect(sim.world.ego.x).toBe(0);
    expect(sim.world.ego.speed).toBe(9);
  });
});
