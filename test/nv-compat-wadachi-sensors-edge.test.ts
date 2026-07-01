/**
 * nv-compat wadachi-sim sensor boundary cases.
 *
 * Coverage for the DriveSim sensor models: sensor mount pose under ego yaw,
 * radar FOV / range exclusion + Doppler sign (closing vs receding, ego-motion
 * contribution), LiDAR maxRange + no-hit + ray count, and oriented-box world
 * AABB expansion.
 *
 *     pnpm exec vitest run test/nv-compat-wadachi-sensors-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 wadachi-sim.
 */

import { describe, it, expect } from "vitest";
import {
  type Actor,
  buildSensorScene,
  sampleLidar,
  sampleRadar,
  sensorPose,
  worldAabb,
} from "../src/wadachi-sim/index.js";
import { createScenario, createLidar, createRadar } from "../src/drive-sim.js";

const actor = (over: Partial<Actor>): Actor => ({
  id: "a", kind: "vehicle", x: 0, y: 0, yaw: 0, vx: 0, vy: 0, extent: [2, 1, 0.75], ...over,
});

describe("sensorPose mount offset", () => {
  it("places the sensor at the mount offset for a forward-facing ego", () => {
    const { origin, heading } = sensorPose(
      { x: 0, y: 0, yaw: 0, speed: 0, extent: [2, 1, 0.75] },
      { forward: 2, left: 1, height: 1.5, yaw: 0 },
    );
    expect(origin).toEqual([2, 1, 1.5]);
    expect(heading).toBe(0);
  });

  it("rotates the mount offset by the ego heading (yaw 90°)", () => {
    const { origin, heading } = sensorPose(
      { x: 0, y: 0, yaw: Math.PI / 2, speed: 0, extent: [2, 1, 0.75] },
      { forward: 2, left: 1, height: 1.5, yaw: 0 },
    );
    // forward → +y, left → -x at yaw 90°.
    expect(origin[0]).toBeCloseTo(-1, 6);
    expect(origin[1]).toBeCloseTo(2, 6);
    expect(heading).toBeCloseTo(Math.PI / 2, 6);
  });
});

describe("radar FOV / range / Doppler", () => {
  const ego0 = { speed: 0 } as const;

  it("drops an actor beyond maxRange", () => {
    const sc = createScenario({ ego: ego0, actors: [actor({ x: 200, y: 0 })] });
    expect(sampleRadar(sc, createRadar({ maxRange: 150 }))).toHaveLength(0);
  });

  it("reports positive range-rate for a receding actor, negative for a closing one", () => {
    const recede = createScenario({ ego: ego0, actors: [actor({ x: 30, vx: 5 })] }); // moving +x, away
    expect(sampleRadar(recede, createRadar())[0].rangeRate).toBeGreaterThan(0);
    const close = createScenario({ ego: ego0, actors: [actor({ x: 30, vx: -5 })] });
    expect(sampleRadar(close, createRadar())[0].rangeRate).toBeLessThan(0);
  });

  it("ego forward motion makes a static actor ahead read as closing", () => {
    const sc = createScenario({ ego: { speed: 10 }, actors: [actor({ x: 30, vx: 0 })] });
    expect(sampleRadar(sc, createRadar())[0].rangeRate).toBeLessThan(0);
  });

  it("a left-offset actor in FOV has positive azimuth", () => {
    const sc = createScenario({ ego: ego0, actors: [actor({ x: 10, y: 10 })] });
    const dets = sampleRadar(sc, createRadar({ azimuthFovDeg: 120 }));
    expect(dets).toHaveLength(1);
    expect(dets[0].azimuth).toBeGreaterThan(0);
  });
});

describe("LiDAR range + hit count", () => {
  const forwardRay = createLidar({ azimuthFovDeg: 0, azimuthSteps: 1, elevationFovDeg: 0, elevationSteps: 1, mount: { forward: 1.5, left: 0, height: 1.5, yaw: 0 } });

  it("returns nothing when the only hit lies beyond maxRange", () => {
    const sc = createScenario({ ego: { speed: 0 }, obstacles: [{ id: "wall", kind: "vehicle", x: 50, y: 0, yaw: 0, extent: [1, 1, 1] }] });
    const scan = sampleLidar(sc, buildSensorScene(sc), { ...forwardRay, maxRange: 30 });
    expect(scan.rays).toBe(1);
    expect(scan.returns).toHaveLength(0); // front face at x=49 → range 47.5 > 30
  });

  it("a horizontal ray over empty ground hits nothing", () => {
    const sc = createScenario({ ego: { speed: 0 } }); // ground only, no obstacles
    const scan = sampleLidar(sc, buildSensorScene(sc), { ...forwardRay, maxRange: 80 });
    expect(scan.returns).toHaveLength(0); // horizontal ray never meets the z=0 ground
  });

  it("casts azimuthSteps × elevationSteps rays", () => {
    const sc = createScenario({ ego: { speed: 0 }, obstacles: [{ id: "w", kind: "vehicle", x: 10, y: 0, yaw: 0, extent: [2, 2, 2] }] });
    const scan = sampleLidar(sc, buildSensorScene(sc), createLidar({ azimuthFovDeg: 90, azimuthSteps: 12, elevationFovDeg: 20, elevationSteps: 4, maxRange: 60 }));
    expect(scan.rays).toBe(12 * 4);
    expect(scan.returns.length).toBeGreaterThan(0);
  });
});

describe("worldAabb of an oriented box", () => {
  it("a 45° rotation expands the axis-aligned footprint", () => {
    const { min, max } = worldAabb(0, 0, [2, 1, 0.75], Math.PI / 4);
    const ex = (max[0] - min[0]) / 2;
    const ey = (max[1] - min[1]) / 2;
    expect(ex).toBeGreaterThan(2); // longer than the 2 half-length
    expect(ey).toBeGreaterThan(1); // wider than the 1 half-width
    expect(max[2]).toBeCloseTo(1.5, 6); // height unchanged (2 × hz)
  });
});
