/**
 * nv-compat kami-drive unicycle physical-accuracy invariants.
 *
 * A constant-curvature command traces a circular arc of radius 1/κ; this
 * checks the BEV unicycle integrator against that geometry: points stay on the
 * turning circle, yaw advances at speed·κ, straight motion holds heading, and
 * arc length tracks speed·time.
 *
 *     pnpm exec vitest run test/nv-compat-unicycle-circle.test.ts
 *
 * AV scope per ADR-2606010600 (kami-autodrive).
 */

import { describe, it, expect } from "vitest";
import { rolloutTrajectory, trajectoryLength } from "../src/kami-drive/index.js";

const yawOf = (wp: { rotation: number[] }) => Math.atan2(wp.rotation[3], wp.rotation[0]);

describe("constant curvature → circular arc", () => {
  const v = 10;
  const k = 0.1; // radius R = 1/κ = 10
  const dt = 0.01;
  const N = 100; // sweeps ~1 rad
  const wps = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: v }, new Array(N).fill({ accel: 0, curvature: k }), dt);

  it("keeps every waypoint near the turning circle (centre at (0, R))", () => {
    const R = 1 / k;
    for (const wp of wps) {
      const d = Math.hypot(wp.translation[0] - 0, wp.translation[1] - R);
      expect(d).toBeCloseTo(R, 0); // within ±0.5 of radius 10 over the arc
    }
  });

  it("advances yaw at rate speed·κ", () => {
    expect(yawOf(wps[N])).toBeCloseTo(v * k * N * dt, 2); // ≈ 1 rad
  });

  it("a negative curvature turns the other way (yaw decreases)", () => {
    const right = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: v }, new Array(N).fill({ accel: 0, curvature: -k }), dt);
    expect(yawOf(right[N])).toBeLessThan(0);
  });
});

describe("straight-line + arc length", () => {
  it("zero curvature holds heading and advances along +x", () => {
    const v = 10;
    const wps = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: v }, new Array(100).fill({ accel: 0, curvature: 0 }), 0.01);
    expect(yawOf(wps[100])).toBeCloseTo(0, 9);
    expect(wps[100].translation[0]).toBeCloseTo(10, 6); // 10 m/s × 1 s
    expect(wps[100].translation[1]).toBeCloseTo(0, 9);
  });

  it("arc length ≈ speed × time at constant speed (straight or turning)", () => {
    const v = 10;
    const straight = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: v }, new Array(100).fill({ accel: 0, curvature: 0 }), 0.01);
    const turning = rolloutTrajectory({ x: 0, y: 0, yaw: 0, speed: v }, new Array(100).fill({ accel: 0, curvature: 0.1 }), 0.01);
    expect(trajectoryLength(straight)).toBeCloseTo(10, 4); // 10 m/s × 1 s
    expect(trajectoryLength(turning)).toBeCloseTo(10, 2); // same arc length, just curved
  });
});
