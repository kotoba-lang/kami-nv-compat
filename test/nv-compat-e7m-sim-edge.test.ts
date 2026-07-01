/**
 * nv-compat e7m-sim RigidPrim / Articulation / World edge cases.
 *
 * Boundary coverage for the Isaac Sim core: rigid-body quaternion integration
 * under angular velocity, pose accessors, World substeps + body lookups, and
 * Articulation action padding (action shorter than the DoF count), unknown-body
 * pose, and accessor shapes.
 *
 *     pnpm exec vitest run test/nv-compat-e7m-sim-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 e7m-sim.
 */

import { describe, it, expect } from "vitest";
import { Articulation, RigidPrim, World, core } from "../src/isaac-sim.js";
import { buildSerialChainUrdf } from "../src/assets/index.js";

describe("RigidPrim quaternion + pose", () => {
  it("integrates orientation under angular velocity, staying unit-norm", () => {
    const world = new World({ physicsDt: 1 / 200, gravity: [0, 0, 0] });
    const body = world.addRigidPrim(new RigidPrim("spinner", 1, [0, 0, 0], [0, 0, 0, 1]));
    body.angularVelocity = [0, 0, 1]; // 1 rad/s about +z
    world.step(100); // 0.5 s
    const q = body.orientation;
    expect(Math.hypot(q[0], q[1], q[2], q[3])).toBeCloseTo(1, 5); // re-normalized
    expect(Math.abs(q[2])).toBeGreaterThan(0); // rotated about z
  });

  it("setLinearVelocity + getPose round-trip", () => {
    const body = new RigidPrim("b", 1, [1, 2, 3]);
    body.setLinearVelocity([0, 0, 5]);
    const world = new World({ physicsDt: 0.1, gravity: [0, 0, 0] });
    world.addRigidPrim(body);
    world.step(1);
    const pose = body.getPose();
    expect(pose.position[2]).toBeCloseTo(3 + 5 * 0.1, 6);
    expect(pose.orientation).toHaveLength(4);
  });
});

describe("World substeps + body lookups", () => {
  it("advances the clock by substeps × dt in one call", () => {
    const world = new World({ physicsDt: 1 / 50 });
    world.addRigidPrim(new RigidPrim("b", 1, [0, 0, 10]));
    world.step(25);
    expect(world.stepCount).toBe(25);
    expect(world.time).toBeCloseTo(0.5, 6);
  });

  it("getRigidPrim / getArticulation resolve by name; unknown → undefined", () => {
    const world = new World();
    world.addRigidPrim(new RigidPrim("box", 1));
    expect(world.getRigidPrim("box")).toBeDefined();
    expect(world.getRigidPrim("nope")).toBeUndefined();
    expect(world.getArticulation("nope")).toBeUndefined();
  });
});

describe("Articulation action padding + accessors", () => {
  const urdf = buildSerialChainUrdf("arm", [
    { name: "j0", type: "revolute", axis: [0, 0, 1] },
    { name: "j1", type: "revolute", axis: [0, 0, 1] },
  ]);

  it("pads a too-short position target to the DoF count without crashing", () => {
    const art = Articulation.fromUrdf("arm", urdf, [0, 0]);
    const world = new World({ physicsDt: 1 / 240, gravity: [0, 0, 0] });
    world.addArticulation(art);
    art.setPdGains(20, 5);
    for (let i = 0; i < 200; i++) {
      art.applyAction({ jointPositions: [0.3] }); // only 1 of 2 → padded with 0
      world.step(1);
    }
    for (const q of art.getJointPositions()) expect(Number.isFinite(q)).toBe(true);
    expect(art.getJointPositions()[0]).toBeGreaterThan(0); // tracked toward 0.3
  });

  it("pads a too-short effort vector", () => {
    const art = Articulation.fromUrdf("arm", urdf);
    const world = new World({ physicsDt: 1 / 240, gravity: [0, 0, 0] });
    world.addArticulation(art);
    art.setJointEfforts([1]); // 1 of 2 → padded
    world.step(10);
    for (const q of art.getJointPositions()) expect(Number.isFinite(q)).toBe(true);
  });

  it("getBodyPose returns null for an unknown joint; accessors have DoF length", () => {
    const art = Articulation.fromUrdf("arm", urdf);
    expect(art.getBodyPose("does-not-exist")).toBeNull();
    expect(art.getJointAccelerations()).toHaveLength(art.numDof);
    expect(art.getJointVelocities()).toHaveLength(art.numDof);
    const named = art.getBodyPose(art.jointNames[0]);
    expect(named!.rotation).toHaveLength(3);
  });
});

describe("isaacsim.core.api namespace", () => {
  it("exposes World / RigidPrim / Articulation", () => {
    expect(core.api.World).toBe(World);
    expect(core.api.RigidPrim).toBe(RigidPrim);
    expect(typeof core.api.Articulation.fromUrdf).toBe("function");
  });
});
