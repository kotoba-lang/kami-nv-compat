/**
 * nv-compat isaacsim.core.api / e7m-sim validation.
 *
 * Exercises the clean-room Isaac Sim core simulation context (World +
 * Articulation + RigidPrim) over the Featherstone articulated-dynamics module,
 * including loading the Franka asset URDF, PD joint tracking, forward
 * kinematics, and rigid-body fall under gravity. Deterministic, CPU-only.
 *
 *     pnpm exec vitest run test/nv-compat-isaac-sim.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.1 isaacsim.core.api surface.
 */

import { describe, it, expect } from "vitest";
import { Articulation, RigidPrim, World, core } from "../src/isaac-sim.js";
import { makeFrankaPanda, buildSerialChainUrdf } from "../src/assets/index.js";

describe("RigidPrim", () => {
  it("falls under gravity (semi-implicit Euler)", () => {
    const world = new World({ physicsDt: 1 / 100, gravity: [0, 0, -9.81] });
    const box = world.addRigidPrim(new RigidPrim("box", 1, [0, 0, 10]));
    const z0 = box.position[2];
    world.step(100); // 1 s
    expect(box.position[2]).toBeLessThan(z0);
    // After ~1 s of free fall, velocity ≈ -g.
    expect(box.linearVelocity[2]).toBeCloseTo(-9.81, 0);
  });

  it("an applied upward force can hold a body against gravity", () => {
    const world = new World({ physicsDt: 1 / 100 });
    const box = new RigidPrim("b", 2, [0, 0, 0]);
    world.addRigidPrim(box);
    for (let i = 0; i < 50; i++) {
      box.applyForce([0, 0, 2 * 9.81]); // mg upward each tick
      world.step(1);
    }
    // Net force ~0 → stays near the origin.
    expect(Math.abs(box.position[2])).toBeLessThan(0.2);
  });

  it("reset restores the initial pose", () => {
    const box = new RigidPrim("b", 1, [1, 2, 3]);
    const world = new World();
    world.addRigidPrim(box);
    world.step(10);
    world.reset();
    expect(box.position).toEqual([1, 2, 3]);
    expect(box.linearVelocity).toEqual([0, 0, 0]);
    expect(world.stepCount).toBe(0);
  });
});

describe("Articulation (Franka Panda)", () => {
  const franka = makeFrankaPanda();

  it("loads the URDF and exposes DoF + joint names", () => {
    const art = Articulation.fromUrdf("franka", franka.urdfText);
    expect(art.numDof).toBeGreaterThanOrEqual(7);
    expect(art.jointNames.length).toBe(art.numDof);
    expect(art.getJointPositions()).toHaveLength(art.numDof);
  });

  it("PD position targets drive the joints toward the target (2-link chain)", () => {
    // A simple 2-revolute chain integrates stably under explicit Euler.
    const urdf = buildSerialChainUrdf("arm", [
      { name: "j0", type: "revolute", axis: [0, 0, 1] },
      { name: "j1", type: "revolute", axis: [0, 0, 1] },
    ]);
    const target = [0.4, -0.3];
    const art = Articulation.fromUrdf("arm", urdf, [0, 0]);
    art.setPdGains(40, 8);
    const err0 = jointError(art.getJointPositions(), target);
    const world = new World({ physicsDt: 1 / 240, gravity: [0, 0, 0] });
    world.addArticulation(art);
    for (let i = 0; i < 2000; i++) {
      art.applyAction({ jointPositions: target });
      world.step(1);
    }
    const q = art.getJointPositions();
    for (const v of q) expect(Number.isFinite(v)).toBe(true);
    const err1 = jointError(q, target);
    expect(err1).toBeLessThan(err0);
    expect(err1).toBeLessThan(0.1); // converges close to the target
  });

  it("direct joint efforts move the joints and stay finite", () => {
    const art = Articulation.fromUrdf("franka", franka.urdfText);
    const world = new World({ physicsDt: 1 / 240, gravity: [0, 0, -9.81] });
    world.addArticulation(art);
    const tau = new Array<number>(art.numDof).fill(0);
    tau[0] = 5;
    art.setJointEfforts(tau);
    world.step(20);
    for (const q of art.getJointPositions()) expect(Number.isFinite(q)).toBe(true);
    expect(art.getJointPositions()[0]).not.toBe(0); // joint 0 moved
  });

  it("forward kinematics returns a world pose per joint", () => {
    const art = Articulation.fromUrdf("franka", franka.urdfText);
    const poses = art.forwardKinematics();
    expect(poses).toHaveLength(art.numDof);
    expect(poses[0].p).toHaveLength(3);
    expect(poses[0].R).toHaveLength(3);
    const named = art.getBodyPose(art.jointNames[0]);
    expect(named).not.toBeNull();
    expect(named!.position).toHaveLength(3);
  });

  it("reset returns joints to the default pose", () => {
    const n = makeFrankaPanda().jointNames.length;
    const home = new Array<number>(n).fill(0.1);
    const art = Articulation.fromUrdf("franka", franka.urdfText, home);
    const world = new World();
    world.addArticulation(art);
    art.setJointEfforts(new Array<number>(n).fill(3));
    world.step(30);
    world.reset();
    expect(art.getJointPositions()[0]).toBeCloseTo(0.1, 9);
    expect(art.getJointVelocities().every((v) => v === 0)).toBe(true);
  });
});

describe("World simulation context", () => {
  it("tracks the scene, advances the clock, and resets", () => {
    const world = new World({ physicsDt: 1 / 60 });
    world.addArticulation(Articulation.fromUrdf("franka", makeFrankaPanda().urdfText));
    world.addRigidPrim(new RigidPrim("box", 1, [0, 0, 5]));
    expect(world.scene().articulations).toContain("franka");
    expect(world.scene().rigidPrims).toContain("box");
    world.step(60);
    expect(world.stepCount).toBe(60);
    expect(world.time).toBeCloseTo(1.0, 6);
    world.reset();
    expect(world.time).toBe(0);
  });

  it("exposes the isaacsim.core.api namespace", () => {
    expect(core.api.World).toBe(World);
    expect(core.api.RigidPrim).toBe(RigidPrim);
    expect(typeof core.api.Articulation.fromUrdf).toBe("function");
  });
});

function jointError(q: readonly number[], target: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < q.length; i++) s += (q[i] - target[i]) ** 2;
  return Math.sqrt(s / q.length);
}
