// @etzhayyim/kami-nv-compat/e7m-sim
//
// Clean-room Isaac Sim core engine (e7m-sim) — the canonical KAMI
// implementation behind `nv-compat/isaac-sim`. Reproduces the documented
// `isaacsim.core.api` surface (World simulation context + Articulation + RigidPrim)
// over the existing Featherstone articulated-dynamics module, so Isaac Sim
// scripts port to KAMI via import-path-only changes.
//
// Clean-room: from-spec simulation-context API over textbook rigid-body +
// reduced-coordinate dynamics. No Isaac Sim source/USD/binaries.
// ADR-2605261800 §D6 / D10.4 e7m-sim (Isaac Sim → e7m-sim).

import {
  type ArticulatedState,
  type BuiltArticulation,
  type JointWorldPose,
  type UrdfArticulatedSystem,
  articulatedStep,
  buildArticulation,
  forwardKinematics,
  makeZeroState,
  parseUrdf,
} from "../dynamics/index.js";

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // [x, y, z, w]

// ── Articulation (isaacsim.core.api.Articulation) ────────────────────────────

export interface ArticulationActionInput {
  /** Per-joint position targets (PD-tracked). */
  jointPositions?: number[];
  /** Per-joint effort (torque/force) applied directly. */
  jointEfforts?: number[];
}

export interface ArticulationGains {
  kp: number;
  kd: number;
}

/** A reduced-coordinate articulation view (mirrors isaacsim Articulation). */
export class Articulation {
  readonly built: BuiltArticulation;
  readonly state: ArticulatedState;
  private readonly defaultQ: number[];
  private gains: ArticulationGains = { kp: 60, kd: 6 };
  private pendingEffort: number[];

  constructor(
    readonly name: string,
    system: UrdfArticulatedSystem,
    defaultJointPositions?: number[],
  ) {
    this.built = buildArticulation(system);
    this.state = makeZeroState(this.built.n);
    this.defaultQ = defaultJointPositions
      ? [...defaultJointPositions]
      : new Array<number>(this.built.n).fill(0);
    for (let i = 0; i < this.built.n; i++) this.state.q[i] = this.defaultQ[i] ?? 0;
    this.pendingEffort = new Array<number>(this.built.n).fill(0);
  }

  get numDof(): number {
    return this.built.n;
  }
  get jointNames(): string[] {
    return [...this.built.jointNames];
  }
  getJointPositions(): number[] {
    return [...this.state.q];
  }
  getJointVelocities(): number[] {
    return [...this.state.qdot];
  }
  getJointAccelerations(): number[] {
    return [...this.state.qddot];
  }

  setPdGains(kp: number, kd: number): void {
    this.gains = { kp, kd };
  }

  /** Queue an action applied on the next World step (effort or PD-tracked
   *  position targets). */
  applyAction(action: ArticulationActionInput): void {
    if (action.jointEfforts) {
      this.pendingEffort = padTo(action.jointEfforts, this.built.n);
    } else if (action.jointPositions) {
      const target = padTo(action.jointPositions, this.built.n);
      const tau = new Array<number>(this.built.n);
      for (let i = 0; i < this.built.n; i++) {
        tau[i] = this.gains.kp * (target[i] - this.state.q[i]) - this.gains.kd * this.state.qdot[i];
      }
      this.pendingEffort = tau;
    }
  }

  setJointEfforts(tau: number[]): void {
    this.pendingEffort = padTo(tau, this.built.n);
  }

  /** Advance the articulation one physics step under `gravity`. */
  step(dt: number, gravity: Vec3): void {
    articulatedStep(this.built, this.state, this.pendingEffort, dt, gravity);
  }

  /** Reset to the default joint positions (zero velocity / effort). */
  reset(): void {
    for (let i = 0; i < this.built.n; i++) {
      this.state.q[i] = this.defaultQ[i] ?? 0;
      this.state.qdot[i] = 0;
      this.state.qddot[i] = 0;
      this.pendingEffort[i] = 0;
    }
  }

  /** Per-joint world poses via forward kinematics. */
  forwardKinematics(): JointWorldPose[] {
    return forwardKinematics(this.built, this.state.q);
  }

  /** World pose of a named joint frame (or null if unknown). */
  getBodyPose(jointName: string): { position: Vec3; rotation: number[][] } | null {
    const idx = this.built.jointNames.indexOf(jointName);
    if (idx < 0) return null;
    const pose = this.forwardKinematics()[idx];
    return { position: [pose.p[0], pose.p[1], pose.p[2]], rotation: pose.R };
  }
}

function padTo(a: readonly number[], n: number): number[] {
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < Math.min(a.length, n); i++) out[i] = a[i];
  return out;
}

/** Build an Articulation from URDF text. */
export function articulationFromUrdf(name: string, urdfText: string, defaultQ?: number[]): Articulation {
  return new Articulation(name, parseUrdf(urdfText), defaultQ);
}

// ── RigidPrim (isaacsim.core.api.RigidPrim) ──────────────────────────────────

/** A single rigid body integrated under gravity + applied force/torque. */
export class RigidPrim {
  position: Vec3;
  orientation: Quat;
  linearVelocity: Vec3 = [0, 0, 0];
  angularVelocity: Vec3 = [0, 0, 0];
  private force: Vec3 = [0, 0, 0];
  private readonly initial: { position: Vec3; orientation: Quat };

  constructor(
    readonly name: string,
    readonly mass = 1,
    position: Vec3 = [0, 0, 0],
    orientation: Quat = [0, 0, 0, 1],
  ) {
    this.position = [...position];
    this.orientation = [...orientation];
    this.initial = { position: [...position], orientation: [...orientation] };
  }

  /** Apply a world-frame force for the next step (cleared after stepping). */
  applyForce(f: Vec3): void {
    this.force = [this.force[0] + f[0], this.force[1] + f[1], this.force[2] + f[2]];
  }
  setLinearVelocity(v: Vec3): void {
    this.linearVelocity = [...v];
  }
  getPose(): { position: Vec3; orientation: Quat } {
    return { position: [...this.position], orientation: [...this.orientation] };
  }

  /** Semi-implicit Euler integration under gravity + applied force. */
  step(dt: number, gravity: Vec3): void {
    for (let i = 0; i < 3; i++) {
      const a = this.force[i] / this.mass + gravity[i];
      this.linearVelocity[i] += a * dt;
      this.position[i] += this.linearVelocity[i] * dt;
    }
    // Integrate orientation by the (small) angular velocity quaternion.
    const [wx, wy, wz] = this.angularVelocity;
    const [qx, qy, qz, qw] = this.orientation;
    const dqx = 0.5 * (wx * qw + wy * qz - wz * qy);
    const dqy = 0.5 * (-wx * qz + wy * qw + wz * qx);
    const dqz = 0.5 * (wx * qy - wy * qx + wz * qw);
    const dqw = 0.5 * (-wx * qx - wy * qy - wz * qz);
    let nx = qx + dqx * dt, ny = qy + dqy * dt, nz = qz + dqz * dt, nw = qw + dqw * dt;
    const len = Math.hypot(nx, ny, nz, nw) || 1;
    this.orientation = [nx / len, ny / len, nz / len, nw / len];
    this.force = [0, 0, 0];
  }

  reset(): void {
    this.position = [...this.initial.position];
    this.orientation = [...this.initial.orientation];
    this.linearVelocity = [0, 0, 0];
    this.angularVelocity = [0, 0, 0];
    this.force = [0, 0, 0];
  }
}

// ── World (isaacsim.core.api.World) ──────────────────────────────────────────

export interface WorldCfg {
  physicsDt: number;
  gravity: Vec3;
}

/** Simulation context owning a scene of articulations + rigid prims. */
export class World {
  readonly physicsDt: number;
  readonly gravity: Vec3;
  private readonly articulations = new Map<string, Articulation>();
  private readonly rigidPrims = new Map<string, RigidPrim>();
  time = 0;
  stepCount = 0;

  constructor(cfg: Partial<WorldCfg> = {}) {
    this.physicsDt = cfg.physicsDt ?? 1 / 60;
    this.gravity = cfg.gravity ?? [0, 0, -9.81];
  }

  addArticulation(a: Articulation): Articulation {
    this.articulations.set(a.name, a);
    return a;
  }
  addRigidPrim(r: RigidPrim): RigidPrim {
    this.rigidPrims.set(r.name, r);
    return r;
  }
  getArticulation(name: string): Articulation | undefined {
    return this.articulations.get(name);
  }
  getRigidPrim(name: string): RigidPrim | undefined {
    return this.rigidPrims.get(name);
  }
  scene(): { articulations: string[]; rigidPrims: string[] } {
    return { articulations: [...this.articulations.keys()], rigidPrims: [...this.rigidPrims.keys()] };
  }

  /** Advance the simulation by `substeps` physics ticks. */
  step(substeps = 1): void {
    for (let s = 0; s < substeps; s++) {
      for (const a of this.articulations.values()) a.step(this.physicsDt, this.gravity);
      for (const r of this.rigidPrims.values()) r.step(this.physicsDt, this.gravity);
      this.time += this.physicsDt;
      this.stepCount++;
    }
  }

  /** Reset all bodies to their initial state and the clock to zero. */
  reset(): void {
    for (const a of this.articulations.values()) a.reset();
    for (const r of this.rigidPrims.values()) r.reset();
    this.time = 0;
    this.stepCount = 0;
  }
}
