// TypeScript port of kotodama.nv_compat.isaaclab.controllers.operational_space
//
// OperationalSpaceController — task-space torque control. Sibling of
// iter 72 DifferentialIKController:
//
//   - DifferentialIKController: target pose → joint POSITION delta via
//     DLS over Jacobian. Pairs with joint-position action.
//   - OperationalSpaceController: target pose / wrench → joint TORQUE
//     via Jacobian transpose. Pairs with joint-effort action directly.
//
// OSC = canonical contact-rich manipulation controller. Cartesian
// impedance (K_p · pos_err + K_d · vel_err) → 6-DOF task-space wrench →
// joint torque τ = Jᵀ F_task. Optional null-space joint regularization
// for redundant arms (n > 6) + optional gravity compensation.
//
// Algorithm-for-algorithm port of the Python iter 63 reference impl
// (~480 LoC Python).
//
// ADR-2605261800 §D6 nv-compat namespace localization. Pure TypeScript,
// zero runtime deps.

import { type Vec3, type Quat, quatMul, quatInverse, axisAngleVec } from "./differential-ik.js";

// ── Config ────────────────────────────────────────────────────────────────

export type OscTargetType =
  | "pose_abs"
  | "pose_rel"
  | "wrench_abs"
  | "force_abs"
  | "torque_abs";

export type OscImpedanceMode = "fixed" | "variable";

export type OscNullspaceControl = "none" | "position";

export interface OperationalSpaceControllerCfg {
  /** Target categories accepted by the controller (composed in order). */
  targetTypes: OscTargetType[];
  /** `"fixed"` — gains constant from cfg; `"variable"` — gains arrive in
   *  the command vector each step. */
  impedanceMode: OscImpedanceMode;
  /** 6-vec K_p_task (3 linear + 3 angular). */
  motionStiffnessTask: number[];
  /** 6-vec damping ratio ζ (1.0 = critically damped). */
  motionDampingRatioTask: number[];
  /** [min, max] gain bounds when impedanceMode="variable". */
  motionStiffnessLimits: [number, number];
  motionDampingLimits: [number, number];
  /** `"none"` or `"position"`. When `"position"`, regularize redundant
   *  DoFs toward `nullspaceTargetPos`. */
  nullspaceControl: OscNullspaceControl;
  /** Scalar gains for the null-space P/D loop. */
  nullspaceStiffness: number;
  nullspaceDampingRatio: number;
  /** When true, `compute` adds gravity-comp torque G(q) to the output. */
  gravityCompensation: boolean;
}

export function makeDefaultOscCfg(
  overrides: Partial<OperationalSpaceControllerCfg> = {},
): OperationalSpaceControllerCfg {
  return {
    targetTypes: ["pose_abs"],
    impedanceMode: "fixed",
    motionStiffnessTask: [100, 100, 100, 100, 100, 100],
    motionDampingRatioTask: [1, 1, 1, 1, 1, 1],
    motionStiffnessLimits: [0, 1000],
    motionDampingLimits: [0, 100],
    nullspaceControl: "none",
    nullspaceStiffness: 10,
    nullspaceDampingRatio: 1,
    gravityCompensation: false,
    ...overrides,
  };
}

// ── Controller ────────────────────────────────────────────────────────────

export class OperationalSpaceController {
  readonly cfg: OperationalSpaceControllerCfg;
  readonly numEnvs: number;
  readonly numDof: number;
  /** Per-env target pose [px, py, pz, qx, qy, qz, qw]. */
  private readonly _target: number[][];
  /** Per-env variable gains (length 12 = 6 stiffness + 6 damping ratio)
   *  or null when in fixed mode. */
  private readonly _variableStiffness: (number[] | null)[];

  constructor(
    cfg: OperationalSpaceControllerCfg,
    numEnvs: number = 1,
    numDof: number = 7,
  ) {
    if (numEnvs <= 0) throw new Error(`numEnvs must be > 0; got ${numEnvs}`);
    if (numDof <= 0) throw new Error(`numDof must be > 0; got ${numDof}`);
    if (cfg.impedanceMode !== "fixed" && cfg.impedanceMode !== "variable") {
      throw new Error(`impedanceMode must be 'fixed' or 'variable'; got '${String(cfg.impedanceMode)}'`);
    }
    if (cfg.nullspaceControl !== "none" && cfg.nullspaceControl !== "position") {
      throw new Error(`nullspaceControl must be 'none' or 'position'; got '${String(cfg.nullspaceControl)}'`);
    }
    if (cfg.motionStiffnessTask.length !== 6) {
      throw new Error(`motionStiffnessTask must be 6-vec; got ${cfg.motionStiffnessTask.length}`);
    }
    if (cfg.motionDampingRatioTask.length !== 6) {
      throw new Error(`motionDampingRatioTask must be 6-vec; got ${cfg.motionDampingRatioTask.length}`);
    }
    this.cfg = cfg;
    this.numEnvs = numEnvs;
    this.numDof = numDof;
    this._target = Array.from({ length: numEnvs }, () => [0, 0, 0, 0, 0, 0, 1]);
    this._variableStiffness = new Array<number[] | null>(numEnvs).fill(null);
  }

  /** Length of one command vector.
   * pose_abs=7, pose_rel=6, wrench_abs=6, force_abs/torque_abs=3.
   * impedanceMode="variable" adds 12 (6 stiffness + 6 damping ratio).
   */
  get actionDim(): number {
    let dim = 0;
    for (const tt of this.cfg.targetTypes) {
      if (tt === "pose_abs") dim += 7;
      else if (tt === "pose_rel") dim += 6;
      else if (tt === "wrench_abs") dim += 6;
      else if (tt === "force_abs" || tt === "torque_abs") dim += 3;
    }
    if (this.cfg.impedanceMode === "variable") dim += 12;
    return dim;
  }

  reset(envIds?: readonly number[]): void {
    const ids = envIds ?? Array.from({ length: this.numEnvs }, (_, i) => i);
    for (const i of ids) {
      this._target[i] = [0, 0, 0, 0, 0, 0, 1];
      this._variableStiffness[i] = null;
    }
  }

  /** Set the target. For pose_abs: command = [px,py,pz,qx,qy,qz,qw].
   * For pose_rel: command = [Δx,Δy,Δz,Δax,Δay,Δaz] axis-angle and
   * eePos/eeQuat MUST be supplied.
   * In `variable` mode, the LAST 12 elements are stiffness (6) + damping (6).
   */
  setCommand(
    command: readonly number[],
    opts: { eePos?: Vec3; eeQuat?: Quat; envIdx?: number } = {},
  ): void {
    const { eePos, eeQuat, envIdx = 0 } = opts;
    if (envIdx < 0 || envIdx >= this.numEnvs) {
      throw new RangeError(`envIdx=${envIdx} out of range [0, ${this.numEnvs})`);
    }
    const expected = this.actionDim;
    if (command.length !== expected) {
      throw new Error(
        `command must be length ${expected} (targetTypes=${JSON.stringify(this.cfg.targetTypes)}, ` +
          `impedanceMode='${this.cfg.impedanceMode}'); got ${command.length}`,
      );
    }
    let cmd: readonly number[] = command;
    if (this.cfg.impedanceMode === "variable") {
      const gainsStart = expected - 12;
      this._variableStiffness[envIdx] = [...command.slice(gainsStart)];
      cmd = command.slice(0, gainsStart);
    }
    let idx = 0;
    for (const tt of this.cfg.targetTypes) {
      if (tt === "pose_abs") {
        this._target[envIdx] = [
          cmd[idx], cmd[idx + 1], cmd[idx + 2],
          cmd[idx + 3], cmd[idx + 4], cmd[idx + 5], cmd[idx + 6],
        ];
        idx += 7;
      } else if (tt === "pose_rel") {
        if (eePos === undefined || eeQuat === undefined) {
          throw new Error("pose_rel requires eePos + eeQuat");
        }
        const delta: number[] = [cmd[idx], cmd[idx + 1], cmd[idx + 2], cmd[idx + 3], cmd[idx + 4], cmd[idx + 5]];
        this._target[envIdx][0] = eePos[0] + delta[0];
        this._target[envIdx][1] = eePos[1] + delta[1];
        this._target[envIdx][2] = eePos[2] + delta[2];
        const angle = Math.sqrt(delta[3] ** 2 + delta[4] ** 2 + delta[5] ** 2);
        let qDelta: Quat;
        if (angle < 1e-9) {
          qDelta = [0, 0, 0, 1];
        } else {
          const ax: Vec3 = [delta[3] / angle, delta[4] / angle, delta[5] / angle];
          const h = angle * 0.5;
          const s = Math.sin(h);
          qDelta = [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(h)];
        }
        const tq = quatMul(qDelta, eeQuat);
        this._target[envIdx][3] = tq[0];
        this._target[envIdx][4] = tq[1];
        this._target[envIdx][5] = tq[2];
        this._target[envIdx][6] = tq[3];
        idx += 6;
      } else if (tt === "wrench_abs") {
        idx += 6;
      } else {
        idx += 3;
      }
    }
  }

  getTarget(envIdx: number = 0): number[] {
    return [...this._target[envIdx]];
  }

  /** Compute joint torques.
   *
   * Pipeline:
   *   1. pose error e = [pos_err (3-vec); ori_err (axis-angle 3-vec)]
   *   2. velocity error de = -[ee_lin_vel; ee_ang_vel] (target rest = 0)
   *   3. F_task = K_p · e + K_d · de  (K_d = 2 · ζ · sqrt(K_p))
   *   4. τ = Jᵀ · F_task
   *   5. (optional) τ_ns = (I - Jᵀ J⁺ᵀ) · (K_p_ns · (q_target - q) - K_d_ns · qdot)
   *   6. (optional) τ += G(q)
   */
  compute(args: {
    eePos: Vec3;
    eeQuat: Quat;
    eeLinVel: Vec3;
    eeAngVel: Vec3;
    jacobian: number[][];
    jointPos?: readonly number[];
    jointVel?: readonly number[];
    massMatrix?: number[][];
    gravityTorque?: readonly number[];
    nullspaceTargetPos?: readonly number[];
    envIdx?: number;
  }): number[] {
    const {
      eePos, eeQuat, eeLinVel, eeAngVel, jacobian,
      jointPos, jointVel, gravityTorque, nullspaceTargetPos, envIdx = 0,
    } = args;
    if (jacobian.length !== 6) {
      throw new Error(`jacobian must have 6 rows; got ${jacobian.length}`);
    }
    const n = jacobian[0].length;
    if (n !== this.numDof) {
      throw new Error(`jacobian width ${n} != cfg.numDof ${this.numDof}`);
    }
    const target = this._target[envIdx];
    const tPos: Vec3 = [target[0], target[1], target[2]];
    const tQuat: Quat = [target[3], target[4], target[5], target[6]];

    // 1. Pose error
    const posErr: Vec3 = [tPos[0] - eePos[0], tPos[1] - eePos[1], tPos[2] - eePos[2]];
    const qErr = quatMul(tQuat, quatInverse(eeQuat));
    const oriErr = axisAngleVec(qErr);
    const error = [posErr[0], posErr[1], posErr[2], oriErr[0], oriErr[1], oriErr[2]];

    // 2. Velocity error
    const dError = [-eeLinVel[0], -eeLinVel[1], -eeLinVel[2], -eeAngVel[0], -eeAngVel[1], -eeAngVel[2]];

    // 3. Task-space wrench
    let kpTask: readonly number[];
    let kdRatio: readonly number[];
    const vs = this._variableStiffness[envIdx];
    if (this.cfg.impedanceMode === "variable" && vs !== null) {
      kpTask = vs.slice(0, 6);
      kdRatio = vs.slice(6, 12);
    } else {
      kpTask = this.cfg.motionStiffnessTask;
      kdRatio = this.cfg.motionDampingRatioTask;
    }
    const kdTask: number[] = [];
    for (let i = 0; i < 6; i++) {
      kdTask.push(2 * kdRatio[i] * Math.sqrt(Math.max(0, kpTask[i])));
    }
    const FTask: number[] = [];
    for (let i = 0; i < 6; i++) {
      FTask.push(kpTask[i] * error[i] + kdTask[i] * dError[i]);
    }

    // 4. Joint torque via Jᵀ F_task
    const tau = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < 6; k++) s += jacobian[k][i] * FTask[k];
      tau[i] = s;
    }

    // 5. Null-space joint regularization
    if (
      this.cfg.nullspaceControl === "position" &&
      nullspaceTargetPos !== undefined &&
      jointPos !== undefined &&
      jointVel !== undefined &&
      n > 6
    ) {
      const kpNs = this.cfg.nullspaceStiffness;
      const kdNs = 2 * this.cfg.nullspaceDampingRatio * Math.sqrt(Math.max(0, kpNs));
      const tauNs: number[] = [];
      for (let i = 0; i < n; i++) {
        tauNs.push(kpNs * (nullspaceTargetPos[i] - jointPos[i]) - kdNs * jointVel[i]);
      }
      const tauNsProj = projectToNullspace(jacobian, tauNs, n, 0.05);
      for (let i = 0; i < n; i++) tau[i] += tauNsProj[i];
    }

    // 6. Gravity compensation
    if (this.cfg.gravityCompensation && gravityTorque !== undefined) {
      const m = Math.min(n, gravityTorque.length);
      for (let i = 0; i < m; i++) tau[i] += gravityTorque[i];
    }
    return tau;
  }
}

// ── Null-space projection (damped pseudoinverse) ─────────────────────────

/**
 * Project tauNs into the null-space of Jacobian J via:
 *   P_null = I - Jᵀ (J Jᵀ + λ²I)⁻¹ J
 *   tau_proj = P_null · tauNs
 *
 * Uses Gauss-Jordan 6×6 inverse with partial pivoting + 1e-18 singularity
 * guard. Damping (λ²=0.0025) gives numerical stability near singular
 * configurations.
 */
function projectToNullspace(
  J: readonly number[][],
  tauNs: readonly number[],
  n: number,
  lam: number,
): number[] {
  // Step 1: J · tauNs (6-vec)
  const Jtau = new Array<number>(6).fill(0);
  for (let k = 0; k < 6; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += J[k][i] * tauNs[i];
    Jtau[k] = s;
  }
  // Step 2: solve (J Jᵀ + λ²I) y = Jtau
  const lam2 = lam * lam;
  const A: number[][] = [];
  for (let i = 0; i < 6; i++) {
    const row = new Array<number>(6).fill(0);
    for (let j = 0; j < 6; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += J[i][k] * J[j][k];
      row[j] = s + (i === j ? lam2 : 0);
    }
    A.push(row);
  }
  const aug: number[][] = A.map((r, i) => [...r, Jtau[i]]);
  for (let col = 0; col < 6; col++) {
    let piv = col;
    let maxAbs = Math.abs(aug[col][col]);
    for (let r = col + 1; r < 6; r++) {
      if (Math.abs(aug[r][col]) > maxAbs) {
        maxAbs = Math.abs(aug[r][col]);
        piv = r;
      }
    }
    if (maxAbs < 1e-18) continue;
    if (piv !== col) {
      const tmp = aug[col];
      aug[col] = aug[piv];
      aug[piv] = tmp;
    }
    const pv = aug[col][col];
    for (let j = col; j < 7; j++) aug[col][j] /= pv;
    for (let r = 0; r < 6; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      if (Math.abs(f) < 1e-18) continue;
      for (let j = col; j < 7; j++) aug[r][j] -= f * aug[col][j];
    }
  }
  const y = aug.map((r) => r[6]);
  // Step 3: Jᵀ · y  (n-vec)
  const Jty = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < 6; k++) s += J[k][i] * y[k];
    Jty[i] = s;
  }
  // Step 4: tau_proj = tauNs - Jty
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = tauNs[i] - Jty[i];
  return out;
}
