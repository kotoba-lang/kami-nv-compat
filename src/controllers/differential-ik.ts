// TypeScript port of kotodama.nv_compat.isaaclab.controllers.differential_ik
//
// DifferentialIKController — Jacobian-based inverse kinematics for arm
// reaching. Maps a 6-DOF task-space command (pose or position) onto a
// joint-space delta via damped least squares (DLS) or Moore-Penrose
// pseudoinverse over the articulation Jacobian.
//
// Mirrors `isaaclab.controllers.DifferentialIKController` (Isaac Lab 1.x).
// Algorithm-for-algorithm port of the Python iter 41 reference impl.
//
// ADR-2605261800 §D6 nv-compat namespace localization. Trademark:
// "Isaac®" is a trademark of NVIDIA Corporation; this module is API
// namespace localization for forward dynamics interoperability per
// Google v. Oracle (2021).

// ── Config ────────────────────────────────────────────────────────────────

export type DifferentialIKCommandType = "pose" | "position";
export type DifferentialIKMethod = "dls" | "pinv";

export interface DifferentialIKControllerCfg {
  /** `"pose"` → 7-elem [x,y,z,qx,qy,qz,qw] (full 6-DOF target).
   * `"position"` → 3-elem [x,y,z] (orientation ignored). */
  commandType: DifferentialIKCommandType;
  /** When true, set_command receives a DELTA from current EE pose
   *  (matches Isaac Lab's residual teleop convention). */
  useRelativeMode: boolean;
  /** `"dls"` damped-least-squares (default), or `"pinv"` Moore-Penrose. */
  ikMethod: DifferentialIKMethod;
  /** DLS damping {"lambdaVal": float}. Defaults to 0.05 — bigger = more
   *  stable near singularities, slower convergence. */
  ikParams: { lambdaVal: number };
}

export function makeDefaultDifferentialIKCfg(
  overrides: Partial<DifferentialIKControllerCfg> = {},
): DifferentialIKControllerCfg {
  return {
    commandType: "pose",
    useRelativeMode: false,
    ikMethod: "dls",
    ikParams: { lambdaVal: 0.05 },
    ...overrides,
  };
}

// ── Quaternion math (Hamilton convention, (x, y, z, w)) ──────────────────

export type Quat = readonly [number, number, number, number];
export type Vec3 = readonly [number, number, number];

export function quatInverse(q: Quat): [number, number, number, number] {
  const [qx, qy, qz, qw] = q;
  const n2 = qx * qx + qy * qy + qz * qz + qw * qw;
  if (n2 < 1e-24) return [0, 0, 0, 1];
  const inv = 1 / n2;
  return [-qx * inv, -qy * inv, -qz * inv, qw * inv];
}

export function quatMul(q1: Quat, q2: Quat): [number, number, number, number] {
  const [x1, y1, z1, w1] = q1;
  const [x2, y2, z2, w2] = q2;
  return [
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
  ];
}

/**
 * Swap a 6×n Jacobian from Featherstone spatial convention `[ω; v]`
 * (angular rows first, used by iter 71 `geometricJacobian`) into Isaac
 * Lab's IK convention `[v; ω]` (linear rows first, used by
 * `DifferentialIKController.compute`).
 *
 * Both impls swap the top 3 rows with the bottom 3 rows.
 */
export function spatialToIsaaclabJacobian(J: number[][]): number[][] {
  if (J.length !== 6) {
    throw new Error(`spatialToIsaaclabJacobian: expected 6 rows; got ${J.length}`);
  }
  return [J[3], J[4], J[5], J[0], J[1], J[2]];
}

export function axisAngleVec(q: Quat): [number, number, number] {
  let [qx, qy, qz, qw] = q;
  if (qw < 0) {
    qx = -qx;
    qy = -qy;
    qz = -qz;
    qw = -qw;
  }
  qw = Math.max(-1, Math.min(1, qw));
  const angle = 2 * Math.acos(qw);
  const s = Math.sqrt(Math.max(0, 1 - qw * qw));
  if (s < 1e-8) return [0, 0, 0];
  const invS = 1 / s;
  return [qx * invS * angle, qy * invS * angle, qz * invS * angle];
}

// ── Controller ────────────────────────────────────────────────────────────

export class DifferentialIKController {
  readonly cfg: DifferentialIKControllerCfg;
  readonly numEnvs: number;
  /** Per-env target pose: [px, py, pz, qx, qy, qz, qw]. */
  private readonly _target: number[][];

  constructor(cfg: DifferentialIKControllerCfg, numEnvs: number = 1) {
    if (numEnvs <= 0) {
      throw new Error(`numEnvs must be > 0; got ${numEnvs}`);
    }
    if (cfg.commandType !== "pose" && cfg.commandType !== "position") {
      throw new Error(
        `commandType must be 'pose' or 'position'; got '${String(cfg.commandType)}'`,
      );
    }
    if (cfg.ikMethod !== "dls" && cfg.ikMethod !== "pinv") {
      throw new Error(`ikMethod must be 'dls' or 'pinv'; got '${String(cfg.ikMethod)}'`);
    }
    this.cfg = cfg;
    this.numEnvs = numEnvs;
    this._target = Array.from({ length: numEnvs }, () => [0, 0, 0, 0, 0, 0, 1]);
  }

  /** Length of one command vector.
   *
   * - `"pose" + abs`: 7  (xyz + quat xyzw)
   * - `"pose" + rel`: 6  (xyz delta + axis-angle delta)
   * - `"position" + *`: 3 (xyz)
   */
  get actionDim(): number {
    if (this.cfg.commandType === "pose") {
      return this.cfg.useRelativeMode ? 6 : 7;
    }
    return 3;
  }

  /** Clear targets (identity pose at origin) for the named envs (or all). */
  reset(envIds?: readonly number[]): void {
    const ids = envIds ?? Array.from({ length: this.numEnvs }, (_, i) => i);
    for (const i of ids) {
      this._target[i] = [0, 0, 0, 0, 0, 0, 1];
    }
  }

  /** Set the target pose for env_idx.
   *
   * - If `useRelativeMode=false` (default): `command` is the absolute
   *   world-frame target (7-elem for "pose", 3-elem for "position").
   * - If `useRelativeMode=true`: `command` is a DELTA added to the current
   *   EE pose. Position delta is added in world frame; orientation delta
   *   (for "pose" mode) is a 3-element axis-angle rotation applied as
   *   `target_q = q_delta * ee_quat`.
   *
   * For relative mode, `eePos` and `eeQuat` MUST be supplied.
   */
  setCommand(
    command: readonly number[],
    opts: { eePos?: Vec3; eeQuat?: Quat; envIdx?: number } = {},
  ): void {
    const { eePos, eeQuat, envIdx = 0 } = opts;
    if (command.length !== this.actionDim) {
      throw new Error(
        `command must be length ${this.actionDim} for ` +
          `commandType='${this.cfg.commandType}'; got ${command.length}`,
      );
    }
    if (envIdx < 0 || envIdx >= this.numEnvs) {
      throw new RangeError(`envIdx=${envIdx} out of range [0, ${this.numEnvs})`);
    }
    let targetPos: Vec3;
    let targetQuat: Quat;
    if (this.cfg.useRelativeMode) {
      if (eePos === undefined || eeQuat === undefined) {
        throw new Error("useRelativeMode=true requires eePos + eeQuat to setCommand");
      }
      targetPos = [
        eePos[0] + command[0],
        eePos[1] + command[1],
        eePos[2] + command[2],
      ];
      if (this.cfg.commandType === "pose") {
        const axang: Vec3 = [command[3], command[4], command[5]];
        const angle = Math.sqrt(axang[0] ** 2 + axang[1] ** 2 + axang[2] ** 2);
        let qDelta: Quat;
        if (angle < 1e-9) {
          qDelta = [0, 0, 0, 1];
        } else {
          const ax: Vec3 = [axang[0] / angle, axang[1] / angle, axang[2] / angle];
          const h = angle * 0.5;
          const s = Math.sin(h);
          qDelta = [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(h)];
        }
        targetQuat = quatMul(qDelta, eeQuat);
      } else {
        targetQuat = eeQuat;
      }
    } else {
      targetPos = [command[0], command[1], command[2]];
      if (this.cfg.commandType === "pose") {
        targetQuat = [command[3], command[4], command[5], command[6]];
      } else {
        const t = this._target[envIdx];
        targetQuat = [t[3], t[4], t[5], t[6]];
      }
    }
    this._target[envIdx] = [
      targetPos[0],
      targetPos[1],
      targetPos[2],
      targetQuat[0],
      targetQuat[1],
      targetQuat[2],
      targetQuat[3],
    ];
  }

  /** Returns the 7-element target pose for env_idx (defensive copy). */
  getTarget(envIdx: number = 0): number[] {
    return [...this._target[envIdx]];
  }

  /** Compute joint-space delta to drive EE → target.
   *
   * @param eePos    current EE position [x,y,z] in world frame
   * @param eeQuat   current EE quaternion [qx,qy,qz,qw]
   * @param jacobian 6×n list-of-lists (rows = task-space DoF in order
   *                 [vx, vy, vz, wx, wy, wz]; cols = n joints)
   * @param envIdx   per-env target index
   * @returns length-n joint-position delta.
   */
  compute(args: {
    eePos: Vec3;
    eeQuat: Quat;
    jacobian: number[][];
    envIdx?: number;
  }): number[] {
    const { eePos, eeQuat, jacobian, envIdx = 0 } = args;
    if (jacobian.length !== 6) {
      throw new Error(
        `jacobian must be 6 rows (linear x/y/z + angular x/y/z); got ${jacobian.length}`,
      );
    }
    const n = jacobian[0].length;
    for (const row of jacobian) {
      if (row.length !== n) {
        throw new Error("jacobian rows must all have the same width");
      }
    }
    // 1. Pose error.
    const target = this._target[envIdx];
    const tPos: Vec3 = [target[0], target[1], target[2]];
    const tQuat: Quat = [target[3], target[4], target[5], target[6]];
    const posErr: Vec3 = [tPos[0] - eePos[0], tPos[1] - eePos[1], tPos[2] - eePos[2]];
    let oriErr: Vec3;
    if (this.cfg.commandType === "pose") {
      const qErr = quatMul(tQuat, quatInverse(eeQuat));
      oriErr = axisAngleVec(qErr);
    } else {
      oriErr = [0, 0, 0];
    }
    const error = [posErr[0], posErr[1], posErr[2], oriErr[0], oriErr[1], oriErr[2]];

    // 2. Solve. Both DLS and pinv use the same DLS solver — for pinv we use
    // a vanishingly small lambda (recovers Moore-Penrose for full-rank J).
    let lam: number;
    if (this.cfg.ikMethod === "dls") {
      lam = this.cfg.ikParams.lambdaVal;
    } else {
      lam = 1e-6;
    }
    return solveDls(jacobian, error, lam, n);
  }
}

// ── DLS solver ────────────────────────────────────────────────────────────

/**
 * Damped least squares: Δq = J^T (J J^T + λ²I)^-1 error.
 *
 * Steps:
 *   1. A = J J^T + λ²I    (6×6)
 *   2. b = error          (6-vec)
 *   3. y = A^-1 b         (6-vec) via Gauss-Jordan
 *   4. Δq = J^T y         (n-vec)
 */
function solveDls(
  J: readonly number[][],
  error: readonly number[],
  lam: number,
  n: number,
): number[] {
  const lam2 = lam * lam;
  // 1. A = J J^T + λ²I.
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
  // 2. Solve A y = error via Gauss-Jordan on augmented 6×7.
  const aug: number[][] = A.map((r, i) => [...r, error[i]]);
  for (let col = 0; col < 6; col++) {
    let piv = col;
    let maxAbs = Math.abs(aug[col][col]);
    for (let r = col + 1; r < 6; r++) {
      if (Math.abs(aug[r][col]) > maxAbs) {
        maxAbs = Math.abs(aug[r][col]);
        piv = r;
      }
    }
    if (maxAbs < 1e-18) continue; // singular column; DLS damping should prevent
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
  // 4. Δq = J^T y.
  const deltaQ = new Array<number>(n).fill(0);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < 6; i++) s += J[i][k] * y[i];
    deltaQ[k] = s;
  }
  return deltaQ;
}
