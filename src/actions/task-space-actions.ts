// TypeScript port of kotodama.nv_compat.isaaclab.envs.mdp.actions
// task-space wrappers (iter 64).
//
// DifferentialInverseKinematicsAction
//   Action layout depends on controller_cfg.command_type ×
//   useRelativeMode:
//     pose + abs: 7 (px,py,pz,qx,qy,qz,qw)
//     pose + rel: 6 (dx,dy,dz, axis-angle rotation 3-vec)
//     position:   3 (xyz)
//   Pipeline:
//     processActions: scale + offset
//     applyActions:
//       1. env.getJacobian(bodyName), getEePose(bodyName), joint state
//       2. controller.setCommand(processed, ee_pos, ee_quat)
//       3. controller.compute() returns joint DELTA
//       4. target = q_arm + delta; PD into effort
//       5. dispatch via writeEffort
//
// OperationalSpaceControllerAction
//   Action vector size = controller.actionDim (varies with target_types
//   × impedance_mode).
//   Pipeline:
//     processActions: scale + offset
//     applyActions:
//       1. read jacobian + ee_pose + ee_velocity + joint state +
//          (optional) gravity_torque
//       2. controller.setCommand(processed)
//       3. controller.compute() returns joint torques DIRECTLY (OSC IS
//          the loop — no PD layer)
//       4. nullspace target defaults to current q if cfg.nullspaceControl
//          != "none" and no explicit target supplied
//       5. dispatch via writeEffort
//
// ADR-2605261800 §D6 nv-compat namespace localization.

import {
  DifferentialIKController,
  type DifferentialIKControllerCfg,
  makeDefaultDifferentialIKCfg,
  type Vec3,
  type Quat,
} from "../controllers/differential-ik.js";
import {
  OperationalSpaceController,
  type OperationalSpaceControllerCfg,
  makeDefaultOscCfg,
} from "../controllers/operational-space.js";
import { ActionTerm, type ActionTermCfgBase } from "./action-term.js";
import { type ArticulatedEnv, writeEffort } from "./articulated-env.js";

// ── DifferentialInverseKinematicsAction ──────────────────────────────────

export interface DifferentialInverseKinematicsActionCfg extends ActionTermCfgBase {
  /** EE link name (passed to env.getJacobian / getEePose). */
  bodyName: string;
  /** Controller cfg; defaults to makeDefaultDifferentialIKCfg(). */
  controllerCfg?: DifferentialIKControllerCfg;
  /** PD gains for converting IK output (joint position target) → effort. */
  pGain?: number;
  dGain?: number;
}

export class DifferentialInverseKinematicsAction extends ActionTerm {
  readonly cfg: DifferentialInverseKinematicsActionCfg;
  readonly controller: DifferentialIKController;
  private readonly _controllerCfg: DifferentialIKControllerCfg;

  constructor(cfg: DifferentialInverseKinematicsActionCfg) {
    const controllerCfg = cfg.controllerCfg ?? makeDefaultDifferentialIKCfg();
    const controller = new DifferentialIKController(controllerCfg, 1);
    const inferred = controller.actionDim;
    if (cfg.actionDim !== undefined && cfg.actionDim !== inferred) {
      throw new Error(
        `DifferentialInverseKinematicsActionCfg: actionDim=${cfg.actionDim} ` +
          `contradicts controller actionDim=${inferred}`,
      );
    }
    super({ ...cfg, actionDim: inferred });
    this.cfg = cfg;
    this._controllerCfg = controllerCfg;
    this.controller = controller;
  }

  reset(): void {
    super.reset();
    this.controller.reset();
  }

  applyActions(env: ArticulatedEnv): void {
    if (!env.getJacobian) {
      throw new Error("DifferentialInverseKinematicsAction: env must expose getJacobian(bodyName)");
    }
    if (!env.getEePose) {
      throw new Error("DifferentialInverseKinematicsAction: env must expose getEePose(bodyName)");
    }
    const jacobian = env.getJacobian(this.cfg.bodyName);
    const [eePos, eeQuat] = env.getEePose(this.cfg.bodyName);
    const qFull = env.getJointPositions();
    const dqFull = env.getJointVelocities();
    const qArm: number[] = this.cfg.jointNames.map((j) => (j < qFull.length ? qFull[j] : 0));
    // 2. Push command into IK.
    this.controller.setCommand(this.processed_actions, {
      eePos,
      eeQuat,
    });
    // 3. Compute joint-space DELTA.
    const jointDelta = this.controller.compute({
      eePos,
      eeQuat,
      jacobian,
    });
    // 4. Joint target = q_arm + delta; PD into effort.
    const pGain = this.cfg.pGain ?? 100.0;
    const dGain = this.cfg.dGain ?? 10.0;
    const torquesToApply: [number, number][] = [];
    for (let slot = 0; slot < this.cfg.jointNames.length; slot++) {
      const joint = this.cfg.jointNames[slot];
      const target = qArm[slot] + jointDelta[slot];
      const qj = joint < qFull.length ? qFull[joint] : 0;
      const dqj = joint < dqFull.length ? dqFull[joint] : 0;
      const tau = pGain * (target - qj) - dGain * dqj;
      torquesToApply.push([joint, tau]);
    }
    writeEffort(env, torquesToApply, false);
  }
}

// ── OperationalSpaceControllerAction ─────────────────────────────────────

export interface OperationalSpaceControllerActionCfg extends ActionTermCfgBase {
  /** EE link name (passed to env getters). */
  bodyName: string;
  /** Controller cfg; defaults to makeDefaultOscCfg(). */
  controllerCfg?: OperationalSpaceControllerCfg;
  /** Optional null-space target (length=jointNames.length). When
   *  controllerCfg.nullspaceControl !== "none" and this is unset, the
   *  null-space target defaults to current joint positions. */
  nullspaceJointTargets?: number[];
}

export class OperationalSpaceControllerAction extends ActionTerm {
  readonly cfg: OperationalSpaceControllerActionCfg;
  readonly controller: OperationalSpaceController;
  private readonly _controllerCfg: OperationalSpaceControllerCfg;

  constructor(cfg: OperationalSpaceControllerActionCfg) {
    const controllerCfg = cfg.controllerCfg ?? makeDefaultOscCfg();
    const controller = new OperationalSpaceController(
      controllerCfg,
      1,
      cfg.jointNames.length,
    );
    const inferred = controller.actionDim;
    if (cfg.actionDim !== undefined && cfg.actionDim !== inferred) {
      throw new Error(
        `OperationalSpaceControllerActionCfg: actionDim=${cfg.actionDim} ` +
          `contradicts controller actionDim=${inferred}`,
      );
    }
    super({ ...cfg, actionDim: inferred });
    this.cfg = cfg;
    this._controllerCfg = controllerCfg;
    this.controller = controller;
  }

  applyActions(env: ArticulatedEnv): void {
    if (!env.getJacobian || !env.getEePose || !env.getEeVelocity) {
      throw new Error(
        "OperationalSpaceControllerAction: env must expose getJacobian / getEePose / getEeVelocity",
      );
    }
    const jacobian = env.getJacobian(this.cfg.bodyName);
    const [eePos, eeQuat] = env.getEePose(this.cfg.bodyName);
    const [eeLinVel, eeAngVel] = env.getEeVelocity(this.cfg.bodyName);
    const qFull = env.getJointPositions();
    const dqFull = env.getJointVelocities();
    const qArm: number[] = this.cfg.jointNames.map((j) => (j < qFull.length ? qFull[j] : 0));
    const dqArm: number[] = this.cfg.jointNames.map((j) => (j < dqFull.length ? dqFull[j] : 0));

    // Push command. Pass current EE pose unconditionally — OSC ignores
    // when not in pose_rel mode.
    this.controller.setCommand(this.processed_actions, {
      eePos,
      eeQuat,
    });

    // Gravity comp source
    let gravityTorque: readonly number[] | undefined;
    if (this._controllerCfg.gravityCompensation) {
      if (!env.getGravityTorque) {
        throw new Error(
          "OperationalSpaceControllerAction: gravityCompensation=true requires env.getGravityTorque",
        );
      }
      gravityTorque = env.getGravityTorque(this.cfg.bodyName);
    }

    // Null-space target source
    let nullspaceTargetPos: number[] | undefined;
    if (this._controllerCfg.nullspaceControl !== "none") {
      nullspaceTargetPos = this.cfg.nullspaceJointTargets
        ? [...this.cfg.nullspaceJointTargets]
        : [...qArm]; // default = "stay where you are"
    }

    // Compute torques.
    const jointTorques = this.controller.compute({
      eePos,
      eeQuat,
      eeLinVel,
      eeAngVel,
      jacobian,
      jointPos: qArm,
      jointVel: dqArm,
      nullspaceTargetPos,
      gravityTorque,
    });

    // Direct write — no PD layer (OSC IS the loop).
    const torquesToApply: [number, number][] = this.cfg.jointNames.map(
      (joint, slot) => [joint, jointTorques[slot]] as [number, number],
    );
    writeEffort(env, torquesToApply, false);
  }
}

// ── BinaryJointPositionAction (gripper open/close) ──────────────────────

export interface BinaryJointPositionActionCfg extends ActionTermCfgBase {
  /** Per-joint open pose. Length must equal jointNames.length. */
  openCommand: number[];
  /** Per-joint close pose. Length must equal jointNames.length. */
  closeCommand: number[];
  /** Threshold for binary decision (default 0.0). action >= threshold → close. */
  threshold?: number;
  pGain?: number;
  dGain?: number;
}

export class BinaryJointPositionAction extends ActionTerm {
  readonly cfg: BinaryJointPositionActionCfg;
  private _isClose: boolean = false;

  constructor(cfg: BinaryJointPositionActionCfg) {
    if (cfg.actionDim !== undefined && cfg.actionDim !== 1) {
      throw new Error(`BinaryJointPositionActionCfg.actionDim must be 1 or undefined; got ${cfg.actionDim}`);
    }
    const n = cfg.jointNames.length;
    if (cfg.openCommand.length !== n) {
      throw new Error(
        `BinaryJointPositionActionCfg.openCommand length ${cfg.openCommand.length} ` +
          `must match jointNames length ${n}`,
      );
    }
    if (cfg.closeCommand.length !== n) {
      throw new Error(
        `BinaryJointPositionActionCfg.closeCommand length ${cfg.closeCommand.length} ` +
          `must match jointNames length ${n}`,
      );
    }
    super({ ...cfg, actionDim: 1 });
    this.cfg = cfg;
  }

  get isClose(): boolean {
    return this._isClose;
  }

  reset(): void {
    super.reset();
    this._isClose = false;
  }

  applyActions(env: ArticulatedEnv): void {
    const threshold = this.cfg.threshold ?? 0.0;
    this._isClose = this.processed_actions[0] >= threshold;
    const targetPose = this._isClose ? this.cfg.closeCommand : this.cfg.openCommand;
    const q = env.getJointPositions();
    const dq = env.getJointVelocities();
    const pGain = this.cfg.pGain ?? 100.0;
    const dGain = this.cfg.dGain ?? 10.0;
    const torquesToApply: [number, number][] = [];
    for (let slot = 0; slot < this.cfg.jointNames.length; slot++) {
      const joint = this.cfg.jointNames[slot];
      const target = targetPose[slot];
      const qj = joint < q.length ? q[joint] : 0;
      const dqj = joint < dq.length ? dq[joint] : 0;
      const tau = pGain * (target - qj) - dGain * dqj;
      torquesToApply.push([joint, tau]);
    }
    writeEffort(env, torquesToApply, false);
  }
}

// ── NonHolonomicAction (differential-drive mobile base) ─────────────────

export interface NonHolonomicActionCfg extends ActionTermCfgBase {
  /** Wheel radius (m). */
  wheelRadius: number;
  /** Lateral distance between wheels (m). */
  wheelSeparation: number;
  /** Velocity-loop P gain (no D since target is velocity). */
  pGain?: number;
}

export class NonHolonomicAction extends ActionTerm {
  readonly cfg: NonHolonomicActionCfg;
  private _wheelVelocityTarget: [number, number] = [0, 0];

  constructor(cfg: NonHolonomicActionCfg) {
    if (cfg.jointNames.length !== 2) {
      throw new Error(
        `NonHolonomicActionCfg.jointNames must be [leftWheel, rightWheel] (length 2); got length ${cfg.jointNames.length}`,
      );
    }
    if (cfg.wheelRadius <= 0) {
      throw new Error(`NonHolonomicActionCfg.wheelRadius must be > 0; got ${cfg.wheelRadius}`);
    }
    if (cfg.wheelSeparation <= 0) {
      throw new Error(`NonHolonomicActionCfg.wheelSeparation must be > 0; got ${cfg.wheelSeparation}`);
    }
    if (cfg.actionDim !== undefined && cfg.actionDim !== 2) {
      throw new Error(`NonHolonomicActionCfg.actionDim must be 2 or undefined; got ${cfg.actionDim}`);
    }
    super({ ...cfg, actionDim: 2 });
    this.cfg = cfg;
  }

  /** Override default processActions — explicit length check. */
  processActions(raw: readonly number[]): void {
    if (raw.length !== 2) {
      throw new Error(`NonHolonomicAction: expected 2 action elements (v_x, ω_z); got ${raw.length}`);
    }
    this.raw_actions = [...raw];
    const s = this.cfg.scale ?? 1.0;
    const o = this.cfg.offset ?? 0.0;
    this.processed_actions = raw.map((r) => r * s + o);
  }

  applyActions(env: ArticulatedEnv): void {
    const v_x = this.processed_actions[0];
    const omega_z = this.processed_actions[1];
    const halfL = this.cfg.wheelSeparation / 2;
    const r = this.cfg.wheelRadius;
    const omega_left = (v_x - omega_z * halfL) / r;
    const omega_right = (v_x + omega_z * halfL) / r;
    this._wheelVelocityTarget = [omega_left, omega_right];
    const dq = env.getJointVelocities();
    const [leftJoint, rightJoint] = this.cfg.jointNames;
    const dq_left = leftJoint < dq.length ? dq[leftJoint] : 0;
    const dq_right = rightJoint < dq.length ? dq[rightJoint] : 0;
    const pGain = this.cfg.pGain ?? 10.0;
    const tau_left = pGain * (omega_left - dq_left);
    const tau_right = pGain * (omega_right - dq_right);
    writeEffort(
      env,
      [
        [leftJoint, tau_left],
        [rightJoint, tau_right],
      ],
      false,
    );
  }

  get wheelVelocityTarget(): readonly [number, number] {
    return this._wheelVelocityTarget;
  }
}
