// @etzhayyim/kami-nv-compat/controllers
//
// TypeScript port of nv_compat.isaaclab.controllers — high-level
// joint-space + task-space controllers.
//
// Iter 72 surface:
//   - DifferentialIKController — DLS / pinv joint-space delta from
//     6-DOF task-space command (paired with iter 71 geometricJacobian).
//
// Future iters will add OperationalSpaceController (iter 63 → TS) and
// the action-wrapper surface (iter 64/65 → TS).
//
// ADR-2605261800 §D6 nv-compat namespace localization.

export {
  type DifferentialIKCommandType,
  type DifferentialIKMethod,
  type DifferentialIKControllerCfg,
  type Quat,
  type Vec3,
  makeDefaultDifferentialIKCfg,
  quatInverse,
  quatMul,
  axisAngleVec,
  spatialToIsaaclabJacobian,
  DifferentialIKController,
} from "./differential-ik.js";

export {
  type OscTargetType,
  type OscImpedanceMode,
  type OscNullspaceControl,
  type OperationalSpaceControllerCfg,
  makeDefaultOscCfg,
  OperationalSpaceController,
} from "./operational-space.js";
