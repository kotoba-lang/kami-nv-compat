// @etzhayyim/kami-nv-compat/actions
//
// TypeScript port of nv_compat.isaaclab.envs.mdp.actions — action
// wrappers + ActionManager. Each wrapper transforms one slice of a
// policy's flat action vector into env-buffer writes.
//
// Iter 74 surface:
//   - ArticulatedEnv interface — contract that wrappers consume
//   - writeEffort dispatch helper (effort buffer routing)
//   - ActionTerm base + ActionTermCfgBase
//   - ActionManager (compose multiple terms)
//   - DifferentialInverseKinematicsAction (iter 64; composes iter 72 DiffIK)
//   - OperationalSpaceControllerAction (iter 64; composes iter 73 OSC)
//   - BinaryJointPositionAction (iter 65; gripper)
//   - NonHolonomicAction (iter 65; differential-drive mobile base)
//
// ADR-2605261800 §D6 nv-compat namespace localization.

export { type ArticulatedEnv, writeEffort } from "./articulated-env.js";
export {
  type ActionTermCfgBase,
  ActionTerm,
  ActionManager,
} from "./action-term.js";
export {
  type DifferentialInverseKinematicsActionCfg,
  DifferentialInverseKinematicsAction,
  type OperationalSpaceControllerActionCfg,
  OperationalSpaceControllerAction,
  type BinaryJointPositionActionCfg,
  BinaryJointPositionAction,
  type NonHolonomicActionCfg,
  NonHolonomicAction,
} from "./task-space-actions.js";
