// Articulated env interface — the contract action wrappers expect.
//
// Both joint-space (effort/position/velocity) and task-space (DiffIK /
// OSC) action wrappers consume an object that conforms to this
// interface. Real Isaac Lab envs implement it via their internal sim
// substrate; for end-to-end browser-side use, callers wire iter 71
// dynamics + iter 70 forward kinematics into a small adapter class.
//
// Conventions:
//   - joint positions / velocities indexed by integer joint index
//     (0..n-1; same indexing as nv_compat dynamics BuiltArticulation).
//   - Jacobian: 6×n list-of-lists, rows = task-space DoF in either
//     `[v; ω]` (Isaac Lab) or `[ω; v]` (Featherstone). The DiffIK
//     wrapper passes through as-is; callers must ensure the Jacobian
//     matches the controller's expected layout (use
//     spatialToIsaaclabJacobian helper from iter 72 if needed).
//   - EE pose: ([x,y,z] position, [qx,qy,qz,qw] quaternion).
//   - Effort buffers: the dispatch helper below writes onto whichever
//     of _appliedTorques / _appliedForce / _actions[0] the env exposes.

import type { Quat, Vec3 } from "../controllers/differential-ik.js";

/** What the action wrappers need to read from the env. */
export interface ArticulatedEnv {
  getJointPositions(): readonly number[];
  getJointVelocities(): readonly number[];
  /** Optional — required by DifferentialInverseKinematicsAction +
   *  OperationalSpaceControllerAction. */
  getJacobian?(bodyName: string): number[][];
  getEePose?(bodyName: string): readonly [Vec3, Quat];
  getEeVelocity?(bodyName: string): readonly [Vec3, Vec3];
  /** Optional — required only when gravityCompensation=true. */
  getGravityTorque?(bodyName: string): readonly number[];
  /** Effort dispatch targets. Action wrappers write to whichever
   *  exists; at least one is required. */
  _appliedTorques?: number[];
  _appliedForce?: number;
  _actions?: number[][];
}

/** Write per-joint torques onto whichever effort buffer the env exposes.
 *
 * Dispatch chain (mirrors Python `_write_effort`):
 *   1. `env._appliedTorques` (multi-DoF revolute)
 *   2. `env._appliedForce` (single-DoF prismatic — only when single
 *      joint=0 AND `singleDofForceOk` is true)
 *   3. `env._actions[0]` (DirectRLEnv per-env buffer)
 */
export function writeEffort(
  env: ArticulatedEnv,
  torquesToApply: ReadonlyArray<readonly [number, number]>,
  singleDofForceOk: boolean = true,
): void {
  if (env._appliedTorques !== undefined) {
    const torques = env._appliedTorques;
    let maxIdx = 0;
    for (const [j] of torquesToApply) if (j > maxIdx) maxIdx = j;
    while (torques.length < maxIdx + 1) torques.push(0);
    for (const [j, t] of torquesToApply) torques[j] = t;
    return;
  }
  if (
    singleDofForceOk &&
    env._appliedForce !== undefined &&
    torquesToApply.length === 1 &&
    torquesToApply[0][0] === 0
  ) {
    env._appliedForce = torquesToApply[0][1];
    return;
  }
  if (env._actions !== undefined && env._actions.length > 0) {
    const actionsPerEnv = env._actions[0];
    let maxIdx = 0;
    for (const [j] of torquesToApply) if (j > maxIdx) maxIdx = j;
    while (actionsPerEnv.length < maxIdx + 1) actionsPerEnv.push(0);
    for (const [j, t] of torquesToApply) actionsPerEnv[j] = t;
    return;
  }
  throw new Error(
    "writeEffort: env has no _appliedTorques / _appliedForce / _actions buffer",
  );
}
