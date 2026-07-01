// @etzhayyim/kami-nv-compat/assets
//
// TypeScript port of nv_compat.isaacsim.assets — canonical Isaac Lab
// benchmark robots + URDF construction helpers.
//
// Iter 75 surface:
//   - FrankaPanda (9-DoF: 7 arm + 2 finger gripper)
//   - AnymalC (12-DoF quadruped: 4 legs × 3 joints)
//   - URDF builder primitives (buildSerialChainUrdf,
//     buildBranchedUrdf, countJoints, jointNames)
//
// Substrate-publishable — generated URDFs contain NO proprietary mesh
// references, NO Isaac Sim USD refs, and rely only on public Franka
// FCI / ANYbotics public docs for joint specs.
//
// ADR-2605261800 §D6 nv-compat namespace localization.

export {
  type JointType,
  type UrdfJointSpec,
  buildSerialChainUrdf,
  buildBranchedUrdf,
  countJoints,
  jointNames,
} from "./urdf-builder.js";

export { type FrankaPanda, makeFrankaPanda } from "./franka-panda.js";
export { type AnymalC, makeAnymalC } from "./anymal-c.js";
export { type Ur10, makeUr10 } from "./ur10.js";
