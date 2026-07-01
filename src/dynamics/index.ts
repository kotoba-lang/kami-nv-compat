// @etzhayyim/kami-nv-compat/dynamics
//
// TypeScript port of nv_compat.dynamics — Featherstone ABA / RNEA /
// CRBA + forward kinematics + geometric Jacobian. Mirrors the Python
// reference impl (iter 68-70) algorithm-for-algorithm so cross-impl
// numerical agreement can be checked.
//
// ADR-2605261800 §D6+§D11 nv-compat namespace localization.

export {
  // URDF types
  type UrdfPose,
  type UrdfInertia,
  type UrdfLink,
  type UrdfJoint,
  type UrdfJointKind,
  type UrdfArticulatedSystem,
  // 3-space helpers
  skew3,
  mat3Mul,
  mat3T,
  mat3Add,
  mat3Scale,
  rotFromRpy,
  rodriguesRotation,
  // Build + state
  type BuiltArticulation,
  type ArticulatedState,
  buildArticulation,
  makeZeroState,
  spatialInertiaFromLink,
  // Dynamics
  abaForward,
  articulatedStep,
  rneaInverseDynamics,
  coriolisGravityVector,
  crbaMassMatrix,
  kineticEnergy,
  // Kinematics
  type JointWorldPose,
  forwardKinematics,
  geometricJacobian,
} from "./articulated-dynamics.js";

export { parseUrdf } from "./urdf-parser.js";
