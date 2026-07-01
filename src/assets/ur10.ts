// TypeScript port-style asset wrapper for the Universal Robots UR10.
//
// 6-DoF industrial arm. Joint origins encode the modified-DH frame
// transforms used by the publicly-distributed `ur_description` ROS
// package (github.com/ros-industrial/universal_robot, BSD-3). axis
// is per the public URDF spec; rpy on joint origins rotates frames
// so each joint's axis stays well-defined.
//
// No proprietary mesh / no Isaac Sim USD refs — joint kinematics
// only, sufficient to drive the generic serial-chain FK kernel
// (iter 97) and validate cross-vendor portability.
//
// Trademark: "Universal Robots" and "UR10" are trademarks of
// Universal Robots A/S; this wrapper is API namespace localization
// only (matching the public ROS spec — Google v. Oracle 2021 API
// fair use).
//
// ADR-2605261800 §D6.

const HALF_PI = Math.PI / 2;

// UR10 joint origins per the publicly-distributed ur_description URDF
// (universal_robot/ur_description/urdf/inc/ur10_macro.xacro, BSD-3).
//
//   shoulder_pan_joint    xyz="0 0 0.1273"      rpy="0 0 0"        axis="0 0 1"
//   shoulder_lift_joint   xyz="0 0 0"           rpy="0 1.570796 0" axis="0 1 0"
//   elbow_joint           xyz="-0.612 0 0"      rpy="0 0 0"        axis="0 1 0"
//   wrist_1_joint         xyz="-0.5723 0 0.163941" rpy="0 1.570796 0" axis="0 1 0"
//   wrist_2_joint         xyz="0 -0.1157 0"     rpy="0 0 0"        axis="0 0 1"
//   wrist_3_joint         xyz="0 0 0.0922"      rpy="0 0 0"        axis="0 1 0"
//
// These transforms map between successive joint frames using the
// modified-DH-equivalent xyz+rpy convention of the ROS URDF. At q=0
// they place the EE at approximately
//   x = -0.612 - 0.5723 - 0.0922 = -1.2765  (when wrist rotations
//   "fold" their offsets into x). After the joint-2 origin rpy=
//   (0, π/2, 0) rotation, link2 + link3 + wrist offsets lay along
//   world-x (negative).
//
// Real numbers are the canonical UR10 spec (max reach ≈ 1.3 m).
const UR10_ORIGINS: ReadonlyArray<{
  xyz: [number, number, number];
  rpy: [number, number, number];
  axis: [number, number, number];
}> = [
  { xyz: [0, 0, 0.1273],      rpy: [0, 0, 0],       axis: [0, 0, 1] },     // shoulder_pan
  { xyz: [0, 0, 0],            rpy: [0, HALF_PI, 0], axis: [0, 1, 0] },     // shoulder_lift
  { xyz: [-0.612, 0, 0],       rpy: [0, 0, 0],       axis: [0, 1, 0] },     // elbow
  { xyz: [-0.5723, 0, 0.163941], rpy: [0, HALF_PI, 0], axis: [0, 1, 0] },   // wrist_1
  { xyz: [0, -0.1157, 0],      rpy: [0, 0, 0],       axis: [0, 0, 1] },     // wrist_2
  { xyz: [0, 0, 0.0922],       rpy: [0, 0, 0],       axis: [0, 1, 0] },     // wrist_3
];

const UR10_JOINT_NAMES: readonly string[] = [
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint",
];

// Joint limits per UR10 datasheet (ur_description URDF):
//   ±2π for all 6 joints (continuous joints in URDF; we cap at ±2π).
//   max velocity: 2.16 rad/s (base/shoulder), 3.2 rad/s (wrists).
const UR10_LOWER: readonly number[] = [-2 * Math.PI, -2 * Math.PI, -Math.PI, -2 * Math.PI, -2 * Math.PI, -2 * Math.PI];
const UR10_UPPER: readonly number[] = [ 2 * Math.PI,  2 * Math.PI,  Math.PI,  2 * Math.PI,  2 * Math.PI,  2 * Math.PI];
const UR10_VEL_LIMIT: readonly number[] = [2.16, 2.16, 3.15, 3.20, 3.20, 3.20];
const UR10_EFFORT: readonly number[] = [330, 330, 150, 54, 54, 54]; // N·m, UR10 public spec

export interface Ur10 {
  primPath: string;
  name: string;
  jointNames: readonly string[];
  dofCount: number;
  defaultJointPositions: readonly number[];
  defaultJointVelocities: readonly number[];
  jointLowerLimits: readonly number[];
  jointUpperLimits: readonly number[];
  jointVelocityLimits: readonly number[];
  effortLimits: readonly number[];
  /** Per-joint origin xyz [m] in modified-DH form (parent → child). */
  jointOriginXyz: ReadonlyArray<readonly [number, number, number]>;
  /** Per-joint origin rpy [rad] applied before the axis rotation. */
  jointOriginRpy: ReadonlyArray<readonly [number, number, number]>;
  /** Per-joint rotation axis in body frame. */
  jointAxis: ReadonlyArray<readonly [number, number, number]>;
  /** Flat storage layout for genericSerialFkKernel: N×3 floats. */
  flatXyz(): readonly number[];
  flatRpy(): readonly number[];
  flatAxis(): readonly number[];
}

export function makeUr10(opts: Partial<{ primPath: string; name: string }> = {}): Ur10 {
  const primPath = opts.primPath ?? "/World/UR10";
  const name = opts.name ?? "ur10";
  const dofCount = 6;
  const xyz: ReadonlyArray<readonly [number, number, number]> = UR10_ORIGINS.map(o => o.xyz);
  const rpy: ReadonlyArray<readonly [number, number, number]> = UR10_ORIGINS.map(o => o.rpy);
  const axis: ReadonlyArray<readonly [number, number, number]> = UR10_ORIGINS.map(o => o.axis);

  return {
    primPath,
    name,
    jointNames: UR10_JOINT_NAMES,
    dofCount,
    defaultJointPositions: [0, -HALF_PI, 0, -HALF_PI, 0, 0],
    defaultJointVelocities: new Array(6).fill(0),
    jointLowerLimits: UR10_LOWER,
    jointUpperLimits: UR10_UPPER,
    jointVelocityLimits: UR10_VEL_LIMIT,
    effortLimits: UR10_EFFORT,
    jointOriginXyz: xyz,
    jointOriginRpy: rpy,
    jointAxis: axis,
    flatXyz() { return xyz.flatMap(v => [...v]); },
    flatRpy() { return rpy.flatMap(v => [...v]); },
    flatAxis() { return axis.flatMap(v => [...v]); },
  };
}
