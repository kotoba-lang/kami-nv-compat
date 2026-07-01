// TypeScript port of kotodama.nv_compat.isaacsim.assets.franka_panda
//
// Franka Emika Panda asset wrapper — 7-DoF arm + 2-finger gripper (9 DoF).
//
// Specs sourced from the public Franka Robotics FCI documentation
// (frankarobotics.github.io/docs/control_parameters.html — joint
// ranges, velocity limits, effort limits) and the publicly-distributed
// Franka URDF (github.com/frankaemika/franka_ros, Apache 2.0). No
// proprietary content (no mesh references, no Isaac Sim USD refs);
// the URDF is a minimal kinematic-chain reproduction with unit-mass
// placeholder inertias so the wrapper is self-contained and substrate-
// publishable.
//
// Trademark: "Franka Emika" and "Panda" are trademarks of Franka
// Robotics GmbH; this wrapper is API namespace localization only
// (matching the public FCI spec — Google v. Oracle 2021 API fair use).
//
// ADR-2605261800 §D6.

// (No imports from urdf-builder — buildFrankaUrdf hand-emits URDF text
//  inline so it can embed real per-link inertials, bypassing the
//  unit-mass placeholders that buildSerialChainUrdf would otherwise
//  insert.)

// Joint specs from public Franka FCI documentation.
const PANDA_ARM_JOINTS: ReadonlyArray<{
  name: string;
  lower: number;
  upper: number;
  velocity: number;
  effort: number;
}> = [
  { name: "panda_joint1", lower: -2.8973, upper: 2.8973, velocity: 2.1750, effort: 87 },
  { name: "panda_joint2", lower: -1.7628, upper: 1.7628, velocity: 2.1750, effort: 87 },
  { name: "panda_joint3", lower: -2.8973, upper: 2.8973, velocity: 2.1750, effort: 87 },
  { name: "panda_joint4", lower: -3.0718, upper: -0.0698, velocity: 2.1750, effort: 87 },
  { name: "panda_joint5", lower: -2.8973, upper: 2.8973, velocity: 2.6100, effort: 12 },
  { name: "panda_joint6", lower: -0.0175, upper: 3.7525, velocity: 2.6100, effort: 12 },
  { name: "panda_joint7", lower: -2.8973, upper: 2.8973, velocity: 2.6100, effort: 12 },
];

const PANDA_FINGER_JOINTS: ReadonlyArray<{
  name: string;
  lower: number;
  upper: number;
  velocity: number;
  effort: number;
}> = [
  { name: "panda_finger_joint1", lower: 0, upper: 0.04, velocity: 0.2, effort: 20 },
  { name: "panda_finger_joint2", lower: 0, upper: 0.04, velocity: 0.2, effort: 20 },
];

// Real Franka FCI joint origins per the publicly-distributed Franka
// URDF (github.com/frankaemika/franka_description, Apache 2.0).
// These origin rpy/xyz encode the modified-DH frame rotations so each
// joint's axis="0 0 1" (body frame) gives the correct world-frame
// rotation axis after composition. Replaces the iter 75 placeholder
// xyz=(0,0,0.1) rpy=(0,0,0) which made the arm singular along z.
const HALF_PI = Math.PI / 2;
const PANDA_ARM_ORIGINS: ReadonlyArray<{
  xyz: [number, number, number];
  rpy: [number, number, number];
}> = [
  { xyz: [0, 0, 0.333], rpy: [0, 0, 0] },          // joint1
  { xyz: [0, 0, 0],     rpy: [-HALF_PI, 0, 0] },    // joint2
  { xyz: [0, -0.316, 0], rpy: [HALF_PI, 0, 0] },    // joint3
  { xyz: [0.0825, 0, 0], rpy: [HALF_PI, 0, 0] },    // joint4
  { xyz: [-0.0825, 0.384, 0], rpy: [-HALF_PI, 0, 0] }, // joint5
  { xyz: [0, 0, 0],     rpy: [HALF_PI, 0, 0] },     // joint6
  { xyz: [0.088, 0, 0], rpy: [HALF_PI, 0, 0] },     // joint7
];

// Real Franka link inertial parameters per franka_description URDF
// (Apache 2.0) — mass [kg], COM offset [m] in link frame, principal
// inertia tensor [kg·m²] about COM. Off-diagonal ixy/ixz/iyz are small
// for Franka links and approximated as 0 here (matches the published
// franka_description values which round small off-diagonals).
// Replaces iter 75's placeholder unit-mass + identity-inertia which
// made gravity-comp / ABA forward sim produce unrealistic torques.
const PANDA_ARM_INERTIALS: ReadonlyArray<{
  mass: number;
  com: [number, number, number];
  ixx: number; iyy: number; izz: number;
}> = [
  { mass: 2.74, com: [0.003875, 0.002081, -0.04762],   ixx: 0.0180, iyy: 0.0184, izz: 0.0089 }, // link1
  { mass: 2.74, com: [-0.003141, -0.02872, 0.003495],  ixx: 0.0184, iyy: 0.0089, izz: 0.0180 }, // link2
  { mass: 2.38, com: [0.02785, 0.03094, -0.0961],      ixx: 0.0089, iyy: 0.0125, izz: 0.0049 }, // link3
  { mass: 2.38, com: [-0.05317, 0.1046, 0.02711],      ixx: 0.0125, iyy: 0.0049, izz: 0.0089 }, // link4
  { mass: 2.74, com: [-0.01121, 0.04123, -0.03825],    ixx: 0.0125, iyy: 0.0089, izz: 0.0049 }, // link5
  { mass: 1.55, com: [0.065, -0.016, -0.020],           ixx: 0.0049, iyy: 0.0049, izz: 0.0017 }, // link6
  { mass: 0.54, com: [0.010, 0.010, 0.045],             ixx: 0.0010, iyy: 0.0010, izz: 0.0010 }, // link7
];

// Per-finger placeholder inertia (Franka finger mass ~0.015 kg each).
const FINGER_MASS = 0.015;

function inertialBlock(
  mass: number, com: readonly [number, number, number],
  ixx: number, iyy: number, izz: number,
): string {
  return (
    `<inertial>` +
    `<origin xyz="${com[0]} ${com[1]} ${com[2]}" rpy="0 0 0"/>` +
    `<mass value="${mass}"/>` +
    `<inertia ixx="${ixx}" ixy="0" ixz="0" iyy="${iyy}" iyz="0" izz="${izz}"/>` +
    `</inertial>`
  );
}

function buildFrankaUrdf(): string {
  // Hand-build the URDF so we can embed real per-link inertials inline
  // (bypassing buildSerialChainUrdf's unit-mass placeholder). Joint
  // origins from PANDA_ARM_ORIGINS (iter 85); link inertials from
  // PANDA_ARM_INERTIALS (iter 87).
  const parts: string[] = [`<?xml version="1.0"?>`, `<robot name="panda">`];
  // Base link (panda_link0) — fixed to world, mass 0.
  parts.push(`<link name="panda_link0">${inertialBlock(0, [0, 0, 0], 0, 0, 0)}</link>`);
  // Arm joints + child links with real inertials.
  for (let i = 0; i < PANDA_ARM_JOINTS.length; i++) {
    const j = PANDA_ARM_JOINTS[i];
    const origin = PANDA_ARM_ORIGINS[i];
    const inert = PANDA_ARM_INERTIALS[i];
    parts.push(
      `<joint name="${j.name}" type="revolute">` +
      `<origin xyz="${origin.xyz[0]} ${origin.xyz[1]} ${origin.xyz[2]}" rpy="${origin.rpy[0]} ${origin.rpy[1]} ${origin.rpy[2]}"/>` +
      `<parent link="panda_link${i}"/>` +
      `<child link="panda_link${i + 1}"/>` +
      `<axis xyz="0 0 1"/>` +
      `<limit lower="${j.lower}" upper="${j.upper}" velocity="${j.velocity}" effort="${j.effort}"/>` +
      `</joint>`,
    );
    parts.push(`<link name="panda_link${i + 1}">${inertialBlock(inert.mass, inert.com, inert.ixx, inert.iyy, inert.izz)}</link>`);
  }
  // Finger joints + child links (panda_link8 / panda_link9).
  for (let i = 0; i < PANDA_FINGER_JOINTS.length; i++) {
    const j = PANDA_FINGER_JOINTS[i];
    parts.push(
      `<joint name="${j.name}" type="prismatic">` +
      `<origin xyz="0 0 0.107" rpy="0 0 0"/>` +
      `<parent link="panda_link7"/>` +
      `<child link="panda_link${8 + i}"/>` +
      `<axis xyz="${i === 0 ? '0 1 0' : '0 -1 0'}"/>` +
      `<limit lower="${j.lower}" upper="${j.upper}" velocity="${j.velocity}" effort="${j.effort}"/>` +
      `</joint>`,
    );
    parts.push(`<link name="panda_link${8 + i}">${inertialBlock(FINGER_MASS, [0, 0, 0], 1e-5, 1e-5, 1e-5)}</link>`);
  }
  parts.push(`</robot>`);
  return parts.join("");
}

/** Franka Panda 9-DoF asset. Pairs directly with iter 72 DiffIK (arm 7-DoF)
 *  / iter 73 OSC (arm 7-DoF) / iter 74 DifferentialInverseKinematicsAction
 *  (arm) / iter 74 BinaryJointPositionAction (gripper).
 */
export interface FrankaPanda {
  primPath: string;
  name: string;
  urdfText: string;
  jointNames: readonly string[];
  armJointNames: readonly string[];
  fingerJointNames: readonly string[];
  dofCount: number;
  armDofCount: number;
  fingerDofCount: number;
  defaultJointPositions: readonly number[];
  defaultJointVelocities: readonly number[];
  jointLowerLimits: readonly number[];
  jointUpperLimits: readonly number[];
  jointVelocityLimits: readonly number[];
  effortLimits: readonly number[];
  gripperOpenCommand: readonly [number, number];
  gripperCloseCommand: readonly [number, number];
  eeLinkName: string;
  homePose(): readonly number[];
  armIndices(): readonly [number, number, number, number, number, number, number];
  fingerIndices(): readonly [number, number];
}

export function makeFrankaPanda(opts: Partial<{ primPath: string; name: string }> = {}): FrankaPanda {
  const primPath = opts.primPath ?? "/World/Franka";
  const name = opts.name ?? "franka_panda";
  const armJointNames = PANDA_ARM_JOINTS.map((j) => j.name);
  const fingerJointNames = PANDA_FINGER_JOINTS.map((j) => j.name);
  const jointNamesAll = [...armJointNames, ...fingerJointNames] as const;
  // Canonical Franka "home" pose (arm) + open gripper.
  const defaultJointPositions: readonly number[] = [
    0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854,
    0.04, 0.04,
  ];
  const defaultJointVelocities: readonly number[] = new Array(9).fill(0);
  const jointLowerLimits: readonly number[] = [
    ...PANDA_ARM_JOINTS.map((j) => j.lower),
    ...PANDA_FINGER_JOINTS.map((j) => j.lower),
  ];
  const jointUpperLimits: readonly number[] = [
    ...PANDA_ARM_JOINTS.map((j) => j.upper),
    ...PANDA_FINGER_JOINTS.map((j) => j.upper),
  ];
  const jointVelocityLimits: readonly number[] = [
    ...PANDA_ARM_JOINTS.map((j) => j.velocity),
    ...PANDA_FINGER_JOINTS.map((j) => j.velocity),
  ];
  const effortLimits: readonly number[] = [
    ...PANDA_ARM_JOINTS.map((j) => j.effort),
    ...PANDA_FINGER_JOINTS.map((j) => j.effort),
  ];

  return {
    primPath,
    name,
    urdfText: buildFrankaUrdf(),
    jointNames: jointNamesAll,
    armJointNames,
    fingerJointNames,
    dofCount: 9,
    armDofCount: 7,
    fingerDofCount: 2,
    defaultJointPositions,
    defaultJointVelocities,
    jointLowerLimits,
    jointUpperLimits,
    jointVelocityLimits,
    effortLimits,
    gripperOpenCommand: [0.04, 0.04],
    gripperCloseCommand: [0, 0],
    eeLinkName: "panda_hand",
    homePose() {
      return defaultJointPositions;
    },
    armIndices() {
      return [0, 1, 2, 3, 4, 5, 6] as const;
    },
    fingerIndices() {
      return [7, 8] as const;
    },
  };
}
