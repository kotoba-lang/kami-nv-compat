// TypeScript port of kotodama.nv_compat.isaacsim.assets.anymal_c
//
// ANYmal C asset wrapper — 12-DoF quadruped (4 legs × 3 joints).
//
// Specs sourced from the publicly-distributed ANYbotics ANYmal C URDF
// (github.com/ANYbotics/anymal_c_simple_description, BSD-3) + academic
// locomotion papers (Hwangbo et al. 2019 "Learning Agile and Dynamic
// Motor Skills for Legged Robots"). No proprietary content; the URDF
// is a minimal kinematic-tree reproduction.
//
// Trademark: "ANYmal" is a trademark of ANYbotics AG; this wrapper is
// API namespace localization only (matching the public ROS URDF spec —
// Google v. Oracle 2021 API fair use).
//
// ADR-2605261800 §D6.

import { buildBranchedUrdf, type UrdfJointSpec } from "./urdf-builder.js";

const LEG_NAMES = ["LF", "LH", "RF", "RH"] as const;
type LegName = (typeof LEG_NAMES)[number];

// (suffix, axis_xyz, lower, upper, vel_limit, effort_limit)
const LEG_JOINTS: ReadonlyArray<{
  suffix: string;
  axis: readonly [number, number, number];
  lower: number;
  upper: number;
  velocity: number;
  effort: number;
}> = [
  // HAA — hip ab/adduction (roll)
  { suffix: "HAA", axis: [1, 0, 0], lower: -0.611, upper: 0.611, velocity: 7.5, effort: 80 },
  // HFE — hip flex/extension (pitch)
  { suffix: "HFE", axis: [0, 1, 0], lower: -9.42, upper: 9.42, velocity: 7.5, effort: 80 },
  // KFE — knee flex/extension
  { suffix: "KFE", axis: [0, 1, 0], lower: -9.42, upper: 9.42, velocity: 7.5, effort: 80 },
];

// Real ANYbotics ANYmal C joint origins (per anymal_c_simple_description
// URDF, BSD-3). Replaces iter 75 placeholder (0, 0, -0.15) uniform
// downward stacking with the canonical quadruped geometry.
//
// Per-leg HAA base attachment (in body frame):
//   LF: (+0.277, +0.116, 0.0)    LH: (-0.277, +0.116, 0.0)
//   RF: (+0.277, -0.116, 0.0)    RH: (-0.277, -0.116, 0.0)
//
// HFE origin in HAA frame (local): (0, +0.0635, 0)
// KFE origin in HFE frame (local): (0, +0.041, -0.317)
// Foot origin in KFE frame (local): (0, 0, -0.317)
//
// At q=0, with default LF HAA at (+0.277, +0.116, 0), foot ends up at
// world (0.277, 0.2205, -0.634).
const ANYMAL_HAA_BASE_OFFSET: Record<LegName, [number, number, number]> = {
  LF: [ 0.277,  0.116, 0.0],
  LH: [-0.277,  0.116, 0.0],
  RF: [ 0.277, -0.116, 0.0],
  RH: [-0.277, -0.116, 0.0],
};

function buildAnymalUrdf(): string {
  const branches: UrdfJointSpec[][] = [];
  const branchLinkPrefixes: string[] = [];
  for (const leg of LEG_NAMES) {
    const haaXyz = ANYMAL_HAA_BASE_OFFSET[leg];
    const legJoints: UrdfJointSpec[] = [
      // HAA — attaches at per-leg base offset
      {
        name: `${leg}_HAA`, type: "revolute", axis: [1, 0, 0],
        lower: -0.611, upper: 0.611, velocity: 7.5, effort: 80,
        originXyz: haaXyz,
      },
      // HFE — origin in HAA frame
      {
        name: `${leg}_HFE`, type: "revolute", axis: [0, 1, 0],
        lower: -9.42, upper: 9.42, velocity: 7.5, effort: 80,
        originXyz: [0, 0.0635, 0],
      },
      // KFE — origin in HFE frame
      {
        name: `${leg}_KFE`, type: "revolute", axis: [0, 1, 0],
        lower: -9.42, upper: 9.42, velocity: 7.5, effort: 80,
        originXyz: [0, 0.041, -0.317],
      },
    ];
    branches.push(legJoints);
    branchLinkPrefixes.push(`${leg}_link`);
  }
  return buildBranchedUrdf("anymal_c", "base", branches, branchLinkPrefixes);
}

// Standing pose: HAA=0 (vertical), HFE=±0.4 (front +, hind −), KFE=∓0.8.
// Convention: front legs flex forward; hind legs flex backward.
const STANDING_POSE: readonly number[] = [
  0,  0.4, -0.8,   // LF_HAA, LF_HFE, LF_KFE
  0, -0.4,  0.8,   // LH_HAA, LH_HFE, LH_KFE
  0,  0.4, -0.8,   // RF_HAA, RF_HFE, RF_KFE
  0, -0.4,  0.8,   // RH_HAA, RH_HFE, RH_KFE
];

function makeJointNames(): readonly string[] {
  const names: string[] = [];
  for (const leg of LEG_NAMES) {
    for (const j of LEG_JOINTS) names.push(`${leg}_${j.suffix}`);
  }
  return names;
}

/** ANYmal C 12-DoF quadruped asset.
 *
 * Pairs with: iter 71 (forward kinematics for any joint) + standard
 * locomotion observation/reward functions (when ported to TS) + iter 72/73
 * controllers (when used at the foot level). Canonical Isaac Lab locomotion
 * benchmark robot.
 */
export interface AnymalC {
  primPath: string;
  name: string;
  urdfText: string;
  jointNames: readonly string[];
  dofCount: number;
  defaultJointPositions: readonly number[];
  defaultJointVelocities: readonly number[];
  jointLowerLimits: readonly number[];
  jointUpperLimits: readonly number[];
  jointVelocityLimits: readonly number[];
  effortLimits: readonly number[];
  footLinkNames: readonly [string, string, string, string];
  baseLinkName: string;
  legNames: typeof LEG_NAMES;
  jointsPerLeg: number;
  legIndices(leg: LegName): readonly [number, number, number];
  haaIndices(): readonly [number, number, number, number];
  hfeIndices(): readonly [number, number, number, number];
  kfeIndices(): readonly [number, number, number, number];
}

export function makeAnymalC(opts: Partial<{ primPath: string; name: string }> = {}): AnymalC {
  const primPath = opts.primPath ?? "/World/Anymal";
  const name = opts.name ?? "anymal_c";
  const jointNamesAll = makeJointNames();
  const dofCount = 12;
  const defaultJointVelocities: readonly number[] = new Array(12).fill(0);
  const jointLowerLimits: readonly number[] = [
    -0.611, -9.42, -9.42,
    -0.611, -9.42, -9.42,
    -0.611, -9.42, -9.42,
    -0.611, -9.42, -9.42,
  ];
  const jointUpperLimits: readonly number[] = [
    0.611, 9.42, 9.42,
    0.611, 9.42, 9.42,
    0.611, 9.42, 9.42,
    0.611, 9.42, 9.42,
  ];
  const jointVelocityLimits: readonly number[] = new Array(12).fill(7.5);
  const effortLimits: readonly number[] = new Array(12).fill(80);

  return {
    primPath,
    name,
    urdfText: buildAnymalUrdf(),
    jointNames: jointNamesAll,
    dofCount,
    defaultJointPositions: STANDING_POSE,
    defaultJointVelocities,
    jointLowerLimits,
    jointUpperLimits,
    jointVelocityLimits,
    effortLimits,
    footLinkNames: ["LF_foot", "LH_foot", "RF_foot", "RH_foot"],
    baseLinkName: "base",
    legNames: LEG_NAMES,
    jointsPerLeg: 3,
    legIndices(leg: LegName) {
      const idx = LEG_NAMES.indexOf(leg);
      if (idx < 0) throw new Error(`AnymalC.legIndices: leg must be one of ${LEG_NAMES.join(',')}; got '${leg}'`);
      const start = idx * 3;
      return [start, start + 1, start + 2];
    },
    haaIndices() {
      return [0, 3, 6, 9];
    },
    hfeIndices() {
      return [1, 4, 7, 10];
    },
    kfeIndices() {
      return [2, 5, 8, 11];
    },
  };
}
