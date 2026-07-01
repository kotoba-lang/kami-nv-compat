// @etzhayyim/kami-nv-compat/isaac-sim
//
// Drop-in NVIDIA Isaac Sim `isaacsim.core.api` API-compat facade — the core
// simulation context. Mirrors the documented World / Articulation / RigidPrim
// surface so Isaac Sim scripts port to KAMI via import-path-only changes —
// e.g.
//
//     import { World, Articulation, RigidPrim } from "@etzhayyim/kami-nv-compat/isaac-sim";
//     import { makeFrankaPanda } from "@etzhayyim/kami-nv-compat/assets";
//
//     const world = new World({ physicsDt: 1/240 });
//     const franka = world.addArticulation(
//       Articulation.fromUrdf("franka", makeFrankaPanda().urdfText));
//     franka.applyAction({ jointPositions: target });
//     world.step(4);
//     franka.getJointPositions();
//
// Backed by the clean-room e7m-sim engine (the existing Featherstone
// articulated-dynamics module + rigid-body integration). No Isaac Sim
// source/USD/binaries; from-spec reproduction (Google v. Oracle, 593 U.S. ___
// (2021)). Canonical engine: e7m-sim.
//
// Trademark: NVIDIA® / Isaac® / Isaac Sim are trademarks of NVIDIA
// Corporation; API-compat identifiers only.
//
// ADR-2605261800 §D1/D6, R1.1 isaacsim.core.api surface.

import {
  type Quat,
  type Vec3,
  type WorldCfg,
  type ArticulationActionInput,
  Articulation as ArticulationImpl,
  RigidPrim,
  World,
  articulationFromUrdf,
} from "./e7m-sim/index.js";

export type { Vec3, Quat, WorldCfg, ArticulationActionInput } from "./e7m-sim/index.js";
export { World, RigidPrim } from "./e7m-sim/index.js";

/** Isaac-Sim-shaped Articulation with a `fromUrdf` static constructor. */
export class Articulation extends ArticulationImpl {
  static fromUrdf(name: string, urdfText: string, defaultQ?: number[]): Articulation {
    return articulationFromUrdf(name, urdfText, defaultQ) as Articulation;
  }
}

/** `isaacsim.core.api` namespace. */
export const core = {
  api: { World, Articulation, RigidPrim },
};

export const KAMI_ENGINE = "e7m-sim";
export const ADR = "ADR-2605261800";
