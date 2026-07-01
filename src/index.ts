// @etzhayyim/kami-nv-compat
// NVIDIA Omniverse stack public-API drop-in compat facade.
// R1.0 path reservation per ADR-2605261800; first implementation
// surface (dynamics) landed iter 71 as TypeScript port of the Python
// nv_compat reference impl (iter 68-70).
// See README.md for trademark notice and sub-phase delivery plan.

export * as dynamics from "./dynamics/index.js";
export * as controllers from "./controllers/index.js";
// R1.1 isaacsim.core.api surface (World / Articulation / RigidPrim), backed by
// the e7m-sim engine over the Featherstone articulated-dynamics module.
export * as isaacSim from "./isaac-sim.js";
export * as e7mSim from "./e7m-sim/index.js";
export * as actions from "./actions/index.js";
export * as assets from "./assets/index.js";
export * as warp from "./warp/index.js";
export * as policies from "./policies/index.js";
// R1.2 OptiX® + RTX Renderer API-compat surface, backed by the kami-rt
// WebGPU ray tracer / kami-rtx WGSL path tracer.
export * as optix from "./optix.js";
export * as rtxRenderer from "./rtx-renderer.js";
export * as kamiRt from "./kami-rt/index.js";
// R1.4 omni.usd / pxr.Usd surface, backed by the kami-usd USDA reader; the
// scene bridge feeds parsed stages straight into kami-rt / kami-rtx.
export * as omniUsd from "./omni-usd.js";
export * as kamiUsd from "./kami-usd/index.js";
// R1.4 omni.kit.app / commands surface (Kit app shell + extension lifecycle),
// backed by the amenominaka engine; hosts the USD→render pipeline.
export * as omniKitApp from "./omni-kit-app.js";
export * as amenominaka from "./amenominaka/index.js";
// R1.3 omni.replicator.core surface (synthetic data + domain randomization),
// backed by the utsushimi engine; ground-truth boxes via kami-rt projection.
export * as omniReplicatorCore from "./omni-replicator-core.js";
export * as utsushimi from "./utsushimi/index.js";
// R1.5 isaaclab.envs / managers surface (manager-based RL envs), backed by the
// e7m-shugyo engine (Cartpole + manager framework).
export * as isaaclabEnvs from "./isaaclab-envs.js";
export * as e7mShugyo from "./e7m-shugyo/index.js";
// R1.9 omni.client / Nucleus surface (content-addressed versioned store),
// backed by the kotoba-datomic-nucleus engine.
export * as omniNucleus from "./omni-nucleus.js";
export * as kotobaDatomicNucleus from "./kotoba-datomic-nucleus/index.js";
// R1.7 Omniverse Cloud surface (managed render farm), backed by murakumo-render
// over the kami renderers; Murakumo-fleet-only (no commercial GPU rental).
export * as omniCloud from "./omni-cloud.js";
export * as murakumoRender from "./murakumo-render/index.js";
// Alpamayo AV VLA surface (separate NVIDIA AV stack, not Omniverse): the
// reasoning planner + closed-loop AlpaSim, backed by kami-drive (michibiki) /
// wadachi-sim. Civilian, SAE-L4 ceiling, sim-only, Murakumo-only inference.
export * as alpamayo from "./alpamayo.js";
export * as alpasim from "./alpasim.js";
export * as kamiDrive from "./kami-drive/index.js";
// DRIVE Sim — sensor-realistic AV simulator (camera/LiDAR/radar grounded in
// kami-rt), backed by wadachi-sim; closes the loop with the Alpamayo model.
export * as driveSim from "./drive-sim.js";
export * as wadachiSim from "./wadachi-sim/index.js";

export const ADR = "ADR-2605261800";
// R1 Omniverse stack complete: R1.2 OptiX/RTX · R1.3 Replicator · R1.4 USD +
// Kit app · R1.5 Isaac Lab · R1.6 DriveSim · R1.7 Omniverse Cloud · R1.9
// Nucleus (+ the separate Alpamayo AV stack). Engines: kami-rt / kami-rtx /
// kami-usd / utsushimi / e7m-shugyo / wadachi-sim / amenominaka /
// murakumo-render / kotoba-datomic-nucleus / michibiki.
export const PHASE = "R1-complete";

/** Canonical KAMI engine names for the Alpamayo AV stack (parallel to
 *  NV_COMPAT_MAP, which covers the Omniverse stack). */
export const ALPAMAYO_COMPAT_MAP: Readonly<Record<string, string>> = Object.freeze({
  "Alpamayo": "michibiki",
  "AlpaSim": "wadachi-sim",
  "AlpaGym": "wadachi-gym",
});

export const NV_COMPAT_MAP: Readonly<Record<string, string>> = Object.freeze({
  "Omniverse Kit":     "amenominaka",
  "Nucleus":           "kotoba-datomic-nucleus",
  "Isaac Sim":         "e7m-sim",
  "Isaac Lab":         "e7m-shugyo",
  "OptiX":             "hikari-rt",
  "RTX Renderer":      "kami-rtx",
  "Replicator":        "utsushimi",
  "DriveSim":          "wadachi-sim",
  "Omniverse Cloud":   "murakumo-render",
});
