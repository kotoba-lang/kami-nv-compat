// @etzhayyim/kami-nv-compat/drive-sim
//
// Drop-in NVIDIA DRIVE Sim API-compat facade — a sensor-realistic, scenario-
// driven autonomous-vehicle simulator. Mirrors the documented DRIVE Sim
// concepts (a scenario world + ego + traffic actors, a sensor rig of
// cameras / LiDAR / radar producing frames + ground truth, stepped open- or
// closed-loop) so AV test scripts port to KAMI via import-path-only changes —
// e.g.
//
//     import { createScenario, createCamera, createLidar, DriveSim }
//       from "@etzhayyim/kami-nv-compat/drive-sim";
//     import { AlpamayoR1 } from "@etzhayyim/kami-nv-compat/alpamayo";
//
//     const sim = new DriveSim({
//       scenario: createScenario({ ego, actors }),
//       rig: { camera: createCamera(), lidar: createLidar(), radar: createRadar() },
//       hz: 10,
//     });
//     const frames = sim.run(AlpamayoR1.fromPretrained(), 50); // closed-loop
//
// Backed by the clean-room wadachi-sim engine (sensors grounded in the kami-rt
// ray tracer + utsushimi projection). No DRIVE Sim source/binaries; from-spec
// reproduction (Google v. Oracle, 593 U.S. ___ (2021)). Canonical engine:
// wadachi-sim.
//
// Charter: civilian, SAE-L4 ceiling, simulation-only (no actuation), and any
// closed-loop driving model runs under the `nv-compat/alpamayo` Murakumo-only
// inference posture.
//
// Trademark: NVIDIA® / DRIVE® / DRIVE Sim are trademarks of NVIDIA
// Corporation; API-compat identifiers only.
//
// ADR-2605261800 §D1/D6 (DriveSim → wadachi-sim).

import {
  type Actor,
  type CameraConfig,
  type EgoState,
  type LidarConfig,
  type RadarConfig,
  type Scenario,
  type StaticObstacle,
  DEFAULT_MOUNT,
  DriveSim,
} from "./wadachi-sim/index.js";
import { type Stage, stageToFlatScene } from "./omni-usd.js";

export {
  DriveSim,
  type Scenario,
  type EgoState,
  type Actor,
  type StaticObstacle,
  type SensorRig,
  type SensorFrame,
  type DriveSimConfig,
  type DrivingModel,
  type CameraFrame,
  type CameraBox,
  type LidarScan,
  type LidarReturn,
  type RadarDetection,
  type SensorMount,
  DEFAULT_MOUNT,
  sensorPose,
  buildSensorScene,
  groundTruth,
} from "./wadachi-sim/index.js";

/** DRIVE-Sim-style scenario alias. */
export type DriveSimScenario = Scenario;

const DEFAULT_EGO: EgoState = { x: 0, y: 0, yaw: 0, speed: 8, extent: [2.4, 1, 0.75] };

/** Build a scenario world with sensible defaults. */
export function createScenario(opts: {
  ego?: Partial<EgoState>;
  actors?: Actor[];
  obstacles?: StaticObstacle[];
  groundHalfSize?: number;
}): Scenario {
  return {
    ego: { ...DEFAULT_EGO, ...opts.ego, extent: opts.ego?.extent ?? DEFAULT_EGO.extent },
    actors: opts.actors ?? [],
    obstacles: opts.obstacles ?? [],
    groundHalfSize: opts.groundHalfSize ?? 100,
  };
}

// ── sensor config builders (DRIVE Sim sensor rig) ────────────────────────────

export function createCamera(cfg: Partial<CameraConfig> = {}): CameraConfig {
  return {
    width: cfg.width ?? 320,
    height: cfg.height ?? 180,
    vfovDeg: cfg.vfovDeg ?? 40,
    mount: cfg.mount ?? DEFAULT_MOUNT,
  };
}

export function createLidar(cfg: Partial<LidarConfig> = {}): LidarConfig {
  return {
    azimuthFovDeg: cfg.azimuthFovDeg ?? 360,
    azimuthSteps: cfg.azimuthSteps ?? 180,
    elevationFovDeg: cfg.elevationFovDeg ?? 30,
    elevationSteps: cfg.elevationSteps ?? 8,
    maxRange: cfg.maxRange ?? 80,
    mount: cfg.mount ?? { ...DEFAULT_MOUNT, height: 1.8 },
  };
}

export function createRadar(cfg: Partial<RadarConfig> = {}): RadarConfig {
  return {
    azimuthFovDeg: cfg.azimuthFovDeg ?? 120,
    maxRange: cfg.maxRange ?? 150,
    mount: cfg.mount ?? DEFAULT_MOUNT,
  };
}

// ── USD scenario bridge (DRIVE Sim scenarios are USD) ────────────────────────

/** Build static obstacles from a parsed USD stage: each UsdGeomMesh becomes a
 *  box obstacle sized to its world AABB. Ties the kami-usd reader (R1.4) to
 *  DriveSim so a USD scene can seed a simulation. */
export function obstaclesFromStage(stage: Stage, kind: Actor["kind"] = "unknown"): StaticObstacle[] {
  const flat = stageToFlatScene(stage);
  const out: StaticObstacle[] = [];
  flat.triangles.forEach((tri, i) => {
    let min = [Infinity, Infinity, Infinity];
    let max = [-Infinity, -Infinity, -Infinity];
    for (const v of tri) {
      for (let c = 0; c < 3; c++) {
        if (v[c] < min[c]) min[c] = v[c];
        if (v[c] > max[c]) max[c] = v[c];
      }
    }
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    out.push({
      id: `usd-${i}`,
      kind,
      x: cx,
      y: cy,
      yaw: 0,
      extent: [Math.max(0.05, (max[0] - min[0]) / 2), Math.max(0.05, (max[1] - min[1]) / 2), Math.max(0.05, (max[2] - min[2]) / 2)],
    });
  });
  return out;
}

export const KAMI_ENGINE = "wadachi-sim";
export const ADR = "ADR-2605261800";
