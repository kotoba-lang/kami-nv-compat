// wadachi-sim — clean-room sensor models (camera / LiDAR / radar).
//
// DriveSim's value is sensor-realistic ground truth; these models reproduce
// that on the kami-rt ray tracer + the utsushimi camera projection:
//   - CameraSensor : kami-rt RGB frame + projected 2D bounding-box ground truth
//   - LidarSensor  : BVH ray-cast scan → range image + 3D point cloud
//   - RadarSensor  : per-object range / azimuth / radial-velocity detections
//
// All sensors are mounted on the ego with a planar offset + yaw and a mast
// height; their forward axis follows the ego heading. Deterministic, CPU-only.
//
// ADR-2605261800 §D1/D6 (DriveSim → wadachi-sim).

import { type Scene, type Vec3, lookAt, traceClosest, traceImageCPU } from "../kami-rt/index.js";
import { makeProjCamera, projectAabb } from "../utsushimi/index.js";
import { type AgentKind } from "../kami-drive/index.js";
import {
  type Actor,
  type EgoState,
  type GtObject,
  type Scenario,
  buildSensorScene,
  worldAabb,
} from "./world.js";

// ── sensor mount ─────────────────────────────────────────────────────────────

export interface SensorMount {
  /** Longitudinal/lateral offset from the ego origin (ego frame, m). */
  forward: number;
  left: number;
  /** Mast height above ground (m). */
  height: number;
  /** Yaw offset from the ego heading (rad). */
  yaw: number;
}

export const DEFAULT_MOUNT: SensorMount = { forward: 1.5, left: 0, height: 1.5, yaw: 0 };

/** World-space sensor pose (origin + forward heading) for an ego + mount. */
export function sensorPose(ego: EgoState, mount: SensorMount): { origin: Vec3; heading: number } {
  const c = Math.cos(ego.yaw), s = Math.sin(ego.yaw);
  const origin: Vec3 = [
    ego.x + mount.forward * c - mount.left * s,
    ego.y + mount.forward * s + mount.left * c,
    mount.height,
  ];
  return { origin, heading: ego.yaw + mount.yaw };
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

// ── camera sensor ────────────────────────────────────────────────────────────

export interface CameraConfig {
  width: number;
  height: number;
  vfovDeg: number;
  mount: SensorMount;
}

export interface CameraBox {
  id: string;
  kind: AgentKind;
  bbox2d: [number, number, number, number];
}

export interface CameraFrame {
  /** RGBA-float framebuffer (width*height*4). */
  rgb: Float32Array;
  /** 2D ground-truth boxes for on-screen objects. */
  boxes: CameraBox[];
  width: number;
  height: number;
}

/** Render an RGB frame + projected 2D ground-truth boxes for the scenario. */
export function sampleCamera(
  scenario: Scenario,
  gt: readonly GtObject[],
  scene: Scene,
  cfg: CameraConfig,
): CameraFrame {
  const { origin, heading } = sensorPose(scenario.ego, cfg.mount);
  const target: Vec3 = [origin[0] + Math.cos(heading), origin[1] + Math.sin(heading), origin[2]];
  const up: Vec3 = [0, 0, 1];
  const aspect = cfg.width / cfg.height;
  const cam = lookAt(origin, target, up, cfg.vfovDeg, aspect);
  const rgb = traceImageCPU(scene, cam, cfg.width, cfg.height).framebuffer;

  const proj = makeProjCamera(origin, target, up, cfg.vfovDeg, aspect);
  const boxes: CameraBox[] = [];
  for (const o of gt) {
    const aabb = worldAabb(o.center[0], o.center[1], o.extent, o.yaw);
    const bbox = projectAabb(proj, aabb.min, aabb.max, cfg.width, cfg.height);
    if (bbox) boxes.push({ id: o.id, kind: o.kind, bbox2d: bbox });
  }
  return { rgb, boxes, width: cfg.width, height: cfg.height };
}

// ── LiDAR sensor ─────────────────────────────────────────────────────────────

export interface LidarConfig {
  /** Horizontal field of view (deg); 360 for a spinning sensor. */
  azimuthFovDeg: number;
  azimuthSteps: number;
  /** Vertical FOV (deg) split into `elevationSteps` rings. */
  elevationFovDeg: number;
  elevationSteps: number;
  maxRange: number;
  mount: SensorMount;
}

export interface LidarReturn {
  /** Hit point in world coordinates. */
  point: Vec3;
  range: number;
  azimuth: number;
  elevation: number;
}

export interface LidarScan {
  origin: Vec3;
  returns: LidarReturn[];
  /** Rays cast (returns ≤ rays; misses are dropped). */
  rays: number;
}

/** Cast a LiDAR scan against the scene BVH. Each (azimuth, elevation) ray that
 *  hits within `maxRange` yields a point. */
export function sampleLidar(scenario: Scenario, scene: Scene, cfg: LidarConfig): LidarScan {
  const { origin, heading } = sensorPose(scenario.ego, cfg.mount);
  const returns: LidarReturn[] = [];
  const azHalf = (cfg.azimuthFovDeg * Math.PI) / 180 / 2;
  const elHalf = (cfg.elevationFovDeg * Math.PI) / 180 / 2;
  const azStep = cfg.azimuthSteps > 1 ? (2 * azHalf) / (cfg.azimuthSteps - 1) : 0;
  const elStep = cfg.elevationSteps > 1 ? (2 * elHalf) / (cfg.elevationSteps - 1) : 0;
  let rays = 0;
  for (let ai = 0; ai < cfg.azimuthSteps; ai++) {
    const az = -azHalf + ai * azStep;
    const worldAz = heading + az;
    for (let ei = 0; ei < cfg.elevationSteps; ei++) {
      const el = -elHalf + ei * elStep;
      const dir: Vec3 = [
        Math.cos(el) * Math.cos(worldAz),
        Math.cos(el) * Math.sin(worldAz),
        Math.sin(el),
      ];
      rays++;
      const hit = traceClosest(scene.soup, scene.bvh, origin, dir, cfg.maxRange);
      if (hit) {
        returns.push({
          point: [origin[0] + dir[0] * hit.t, origin[1] + dir[1] * hit.t, origin[2] + dir[2] * hit.t],
          range: hit.t,
          azimuth: az,
          elevation: el,
        });
      }
    }
  }
  return { origin, returns, rays };
}

// ── radar sensor ─────────────────────────────────────────────────────────────

export interface RadarConfig {
  azimuthFovDeg: number;
  maxRange: number;
  mount: SensorMount;
}

export interface RadarDetection {
  id: string;
  range: number;
  /** Azimuth relative to sensor forward (rad, +left). */
  azimuth: number;
  /** Radial velocity (m/s), +ve = receding. */
  rangeRate: number;
}

/** Per-actor radar detections (range / azimuth / Doppler range-rate). Static
 *  obstacles have zero range-rate; only objects within FOV + range are
 *  returned. The ego's own velocity is included in the relative motion. */
export function sampleRadar(scenario: Scenario, cfg: RadarConfig): RadarDetection[] {
  const { origin, heading } = sensorPose(scenario.ego, cfg.mount);
  const egoVx = scenario.ego.speed * Math.cos(scenario.ego.yaw);
  const egoVy = scenario.ego.speed * Math.sin(scenario.ego.yaw);
  const azHalf = (cfg.azimuthFovDeg * Math.PI) / 180 / 2;
  const out: RadarDetection[] = [];
  const consider = (id: string, x: number, y: number, vx: number, vy: number): void => {
    const dx = x - origin[0], dy = y - origin[1];
    const range = Math.hypot(dx, dy);
    if (range < 1e-3 || range > cfg.maxRange) return;
    let az = Math.atan2(dy, dx) - heading;
    az = Math.atan2(Math.sin(az), Math.cos(az)); // wrap to [-π, π]
    if (Math.abs(az) > azHalf) return;
    const losx = dx / range, losy = dy / range;
    // Relative velocity (object − ego) projected on the line of sight.
    const rangeRate = (vx - egoVx) * losx + (vy - egoVy) * losy;
    out.push({ id, range, azimuth: az, rangeRate });
  };
  for (const a of scenario.actors) consider(a.id, a.x, a.y, a.vx, a.vy);
  for (const o of scenario.obstacles) consider(o.id, o.x, o.y, 0, 0);
  return out;
}

// ── helper re-exports ────────────────────────────────────────────────────────

export { buildSensorScene };
export type { Actor };
