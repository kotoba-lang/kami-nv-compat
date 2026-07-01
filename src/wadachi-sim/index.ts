// @etzhayyim/kami-nv-compat/wadachi-sim
//
// Clean-room AV simulation engine (wadachi-sim 轍) — the canonical KAMI
// implementation behind the `nv-compat/drive-sim` facade. Owns the scenario
// world, the per-tick physics advance, the sensor rig, and the closed-loop
// driving step that plugs an Alpamayo VLA model in to steer the ego.
//
// Civilian, SAE-L4 ceiling, simulation-only (no actuation). ADR-2605261800 D1
// (DriveSim → wadachi-sim); AV scope per ADR-2605242000 / ADR-2606010600.

import { type Scene } from "../kami-rt/index.js";
import {
  type DrivingObservation,
  type NavigationCommand,
  type PerceivedAgent,
  stepUnicycle,
} from "../kami-drive/index.js";
import {
  type Actor,
  type GtObject,
  type Scenario,
  buildSensorScene,
  groundTruth,
} from "./world.js";
import {
  type CameraConfig,
  type CameraFrame,
  type LidarConfig,
  type LidarScan,
  type RadarConfig,
  type RadarDetection,
  sampleCamera,
  sampleLidar,
  sampleRadar,
} from "./sensors.js";

export * from "./world.js";
export * from "./sensors.js";

// ── sensor rig ───────────────────────────────────────────────────────────────

export interface SensorRig {
  camera?: CameraConfig;
  lidar?: LidarConfig;
  radar?: RadarConfig;
}

export interface SensorFrame {
  tick: number;
  time: number;
  camera?: CameraFrame;
  lidar?: LidarScan;
  radar?: RadarDetection[];
  groundTruth: GtObject[];
}

/** A model that can drive the ego closed-loop (the Alpamayo facade fits). */
export interface DrivingModel {
  predict(obs: DrivingObservation): { trajectory: { accel: number; curvature: number }[] };
}

// ── DriveSim ─────────────────────────────────────────────────────────────────

export interface DriveSimConfig {
  scenario: Scenario;
  rig: SensorRig;
  /** Simulation rate (Hz). */
  hz: number;
  /** Navigation command issued to a closed-loop model. */
  command?: NavigationCommand;
  speedLimit?: number;
}

/** Closed-loop / open-loop AV simulator. `step()` advances the world one tick;
 *  `sense()` samples the configured sensors against the current world. */
export class DriveSim {
  private scenario: Scenario;
  private readonly init: Scenario;
  private readonly rig: SensorRig;
  private readonly dt: number;
  private readonly command: NavigationCommand;
  private readonly speedLimit?: number;
  tick = 0;

  constructor(cfg: DriveSimConfig) {
    this.init = cloneScenario(cfg.scenario);
    this.scenario = cloneScenario(cfg.scenario);
    this.rig = cfg.rig;
    this.dt = 1 / cfg.hz;
    this.command = cfg.command ?? "keep_lane";
    this.speedLimit = cfg.speedLimit;
  }

  /** Reset to the initial scenario. */
  reset(): void {
    this.scenario = cloneScenario(this.init);
    this.tick = 0;
  }

  get world(): Scenario {
    return this.scenario;
  }

  /** Build the current kami-rt scene (ground + actors + obstacles). */
  scene(): Scene {
    return buildSensorScene(this.scenario);
  }

  /** Sample all configured sensors against the current world. */
  sense(): SensorFrame {
    const gt = groundTruth(this.scenario);
    const frame: SensorFrame = { tick: this.tick, time: this.tick * this.dt, groundTruth: gt };
    if (this.rig.camera || this.rig.lidar) {
      const sc = this.scene();
      if (this.rig.camera) frame.camera = sampleCamera(this.scenario, gt, sc, this.rig.camera);
      if (this.rig.lidar) frame.lidar = sampleLidar(this.scenario, sc, this.rig.lidar);
    }
    if (this.rig.radar) frame.radar = sampleRadar(this.scenario, this.rig.radar);
    return frame;
  }

  /** The ego-frame observation a driving model consumes (actors → ego frame). */
  observation(): DrivingObservation {
    return {
      ego: { x: 0, y: 0, yaw: 0, speed: this.scenario.ego.speed },
      command: this.command,
      agents: this.scenario.actors.map((a) => toEgoFrame(this.scenario.ego, a)),
      speedLimit: this.speedLimit,
    };
  }

  /** Advance one tick. With `model`, the ego is driven closed-loop by the
   *  model's first dynamic action; otherwise an explicit `action` is applied
   *  (default: hold). Actors advance by their scripted constant velocity. */
  step(opts: { model?: DrivingModel; action?: { accel: number; curvature: number } } = {}): void {
    let action = opts.action ?? { accel: 0, curvature: 0 };
    if (opts.model) {
      const out = opts.model.predict(this.observation());
      if (out.trajectory[1]) action = { accel: out.trajectory[1].accel, curvature: out.trajectory[1].curvature };
    }
    const e = this.scenario.ego;
    const bev = stepUnicycle({ x: e.x, y: e.y, yaw: e.yaw, speed: e.speed }, action, this.dt);
    this.scenario.ego = { ...e, x: bev.x, y: bev.y, yaw: bev.yaw, speed: bev.speed };
    for (const a of this.scenario.actors) {
      a.x += a.vx * this.dt;
      a.y += a.vy * this.dt;
    }
    this.tick++;
  }

  /** Run a closed-loop rollout: at each tick sense → drive → advance. Returns
   *  the per-tick sensor frames. */
  run(model: DrivingModel, numTicks: number): SensorFrame[] {
    const frames: SensorFrame[] = [];
    for (let i = 0; i < numTicks; i++) {
      frames.push(this.sense());
      this.step({ model });
    }
    return frames;
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────

function cloneScenario(s: Scenario): Scenario {
  return {
    ego: { ...s.ego, extent: [...s.ego.extent] },
    actors: s.actors.map((a) => ({ ...a, extent: [...a.extent] })),
    obstacles: s.obstacles.map((o) => ({ ...o, extent: [...o.extent] })),
    groundHalfSize: s.groundHalfSize,
  };
}

function toEgoFrame(ego: { x: number; y: number; yaw: number; speed: number }, a: Actor): PerceivedAgent {
  const c = Math.cos(-ego.yaw), s = Math.sin(-ego.yaw);
  const dx = a.x - ego.x, dy = a.y - ego.y;
  const rvx = a.vx - ego.speed * Math.cos(ego.yaw);
  const rvy = a.vy - ego.speed * Math.sin(ego.yaw);
  return {
    id: a.id,
    kind: a.kind,
    x: dx * c - dy * s,
    y: dx * s + dy * c,
    vx: rvx * c - rvy * s,
    vy: rvx * s + rvy * c,
  };
}
