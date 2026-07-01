// @etzhayyim/kami-nv-compat/alpasim
//
// Drop-in NVIDIA AlpaSim API-compat facade — an open-loop-free closed-loop
// validation harness for reasoning AV models. Mirrors the documented AlpaSim
// behaviour (step a model through a scenario, scoring its decisions against
// consequences) and the AlpaGym closed-loop reward shape, backed by the
// clean-room kami-drive BEV simulator + the `nv-compat/alpamayo` planner.
//
// The world coordinate frame is global; at each tick the ego's neighbourhood
// is transformed into the ego frame, the model predicts a trajectory, the
// first dynamic action is applied to the ego via the unicycle model, scripted
// agents advance, and collision / progress / comfort are accumulated into an
// AlpaGym-style reward.
//
// Clean-room: from-spec simulator. No AlpaSim / AlpaGym / DRIVE source or
// binaries. Civilian, SAE-L4 ceiling, sim-only (no actuation). Canonical KAMI
// engine: wadachi-sim (DriveSim lineage, ADR-2605261800 D1) + michibiki.
//
// nv-compat namespace; AV scope per wadachi / kami-autodrive ADRs.

import {
  type AgentKind,
  type BevState,
  type DynamicAction,
  type NavigationCommand,
  type PerceivedAgent,
  stepUnicycle,
} from "./kami-drive/index.js";
import { type AlpamayoR1 } from "./alpamayo.js";

// ── world types ────────────────────────────────────────────────────────────

/** A scripted agent in WORLD coordinates. Its pose advances by constant
 *  velocity (a deterministic, replayable script). */
export interface WorldAgent {
  id: string;
  kind: AgentKind;
  x: number;
  y: number;
  /** World-frame velocity (m/s). */
  vx: number;
  vy: number;
  /** Collision radius (m). */
  radius: number;
}

/** Ego pose in WORLD coordinates. */
export interface WorldEgo {
  x: number;
  y: number;
  yaw: number;
  speed: number;
  radius: number;
}

export interface Scenario {
  ego: WorldEgo;
  agents: WorldAgent[];
  /** Navigation command schedule; index `floor(t * hz / 1)` clamped. A single
   *  command applies for the whole rollout. */
  command: NavigationCommand;
  speedLimit?: number;
  /** Rollout duration (s). */
  durationS: number;
  /** Simulation rate (Hz). */
  hz: number;
}

export interface RolloutStep {
  t: number;
  ego: WorldEgo;
  /** First applied dynamic action. */
  action: DynamicAction;
  /** Minimum distance to any agent this step (m). */
  minClearance: number;
}

export interface RolloutMetrics {
  /** Distance the ego progressed along its path (m). */
  progress: number;
  /** True if the ego ever came within (egoR + agentR) of an agent. */
  collision: boolean;
  /** Minimum clearance over the whole rollout (m). */
  minClearance: number;
  /** RMS longitudinal jerk (m/s³) — comfort proxy. */
  jerkRms: number;
}

export interface RolloutResult {
  steps: RolloutStep[];
  metrics: RolloutMetrics;
  /** AlpaGym-style scalar reward in roughly [0, 1] (1 = ideal). */
  reward: number;
}

// ── world → ego-frame transform ──────────────────────────────────────────────

function toEgoFrame(ego: WorldEgo, a: WorldAgent): PerceivedAgent {
  const c = Math.cos(-ego.yaw), s = Math.sin(-ego.yaw);
  const dx = a.x - ego.x, dy = a.y - ego.y;
  // Ego-relative velocity, then rotate both into the ego frame.
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

// ── closed-loop rollout ──────────────────────────────────────────────────────

/** Run `model` closed-loop through `scenario`. Each tick: perceive (world →
 *  ego frame), predict, apply the first dynamic action, advance agents, score. */
export function runClosedLoop(model: AlpamayoR1, scenario: Scenario): RolloutResult {
  const hz = scenario.hz;
  const dt = 1 / hz;
  const nSteps = Math.max(1, Math.round(scenario.durationS * hz));

  let ego: WorldEgo = { ...scenario.ego };
  const agents = scenario.agents.map((a) => ({ ...a }));
  const steps: RolloutStep[] = [];

  let progress = 0;
  let collision = false;
  let minClearance = Infinity;
  let prevAccel = 0;
  let jerkSq = 0;

  for (let i = 0; i < nSteps; i++) {
    const t = i * dt;

    // Perceive: agents into the ego frame.
    const perceived = agents.map((a) => toEgoFrame(ego, a));
    const obs = {
      ego: { x: 0, y: 0, yaw: 0, speed: ego.speed } as BevState,
      command: scenario.command,
      agents: perceived,
      speedLimit: scenario.speedLimit,
    };

    // Predict + take the first dynamic action.
    const out = model.predict(obs);
    const action = out.trajectory[1]
      ? { accel: out.trajectory[1].accel, curvature: out.trajectory[1].curvature }
      : { accel: 0, curvature: 0 };

    // Integrate the ego in the world frame.
    const before = { x: ego.x, y: ego.y };
    const bev = stepUnicycle({ x: ego.x, y: ego.y, yaw: ego.yaw, speed: ego.speed }, action, dt);
    ego = { ...ego, x: bev.x, y: bev.y, yaw: bev.yaw, speed: bev.speed };
    progress += Math.hypot(ego.x - before.x, ego.y - before.y);

    // Advance scripted agents (constant velocity).
    for (const a of agents) {
      a.x += a.vx * dt;
      a.y += a.vy * dt;
    }

    // Clearance / collision.
    let stepMin = Infinity;
    for (const a of agents) {
      const d = Math.hypot(a.x - ego.x, a.y - ego.y) - (a.radius + ego.radius);
      if (d < stepMin) stepMin = d;
      if (d <= 0) collision = true;
    }
    if (stepMin < minClearance) minClearance = stepMin;

    // Comfort: jerk = d(accel)/dt.
    const jerk = (action.accel - prevAccel) / dt;
    jerkSq += jerk * jerk;
    prevAccel = action.accel;

    steps.push({ t, ego: { ...ego }, action, minClearance: stepMin });
  }

  const jerkRms = Math.sqrt(jerkSq / nSteps);
  const metrics: RolloutMetrics = {
    progress,
    collision,
    minClearance: Number.isFinite(minClearance) ? minClearance : Infinity,
    jerkRms,
  };
  return { steps, metrics, reward: reward(metrics, scenario) };
}

/** AlpaGym-style reward: progress (normalized) − comfort penalty, zeroed on
 *  collision. Roughly [0, 1]. */
function reward(m: RolloutMetrics, scenario: Scenario): number {
  if (m.collision) return 0;
  const maxProgress = (scenario.speedLimit ?? 30) * scenario.durationS;
  const progressScore = clamp01(m.progress / Math.max(1e-3, maxProgress));
  const comfortPenalty = clamp01(m.jerkRms / 20); // 20 m/s³ ≈ harsh
  return clamp01(progressScore * (1 - 0.3 * comfortPenalty));
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export const KAMI_ENGINE = "wadachi-sim";
export const ADR = "ADR-2606010600";
