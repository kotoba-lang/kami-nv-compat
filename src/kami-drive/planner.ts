// kami-drive — clean-room reasoning planner (michibiki 導き).
//
// The canonical planner behind the `nv-compat/alpamayo` VLA facade. Given the
// ego state, a navigation command, and the perceived agents (in the ego
// frame), it produces (a) a dynamic-action sequence rolled out to an
// Alpamayo-format trajectory and (b) a Chain-of-Causation trace explaining the
// decision — the same dual output Alpamayo's VLA emits.
//
// The policy is deliberately simple, deterministic, and auditable: longitudinal
// control tracks a target speed (from the command + a yield check against
// agents in the ego path); lateral control tracks a target curvature (from the
// turn command). Every decision writes a CoC step, so the plan is explainable.
//
// Clean-room: from-spec rule-based planning + textbook kinematics. No Alpamayo
// weights/data. Civilian, SAE-L4 ceiling, NO actuation — the planner returns a
// recommended trajectory; it never sends controls to a vehicle.
//
// ADR-2605261800-adjacent (nv-compat); AV scope per wadachi / kami-autodrive.

import {
  type CausationStep,
  type ChainOfCausation,
  type EventCluster,
  CausationBuilder,
} from "./coc.js";
import {
  type BevState,
  type DynamicAction,
  type Waypoint,
  DEFAULT_MAX_SPEED,
  rolloutTrajectory,
} from "./unicycle.js";

// ── observation ───────────────────────────────────────────────────────────

export type NavigationCommand =
  | "keep_lane"
  | "turn_left"
  | "turn_right"
  | "stop"
  | "go_straight";

export type AgentKind = "pedestrian" | "cyclist" | "vehicle" | "unknown";

/** A perceived agent in the EGO frame: +x ahead, +y to the left (metres). */
export interface PerceivedAgent {
  id: string;
  kind: AgentKind;
  /** Position in ego frame, metres. */
  x: number;
  y: number;
  /** Velocity in ego frame, m/s. */
  vx: number;
  vy: number;
}

/** The structured driving observation the planner reasons over. (Upstream a
 *  vision encoder — e.g. the `manako` browser-local detector — produces the
 *  agent list from multi-camera frames; here we take it pre-parsed.) */
export interface DrivingObservation {
  ego: BevState;
  command: NavigationCommand;
  /** Free-text natural-language instruction (optional; mapped to a command
   *  when `command` is not authoritative). */
  instruction?: string;
  agents: PerceivedAgent[];
  /** Posted speed limit (m/s). */
  speedLimit?: number;
}

// ── planner config ────────────────────────────────────────────────────────

export interface PlannerConfig {
  /** Trajectory horizon (s). Alpamayo predicts 6.4 s. */
  horizonS: number;
  /** Control rate (Hz). Alpamayo runs at 10 Hz (64 waypoints over 6.4 s). */
  hz: number;
  /** Comfortable longitudinal accel/decel magnitude (m/s²). */
  comfortAccel: number;
  /** Curvature magnitude commanded for a turn (1/m). */
  turnCurvature: number;
  /** Longitudinal look-ahead for the yield check (m). */
  lookAhead: number;
  /** Half-width of the ego corridor for the in-path test (m). */
  laneHalfWidth: number;
  /** Desired gap kept behind a leading/crossing agent (m). */
  safeGap: number;
  maxSpeed: number;
}

export const DEFAULT_PLANNER: PlannerConfig = {
  horizonS: 6.4,
  hz: 10,
  comfortAccel: 1.5,
  turnCurvature: 0.05,
  lookAhead: 60, // real AV perception ranges far past the stopping distance
  laneHalfWidth: 1.5,
  safeGap: 5,
  maxSpeed: DEFAULT_MAX_SPEED,
};

export interface PlanResult {
  trajectory: Waypoint[];
  reasoning: ChainOfCausation;
  actions: DynamicAction[];
}

// ── instruction → command mapping (lightweight; no model) ──────────────────

const STOP_WORDS = ["stop", "halt", "brake", "止ま", "停止"];
const LEFT_WORDS = ["left", "左"];
const RIGHT_WORDS = ["right", "右"];

export function commandFromInstruction(text: string): NavigationCommand | null {
  const t = text.toLowerCase();
  if (STOP_WORDS.some((w) => t.includes(w))) return "stop";
  if (LEFT_WORDS.some((w) => t.includes(w)) && t.includes("turn")) return "turn_left";
  if (RIGHT_WORDS.some((w) => t.includes(w)) && t.includes("turn")) return "turn_right";
  if (LEFT_WORDS.some((w) => t.includes(w))) return "turn_left";
  if (RIGHT_WORDS.some((w) => t.includes(w))) return "turn_right";
  return null;
}

// ── yield reasoning ─────────────────────────────────────────────────────────

interface YieldDecision {
  /** Target speed cap imposed by the most constraining agent (m/s). */
  targetSpeed: number;
  agent: PerceivedAgent | null;
  gap: number;
}

/** Find the most constraining in-path agent and the speed it allows. An agent
 *  is "in path" when it is ahead (x>0) within the look-ahead and its lateral
 *  offset is within the corridor (accounting for its lateral motion). */
function yieldCheck(obs: DrivingObservation, cfg: PlannerConfig): YieldDecision {
  let decision: YieldDecision = { targetSpeed: cfg.maxSpeed, agent: null, gap: Infinity };
  for (const a of obs.agents) {
    if (a.x <= 0 || a.x > cfg.lookAhead) continue;
    // Project the agent ~1 s forward to catch crossing VRUs.
    const yNow = a.y;
    const ySoon = a.y + a.vy * 1.0;
    const inCorridor =
      Math.min(Math.abs(yNow), Math.abs(ySoon)) <= cfg.laneHalfWidth ||
      yNow * ySoon < 0; // crossed the centerline within 1 s
    if (!inCorridor) continue;
    const gap = a.x - cfg.safeGap;
    // Vulnerable road users → come to a stop; vehicles → the safe-approach
    // speed that can still decelerate to the safe gap under comfort braking
    // (v = √(2·a·gap)), offset by the lead's own speed.
    const allowed =
      a.kind === "pedestrian" || a.kind === "cyclist"
        ? 0
        : Math.max(0, Math.min(cfg.maxSpeed, a.vx + Math.sqrt(2 * cfg.comfortAccel * Math.max(0, gap))));
    if (allowed < decision.targetSpeed) {
      decision = { targetSpeed: allowed, agent: a, gap };
    }
  }
  return decision;
}

// ── plan ─────────────────────────────────────────────────────────────────

/** Produce a recommended trajectory + Chain-of-Causation for an observation. */
export function plan(obs: DrivingObservation, config: Partial<PlannerConfig> = {}): PlanResult {
  const cfg = { ...DEFAULT_PLANNER, ...config };
  const steps = Math.max(1, Math.round(cfg.horizonS * cfg.hz));
  const dt = 1 / cfg.hz;

  // Resolve the effective command (explicit command unless an instruction
  // overrides it).
  let command = obs.command;
  if (obs.instruction) {
    const c = commandFromInstruction(obs.instruction);
    if (c) command = c;
  }

  const builder = new CausationBuilder("nominal");
  let cluster: EventCluster = "nominal";

  // Lateral target from the command.
  let targetCurvature = 0;
  if (command === "turn_left") {
    targetCurvature = cfg.turnCurvature;
    cluster = "intersection";
    builder.add(
      "Navigation command is turn-left",
      "the ego is approaching a left turn",
      `steer left at curvature ${cfg.turnCurvature.toFixed(3)} /m`,
      0,
    );
  } else if (command === "turn_right") {
    targetCurvature = -cfg.turnCurvature;
    cluster = "intersection";
    builder.add(
      "Navigation command is turn-right",
      "the ego is approaching a right turn",
      `steer right at curvature ${cfg.turnCurvature.toFixed(3)} /m`,
      0,
    );
  }

  // Longitudinal target from the command + posted limit.
  const limit = obs.speedLimit ?? cfg.maxSpeed;
  let targetSpeed = command === "stop" ? 0 : Math.min(cfg.maxSpeed, limit);
  if (command === "stop") {
    cluster = "stop";
    builder.add(
      "Navigation command is stop",
      "the ego must come to a controlled halt",
      "decelerate to zero speed",
      0,
    );
  }

  // Yield reasoning against perceived agents.
  const yld = yieldCheck(obs, cfg);
  if (yld.agent && yld.targetSpeed < targetSpeed) {
    targetSpeed = yld.targetSpeed;
    const a = yld.agent;
    const vru = a.kind === "pedestrian" || a.kind === "cyclist";
    cluster = vru ? "vru_interaction" : "yield";
    const kf = Math.min(steps, Math.max(1, Math.round((a.x / Math.max(0.1, obs.ego.speed)) * cfg.hz)));
    builder.add(
      `${labelKind(a.kind)} ${a.x.toFixed(1)} m ahead within the ego corridor`,
      vru ? "must yield to a vulnerable road user" : "must keep a safe following gap",
      yld.targetSpeed === 0 ? "decelerate to a stop" : `reduce speed to ${yld.targetSpeed.toFixed(1)} m/s`,
      kf,
    );
  } else if (steps > 0 && command === "keep_lane" && builderEmpty(builder)) {
    builder.add(
      "Lane is clear ahead",
      "nominal cruising conditions hold",
      `maintain target speed ${targetSpeed.toFixed(1)} m/s`,
      0,
    );
  }

  // Synthesize the action sequence: P-control toward target speed + curvature.
  const actions: DynamicAction[] = [];
  let speed = obs.ego.speed;
  for (let i = 0; i < steps; i++) {
    const speedErr = targetSpeed - speed;
    const accel = clampAbs(speedErr / dt, cfg.comfortAccel);
    // Only steer once moving; ease curvature in over the first second.
    const ramp = Math.min(1, (i * dt) / 1.0);
    const curvature = speed > 0.1 ? targetCurvature * ramp : 0;
    actions.push({ accel, curvature });
    speed = Math.min(cfg.maxSpeed, Math.max(0, speed + accel * dt));
  }

  // Plan in the ego frame: origin, 0 yaw, current speed (Alpamayo convention).
  const trajectory = rolloutTrajectory(
    { x: 0, y: 0, yaw: 0, speed: obs.ego.speed },
    actions,
    dt,
    cfg.maxSpeed,
  );

  const reasoning = builder.setCluster(cluster).build();
  return { trajectory, reasoning, actions };
}

function clampAbs(v: number, m: number): number {
  return v > m ? m : v < -m ? -m : v;
}
function labelKind(k: AgentKind): string {
  return k === "pedestrian" ? "Pedestrian" : k === "cyclist" ? "Cyclist" : k === "vehicle" ? "Vehicle" : "Object";
}
function builderEmpty(b: CausationBuilder): boolean {
  return b.build().steps.length === 0;
}

export type { CausationStep };
