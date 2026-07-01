// @etzhayyim/kami-nv-compat/alpamayo
//
// Drop-in NVIDIA Alpamayo VLA API-compat facade. Mirrors the documented
// public inference surface of the Alpamayo reasoning Vision-Language-Action
// model (multi-camera images + navigation command + egomotion history →
// future trajectory + Chain-of-Causation reasoning), so existing Alpamayo
// host code ports to KAMI via import-path-only changes — e.g.
//
//     import { AlpamayoR1 } from "@etzhayyim/kami-nv-compat/alpamayo";
//
//     const model = AlpamayoR1.fromPretrained("nvidia/Alpamayo-R1-10B");
//     const out = model.predict(observation);
//     out.trajectory;       // 64 waypoints @10Hz, 6.4 s, ego frame
//     out.reasoning;        // Chain-of-Causation narrative
//
// Backed by the clean-room kami-drive reasoning planner (michibiki) + BEV
// unicycle model — NO Alpamayo / Cosmos / DRIVE weights, source, or binaries
// are used. This is a from-spec reproduction of the public I/O contract
// (Google v. Oracle, 593 U.S. ___ (2021)).
//
// Charter posture (religious-corp AV invariants):
//   - SAE-L4 ceiling — L5 / unconditional autonomy is unrepresentable
//     (ADR-2605242000 wadachi, ADR-2606010600 kami-autodrive).
//   - NO actuation — `predict` returns a RECOMMENDED trajectory + reasoning;
//     it never sends controls to a vehicle.
//   - Murakumo-only inference — any language verbalization routes through an
//     injected Murakumo callback (LiteLLM 127.0.0.1:4000), never a vendor
//     endpoint (ADR-2605215000). Default is a deterministic template.
//
// Trademark: NVIDIA®, Alpamayo, DRIVE®, Cosmos are trademarks of NVIDIA
// Corporation; this project is not affiliated with or endorsed by NVIDIA.
//
// nv-compat namespace (ADR-2605261800 D1/D6); AV scope per wadachi /
// kami-autodrive ADRs.

import {
  type ChainOfCausation,
  type DrivingObservation,
  type PlannerConfig,
  type Waypoint,
  plan,
} from "./kami-drive/index.js";

// ── I/O types (mirror the Alpamayo model card) ───────────────────────────────

/** The four Alpamayo input cameras. */
export const CAMERA_NAMES = ["front_wide", "front_tele", "cross_left", "cross_right"] as const;
export type CameraName = (typeof CAMERA_NAMES)[number];

/** A multi-camera, multi-timestep image stack (0.4 s history @10Hz = 4 frames
 *  per camera). The pixels themselves are an opaque handle here — KAMI does
 *  not bundle a vision encoder in this facade; agents are supplied parsed via
 *  the observation. */
export interface CameraStack {
  name: CameraName;
  /** Frame references (URLs / tensors / ids), newest last. */
  frames: unknown[];
}

/** One egomotion sample: 3D translation + 3×3 (9D) rotation + timestamp. */
export interface EgomotionWaypoint {
  translation: [number, number, number];
  rotation: [number, number, number, number, number, number, number, number, number];
  timestamp: number;
}

/** The raw Alpamayo input tuple (images + text command + egomotion history). */
export interface AlpamayoInput {
  images: CameraStack[];
  /** Natural-language command string. */
  command: string;
  /** Egomotion history (16 waypoints @10Hz in the model card). */
  egomotionHistory: EgomotionWaypoint[];
}

/** Model output: future trajectory + Chain-of-Causation reasoning trace. */
export interface AlpamayoOutput {
  /** 64 waypoints @10Hz over 6.4 s, ego frame (3D translation + 3×3 rotation). */
  trajectory: Waypoint[];
  /** Chain-of-Causation reasoning trace + narrative. */
  reasoning: ChainOfCausation;
  /** Convenience: the verbalized explanation string. */
  explanation: string;
}

/** Alpamayo trajectory constants (from the model card). */
export const TRAJECTORY_HORIZON_S = 6.4;
export const TRAJECTORY_HZ = 10;
export const TRAJECTORY_WAYPOINTS = 64;
export const HISTORY_S = 0.4;

/** Highest SAE level the facade will represent. L5 is intentionally absent. */
export const SAE_CEILING = 4 as const;

/** Murakumo verbalizer: turn a CoC trace into a natural-language explanation.
 *  Implemented by the caller against the Murakumo LiteLLM gateway; the model
 *  never calls a vendor endpoint itself. */
export type NarrateFn = (coc: ChainOfCausation, obs: DrivingObservation) => Promise<string>;

export interface AlpamayoConfig {
  /** Canonical KAMI engine name reported by this instance. */
  pretrainedName: string;
  planner: Partial<PlannerConfig>;
}

// ── model ────────────────────────────────────────────────────────────────────

/** Clean-room Alpamayo-R1-shaped VLA model backed by the kami-drive planner. */
export class AlpamayoR1 {
  readonly pretrainedName: string;
  private readonly plannerCfg: Partial<PlannerConfig>;

  private constructor(cfg: AlpamayoConfig) {
    this.pretrainedName = cfg.pretrainedName;
    this.plannerCfg = {
      horizonS: TRAJECTORY_HORIZON_S,
      hz: TRAJECTORY_HZ,
      ...cfg.planner,
    };
  }

  /** `AlpamayoR1.from_pretrained(...)` mirror. The checkpoint name is accepted
   *  for API parity; the canonical KAMI engine (michibiki) is always used. */
  static fromPretrained(
    name = "nvidia/Alpamayo-R1-10B",
    opts: { planner?: Partial<PlannerConfig> } = {},
  ): AlpamayoR1 {
    return new AlpamayoR1({ pretrainedName: name, planner: opts.planner ?? {} });
  }

  /** Canonical KAMI engine name behind the facade. */
  get engine(): string {
    return "michibiki";
  }

  /** Synchronous inference over a structured driving observation. Returns a
   *  RECOMMENDED 6.4 s trajectory + Chain-of-Causation (deterministic
   *  narrative). */
  predict(obs: DrivingObservation): AlpamayoOutput {
    const { trajectory, reasoning } = plan(obs, this.plannerCfg);
    return { trajectory, reasoning, explanation: reasoning.narrative };
  }

  /** Async inference with an optional Murakumo verbalizer for the explanation.
   *  Without `narrate`, falls back to the deterministic template (identical to
   *  {@link predict}). */
  async predictAsync(
    obs: DrivingObservation,
    narrate?: NarrateFn,
  ): Promise<AlpamayoOutput> {
    const out = this.predict(obs);
    if (narrate) {
      try {
        const explanation = await narrate(out.reasoning, obs);
        return { ...out, explanation, reasoning: { ...out.reasoning, narrative: explanation } };
      } catch {
        // Murakumo unavailable → keep the deterministic narrative (fail-open).
      }
    }
    return out;
  }

  /** Lower-level overload accepting the raw Alpamayo input tuple. The egomotion
   *  history seeds the ego speed; agents must be provided via `agents` since
   *  this facade ships no vision encoder. */
  predictFromInput(input: AlpamayoInput, obs: Omit<DrivingObservation, "ego" | "instruction">): AlpamayoOutput {
    return this.predict({
      ...obs,
      instruction: input.command,
      ego: { x: 0, y: 0, yaw: 0, speed: egoSpeedFromHistory(input.egomotionHistory) },
    });
  }
}

/** Estimate current ego speed from the last two egomotion samples. */
function egoSpeedFromHistory(history: readonly EgomotionWaypoint[]): number {
  if (history.length < 2) return 0;
  const a = history[history.length - 2];
  const b = history[history.length - 1];
  const dt = b.timestamp - a.timestamp;
  if (dt <= 0) return 0;
  const dx = b.translation[0] - a.translation[0];
  const dy = b.translation[1] - a.translation[1];
  return Math.hypot(dx, dy) / dt;
}

export const KAMI_ENGINE = "michibiki";
export const ADR = "ADR-2606010600"; // kami-autodrive (AV autonomy)
