// e7m-shugyo — ManagerBasedRLEnv (Isaac Lab classic Cartpole).
//
// Mirrors `isaaclab.envs.ManagerBasedRLEnv`: a Gym-style RL environment driven
// by declarative managers. The constructor takes a CartpoleEnvCfg plus
// optional Observation / Reward / Termination / Event managers (or their term
// groups, auto-wrapped); `reset` / `step` run the standard Isaac Lab loop —
// apply action → decimated physics → observations / rewards / terminations.
//
// Vectorized over `cfg.numEnvs`: per-env states are stepped in lockstep, with
// per-env auto-reset of done envs (the Isaac Lab vectorized convention).
//
// ADR-2605261800 §D6 / D10.4 e7m-shugyo.

import { Sampler } from "../utsushimi/index.js";
import {
  EventManager,
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
  TerminationManager,
} from "./managers.js";
import {
  type CartpoleEnvCfg,
  type CartpoleEnvView,
  type CartpoleState,
  cartpoleEventTerms,
  cartpoleObsTerms,
  cartpoleRewTerms,
  cartpoleStep,
  cartpoleTerminationTerms,
  defaultCartpoleCfg,
  resetState,
} from "./cartpole.js";

export interface StepResult {
  observations: Record<string, number[]>;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, boolean>;
}

export interface ManagerBundle {
  observations?: ObservationManager<CartpoleEnvView> | Record<string, ObsGroup<CartpoleEnvView>>;
  rewards?: RewardManager<CartpoleEnvView> | RewGroup<CartpoleEnvView>;
  terminations?: TerminationManager<CartpoleEnvView>;
  events?: EventManager<CartpoleEnvView>;
}

/** Per-env mutable record (the view MDP terms read). */
interface EnvSlot extends CartpoleEnvView {
  truncated: boolean;
}

export class ManagerBasedRLEnv {
  readonly cfg: CartpoleEnvCfg;
  readonly maxSteps: number;
  private readonly obsMgr: ObservationManager<CartpoleEnvView>;
  private readonly rewMgr: RewardManager<CartpoleEnvView>;
  private readonly termMgr: TerminationManager<CartpoleEnvView>;
  private readonly eventMgr: EventManager<CartpoleEnvView>;
  private readonly slots: EnvSlot[];

  constructor(cfg: Partial<CartpoleEnvCfg> = {}, bundle: ManagerBundle = {}) {
    this.cfg = defaultCartpoleCfg(cfg);
    this.maxSteps = Math.round(this.cfg.maxEpisodeLengthS / this.cfg.physicsDt);

    this.obsMgr =
      bundle.observations instanceof ObservationManager
        ? bundle.observations
        : new ObservationManager(bundle.observations ?? { policy: new ObsGroup(cartpoleObsTerms()) });
    this.rewMgr =
      bundle.rewards instanceof RewardManager
        ? bundle.rewards
        : new RewardManager(bundle.rewards ?? new RewGroup(cartpoleRewTerms(this.cfg)));
    this.termMgr = bundle.terminations ?? new TerminationManager(cartpoleTerminationTerms());
    this.eventMgr = bundle.events ?? new EventManager(cartpoleEventTerms());

    this.slots = Array.from({ length: Math.max(1, this.cfg.numEnvs) }, (_, i) => ({
      state: { x: 0, xDot: 0, theta: 0, thetaDot: 0 },
      lastAction: [0],
      terminated: false,
      truncated: false,
      stepCount: 0,
      maxSteps: this.maxSteps,
      cfg: this.cfg,
      rng: new Sampler(i),
    }));
    this.eventMgr.apply(this.slots[0], "startup");
  }

  get numEnvs(): number {
    return this.cfg.numEnvs;
  }
  get observationManager(): ObservationManager<CartpoleEnvView> {
    return this.obsMgr;
  }
  get rewardManager(): RewardManager<CartpoleEnvView> {
    return this.rewMgr;
  }

  /** Single-env state view (slot 0). */
  get state(): CartpoleState {
    return this.slots[0].state;
  }

  // ── vectorized API ─────────────────────────────────────────────────────────

  /** Reset all envs (optionally re-seeding env i with `seed + i`). Returns the
   *  per-env observation dicts. */
  resetAll(seed?: number): Record<string, number[]>[] {
    return this.slots.map((slot, i) => {
      if (seed !== undefined) slot.rng = new Sampler(seed + i);
      slot.state = resetState(slot.rng, this.cfg);
      slot.lastAction = [0];
      slot.terminated = false;
      slot.truncated = false;
      slot.stepCount = 0;
      this.eventMgr.apply(slot, "reset");
      return this.obsMgr.compute(slot);
    });
  }

  /** Step all envs with per-env action vectors. Done envs auto-reset and the
   *  returned observation is the post-reset observation (Isaac Lab convention),
   *  while `terminated`/`truncated` flag the transition that just ended. */
  stepAll(actions: number[][]): StepResult[] {
    return this.slots.map((slot, i) => this.stepSlot(slot, actions[i] ?? [0]));
  }

  // ── single-env convenience ──────────────────────────────────────────────────

  reset(seed?: number): Record<string, number[]> {
    return this.resetAll(seed)[0];
  }

  step(action: number[]): StepResult {
    return this.stepSlot(this.slots[0], action);
  }

  // ── core step ───────────────────────────────────────────────────────────────

  private stepSlot(slot: EnvSlot, action: number[]): StepResult {
    const force = Math.max(-this.cfg.forceMag, Math.min(this.cfg.forceMag, (action[0] ?? 0) * this.cfg.forceMag));
    slot.lastAction = [...action];
    // Decimated physics.
    for (let d = 0; d < this.cfg.decimation; d++) {
      slot.state = cartpoleStep(slot.state, force, this.cfg);
    }
    slot.stepCount++;

    const term = this.termMgr.compute(slot);
    slot.terminated = term.terminated;
    slot.truncated = term.truncated;
    const reward = this.rewMgr.compute(slot);
    let observations = this.obsMgr.compute(slot);

    if (term.terminated || term.truncated) {
      // Auto-reset this env; report the post-reset observation.
      slot.state = resetState(slot.rng, this.cfg);
      slot.lastAction = [0];
      slot.stepCount = 0;
      slot.terminated = false;
      slot.truncated = false;
      this.eventMgr.apply(slot, "reset");
      observations = this.obsMgr.compute(slot);
    }
    return { observations, reward, terminated: term.terminated, truncated: term.truncated, info: term.info };
  }
}
