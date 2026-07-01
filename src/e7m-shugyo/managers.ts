// e7m-shugyo — clean-room Isaac Lab manager framework (RL env managers).
//
// The canonical KAMI implementation behind `nv-compat/isaaclab-envs`. NVIDIA
// Isaac Lab drives manager-based RL environments from declarative term groups;
// this module reproduces the documented manager API
// (isaaclab.managers.{Observation,Reward,Termination,Event}Manager) so Isaac
// Lab task configs port to KAMI via import-path-only changes.
//
// Each manager is generic over the env type `E`; terms are plain functions
// `(env, params) => value`, so the same managers drive any KAMI env (the
// cartpole env ships in cartpole.ts).
//
// Clean-room: from-spec manager semantics. No Isaac Lab source/binaries.
// ADR-2605261800 §D6 / D10.4 e7m-shugyo.

// ── observation manager ──────────────────────────────────────────────────────

export interface ObsTerm<E> {
  /** Returns a scalar or vector observation. */
  func: (env: E, params?: Record<string, unknown>) => number | number[];
  /** Multiplicative scale applied to the term's output. */
  scale?: number;
  params?: Record<string, unknown>;
}

/** A named group of observation terms, concatenated into one flat vector. */
export class ObsGroup<E> {
  constructor(readonly terms: Record<string, ObsTerm<E>>) {}

  evaluate(env: E): number[] {
    const out: number[] = [];
    for (const term of Object.values(this.terms)) {
      const v = term.func(env, term.params);
      const scale = term.scale ?? 1;
      if (Array.isArray(v)) out.push(...v.map((x) => x * scale));
      else out.push(v * scale);
    }
    return out;
  }
}

export class ObservationManager<E> {
  constructor(readonly groups: Record<string, ObsGroup<E>> = {}) {}

  compute(env: E): Record<string, number[]> {
    const out: Record<string, number[]> = {};
    for (const [name, group] of Object.entries(this.groups)) out[name] = group.evaluate(env);
    return out;
  }
  groupNames(): string[] {
    return Object.keys(this.groups);
  }
}

// ── reward manager ───────────────────────────────────────────────────────────

export interface RewTerm<E> {
  func: (env: E, params?: Record<string, unknown>) => number;
  weight: number;
  params?: Record<string, unknown>;
}

export class RewGroup<E> {
  constructor(readonly terms: Record<string, RewTerm<E>>) {}

  evaluate(env: E): number {
    let r = 0;
    for (const term of Object.values(this.terms)) r += term.weight * term.func(env, term.params);
    return r;
  }
  evaluateBreakdown(env: E): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [name, term] of Object.entries(this.terms)) out[name] = term.weight * term.func(env, term.params);
    return out;
  }
}

export class RewardManager<E> {
  private episodeSum = 0;
  private readonly episodeBreakdown: Record<string, number> = {};
  private stepCount = 0;

  constructor(readonly group: RewGroup<E>) {}

  compute(env: E): number {
    const r = this.group.evaluate(env);
    this.episodeSum += r;
    this.stepCount++;
    for (const [name, val] of Object.entries(this.group.evaluateBreakdown(env))) {
      this.episodeBreakdown[name] = (this.episodeBreakdown[name] ?? 0) + val;
    }
    return r;
  }
  getBreakdown(env: E): Record<string, number> {
    return this.group.evaluateBreakdown(env);
  }
  logEpisodeReward(): Record<string, number> {
    return { total: this.episodeSum, steps: this.stepCount, ...this.episodeBreakdown };
  }
  resetEpisodeLog(): void {
    this.episodeSum = 0;
    this.stepCount = 0;
    for (const k of Object.keys(this.episodeBreakdown)) delete this.episodeBreakdown[k];
  }
}

// ── termination manager ──────────────────────────────────────────────────────

export interface TerminationTerm<E> {
  func: (env: E, params?: Record<string, unknown>) => boolean;
  /** When true, this term marks truncation (e.g. time-out) rather than a hard
   *  termination. */
  timeOut?: boolean;
  params?: Record<string, unknown>;
}

export interface TerminationResult {
  terminated: boolean;
  truncated: boolean;
  info: Record<string, boolean>;
}

export class TerminationManager<E> {
  constructor(readonly terms: Record<string, TerminationTerm<E>> = {}) {}

  compute(env: E): TerminationResult {
    let terminated = false;
    let truncated = false;
    const info: Record<string, boolean> = {};
    for (const [name, term] of Object.entries(this.terms)) {
      const v = term.func(env, term.params);
      info[name] = v;
      if (v) {
        if (term.timeOut) truncated = true;
        else terminated = true;
      }
    }
    return { terminated, truncated, info };
  }
}

// ── event manager ────────────────────────────────────────────────────────────

export type EventMode = "startup" | "reset" | "interval";

export interface EventTerm<E> {
  /** Mutates the env in place. */
  func: (env: E, params?: Record<string, unknown>) => void;
  mode: EventMode;
  /** For `interval` mode: fire every `intervalS` seconds (sim time). */
  intervalS?: number;
  params?: Record<string, unknown>;
}

export class EventManager<E> {
  private readonly lastFired: Record<string, number> = {};

  constructor(readonly terms: Record<string, EventTerm<E>> = {}) {}

  /** Apply all terms matching `mode`. For interval terms, `simTime` gates
   *  firing by `intervalS`. */
  apply(env: E, mode: EventMode, simTime = 0): void {
    for (const [name, term] of Object.entries(this.terms)) {
      if (term.mode !== mode) continue;
      if (mode === "interval") {
        const last = this.lastFired[name] ?? -Infinity;
        if (simTime - last < (term.intervalS ?? 0)) continue;
        this.lastFired[name] = simTime;
      }
      term.func(env, term.params);
    }
  }
  resetIntervals(): void {
    for (const k of Object.keys(this.lastFired)) delete this.lastFired[k];
  }
}
