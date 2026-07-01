// ActionTerm base + ActionTermCfgBase.
//
// Mirror of `isaaclab.envs.mdp.actions.ActionTerm` (Isaac Lab 1.x). One
// ActionTerm processes one slice of the policy's flat action vector
// (`process_actions(raw)` → `processed_actions` via scale + offset),
// then writes the result onto the env (`apply_actions(env)`).
//
// `ActionManager` composes multiple ActionTerm instances into a single
// combined action vector — slices `raw` by per-term offsets and
// dispatches each slice.

import type { ArticulatedEnv } from "./articulated-env.js";

export interface ActionTermCfgBase {
  assetName?: string;
  /** Integer joint indices (env-specific) the action term targets. */
  jointNames: number[];
  /** Multiplied element-wise with the raw action. Default 1.0. */
  scale?: number;
  /** Added element-wise after scale. Default 0.0. */
  offset?: number;
  /** Optional override; if undefined, inferred from jointNames length. */
  actionDim?: number;
}

export abstract class ActionTerm {
  cfg: ActionTermCfgBase;
  readonly actionDim: number;
  raw_actions: number[];
  processed_actions: number[];

  constructor(cfg: ActionTermCfgBase) {
    if (!cfg.jointNames || cfg.jointNames.length === 0) {
      throw new Error(`${this.constructor.name}.cfg.jointNames must be non-empty`);
    }
    this.cfg = cfg;
    this.actionDim = cfg.actionDim ?? cfg.jointNames.length;
    this.raw_actions = new Array<number>(this.actionDim).fill(0);
    this.processed_actions = new Array<number>(this.actionDim).fill(0);
  }

  /** Default impl: scale + offset element-wise into processed_actions. */
  processActions(raw: readonly number[]): void {
    if (raw.length !== this.actionDim) {
      throw new Error(
        `${this.constructor.name}: expected ${this.actionDim} action elements, got ${raw.length}`,
      );
    }
    this.raw_actions = [...raw];
    const s = this.cfg.scale ?? 1.0;
    const o = this.cfg.offset ?? 0.0;
    this.processed_actions = raw.map((r) => r * s + o);
  }

  abstract applyActions(env: ArticulatedEnv): void;

  reset(): void {
    this.raw_actions = new Array<number>(this.actionDim).fill(0);
    this.processed_actions = new Array<number>(this.actionDim).fill(0);
  }
}

// ── ActionManager ────────────────────────────────────────────────────────

export class ActionManager {
  readonly terms: ActionTerm[];
  readonly termNames: string[];
  readonly totalActionDim: number;
  private readonly _offsets: number[];

  constructor(terms: ActionTerm[]) {
    if (!terms || terms.length === 0) {
      throw new Error("ActionManager requires at least one ActionTerm");
    }
    this.terms = terms;
    this.termNames = terms.map((t) => t.constructor.name);
    this._offsets = [];
    let off = 0;
    for (const t of terms) {
      this._offsets.push(off);
      off += t.actionDim;
    }
    this.totalActionDim = off;
  }

  /** Slice `raw` and dispatch to each term in order. */
  processActions(raw: readonly number[]): void {
    if (raw.length !== this.totalActionDim) {
      throw new Error(
        `ActionManager: expected ${this.totalActionDim} action elements, got ${raw.length}`,
      );
    }
    for (let i = 0; i < this.terms.length; i++) {
      const start = this._offsets[i];
      const end = start + this.terms[i].actionDim;
      this.terms[i].processActions(raw.slice(start, end));
    }
  }

  applyActions(env: ArticulatedEnv): void {
    for (const t of this.terms) t.applyActions(env);
  }

  reset(_envIds?: readonly number[]): void {
    for (const t of this.terms) t.reset();
  }

  numTerms(): number {
    return this.terms.length;
  }
}
