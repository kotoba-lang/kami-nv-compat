// e7m-shugyo — clean-room cart-pole dynamics + MDP terms (Isaac Lab classic).
//
// Reproduces the Isaac Lab classic Cartpole task: cart-pole ODE integration,
// CartpoleEnvCfg, LCG-seeded reset (the same 64-bit LCG as utsushimi, so seeds
// are reproducible across the SDK), and the MDP term functions the managers
// consume (observations / rewards / terminations / events).
//
// State: x (cart position), xDot, theta (pole angle from upright, 0 = up),
// thetaDot. Dynamics are the textbook Barto–Sutton cart-pole; no Isaac Lab
// source/weights are used.
//
// ADR-2605261800 §D6 / D10.4 e7m-shugyo.

import { Sampler } from "../utsushimi/index.js";
import {
  type EventTerm,
  type ObsTerm,
  type RewTerm,
  type TerminationTerm,
} from "./managers.js";

export interface CartpoleState {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
}

export function zeroState(): CartpoleState {
  return { x: 0, xDot: 0, theta: 0, thetaDot: 0 };
}

export interface CartpoleEnvCfg {
  numEnvs: number;
  physicsDt: number;
  decimation: number;
  gravity: number;
  cartMass: number;
  poleMass: number;
  /** Pole half-length to the centre of mass (m). */
  poleLength: number;
  forceMag: number;
  // reward weights
  alive: number;
  terminating: number;
  polePosPenalty: number;
  cartVelPenalty: number;
  poleVelPenalty: number;
  // termination
  maxEpisodeLengthS: number;
  poleBound: number;
  cartBound: number;
  /** Half-range of the uniform reset perturbation. */
  resetNoise: number;
}

export function defaultCartpoleCfg(over: Partial<CartpoleEnvCfg> = {}): CartpoleEnvCfg {
  return {
    numEnvs: 1,
    physicsDt: 1 / 60,
    decimation: 2,
    gravity: 9.81,
    cartMass: 1.0,
    poleMass: 0.1,
    poleLength: 0.5,
    forceMag: 10,
    alive: 1.0,
    terminating: -2.0,
    polePosPenalty: -1.0,
    cartVelPenalty: -0.01,
    poleVelPenalty: -0.005,
    maxEpisodeLengthS: 5.0,
    poleBound: 0.6,
    cartBound: 2.4,
    resetNoise: 0.05,
    ...over,
  };
}

/** One physics step of the cart-pole (Barto–Sutton, theta from upright). */
export function cartpoleStep(s: CartpoleState, force: number, cfg: CartpoleEnvCfg): CartpoleState {
  const { gravity: g, cartMass: mc, poleMass: mp, poleLength: l, physicsDt: dt } = cfg;
  const total = mc + mp;
  const ct = Math.cos(s.theta), st = Math.sin(s.theta);
  const temp = (force + mp * l * s.thetaDot * s.thetaDot * st) / total;
  const thetaAcc = (g * st - ct * temp) / (l * (4 / 3 - (mp * ct * ct) / total));
  const xAcc = temp - (mp * l * thetaAcc * ct) / total;
  return {
    x: s.x + dt * s.xDot,
    xDot: s.xDot + dt * xAcc,
    theta: s.theta + dt * s.thetaDot,
    thetaDot: s.thetaDot + dt * thetaAcc,
  };
}

/** Centered uniform draw in [-half, half] from a {@link Sampler} (matches the
 *  Python `_Lcg.next_f32_centered`). */
export function nextCentered(rng: Sampler, half: number): number {
  return (rng.nextU01() * 2 - 1) * half;
}

/** Seed a fresh randomized cart-pole state. */
export function resetState(rng: Sampler, cfg: CartpoleEnvCfg): CartpoleState {
  return {
    x: nextCentered(rng, cfg.resetNoise),
    xDot: nextCentered(rng, cfg.resetNoise),
    theta: nextCentered(rng, cfg.resetNoise),
    thetaDot: nextCentered(rng, cfg.resetNoise),
  };
}

// ── env surface the MDP terms read ───────────────────────────────────────────

/** The minimal env surface MDP term functions consume. Satisfied by
 *  ManagerBasedRLEnv (single-env view) and the vectorized per-env view. */
export interface CartpoleEnvView {
  state: CartpoleState;
  lastAction: number[];
  terminated: boolean;
  stepCount: number;
  maxSteps: number;
  cfg: CartpoleEnvCfg;
  rng: Sampler;
}

// ── MDP term functions ───────────────────────────────────────────────────────

export const mdp = {
  // observations
  jointPosRel(env: CartpoleEnvView): number[] {
    return [env.state.x, env.state.theta];
  },
  jointVelRel(env: CartpoleEnvView): number[] {
    return [env.state.xDot, env.state.thetaDot];
  },
  lastAction(env: CartpoleEnvView): number[] {
    return [...env.lastAction];
  },

  // rewards (raw terms; weights live in the RewTerm)
  isAlive(env: CartpoleEnvView): number {
    return env.terminated ? 0 : 1;
  },
  isTerminated(env: CartpoleEnvView): number {
    return env.terminated ? 1 : 0;
  },
  polePosL2(env: CartpoleEnvView): number {
    return env.state.theta * env.state.theta;
  },
  cartVelL2(env: CartpoleEnvView): number {
    return env.state.xDot * env.state.xDot;
  },
  poleVelL2(env: CartpoleEnvView): number {
    return env.state.thetaDot * env.state.thetaDot;
  },

  // terminations
  poleOutOfBounds(env: CartpoleEnvView): boolean {
    return Math.abs(env.state.theta) > env.cfg.poleBound;
  },
  cartOutOfBounds(env: CartpoleEnvView): boolean {
    return Math.abs(env.state.x) > env.cfg.cartBound;
  },
  timeOut(env: CartpoleEnvView): boolean {
    return env.stepCount >= env.maxSteps;
  },

  // events
  resetJointsByOffset(env: CartpoleEnvView): void {
    env.state = resetState(env.rng, env.cfg);
  },
};

// ── default Isaac Lab Cartpole term groups ───────────────────────────────────

export function cartpoleObsTerms(): Record<string, ObsTerm<CartpoleEnvView>> {
  return {
    jointPos: { func: mdp.jointPosRel },
    jointVel: { func: mdp.jointVelRel },
    lastAction: { func: mdp.lastAction, scale: 0.1 },
  };
}

export function cartpoleRewTerms(cfg: CartpoleEnvCfg): Record<string, RewTerm<CartpoleEnvView>> {
  return {
    alive: { func: mdp.isAlive, weight: cfg.alive },
    terminating: { func: mdp.isTerminated, weight: cfg.terminating },
    polePos: { func: mdp.polePosL2, weight: cfg.polePosPenalty },
    cartVel: { func: mdp.cartVelL2, weight: cfg.cartVelPenalty },
    poleVel: { func: mdp.poleVelL2, weight: cfg.poleVelPenalty },
  };
}

export function cartpoleTerminationTerms(): Record<string, TerminationTerm<CartpoleEnvView>> {
  return {
    poleOob: { func: mdp.poleOutOfBounds },
    cartOob: { func: mdp.cartOutOfBounds },
    timeOut: { func: mdp.timeOut, timeOut: true },
  };
}

export function cartpoleEventTerms(): Record<string, EventTerm<CartpoleEnvView>> {
  return {
    resetPose: { func: mdp.resetJointsByOffset, mode: "reset" },
  };
}
