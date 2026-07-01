// @etzhayyim/kami-nv-compat/e7m-shugyo
//
// Clean-room Isaac Lab RL-environment engine (e7m-shugyo 修行) — the canonical
// KAMI implementation behind `nv-compat/isaaclab-envs`. Manager framework +
// classic Cartpole task + the ManagerBasedRLEnv Gym loop.
//
// ADR-2605261800 §D6 / D10.4 e7m-shugyo.

export {
  type ObsTerm,
  type RewTerm,
  type TerminationTerm,
  type TerminationResult,
  type EventTerm,
  type EventMode,
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
  TerminationManager,
  EventManager,
} from "./managers.js";

export {
  type CartpoleState,
  type CartpoleEnvCfg,
  type CartpoleEnvView,
  zeroState,
  defaultCartpoleCfg,
  cartpoleStep,
  nextCentered,
  resetState,
  mdp,
  cartpoleObsTerms,
  cartpoleRewTerms,
  cartpoleTerminationTerms,
  cartpoleEventTerms,
} from "./cartpole.js";

export { type StepResult, type ManagerBundle, ManagerBasedRLEnv } from "./env.js";
