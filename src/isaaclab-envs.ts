// @etzhayyim/kami-nv-compat/isaaclab-envs
//
// Drop-in NVIDIA Isaac Lab `isaaclab.envs` / `isaaclab.managers` API-compat
// facade — manager-based reinforcement-learning environments. Mirrors the
// documented surface (ManagerBasedRLEnv + the Observation / Reward /
// Termination / Event managers + the classic Cartpole MDP terms), so Isaac Lab
// task definitions port to KAMI via import-path-only changes — e.g.
//
//     import { envs, managers, mdp } from "@etzhayyim/kami-nv-compat/isaaclab-envs";
//
//     const env = new envs.ManagerBasedRLEnv({ numEnvs: 4 });
//     const obs = env.resetAll(0);
//     const out = env.stepAll(actions);   // per-env {observations, reward, ...}
//
// Backed by the clean-room e7m-shugyo engine (Cartpole dynamics + manager
// framework). No Isaac Lab source/weights/binaries; from-spec reproduction
// (Google v. Oracle, 593 U.S. ___ (2021)). Canonical engine: e7m-shugyo.
//
// Trademark: NVIDIA® / Isaac® / Isaac Lab are trademarks of NVIDIA
// Corporation; API-compat identifiers only.
//
// ADR-2605261800 §D1/D6, R1.5 isaaclab/envs surface.

import {
  ManagerBasedRLEnv,
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
  TerminationManager,
  EventManager,
  defaultCartpoleCfg,
  cartpoleObsTerms,
  cartpoleRewTerms,
  cartpoleTerminationTerms,
  cartpoleEventTerms,
  mdp,
} from "./e7m-shugyo/index.js";

export type {
  CartpoleState,
  CartpoleEnvCfg,
  CartpoleEnvView,
  StepResult,
  ManagerBundle,
  ObsTerm,
  RewTerm,
  TerminationTerm,
  TerminationResult,
  EventTerm,
  EventMode,
} from "./e7m-shugyo/index.js";

/** `isaaclab.envs` namespace. */
export const envs = {
  ManagerBasedRLEnv,
  CartpoleEnvCfg: defaultCartpoleCfg,
};

/** `isaaclab.managers` namespace. */
export const managers = {
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
  TerminationManager,
  EventManager,
};

/** `isaaclab.envs.mdp` namespace + the default Cartpole term groups. */
export const mdpNs = {
  ...mdp,
  cartpoleObsTerms,
  cartpoleRewTerms,
  cartpoleTerminationTerms,
  cartpoleEventTerms,
};

export { mdpNs as mdp };
export {
  ManagerBasedRLEnv,
  ObsGroup,
  ObservationManager,
  RewGroup,
  RewardManager,
  TerminationManager,
  EventManager,
  defaultCartpoleCfg,
} from "./e7m-shugyo/index.js";

export const KAMI_ENGINE = "e7m-shugyo";
export const ADR = "ADR-2605261800";
