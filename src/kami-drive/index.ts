// @etzhayyim/kami-nv-compat/kami-drive
//
// Clean-room autonomous-driving reasoning engine (michibiki 導き) — the
// canonical KAMI implementation behind the `nv-compat/alpamayo` VLA facade and
// the `nv-compat/alpasim` closed-loop harness. BEV unicycle kinematics +
// rule-based reasoning planner + Chain-of-Causation trace schema.
//
// Civilian autonomous-mobility only, SAE-L4 ceiling, NO actuation (plans /
// sim only). ADR-2605242000 (wadachi) / ADR-2606010600 (kami-autodrive) scope.

export {
  type BevState,
  type DynamicAction,
  type Waypoint,
  DEFAULT_MAX_SPEED,
  yawToMat3,
  stepUnicycle,
  rolloutTrajectory,
  trajectoryLength,
} from "./unicycle.js";

export {
  type EventCluster,
  type CausationStep,
  type ChainOfCausation,
  type ReasoningRecord,
  type Datom,
  CausationBuilder,
  renderNarrative,
  parseReasoningRecord,
  recordFromTrace,
  recordToDatoms,
} from "./coc.js";

export {
  type NavigationCommand,
  type AgentKind,
  type PerceivedAgent,
  type DrivingObservation,
  type PlannerConfig,
  type PlanResult,
  DEFAULT_PLANNER,
  commandFromInstruction,
  plan,
} from "./planner.js";
