// kami-drive — Chain-of-Causation (CoC) reasoning schema.
//
// Clean-room schema mirroring NVIDIA Alpamayo's "Chain-of-Causation reasoning
// traces" and the PhysicalAI-Autonomous-Vehicles `ood_reasoning` label record
// (clip uuid / event-cluster / human-refined narrative / keyframe indices),
// so existing CoC-labelled data ports by field-name mapping and so the KAMI
// planner can EMIT the same trace structure that backs Alpamayo's verbalized
// explanations.
//
// A CoC trace is an ordered list of causal steps: each step is an
// observation → inference → action triple, temporally grounded to a keyframe.
// This makes the planner's decisions auditable (which is also the religious-
// corp transparency posture — explainable, on-record reasoning).
//
// No Alpamayo data, weights, or text are copied; this is a from-spec schema.
//
// ADR-2605261800-adjacent (nv-compat); AV scope per wadachi / kami-autodrive.

// ── trace schema ──────────────────────────────────────────────────────────

/** Coarse driving-event taxonomy (mirrors the dataset "Event Cluster"). */
export type EventCluster =
  | "nominal"
  | "vru_interaction" // vulnerable road user (pedestrian / cyclist)
  | "vehicle_cut_in"
  | "intersection"
  | "yield"
  | "lane_change"
  | "stop"
  | "obstacle"
  | "merge";

/** One causal step: what was observed, what it implies, what action follows. */
export interface CausationStep {
  /** 0-based index within the trace. */
  index: number;
  /** Perceived fact, e.g. "pedestrian 8.0 m ahead within the ego lane". */
  observation: string;
  /** Causal inference, e.g. "must yield to a vulnerable road user". */
  inference: string;
  /** Resulting action, e.g. "decelerate to a stop". */
  action: string;
  /** Index of the trajectory/keyframe this step is grounded to. */
  keyframeIndex: number;
}

/** An ordered Chain-of-Causation trace + its rendered narrative. */
export interface ChainOfCausation {
  eventCluster: EventCluster;
  steps: CausationStep[];
  /** Human-readable narrative (template by default; Murakumo-verbalized on
   *  request). */
  narrative: string;
}

/** Render a default narrative from the steps (deterministic; the Murakumo
 *  verbalizer can replace this). */
export function renderNarrative(steps: readonly CausationStep[]): string {
  if (steps.length === 0) return "Proceeding under nominal conditions.";
  return steps
    .map((s) => `Because ${lower(s.observation)}, ${lower(s.inference)}; therefore ${lower(s.action)}.`)
    .join(" ");
}

function lower(s: string): string {
  return s.length > 0 ? s[0].toLowerCase() + s.slice(1) : s;
}

/** Builder for assembling a trace step-by-step. */
export class CausationBuilder {
  private readonly steps: CausationStep[] = [];
  constructor(private cluster: EventCluster = "nominal") {}

  setCluster(c: EventCluster): this {
    this.cluster = c;
    return this;
  }

  add(observation: string, inference: string, action: string, keyframeIndex = 0): this {
    this.steps.push({ index: this.steps.length, observation, inference, action, keyframeIndex });
    return this;
  }

  build(): ChainOfCausation {
    return {
      eventCluster: this.cluster,
      steps: [...this.steps],
      narrative: renderNarrative(this.steps),
    };
  }
}

// ── dataset record (mirrors ood_reasoning.parquet) ───────────────────────────

/** A CoC annotation row, mirroring the dataset's `ood_reasoning` schema. */
export interface ReasoningRecord {
  clipUuid: string;
  eventCluster: EventCluster;
  /** "Human-Refined Chain of Causation" narrative. */
  narrative: string;
  keyframeIndices: number[];
  /** Optional structured steps (KAMI extension; absent in the raw dataset). */
  steps?: CausationStep[];
}

const EVENT_CLUSTERS: ReadonlySet<string> = new Set<EventCluster>([
  "nominal", "vru_interaction", "vehicle_cut_in", "intersection",
  "yield", "lane_change", "stop", "obstacle", "merge",
]);

/** Parse + validate a loosely-typed object into a {@link ReasoningRecord}.
 *  Throws on a missing/invalid required field. Accepts snake_case or
 *  camelCase keys (dataset uses snake_case). */
export function parseReasoningRecord(obj: Record<string, unknown>): ReasoningRecord {
  const uuid = obj.clipUuid ?? obj.clip_uuid ?? obj.uuid;
  if (typeof uuid !== "string" || uuid.length === 0) {
    throw new Error("ReasoningRecord: clipUuid (string) is required");
  }
  const clusterRaw = String(obj.eventCluster ?? obj.event_cluster ?? "nominal");
  if (!EVENT_CLUSTERS.has(clusterRaw)) {
    throw new Error(`ReasoningRecord: unknown eventCluster '${clusterRaw}'`);
  }
  const narrative = String(obj.narrative ?? obj.chain_of_causation ?? "");
  const kfRaw = obj.keyframeIndices ?? obj.keyframe_indices ?? [];
  if (!Array.isArray(kfRaw)) {
    throw new Error("ReasoningRecord: keyframeIndices must be an array");
  }
  return {
    clipUuid: uuid,
    eventCluster: clusterRaw as EventCluster,
    narrative,
    keyframeIndices: kfRaw.map((n) => Number(n)),
  };
}

/** Build a dataset record from a clip id + a planner-emitted CoC trace. */
export function recordFromTrace(clipUuid: string, coc: ChainOfCausation): ReasoningRecord {
  return {
    clipUuid,
    eventCluster: coc.eventCluster,
    narrative: coc.narrative,
    keyframeIndices: coc.steps.map((s) => s.keyframeIndex),
    steps: coc.steps,
  };
}

// ── kotoba Datom bridge ──────────────────────────────────────────────────────

/** A single EAVT datom (entity, attribute, value) for the kotoba log. */
export interface Datom {
  e: string;
  a: string;
  v: string | number;
}

/** Project a reasoning record to append-only `:coc/*` datoms so a trace is
 *  queryable on the kotoba Datom log (auditable reasoning history). */
export function recordToDatoms(rec: ReasoningRecord): Datom[] {
  const e = `coc:${rec.clipUuid}`;
  const out: Datom[] = [
    { e, a: ":coc/clip", v: rec.clipUuid },
    { e, a: ":coc/event-cluster", v: rec.eventCluster },
    { e, a: ":coc/narrative", v: rec.narrative },
  ];
  rec.keyframeIndices.forEach((k) => out.push({ e, a: ":coc/keyframe", v: k }));
  (rec.steps ?? []).forEach((s) => {
    const se = `${e}:step:${s.index}`;
    out.push(
      { e: se, a: ":coc.step/of", v: e },
      { e: se, a: ":coc.step/observation", v: s.observation },
      { e: se, a: ":coc.step/inference", v: s.inference },
      { e: se, a: ":coc.step/action", v: s.action },
      { e: se, a: ":coc.step/keyframe", v: s.keyframeIndex },
    );
  });
  return out;
}
