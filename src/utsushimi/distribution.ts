// utsushimi — Replicator distribution primitives.
//
// Mirrors `omni.replicator.core.distribution.*`: each constructor returns a
// tagged object captured at script time and materialized per-frame by
// {@link sample} (Replicator's lazy distribution semantics). Bit-reproducible
// via the BigInt {@link Sampler}.
//
// ADR-2605261800 §D6 / D10.4 utsushimi.

import { type Sampler, globalSampler } from "./sampler.js";

export type Dist =
  | { _kind: "uniform"; low: number[]; high: number[] }
  | { _kind: "normal"; mean: number[]; std: number[] }
  | { _kind: "truncated_normal"; mean: number[]; std: number[]; low: number[]; high: number[] }
  | { _kind: "choice"; options: unknown[] }
  | { _kind: "sequence"; values: unknown[]; _index: [number] }
  | { _kind: "combine"; distributions: Dist[] };

export type Sampled = number[] | unknown;

export const distribution = {
  uniform(low: number[], high: number[]): Dist {
    return { _kind: "uniform", low: [...low], high: [...high] };
  },
  normal(mean: number[], std: number[]): Dist {
    return { _kind: "normal", mean: [...mean], std: [...std] };
  },
  truncated_normal(mean: number[], std: number[], low: number[], high: number[]): Dist {
    return { _kind: "truncated_normal", mean: [...mean], std: [...std], low: [...low], high: [...high] };
  },
  choice(options: unknown[]): Dist {
    return { _kind: "choice", options: [...options] };
  },
  sequence(values: unknown[]): Dist {
    return { _kind: "sequence", values: [...values], _index: [0] };
  },
  combine(distributions: Dist[]): Dist {
    return { _kind: "combine", distributions: [...distributions] };
  },
};

/** Materialize a distribution to a concrete value (uses the global sampler
 *  when none is given). Mirrors `omni.replicator.core.sample`. */
export function sample(dist: Dist, sampler?: Sampler): Sampled {
  const s = sampler ?? globalSampler();
  switch (dist._kind) {
    case "uniform":
      return dist.low.map((lo, i) => s.nextUniform(lo, dist.high[i]));
    case "normal":
      return dist.mean.map((m, i) => s.nextNormal(m, dist.std[i]));
    case "truncated_normal":
      return dist.mean.map((m, i) => s.nextTruncatedNormal(m, dist.std[i], dist.low[i], dist.high[i]));
    case "choice": {
      const idx = Math.min(dist.options.length - 1, Math.floor(s.nextU01() * dist.options.length));
      return dist.options[idx];
    }
    case "sequence": {
      const i = dist._index[0] % dist.values.length;
      dist._index[0] = (dist._index[0] + 1) % dist.values.length;
      return dist.values[i];
    }
    case "combine": {
      const out: unknown[] = [];
      for (const sub of dist.distributions) {
        const v = sample(sub, s);
        if (Array.isArray(v)) out.push(...v);
        else out.push(v);
      }
      return out;
    }
  }
}
