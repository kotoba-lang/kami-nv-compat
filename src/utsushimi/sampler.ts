// utsushimi — clean-room synthetic-data sampler (Replicator DR core).
//
// The canonical KAMI implementation behind `nv-compat/omni-replicator-core`.
// NVIDIA Omniverse Replicator drives domain-randomization from seedable
// distributions; this module reproduces the documented DR primitives plus a
// 64-bit LCG sampler that is BIT-IDENTICAL to the Python reference
// (kotodama.nv_compat.omni.replicator.core._Sampler), so a randomization seed
// produces the same stream in TS and Python — the cross-language
// reproducibility the Replicator G5 gate requires.
//
// The LCG is the PCG/Knuth multiplier (6364136223846793005) + increment
// (1442695040888963407) modulo 2⁶⁴; JS Number cannot hold 64-bit products, so
// the state is carried in BigInt and matched to the Python `& 0xFFFF…` math.
//
// Clean-room: from-spec PRNG + textbook distributions. No Replicator source or
// binaries. ADR-2605261800 §D6 / D10.4 utsushimi.

const MASK64 = (1n << 64n) - 1n;
const MUL = 6364136223846793005n;
const INC = 1442695040888963407n;
const TWO31 = 2147483648; // 2³¹

/** Seedable 64-bit LCG, bit-identical to the Python `_Sampler`. */
export class Sampler {
  private state: bigint;

  constructor(seed = 0) {
    this.state = (BigInt(seed) * MUL + INC) & MASK64;
  }

  /** Next uniform in [0, 1) — `((state>>33) & 0x7FFFFFFF) / 2³¹`. */
  nextU01(): number {
    this.state = (this.state * MUL + INC) & MASK64;
    const top = Number((this.state >> 33n) & 0x7fffffffn);
    return top / TWO31;
  }

  nextUniform(low: number, high: number): number {
    return low + (high - low) * this.nextU01();
  }

  /** Box–Muller normal (consumes two uniforms, matching the reference). */
  nextNormal(mean: number, std: number): number {
    const u1 = Math.max(this.nextU01(), 1e-12);
    const u2 = this.nextU01();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }

  /** Rejection-sampled truncated normal (caps at 20 attempts like the ref). */
  nextTruncatedNormal(mean: number, std: number, low: number, high: number): number {
    for (let i = 0; i < 20; i++) {
      const v = this.nextNormal(mean, std);
      if (v >= low && v <= high) return v;
    }
    return Math.max(low, Math.min(high, mean));
  }
}

// ── module-level shared sampler (rep.distribution.sample default) ───────────

let _global = new Sampler(0);

/** Re-seed the module-level sampler used by {@link sample} with no explicit
 *  sampler (mirrors `seed_global`). */
export function seedGlobal(seed: number): void {
  _global = new Sampler(seed);
}

export function globalSampler(): Sampler {
  return _global;
}
