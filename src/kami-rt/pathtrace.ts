// kami-rtx — clean-room Monte-Carlo path tracer (CPU reference path).
//
// The canonical engine behind `nv-compat/rtx-renderer.ts`. NVIDIA's RTX
// Renderer is the Omniverse real-time/offline path tracer; kami-rtx
// reproduces its *behaviour* (progressive Monte-Carlo path tracing with
// emissive lights + Lambertian materials) on WebGPU + WGSL, with this module
// as the byte-compatible CPU fallback that runs everywhere.
//
// Per ADR-2605261800 D10.4 this is the `kami-rtx-native` path — a path tracer
// built from-scratch on top of kami-rt's BVH/intersection (the fallback to
// the Mitsuba 3 wgpu upstream route, D3). Nothing here derives from RTX /
// OptiX / Mitsuba source; it is textbook unidirectional path tracing
// (cosine-weighted importance sampling, next-event-free, Russian-roulette-
// free for determinism).
//
// The RNG and hemisphere sampling are written to be reproducible bit-for-bit
// between CPU and WGSL: a per-(pixel, sample) xorshift32 stream, advanced in
// the exact same order, so the GPU image matches the CPU reference.
//
// ADR-2605261800 §D6 / D10.4 kami-rtx.

import {
  type Bvh,
  type Camera,
  type TriangleSoup,
  type Vec3,
  buildBvh,
  traceClosest,
  triangleSoup,
  triNormal,
} from "./bvh.js";

// ── materials ─────────────────────────────────────────────────────────────

export interface Material {
  /** Lambertian reflectance (0..1 per channel). */
  albedo: Vec3;
  /** Emitted radiance (≥0 per channel); nonzero ⇒ the surface is a light. */
  emission: Vec3;
}

export function material(albedo: Vec3, emission: Vec3 = [0, 0, 0]): Material {
  return { albedo, emission };
}

/** Per-triangle materials, parallel to the {@link TriangleSoup}. Stored as
 *  flat Float32Arrays (3 floats/triangle) so they upload verbatim to WGSL. */
export interface MaterialSoup {
  /** 3 floats/triangle. */
  albedo: Float32Array;
  /** 3 floats/triangle. */
  emission: Float32Array;
}

export function materialSoup(materials: readonly Material[]): MaterialSoup {
  const albedo = new Float32Array(materials.length * 3);
  const emission = new Float32Array(materials.length * 3);
  materials.forEach((m, i) => {
    albedo[i * 3] = m.albedo[0];
    albedo[i * 3 + 1] = m.albedo[1];
    albedo[i * 3 + 2] = m.albedo[2];
    emission[i * 3] = m.emission[0];
    emission[i * 3 + 1] = m.emission[1];
    emission[i * 3 + 2] = m.emission[2];
  });
  return { albedo, emission };
}

/** A path-traceable scene: geometry + acceleration structure + materials. */
export interface PathScene {
  soup: TriangleSoup;
  bvh: Bvh;
  mats: MaterialSoup;
}

/** Build a {@link PathScene} from `[v0,v1,v2]` triangles + a parallel
 *  per-triangle material list. */
export function buildPathScene(
  triangles: readonly Vec3[][],
  materials: readonly Material[],
): PathScene {
  if (triangles.length !== materials.length) {
    throw new Error(
      `buildPathScene: ${triangles.length} triangles vs ${materials.length} materials — must be 1:1`,
    );
  }
  const soup = triangleSoup(triangles);
  return { soup, bvh: buildBvh(soup), mats: materialSoup(materials) };
}

// ── deterministic RNG (xorshift32; identical to the WGSL stream) ───────────

/** Mix three integers into a 32-bit seed (a small, well-dispersing hash). */
export function seedHash(px: number, py: number, sample: number): number {
  let h = (px * 1973 + py * 9277 + sample * 26699 + 1) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return h === 0 ? 1 : h; // xorshift must not be seeded with 0
}

/** Mutable single-stream xorshift32. `s.v` holds the state. */
export interface Rng {
  v: number;
}

export function nextFloat(s: Rng): number {
  let x = s.v >>> 0;
  x ^= x << 13;
  x >>>= 0;
  x ^= x >>> 17;
  x ^= x << 5;
  x >>>= 0;
  s.v = x;
  return x / 4294967296; // [0, 1)
}

// ── small vector helpers ──────────────────────────────────────────────────

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, b: Vec3): Vec3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** Branchless orthonormal basis from a unit normal (Duff et al. 2017).
 *  Returned as `[tangent, bitangent]`; identical formula in WGSL. */
export function onb(n: Vec3): [Vec3, Vec3] {
  const sign = n[2] >= 0 ? 1 : -1;
  const a = -1 / (sign + n[2]);
  const b = n[0] * n[1] * a;
  const t: Vec3 = [1 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]];
  const bt: Vec3 = [b, sign + n[1] * n[1] * a, -n[1]];
  return [t, bt];
}

/** Cosine-weighted hemisphere sample around `n`, drawing 2 floats from `s`. */
function cosineSample(n: Vec3, s: Rng): Vec3 {
  const r1 = nextFloat(s);
  const r2 = nextFloat(s);
  const phi = 2 * Math.PI * r1;
  const sinT = Math.sqrt(r2);
  const cosT = Math.sqrt(1 - r2);
  const [t, bt] = onb(n);
  const x = Math.cos(phi) * sinT;
  const y = Math.sin(phi) * sinT;
  return norm(add(add(scale(t, x), scale(bt, y)), scale(n, cosT)));
}

// ── path-trace settings ───────────────────────────────────────────────────

export interface PathSettings {
  /** Samples per pixel accumulated this call. */
  samplesPerPixel: number;
  /** Max path length (bounces) before the path is terminated. */
  maxBounces: number;
  /** Radiance returned when a ray escapes the scene (sky / background). */
  background: Vec3;
}

export const DEFAULT_PATH_SETTINGS: PathSettings = {
  samplesPerPixel: 16,
  maxBounces: 6,
  background: [0, 0, 0],
};

function triEmission(mats: MaterialSoup, tri: number): Vec3 {
  return [mats.emission[tri * 3], mats.emission[tri * 3 + 1], mats.emission[tri * 3 + 2]];
}
function triAlbedo(mats: MaterialSoup, tri: number): Vec3 {
  return [mats.albedo[tri * 3], mats.albedo[tri * 3 + 1], mats.albedo[tri * 3 + 2]];
}

/** Trace one camera path and return the radiance it carries. */
function radiance(scene: PathScene, ro: Vec3, rd: Vec3, settings: PathSettings, s: Rng): Vec3 {
  let throughput: Vec3 = [1, 1, 1];
  let acc: Vec3 = [0, 0, 0];
  let o = ro;
  let d = rd;
  for (let bounce = 0; bounce <= settings.maxBounces; bounce++) {
    const hit = traceClosest(scene.soup, scene.bvh, o, d);
    if (!hit) {
      acc = add(acc, mul(throughput, settings.background));
      break;
    }
    acc = add(acc, mul(throughput, triEmission(scene.mats, hit.tri)));
    if (bounce === settings.maxBounces) break;

    // Geometric normal facing the incoming ray.
    let nrm = triNormal(scene.soup, hit.tri);
    if (nrm[0] * d[0] + nrm[1] * d[1] + nrm[2] * d[2] > 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];

    // Lambertian bounce: cosine-importance-sampled ⇒ throughput *= albedo.
    throughput = mul(throughput, triAlbedo(scene.mats, hit.tri));
    const hitP: Vec3 = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
    d = cosineSample(nrm, s);
    o = [hitP[0] + nrm[0] * 1e-4, hitP[1] + nrm[1] * 1e-4, hitP[2] + nrm[2] * 1e-4];
  }
  return acc;
}

/** Progressive CPU path trace into a `width × height` RGBA-float framebuffer.
 *  Reproducible: same scene + camera + settings ⇒ identical pixels (the RNG
 *  is seeded purely from pixel + sample index). Identical math to the WGSL
 *  kernel within f32 rounding. */
export function pathTraceSync(
  scene: PathScene,
  cam: Camera,
  width: number,
  height: number,
  settings: PathSettings = DEFAULT_PATH_SETTINGS,
): Float32Array {
  const fb = new Float32Array(width * height * 4);
  const spp = Math.max(1, settings.samplesPerPixel);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let col: Vec3 = [0, 0, 0];
      for (let sIdx = 0; sIdx < spp; sIdx++) {
        const rng: Rng = { v: seedHash(px, py, sIdx) };
        // Jitter inside the pixel for anti-aliasing.
        const u = (px + nextFloat(rng)) / width;
        const v = (py + nextFloat(rng)) / height;
        const dir = norm([
          cam.lowerLeft[0] + u * cam.horizontal[0] + v * cam.vertical[0] - cam.origin[0],
          cam.lowerLeft[1] + u * cam.horizontal[1] + v * cam.vertical[1] - cam.origin[1],
          cam.lowerLeft[2] + u * cam.horizontal[2] + v * cam.vertical[2] - cam.origin[2],
        ]);
        col = add(col, radiance(scene, cam.origin, dir, settings, rng));
      }
      const inv = 1 / spp;
      const o = (py * width + px) * 4;
      fb[o] = col[0] * inv;
      fb[o + 1] = col[1] * inv;
      fb[o + 2] = col[2] * inv;
      fb[o + 3] = 1;
    }
  }
  return fb;
}
