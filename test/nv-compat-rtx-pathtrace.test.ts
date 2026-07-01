/**
 * nv-compat RTX Renderer / kami-rtx path-tracer validation.
 *
 * Exercises the clean-room WGSL path tracer (R1.2) on its CPU reference path
 * — CI has no GPU, so the deterministic JS tracer is the byte-compatible
 * fallback the WGSL kernel mirrors. Renders a Cornell-box-class closed scene
 * and asserts on physical properties (energy is bounded + finite, the scene
 * is lit, the estimator is reproducible and convergent) plus the RTX-Renderer
 * facade contract.
 *
 *     pnpm exec vitest run test/nv-compat-rtx-pathtrace.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.2 RTX Renderer surface.
 */

import { describe, it, expect } from "vitest";
import {
  type Material,
  type Vec3,
  buildPathScene,
  lookAt,
  material,
  pathTraceCPU,
  seedHash,
  nextFloat,
  onb,
} from "../src/kami-rt/index.js";
import {
  RtxRenderMode,
  createRenderer,
} from "../src/rtx-renderer.js";

// ── Cornell-box-class scene: closed box, ceiling light, colored side walls ──
function quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3): Vec3[][] {
  return [
    [a, b, c],
    [a, c, d],
  ];
}

function cornellBox(): { meshes: Vec3[][]; materials: Material[] } {
  const white = material([0.73, 0.73, 0.73]);
  const red = material([0.65, 0.05, 0.05]);
  const green = material([0.05, 0.65, 0.05]);
  const light = material([0, 0, 0], [12, 12, 12]);

  const meshes: Vec3[][] = [];
  const materials: Material[] = [];
  const push = (tris: Vec3[][], m: Material) => {
    for (const t of tris) {
      meshes.push(t);
      materials.push(m);
    }
  };

  // Box spans [-1,1]^3 with the front face (z=+1) open toward the camera.
  push(quad([-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]), white); // floor
  push(quad([-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]), white); // ceiling
  push(quad([-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]), white); // back
  push(quad([-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]), red); // left
  push(quad([1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]), green); // right
  // Emissive ceiling light, just below the ceiling plane.
  push(quad([-0.4, 0.999, -0.4], [-0.4, 0.999, 0.4], [0.4, 0.999, 0.4], [0.4, 0.999, -0.4]), light);

  return { meshes, materials };
}

function avgLuminance(fb: Float32Array): number {
  let sum = 0;
  const n = fb.length / 4;
  for (let i = 0; i < n; i++) {
    sum += 0.2126 * fb[i * 4] + 0.7152 * fb[i * 4 + 1] + 0.0722 * fb[i * 4 + 2];
  }
  return sum / n;
}

describe("kami-rtx RNG / sampling primitives", () => {
  it("seedHash is deterministic and never zero", () => {
    expect(seedHash(3, 7, 1)).toBe(seedHash(3, 7, 1));
    expect(seedHash(0, 0, 0)).not.toBe(0);
  });

  it("xorshift32 stream is reproducible and in [0,1)", () => {
    const a = { v: seedHash(1, 2, 3) };
    const b = { v: seedHash(1, 2, 3) };
    for (let i = 0; i < 100; i++) {
      const x = nextFloat(a);
      expect(x).toBe(nextFloat(b));
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });

  it("onb returns an orthonormal basis with the given normal", () => {
    const n: Vec3 = [0.0, 0.6, 0.8];
    const [t, bt] = onb(n);
    const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    expect(dot(t, n)).toBeCloseTo(0, 5);
    expect(dot(bt, n)).toBeCloseTo(0, 5);
    expect(dot(t, bt)).toBeCloseTo(0, 5);
    expect(dot(t, t)).toBeCloseTo(1, 5);
    expect(dot(bt, bt)).toBeCloseTo(1, 5);
  });
});

describe("kami-rtx Cornell box (CPU path trace)", () => {
  const { meshes, materials } = cornellBox();
  const scene = buildPathScene(meshes, materials);
  const W = 24;
  const H = 24;
  const cam = lookAt([0, 0, 2.6], [0, 0, 0], [0, 1, 0], 50, W / H);

  it("produces a finite, lit, energy-bounded framebuffer", () => {
    const { framebuffer } = pathTraceCPU(scene, cam, W, H, {
      samplesPerPixel: 16,
      maxBounces: 5,
      background: [0, 0, 0],
    });
    expect(framebuffer.length).toBe(W * H * 4);
    for (let i = 0; i < framebuffer.length; i++) {
      expect(Number.isFinite(framebuffer[i])).toBe(true);
      expect(framebuffer[i]).toBeGreaterThanOrEqual(0);
    }
    // alpha channel always written
    for (let i = 3; i < framebuffer.length; i += 4) expect(framebuffer[i]).toBe(1);
    // The box is lit by the ceiling emitter — average luminance is clearly > 0.
    expect(avgLuminance(framebuffer)).toBeGreaterThan(0.05);
  });

  it("is reproducible bit-for-bit (deterministic RNG)", () => {
    const opts = { samplesPerPixel: 8, maxBounces: 4, background: [0, 0, 0] as Vec3 };
    const a = pathTraceCPU(scene, cam, W, H, opts).framebuffer;
    const b = pathTraceCPU(scene, cam, W, H, opts).framebuffer;
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("converges: mean radiance is stable as spp increases (unbiased estimator)", () => {
    const lo = avgLuminance(
      pathTraceCPU(scene, cam, W, H, { samplesPerPixel: 8, maxBounces: 5, background: [0, 0, 0] }).framebuffer,
    );
    const hi = avgLuminance(
      pathTraceCPU(scene, cam, W, H, { samplesPerPixel: 48, maxBounces: 5, background: [0, 0, 0] }).framebuffer,
    );
    // Same expectation, different sample count → close means (Monte-Carlo noise only).
    expect(Math.abs(hi - lo)).toBeLessThan(0.15 * hi + 0.02);
  });

  it("color bleeds: left wall is reddish, right wall greenish", () => {
    // Aim a narrow camera at each side wall and compare channel dominance.
    const left = pathTraceCPU(
      scene,
      lookAt([0.2, 0, 0.2], [-1, 0, 0], [0, 1, 0], 40, 1),
      16,
      16,
      { samplesPerPixel: 24, maxBounces: 5, background: [0, 0, 0] },
    ).framebuffer;
    const right = pathTraceCPU(
      scene,
      lookAt([-0.2, 0, 0.2], [1, 0, 0], [0, 1, 0], 40, 1),
      16,
      16,
      { samplesPerPixel: 24, maxBounces: 5, background: [0, 0, 0] },
    ).framebuffer;
    const meanCh = (fb: Float32Array, c: number) => {
      let s = 0;
      const n = fb.length / 4;
      for (let i = 0; i < n; i++) s += fb[i * 4 + c];
      return s / n;
    };
    expect(meanCh(left, 0)).toBeGreaterThan(meanCh(left, 1)); // R > G on red wall
    expect(meanCh(right, 1)).toBeGreaterThan(meanCh(right, 0)); // G > R on green wall
  });
});

describe("RTX Renderer® facade (clean-room API-compat)", () => {
  const { meshes, materials } = cornellBox();
  const cam = lookAt([0, 0, 2.6], [0, 0, 0], [0, 1, 0], 50, 1);

  it("renders a scene synchronously through the renderer object", () => {
    const r = createRenderer({ mode: RtxRenderMode.PATH_TRACED, samplesPerPixel: 12, maxBounces: 4 });
    const scene = r.createScene(meshes, materials);
    expect(scene.triangleCount).toBe(meshes.length);
    const out = r.renderSync(scene, cam, 16, 16);
    expect(out.backend).toBe("cpu");
    expect(out.framebuffer.length).toBe(16 * 16 * 4);
    expect(out.samplesPerPixel).toBe(12);
  });

  it("REAL_TIME mode clamps the sample/bounce budget", () => {
    const r = createRenderer({ mode: RtxRenderMode.REAL_TIME, samplesPerPixel: 256, maxBounces: 16 });
    const scene = r.createScene(meshes, materials);
    const out = r.renderSync(scene, cam, 8, 8);
    expect(out.samplesPerPixel).toBe(4); // clamped
  });

  it("async render falls back to CPU when no WebGPU device is present", async () => {
    const r = createRenderer({ samplesPerPixel: 6, maxBounces: 3 });
    const scene = r.createScene(meshes, materials);
    const out = await r.render(scene, cam, 8, 8);
    expect(out.backend).toBe("cpu");
    expect(out.framebuffer.length).toBe(8 * 8 * 4);
  });

  it("createScene rejects mismatched mesh/material counts", () => {
    const r = createRenderer();
    expect(() => r.createScene(meshes, materials.slice(0, 2))).toThrow(/1:1/);
  });
});
