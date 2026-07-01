/**
 * nv-compat kami-rt BVH + ray/path-trace edge cases.
 *
 * Degenerate-scene + boundary coverage for the renderer core: an empty scene
 * (regression guard for the BVH empty-soup fix), single-triangle leaves,
 * parallel / behind / out-of-range rays, the background gradient on a miss,
 * geometric normals, and the path tracer's spp clamp + zero-bounce emission.
 *
 *     pnpm exec vitest run test/nv-compat-kami-rt-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 kami-rt.
 */

import { describe, it, expect } from "vitest";
import {
  type Vec3,
  buildBvh,
  buildScene,
  buildPathScene,
  lookAt,
  material,
  pathTraceSync,
  traceClosest,
  traceImageCPU,
  triangleSoup,
  triNormal,
} from "../src/kami-rt/index.js";

const TRI: Vec3[] = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]];

describe("empty / degenerate scenes", () => {
  it("buildBvh on an empty soup yields an empty BVH (no crash)", () => {
    const bvh = buildBvh(triangleSoup([]));
    expect(bvh.nodeCount).toBe(0);
    expect(bvh.nodes.length).toBe(0);
  });

  it("traceClosest on an empty scene returns null", () => {
    const scene = buildScene([]);
    expect(traceClosest(scene.soup, scene.bvh, [0, 0, 5], [0, 0, -1])).toBeNull();
  });

  it("rendering an empty scene fills the background gradient (alpha = 1)", () => {
    const scene = buildScene([]);
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1);
    const { framebuffer } = traceImageCPU(scene, cam, 8, 8);
    for (let i = 0; i < framebuffer.length; i++) expect(Number.isFinite(framebuffer[i])).toBe(true);
    for (let i = 3; i < framebuffer.length; i += 4) expect(framebuffer[i]).toBe(1);
    // No geometry → every pixel is the sky gradient (all channels > 0).
    expect(framebuffer[0]).toBeGreaterThan(0);
  });

  it("a single triangle builds a one-node leaf BVH", () => {
    const bvh = buildBvh(triangleSoup([TRI]));
    expect(bvh.nodeCount).toBe(1);
  });
});

describe("ray intersection boundaries", () => {
  const scene = buildScene([TRI]);

  it("misses a ray coplanar with the triangle (parallel, det ≈ 0)", () => {
    // Ray travelling along +x within the z=0 plane is parallel to the tri.
    expect(traceClosest(scene.soup, scene.bvh, [-5, 0, 0], [1, 0, 0])).toBeNull();
  });

  it("misses a ray pointing away from the triangle", () => {
    expect(traceClosest(scene.soup, scene.bvh, [0, 0, 5], [0, 0, 1])).toBeNull();
  });

  it("misses geometry behind the ray origin", () => {
    expect(traceClosest(scene.soup, scene.bvh, [0, 0, -5], [0, 0, -1])).toBeNull();
  });

  it("respects tMax (a hit beyond tMax is rejected)", () => {
    expect(traceClosest(scene.soup, scene.bvh, [0, 0, 5], [0, 0, -1])).not.toBeNull(); // t = 5
    expect(traceClosest(scene.soup, scene.bvh, [0, 0, 5], [0, 0, -1], 3)).toBeNull(); // tMax < 5
  });

  it("geometric normal of a z=0 triangle points along ±z", () => {
    const n = triNormal(scene.soup, 0);
    expect(Math.abs(n[2])).toBeCloseTo(1, 6);
    expect(n[0]).toBeCloseTo(0, 6);
    expect(n[1]).toBeCloseTo(0, 6);
  });
});

describe("path tracer edge cases", () => {
  it("clamps samplesPerPixel to ≥ 1 and stays finite", () => {
    const scene = buildPathScene([TRI], [material([0.8, 0.8, 0.8])]);
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1);
    const fb = pathTraceSync(scene, cam, 8, 8, { samplesPerPixel: 0, maxBounces: 2, background: [0, 0, 0] });
    expect(fb.length).toBe(8 * 8 * 4);
    for (const v of fb) expect(Number.isFinite(v)).toBe(true);
  });

  it("zero bounces returns direct emission only (lit emitter, black background)", () => {
    // A bright emissive triangle facing the camera; background pure black.
    const scene = buildPathScene([TRI], [material([0, 0, 0], [5, 5, 5])]);
    const cam = lookAt([0, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1);
    const fb = pathTraceSync(scene, cam, 16, 16, { samplesPerPixel: 1, maxBounces: 0, background: [0, 0, 0] });
    // Center pixel hits the emitter → bright; a corner misses → black.
    const center = ((16 / 2) * 16 + 16 / 2) * 4;
    expect(fb[center]).toBeGreaterThan(1);
    expect(fb[0]).toBe(0); // background contributes nothing
  });
});
