/**
 * nv-compat utsushimi render-bridge projection + scatter edge cases.
 *
 * Coverage for the synthetic-data ground-truth bridge: camera projection
 * (center, behind-camera null, off-screen clamp), AABB → 2D box, annotateFrame
 * over mixed prim kinds, prim tessellation counts, RGB render, and the
 * scatter_2d xz-plane resolve path.
 *
 *     pnpm exec vitest run test/nv-compat-render-bridge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 utsushimi.
 */

import { describe, it, expect } from "vitest";
import {
  Sampler,
  annotateFrame,
  create,
  makeProjCamera,
  primsToScene,
  projectAabb,
  projectPoint,
  randomize,
  renderFrameCPU,
  resolve,
} from "../src/utsushimi/index.js";

const cam = makeProjCamera([0, 0, 5], [0, 0, 0], [0, 1, 0], 60, 1);

describe("camera projection", () => {
  it("projects a point straight ahead to the image center", () => {
    const px = projectPoint(cam, [0, 0, 0], 256, 256);
    expect(px).not.toBeNull();
    expect(px![0]).toBeCloseTo(128, 4);
    expect(px![1]).toBeCloseTo(128, 4);
  });

  it("returns null for a point behind the camera", () => {
    expect(projectPoint(cam, [0, 0, 10], 256, 256)).toBeNull(); // +z is behind
  });

  it("projects a centered unit cube to an on-screen box", () => {
    const box = projectAabb(cam, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 256, 256);
    expect(box).not.toBeNull();
    const [x, y, w, h] = box!;
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x + w).toBeLessThanOrEqual(256);
    expect(y + h).toBeLessThanOrEqual(256);
    expect(w).toBeGreaterThan(0);
    expect(h).toBeGreaterThan(0);
  });

  it("returns null for an AABB entirely behind the camera", () => {
    expect(projectAabb(cam, [-0.5, -0.5, 9], [0.5, 0.5, 11], 256, 256)).toBeNull();
  });
});

describe("annotateFrame over mixed prim kinds", () => {
  it("annotates an on-screen semantic cube and skips off-screen / non-geometry prims", () => {
    const prims = [
      create.cube([0, 0, 0], [["class", "box"]]),
      create.cube([1000, 0, 0], [["class", "far"]]), // way off to the side
      create.camera(), // no class, no extent
    ];
    const annotated = annotateFrame(cam, prims, 128, 128);
    expect(annotated[0].bbox2d).toBeDefined();
    expect(annotated[1].bbox2d).toBeUndefined(); // off-screen → no box
    expect(annotated[2].bbox2d).toBeUndefined(); // camera prim → no extent
  });
});

describe("prim tessellation + RGB render", () => {
  it("tessellates a cube to 12 triangles and a sphere on top", () => {
    expect(primsToScene([create.cube()]).soup.count).toBe(12);
    const both = primsToScene([create.cube(), create.sphere([3, 0, 0], 1)]).soup.count;
    expect(both).toBeGreaterThan(12); // cube (12) + sphere mesh
  });

  it("renders an RGBA-float framebuffer of the prims", () => {
    const fb = renderFrameCPU([0, 0, 5], [0, 0, 0], [0, 1, 0], 50, [create.cube([0, 0, 0], [["class", "box"]])], 32, 32);
    expect(fb.length).toBe(32 * 32 * 4);
    for (let i = 0; i < fb.length; i++) expect(Number.isFinite(fb[i])).toBe(true);
  });
});

describe("scatter_2d plane variants", () => {
  it("places prims on the xz plane (y = 0)", () => {
    const res = resolve(randomize.scatter_2d([create.cube(), create.cube()], "xz", [[-2, -2], [2, 2]]), new Sampler(3));
    expect(res.kind).toBe("scatter_2d");
    if (res.kind === "scatter_2d") {
      for (const p of res.poses) {
        expect(p.position[1]).toBe(0); // xz plane → y fixed at 0
        expect(p.position[0]).toBeGreaterThanOrEqual(-2);
        expect(p.position[0]).toBeLessThanOrEqual(2);
      }
    }
  });

  it("places prims on the xy plane (z = 0) by default", () => {
    const res = resolve(randomize.scatter_2d([create.cube()], "xy", [[-1, -1], [1, 1]]), new Sampler(4));
    if (res.kind === "scatter_2d") expect(res.poses[0].position[2]).toBe(0);
  });
});
