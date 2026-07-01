/**
 * nv-compat utsushimi DR primitives + writer output edge cases.
 *
 * Boundary coverage for the synthetic-data engine: the truncated-normal
 * rejection fallback, combine() over mixed scalar/array distributions, the
 * remaining resolve() ops (physics / scatter_3d), the on-disk schema emitted by
 * BasicWriter / KittiWriter `toFiles()`, and WriterRegistry registration.
 *
 *     pnpm exec vitest run test/nv-compat-utsushimi-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 utsushimi.
 */

import { describe, it, expect } from "vitest";
import * as rep from "../src/omni-replicator-core.js";
import { type FrameSample, type Writer, Sampler } from "../src/utsushimi/index.js";

describe("Sampler edge cases", () => {
  it("truncated normal clamps to [low,high] when the bounds are unreachable", () => {
    const s = new Sampler(3);
    // mean 0, tiny std, bounds [5,6] → no sample lands in range → clamp to
    // max(low, min(high, mean)) = 5.
    for (let i = 0; i < 20; i++) {
      const v = s.nextTruncatedNormal(0, 0.001, 5, 6);
      expect(v).toBe(5);
    }
  });

  it("truncated normal returns an in-range sample when the bounds are wide", () => {
    const s = new Sampler(4);
    for (let i = 0; i < 50; i++) {
      const v = s.nextTruncatedNormal(0, 1, -5, 5);
      expect(v).toBeGreaterThanOrEqual(-5);
      expect(v).toBeLessThanOrEqual(5);
    }
  });
});

describe("distribution.combine over mixed scalar/array", () => {
  it("flattens a scalar choice + an array uniform into one vector", () => {
    const s = new Sampler(11);
    const combined = rep.sample(
      rep.distribution.combine([rep.distribution.choice([7]), rep.distribution.uniform([0, 0], [1, 1])]),
      s,
    ) as number[];
    expect(combined).toHaveLength(3); // 1 scalar + 2-vector
    expect(combined[0]).toBe(7);
  });
});

describe("resolve() remaining ops", () => {
  it("randomize_physics yields mass + friction within range", () => {
    const res = rep.resolve(rep.randomize.physics_properties(rep.create.cube()), new Sampler(5));
    expect(res.kind).toBe("randomize_physics");
    if (res.kind === "randomize_physics") {
      expect(res.mass).toBeGreaterThanOrEqual(0.5);
      expect(res.mass).toBeLessThanOrEqual(2);
      expect(res.friction).toBeGreaterThanOrEqual(0.3);
      expect(res.friction).toBeLessThanOrEqual(0.9);
    }
  });

  it("scatter_3d places prims inside the volume with a 3-vector rotation", () => {
    const prims = [rep.create.cube(), rep.create.cube()];
    const res = rep.resolve(rep.randomize.scatter_3d(prims, [[-1, -1, 0], [1, 1, 2]]), new Sampler(8));
    expect(res.kind).toBe("scatter_3d");
    if (res.kind === "scatter_3d") {
      expect(res.poses).toHaveLength(2);
      for (const p of res.poses) {
        expect(p.position[0]).toBeGreaterThanOrEqual(-1);
        expect(p.position[0]).toBeLessThanOrEqual(1);
        expect(p.position[2]).toBeGreaterThanOrEqual(0);
        expect(p.position[2]).toBeLessThanOrEqual(2);
        expect(p.rotation).toHaveLength(3);
      }
    }
  });
});

describe("writer toFiles() on-disk schema", () => {
  const frame: FrameSample = {
    frame: 0,
    primitives: [
      { _kind: "cube", position: [0, 0, 0], semantics: [["class", "vehicle"]], bbox2d: [10, 20, 30, 40] },
    ],
  };

  it("BasicWriter emits frame_{0000}.json with frame/cameras/sample", () => {
    const w = new rep.BasicWriter();
    w.initialize({ outputDir: "out" });
    w.attach(["cam0"]);
    w.writeFrame(0, frame);
    const files = w.toFiles();
    const path = "out/frame_0000.json";
    expect(files[path]).toBeDefined();
    const parsed = JSON.parse(files[path]);
    expect(parsed.frame).toBe(0);
    expect(parsed.cameras).toEqual(["cam0"]);
    expect(parsed.sample.primitives).toHaveLength(1);
  });

  it("KittiWriter emits label_2 + image_2 files with a per-object label line", () => {
    const w = new rep.KittiWriter();
    w.initialize({ outputDir: "out", imageWidth: 1242, imageHeight: 375 });
    w.writeFrame(0, frame);
    const files = w.toFiles();
    expect(files["out/label_2/000000.txt"]).toMatch(/^vehicle /);
    expect(files["out/image_2/000000.json"]).toContain("placeholder");
    // bbox2d [10,20,30,40] → right=40, bottom=60.
    expect(files["out/label_2/000000.txt"]).toContain("10.00 20.00 40.00 60.00");
  });
});

describe("WriterRegistry registration", () => {
  it("registers + retrieves a custom writer", () => {
    class NullWriter implements Writer {
      calls = 0;
      initialize(): void {}
      attach(): void {}
      writeFrame(): void {
        this.calls++;
      }
      finalize(): { ok: boolean } {
        return { ok: true };
      }
      toFiles(): Record<string, string> {
        return {};
      }
    }
    rep.WriterRegistry.register("NullWriter", NullWriter);
    const w = rep.WriterRegistry.get("NullWriter");
    expect(w).toBeInstanceOf(NullWriter);
    expect((w.finalize() as { ok: boolean }).ok).toBe(true);
  });

  it("still throws on an unknown writer name", () => {
    expect(() => rep.WriterRegistry.get("Nope")).toThrow(/unknown writer/);
  });
});
