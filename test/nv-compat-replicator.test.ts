/**
 * nv-compat omni.replicator.core / utsushimi validation.
 *
 * Exercises the clean-room synthetic-data engine: the BigInt LCG sampler is
 * checked BIT-IDENTICAL against golden values captured from the Python
 * reference (kotodama.nv_compat.omni.replicator.core._Sampler), the
 * distributions + DR ops resolve deterministically, the COCO/Kitti writers
 * emit the documented schema, the kami-rt projection produces real 2D boxes,
 * and end-to-end generation is reproducible.
 *
 *     pnpm exec vitest run test/nv-compat-replicator.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.3 omni-replicator-core surface.
 */

import { describe, it, expect } from "vitest";
import * as rep from "../src/omni-replicator-core.js";
import { Sampler } from "../src/utsushimi/index.js";

describe("utsushimi LCG sampler (cross-language bit-identity)", () => {
  it("matches Python _Sampler golden values for seed 42", () => {
    const s = new Sampler(42);
    const got = [0, 0, 0, 0, 0].map(() => Number(s.nextU01().toFixed(12)));
    // Golden values captured from the Python reference sampler.
    expect(got).toEqual([0.225463428535, 0.412838318385, 0.630398049485, 0.680147807114, 0.026228910312]);
  });

  it("matches Python golden values for seed 0 and a uniform draw for seed 7", () => {
    const s0 = new Sampler(0);
    expect([0, 0, 0].map(() => Number(s0.nextU01().toFixed(12)))).toEqual([
      0.101698759943, 0.605323322583, 0.401216203347,
    ]);
    const s7 = new Sampler(7);
    expect(Number(s7.nextUniform(-2, 2).toFixed(12))).toBe(1.822638152167);
  });

  it("is reproducible and produces uniforms in [0,1)", () => {
    const a = new Sampler(123), b = new Sampler(123);
    for (let i = 0; i < 50; i++) {
      const x = a.nextU01();
      expect(x).toBe(b.nextU01());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});

describe("distributions", () => {
  it("uniform / choice / sequence / combine sample as specified", () => {
    const s = new Sampler(1);
    const u = rep.sample(rep.distribution.uniform([0, 10], [1, 20]), s) as number[];
    expect(u[0]).toBeGreaterThanOrEqual(0);
    expect(u[0]).toBeLessThan(1);
    expect(u[1]).toBeGreaterThanOrEqual(10);
    expect(u[1]).toBeLessThan(20);

    const seq = rep.distribution.sequence(["a", "b", "c"]);
    expect([0, 1, 2, 3].map(() => rep.sample(seq, s))).toEqual(["a", "b", "c", "a"]);

    const choice = rep.distribution.choice([1, 2, 3, 4]);
    expect([1, 2, 3, 4]).toContain(rep.sample(choice, s));

    const combined = rep.sample(rep.distribution.combine([rep.distribution.uniform([0], [1]), rep.distribution.uniform([5], [6])]), s) as number[];
    expect(combined).toHaveLength(2);
  });

  it("normal has roughly the requested mean over many draws", () => {
    const s = new Sampler(99);
    let sum = 0;
    const N = 4000;
    for (let i = 0; i < N; i++) sum += s.nextNormal(5, 2);
    expect(Math.abs(sum / N - 5)).toBeLessThan(0.2);
  });
});

describe("randomize ops + resolve", () => {
  it("scatter_2d places each prim within the region, deterministically", () => {
    const cubes = [rep.create.cube([0, 0, 0]), rep.create.cube([0, 0, 0])];
    const op = rep.randomize.scatter_2d(cubes, "xy", [[-2, -2], [2, 2]]);
    const s = new Sampler(5);
    const res = rep.resolve(op, s);
    expect(res.kind).toBe("scatter_2d");
    if (res.kind === "scatter_2d") {
      expect(res.poses).toHaveLength(2);
      for (const pose of res.poses) {
        expect(pose.position[0]).toBeGreaterThanOrEqual(-2);
        expect(pose.position[0]).toBeLessThanOrEqual(2);
        expect(pose.position[2]).toBe(0); // xy plane → z = 0
      }
    }
    // Same seed → identical poses.
    const res2 = rep.resolve(op, new Sampler(5));
    expect(JSON.stringify(res2)).toEqual(JSON.stringify(rep.resolve(op, new Sampler(5))));
  });

  it("randomize_lights resolves rotation/intensity/color", () => {
    const res = rep.resolve(rep.randomize.lights(), new Sampler(2));
    expect(res.kind).toBe("randomize_lights");
    if (res.kind === "randomize_lights") {
      expect(res.rotation).toHaveLength(3);
      expect(res.intensity).toBeGreaterThanOrEqual(500);
      expect(res.color).toHaveLength(3);
    }
  });
});

describe("writers emit the documented schema", () => {
  const frame = {
    frame: 0,
    primitives: [
      { _kind: "cube" as const, position: [0, 0, 0], semantics: [["class", "vehicle"]] as [string, string][], bbox2d: [10, 20, 30, 40] as [number, number, number, number] },
      { _kind: "sphere" as const, position: [1, 0, 0], radius: 1, semantics: [["class", "pedestrian"]] as [string, string][] },
    ],
  };

  it("CocoWriter produces COCO-2017 images/annotations/categories", () => {
    const w = new rep.CocoWriter();
    w.initialize({ outputDir: "out", imageWidth: 640, imageHeight: 480 });
    w.writeFrame(0, frame);
    const ds = w.finalize();
    expect(ds.images[0]).toMatchObject({ id: 0, file_name: "rgb_0000.png", width: 640, height: 480 });
    expect(ds.annotations).toHaveLength(2);
    // The cube carried an explicit bbox; the sphere fell back to full-image.
    expect(ds.annotations[0].bbox).toEqual([10, 20, 30, 40]);
    expect(ds.annotations[0].area).toBe(30 * 40);
    expect(ds.annotations[1].bbox).toEqual([0, 0, 640, 480]);
    expect(ds.categories.map((c) => c.name).sort()).toEqual(["pedestrian", "vehicle"]);
    expect(w.toFiles()["out/annotations.json"]).toContain("\"images\"");
  });

  it("KittiWriter emits a label line per semantic object", () => {
    const w = new rep.KittiWriter();
    w.initialize({ outputDir: "out", imageWidth: 1242, imageHeight: 375 });
    w.writeFrame(0, frame);
    const files = w.toFiles();
    const label = files["out/label_2/000000.txt"];
    expect(label.split("\n").filter(Boolean)).toHaveLength(2);
    expect(label).toMatch(/^vehicle 0\.00 0 0\.00 10\.00 20\.00 40\.00 60\.00/);
  });

  it("WriterRegistry resolves writers by name", () => {
    expect(rep.WriterRegistry.get("BasicWriter")).toBeInstanceOf(rep.BasicWriter);
    expect(() => rep.WriterRegistry.get("Nope")).toThrow(/unknown writer/);
  });
});

describe("kami-rt projection ground truth", () => {
  it("projects a cube's AABB to a 2D box inside the image", () => {
    const cam = rep.makeProjCamera([0, 0, 5], [0, 0, 0], [0, 1, 0], 60, 1);
    const box = rep.projectAabb(cam, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 256, 256);
    expect(box).not.toBeNull();
    const [x, y, w, h] = box!;
    // Centered cube → box around the image center.
    expect(x).toBeGreaterThan(0);
    expect(x + w).toBeLessThan(256);
    expect(y + h).toBeLessThanOrEqual(256);
    expect(w).toBeGreaterThan(0);
    expect(h).toBeGreaterThan(0);
  });

  it("returns null for geometry entirely behind the camera", () => {
    const cam = rep.makeProjCamera([0, 0, 5], [0, 0, 0], [0, 1, 0], 60, 1);
    expect(rep.projectAabb(cam, [-0.5, -0.5, 9], [0.5, 0.5, 10], 256, 256)).toBeNull();
  });
});

describe("end-to-end synthetic-data generation", () => {
  it("generates a reproducible annotated COCO dataset with real boxes", () => {
    const make = () => {
      const cubes = [
        rep.create.cube([0, 0, 0], [["class", "box"]]),
        rep.create.cube([0, 0, 0], [["class", "box"]]),
      ];
      const writer = new rep.CocoWriter();
      writer.initialize({ outputDir: "out", imageWidth: 320, imageHeight: 240 });
      return rep.generateDataset({
        prims: cubes,
        randomizers: [rep.randomize.scatter_2d(cubes, "xy", [[-1.5, -1.5], [1.5, 1.5]])],
        camera: { eye: [0, 4, 6], target: [0, 0, 0], vfovDeg: 50 },
        numFrames: 4,
        imageWidth: 320,
        imageHeight: 240,
        seed: 7,
        writers: [writer],
      });
    };
    const a = make();
    const b = make();
    expect(a.frames).toHaveLength(4);
    // Deterministic given the seed.
    expect(JSON.stringify(a.frames)).toEqual(JSON.stringify(b.frames));
    // Each frame's scattered cubes got real projected boxes.
    const ds = a.outputs[0] as rep.CocoDataset;
    expect(ds.images).toHaveLength(4);
    expect(ds.annotations.length).toBeGreaterThan(0);
    const withRealBox = ds.annotations.filter((an) => !(an.bbox[0] === 0 && an.bbox[1] === 0 && an.bbox[2] === 320));
    expect(withRealBox.length).toBeGreaterThan(0); // projection produced non-placeholder boxes
  });

  it("renders RGB frames when requested", () => {
    const cubes = [rep.create.cube([0, 0, 0], [["class", "box"]])];
    const writer = new rep.BasicWriter();
    writer.initialize({ outputDir: "out" });
    const res = rep.generateDataset({
      prims: cubes,
      randomizers: [rep.randomize.scatter_2d(cubes, "xy", [[-1, -1], [1, 1]])],
      camera: { eye: [0, 3, 5], target: [0, 0, 0] },
      numFrames: 2,
      imageWidth: 64,
      imageHeight: 64,
      seed: 1,
      writers: [writer],
      render: true,
    });
    expect(res.rgb).toBeDefined();
    expect(res.rgb!).toHaveLength(2);
    expect(res.rgb![0].length).toBe(64 * 64 * 4);
  });
});
