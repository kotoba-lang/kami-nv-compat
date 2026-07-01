/**
 * nv-compat murakumo-render farm + OmniCloudSession edge cases.
 *
 * Branch coverage for the cloud render farm: queue lifecycle (pending / status
 * transitions / result-before-done), runAll FIFO, unknown-job throw, pathtrace
 * mode over a PathScene, progress, streaming, and the session facade.
 *
 *     pnpm exec vitest run test/nv-compat-murakumo-render-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 murakumo-render.
 */

import { describe, it, expect } from "vitest";
import { OmniCloudSession, RenderFarm, turntableCameras } from "../src/omni-cloud.js";
import { type Vec3, buildScene, buildPathScene, lookAt, material } from "../src/kami-rt/index.js";

const TRI: Vec3[] = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]];
const cam = lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1);

describe("RenderFarm queue lifecycle", () => {
  it("tracks pending jobs and transitions queued → done", () => {
    const farm = new RenderFarm();
    const id = farm.submit({ scene: buildScene(TRI), cameras: [cam], width: 8, height: 8, mode: "rtx" });
    expect(farm.pending()).toEqual([id]);
    expect(farm.status(id)).toBe("queued");
    expect(farm.result(id)).toBeUndefined(); // not done yet
    farm.run(id);
    expect(farm.status(id)).toBe("done");
    expect(farm.pending()).toEqual([]);
    expect(farm.result(id)).toHaveLength(1);
    expect(farm.get(id)!.progress).toBe(1);
  });

  it("runAll renders every queued job (FIFO)", () => {
    const farm = new RenderFarm();
    const scene = buildScene(TRI);
    const a = farm.submit({ scene, cameras: [cam], width: 4, height: 4, mode: "rtx" });
    const b = farm.submit({ scene, cameras: [cam, cam], width: 4, height: 4, mode: "rtx" });
    const jobs = farm.runAll();
    expect(jobs.map((j) => j.id)).toEqual([a, b]);
    expect(farm.result(a)).toHaveLength(1);
    expect(farm.result(b)).toHaveLength(2);
  });

  it("throws when running an unknown job id", () => {
    expect(() => new RenderFarm().run("job-999")).toThrow(/unknown job/);
  });

  it("renders a PathScene in pathtrace mode", () => {
    const farm = new RenderFarm();
    const pscene = buildPathScene([TRI], [material([0.8, 0.2, 0.2], [1, 1, 1])]);
    const id = farm.submit({
      scene: pscene,
      cameras: [cam],
      width: 8,
      height: 8,
      mode: "pathtrace",
      pathSettings: { samplesPerPixel: 2, maxBounces: 2, background: [0, 0, 0] },
    });
    farm.run(id);
    expect(farm.status(id)).toBe("done");
    expect(farm.result(id)![0].length).toBe(8 * 8 * 4);
  });

  it("flags an error (and no result) on a scene/mode mismatch", () => {
    const farm = new RenderFarm();
    const id = farm.submit({ scene: buildScene(TRI), cameras: [cam], width: 4, height: 4, mode: "pathtrace" });
    const job = farm.run(id);
    expect(job.status).toBe("error");
    expect(job.error).toMatch(/PathScene/);
    expect(farm.result(id)).toBeUndefined(); // result() only returns for done jobs
  });

  it("streams each frame via the onFrame callback", () => {
    const farm = new RenderFarm();
    const id = farm.submit({ scene: buildScene(TRI), cameras: [cam, cam, cam], width: 4, height: 4, mode: "rtx" });
    const indices: number[] = [];
    farm.run(id, (_f, i) => indices.push(i));
    expect(indices).toEqual([0, 1, 2]);
  });
});

describe("turntableCameras", () => {
  it("generates an orbit of the requested size at the given radius/height", () => {
    const cams = turntableCameras((eye, target) => lookAt(eye, target, [0, 1, 0], 45, 1), [0, 0, 0], 5, 2, 6);
    expect(cams).toHaveLength(6);
    // First camera sits on +x at the requested radius/height.
    expect(cams[0].origin[1]).toBeCloseTo(2, 6); // height
    expect(Math.hypot(cams[0].origin[0], cams[0].origin[2])).toBeCloseTo(5, 6); // radius
  });
});

describe("OmniCloudSession facade", () => {
  it("submits, processes, and retrieves results", () => {
    const s = new OmniCloudSession();
    const id = s.submitRender({ scene: buildScene(TRI), cameras: [cam, cam], width: 6, height: 6, mode: "rtx" });
    expect(s.getStatus(id)).toBe("queued");
    expect(s.getResult(id)).toBeUndefined();
    s.process();
    expect(s.getStatus(id)).toBe("done");
    expect(s.getResult(id)).toHaveLength(2);
    expect(s.getJob(id)!.progress).toBe(1);
  });

  it("renders a single job with streaming", () => {
    const s = new OmniCloudSession();
    const id = s.submitRender({ scene: buildScene(TRI), cameras: [cam], width: 4, height: 4, mode: "rtx" });
    const seen: number[] = [];
    const job = s.render(id, (_f, i) => seen.push(i));
    expect(job.status).toBe("done");
    expect(seen).toEqual([0]);
  });
});
