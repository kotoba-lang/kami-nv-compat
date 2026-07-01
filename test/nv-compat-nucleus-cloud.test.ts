/**
 * nv-compat omni.client (Nucleus) + Omniverse Cloud validation.
 *
 * Exercises the clean-room content-addressed versioned store + omni.client API
 * (kotoba-datomic-nucleus) and the Murakumo render farm (murakumo-render),
 * including the end-to-end path: write a USDA layer to Nucleus → read it back →
 * submit a turntable render job to the cloud → retrieve frames.
 *
 *     pnpm exec vitest run test/nv-compat-nucleus-cloud.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.7 + R1.9 surfaces.
 */

import { describe, it, expect } from "vitest";
import {
  Client,
  NucleusStore,
  Result,
  cidOf,
  parseUrl,
  readSceneFromNucleus,
} from "../src/omni-nucleus.js";
import { OmniCloudSession, RenderFarm, turntableCameras } from "../src/omni-cloud.js";
import { lookAt, buildScene } from "../src/kami-rt/index.js";

const TRI: [number, number, number][][] = [[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]];

describe("content-addressed store", () => {
  it("hashes content deterministically; identical content → no new version", () => {
    expect(cidOf("hello")).toBe(cidOf("hello"));
    expect(cidOf("hello")).not.toBe(cidOf("world"));
    const store = new NucleusStore();
    const v1 = store.write("/a", "x");
    const v2 = store.write("/a", "x"); // unchanged
    expect(v2).toBe(v1);
    expect(store.history("/a")).toHaveLength(1);
    const v3 = store.write("/a", "y");
    expect(v3.index).toBe(1);
    expect(store.history("/a")).toHaveLength(2);
  });

  it("restore appends a prior version as the new head", () => {
    const store = new NucleusStore();
    store.write("/f", "v0");
    store.write("/f", "v1");
    const restored = store.restore("/f", 0);
    expect(restored?.content).toBe("v0");
    expect(store.read("/f")).toBe("v0");
    expect(store.history("/f")).toHaveLength(3); // append-only
  });

  it("notifies subscribers on write and delete (exact + prefix)", () => {
    const store = new NucleusStore();
    const events: string[] = [];
    const unsubExact = store.subscribe("/scenes/a", (e) => events.push(`exact:${e.kind}`));
    const unsubPrefix = store.subscribe("/scenes/", (e) => events.push(`prefix:${e.kind}`));
    store.write("/scenes/a", "1");
    store.write("/scenes/a", "2");
    store.delete("/scenes/a");
    expect(events).toContain("exact:created");
    expect(events).toContain("exact:modified");
    expect(events).toContain("exact:deleted");
    expect(events.filter((e) => e.startsWith("prefix:")).length).toBe(3);
    const before = events.length;
    unsubExact();
    unsubPrefix();
    store.write("/scenes/a", "3");
    expect(events.length).toBe(before); // no new events after unsubscribe
  });
});

describe("omni.client URL handling + API", () => {
  it("parses omniverse:// URLs", () => {
    expect(parseUrl("omniverse://kami/scenes/box.usda")).toEqual({ server: "kami", path: "/scenes/box.usda" });
    expect(parseUrl("/local/path")).toEqual({ server: "", path: "/local/path" });
    expect(parseUrl("http://nope")).toBeNull();
  });

  it("write / stat / read / list / copy / delete round-trip", () => {
    const c = new Client();
    expect(c.stat("omniverse://kami/a.usda").result).toBe(Result.ERROR_NOT_FOUND);
    c.writeFile("omniverse://kami/a.usda", "hello");
    const stat = c.stat("omniverse://kami/a.usda");
    expect(stat.result).toBe(Result.OK);
    expect(stat.info!.size).toBe(5);
    expect(c.readFile("omniverse://kami/a.usda").content).toBe("hello");
    c.copy("omniverse://kami/a.usda", "omniverse://kami/b.usda");
    expect(c.readFile("omniverse://kami/b.usda").content).toBe("hello");
    const list = c.list("omniverse://kami/");
    expect(list.entries.map((e) => e.relativePath).sort()).toEqual(["a.usda", "b.usda"]);
    expect(c.delete("omniverse://kami/a.usda").result).toBe(Result.OK);
    expect(c.readFile("omniverse://kami/a.usda").result).toBe(Result.ERROR_NOT_FOUND);
    expect(c.writeFile("http://bad", "x").result).toBe(Result.ERROR_INVALID_URL);
  });

  it("checkpoints expose the append-only version history", () => {
    const c = new Client();
    const url = "omniverse://kami/scene.usda";
    c.writeFile(url, "rev0");
    c.createCheckpoint(url, "first");
    c.writeFile(url, "rev1");
    const cps = c.getCheckpoints(url);
    expect(cps.result).toBe(Result.OK);
    expect(cps.checkpoints.some((v) => v.message === "first")).toBe(true);
    expect(cps.checkpoints.length).toBeGreaterThanOrEqual(2);
  });
});

describe("Murakumo render farm (Omniverse Cloud)", () => {
  it("submits, runs, and returns one frame per camera (rtx)", () => {
    const farm = new RenderFarm();
    const scene = buildScene(TRI);
    const cams = [lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1), lookAt([3, 0, 3], [0, 0, 0], [0, 1, 0], 45, 1)];
    const id = farm.submit({ scene, cameras: cams, width: 16, height: 16, mode: "rtx" });
    expect(farm.status(id)).toBe("queued");
    const streamed: number[] = [];
    farm.run(id, (_f, i) => streamed.push(i));
    expect(farm.status(id)).toBe("done");
    const frames = farm.result(id)!;
    expect(frames).toHaveLength(2);
    expect(frames[0].length).toBe(16 * 16 * 4);
    expect(streamed).toEqual([0, 1]); // streamed both frames
  });

  it("flags an error when the scene/mode mismatch", () => {
    const farm = new RenderFarm();
    const id = farm.submit({ scene: buildScene(TRI), cameras: [lookAt([0, 0, 4], [0, 0, 0], [0, 1, 0], 45, 1)], width: 8, height: 8, mode: "pathtrace" });
    const job = farm.run(id);
    expect(job.status).toBe("error");
    expect(job.error).toMatch(/PathScene/);
  });

  it("turntableCameras generates an orbit of the requested size", () => {
    const cams = turntableCameras((eye, target) => lookAt(eye, target, [0, 1, 0], 45, 1), [0, 0, 0], 5, 2, 8);
    expect(cams).toHaveLength(8);
  });
});

describe("end-to-end: Nucleus USD → Cloud render", () => {
  it("stores a USDA layer in Nucleus, reads the scene, and renders it on the farm", () => {
    const usda = `#usda 1.0
def Xform "World" {
    def Mesh "tri" {
        point3f[] points = [(-1,-1,0),(1,-1,0),(0,1,0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
    }
}`;
    const client = new Client();
    client.writeFile("omniverse://kami/scenes/tri.usda", usda);

    const scene = readSceneFromNucleus(client, "omniverse://kami/scenes/tri.usda");
    expect(scene).not.toBeNull();
    expect(scene!.soup.count).toBe(1);

    const session = new OmniCloudSession();
    const jobId = session.submitRender({
      scene: scene!,
      cameras: turntableCameras((eye, t) => lookAt(eye, t, [0, 1, 0], 45, 1), [0, 0, 0], 4, 1, 3),
      width: 16,
      height: 16,
      mode: "rtx",
    });
    session.process();
    expect(session.getStatus(jobId)).toBe("done");
    expect(session.getResult(jobId)).toHaveLength(3);
  });
});
