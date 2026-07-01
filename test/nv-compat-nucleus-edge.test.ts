/**
 * nv-compat kotoba-datomic-nucleus store/client edge + error paths.
 *
 * Boundary coverage for the content-addressed versioned store and the
 * omni.client wrapper: read-by-CID across history, invalid restore, copy from a
 * missing source, head/exists, prefix listing order, and the client-level
 * restore / checkpoint-on-missing / invalid-URL paths.
 *
 *     pnpm exec vitest run test/nv-compat-nucleus-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 kotoba-datomic-nucleus.
 */

import { describe, it, expect } from "vitest";
import { Client, NucleusStore, Result } from "../src/omni-nucleus.js";

describe("NucleusStore version history", () => {
  it("reads a specific historical version by CID", () => {
    const s = new NucleusStore();
    const v0 = s.write("/f", "alpha");
    s.write("/f", "beta");
    expect(s.read("/f")).toBe("beta"); // head
    expect(s.readByCid("/f", v0.cid)).toBe("alpha"); // historical
    expect(s.readByCid("/f", "sha2-256:deadbeef")).toBeNull(); // unknown cid
  });

  it("head / exists reflect presence and deletion", () => {
    const s = new NucleusStore();
    expect(s.exists("/x")).toBe(false);
    expect(s.head("/x")).toBeNull();
    s.write("/x", "v");
    expect(s.exists("/x")).toBe(true);
    expect(s.head("/x")!.index).toBe(0);
    s.delete("/x");
    expect(s.exists("/x")).toBe(false);
  });

  it("restore with an out-of-range index returns null and adds no version", () => {
    const s = new NucleusStore();
    s.write("/f", "v0");
    expect(s.restore("/f", 5)).toBeNull();
    expect(s.restore("/f", -1)).toBeNull();
    expect(s.history("/f")).toHaveLength(1);
  });

  it("copy from a missing source returns null", () => {
    const s = new NucleusStore();
    expect(s.copy("/missing", "/dst")).toBeNull();
    expect(s.exists("/dst")).toBe(false);
  });

  it("list returns matching paths in sorted order", () => {
    const s = new NucleusStore();
    s.write("/scenes/b", "1");
    s.write("/scenes/a", "1");
    s.write("/other/c", "1");
    expect(s.list("/scenes/")).toEqual(["/scenes/a", "/scenes/b"]);
    expect(s.list()).toEqual(["/other/c", "/scenes/a", "/scenes/b"]);
  });

  it("a deleted path drops out of listings and reads as null", () => {
    const s = new NucleusStore();
    s.write("/a", "1");
    expect(s.delete("/a")).toBe(true);
    expect(s.delete("/a")).toBe(false); // already gone
    expect(s.read("/a")).toBeNull();
    expect(s.list()).toEqual([]);
  });
});

describe("Client error + restore paths", () => {
  it("restores a prior version through the client", () => {
    const c = new Client();
    const url = "omniverse://kami/f.usda";
    c.writeFile(url, "v0");
    c.writeFile(url, "v1");
    const r = c.restore(url, 0);
    expect(r.result).toBe(Result.OK);
    expect(c.readFile(url).content).toBe("v0");
  });

  it("restore / getCheckpoints on a missing file report NOT_FOUND", () => {
    const c = new Client();
    expect(c.restore("omniverse://kami/missing", 0).result).toBe(Result.ERROR_NOT_FOUND);
    expect(c.getCheckpoints("omniverse://kami/missing").result).toBe(Result.ERROR_NOT_FOUND);
    expect(c.createCheckpoint("omniverse://kami/missing", "x").result).toBe(Result.ERROR_NOT_FOUND);
  });

  it("rejects invalid URLs across the API", () => {
    const c = new Client();
    expect(c.stat("ftp://x").result).toBe(Result.ERROR_INVALID_URL);
    expect(c.readFile("ftp://x").result).toBe(Result.ERROR_INVALID_URL);
    expect(c.copy("ftp://x", "omniverse://k/y").result).toBe(Result.ERROR_INVALID_URL);
    expect(c.delete("ftp://x").result).toBe(Result.ERROR_INVALID_URL);
    expect(c.list("ftp://x").result).toBe(Result.ERROR_INVALID_URL);
  });

  it("a bare /path URL (no scheme) is accepted on the default mount", () => {
    const c = new Client();
    expect(c.writeFile("/local/a.usda", "hi").result).toBe(Result.OK);
    expect(c.readFile("/local/a.usda").content).toBe("hi");
  });

  it("client subscribe fires on writes under a prefix and unsubscribes cleanly", () => {
    const c = new Client();
    const seen: string[] = [];
    const unsub = c.subscribe("omniverse://kami/scenes/", (ev) => seen.push(ev.kind));
    c.writeFile("omniverse://kami/scenes/a.usda", "1");
    c.writeFile("omniverse://kami/scenes/a.usda", "2");
    expect(seen).toEqual(["created", "modified"]);
    unsub();
    c.writeFile("omniverse://kami/scenes/b.usda", "1");
    expect(seen).toEqual(["created", "modified"]); // no further events
  });
});
