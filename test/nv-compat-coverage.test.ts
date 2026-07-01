/**
 * nv-compat surface coverage + consistency.
 *
 * A maturity guard over the whole nv-compat facade: every component namespace
 * must be exported from the barrel, each facade's canonical KAMI engine name
 * must agree with NV_COMPAT_MAP / ALPAMAYO_COMPAT_MAP, and the WGSL shader
 * strings must carry their required compute entry points + bindings. Catches
 * broken re-exports, mis-wired modules, and drifted compat-map values without
 * needing a GPU.
 *
 *     pnpm exec vitest run test/nv-compat-coverage.test.ts
 *
 * ADR-2605261800 §D1/D6.
 */

import { describe, it, expect } from "vitest";
import * as nv from "../src/index.js";

describe("barrel exports — every component namespace is wired", () => {
  const expected = [
    // base ports
    "dynamics", "controllers", "actions", "assets", "warp", "policies",
    // Omniverse stack facades
    "optix", "rtxRenderer", "kamiRt",
    "omniUsd", "kamiUsd",
    "omniReplicatorCore", "utsushimi",
    "isaaclabEnvs", "e7mShugyo",
    "isaacSim", "e7mSim",
    "omniNucleus", "kotobaDatomicNucleus",
    "omniCloud", "murakumoRender",
    "omniKitApp", "amenominaka",
    "driveSim", "wadachiSim",
    // AV stack
    "alpamayo", "alpasim", "kamiDrive",
  ];

  it.each(expected)("exports %s", (name) => {
    expect((nv as Record<string, unknown>)[name]).toBeDefined();
  });

  it("declares the completed R1 phase + ADR", () => {
    expect(nv.PHASE).toBe("R1-complete");
    expect(nv.ADR).toBe("ADR-2605261800");
  });

  it("NV_COMPAT_MAP covers the 9 Omniverse components", () => {
    expect(Object.keys(nv.NV_COMPAT_MAP).sort()).toEqual(
      ["DriveSim", "Isaac Lab", "Isaac Sim", "Nucleus", "Omniverse Cloud", "Omniverse Kit", "OptiX", "RTX Renderer", "Replicator"].sort(),
    );
    expect(nv.ALPAMAYO_COMPAT_MAP.Alpamayo).toBe("michibiki");
  });
});

describe("compat-map consistency — each facade reports its canonical engine", () => {
  const cases: Array<[string, string]> = [
    [(nv.optix as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["OptiX"]],
    [(nv.rtxRenderer as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["RTX Renderer"]],
    [(nv.omniReplicatorCore as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["Replicator"]],
    [(nv.isaaclabEnvs as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["Isaac Lab"]],
    [(nv.isaacSim as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["Isaac Sim"]],
    [(nv.driveSim as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["DriveSim"]],
    [(nv.omniCloud as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["Omniverse Cloud"]],
    [(nv.omniNucleus as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["Nucleus"]],
    [(nv.omniKitApp as { KAMI_ENGINE: string }).KAMI_ENGINE, nv.NV_COMPAT_MAP["Omniverse Kit"]],
  ];

  it.each(cases)("facade engine %s matches the compat map (%s)", (engine, mapped) => {
    expect(engine).toBe(mapped);
  });

  it("the Alpamayo AV facade reports michibiki", () => {
    expect((nv.alpamayo as { KAMI_ENGINE: string }).KAMI_ENGINE).toBe(nv.ALPAMAYO_COMPAT_MAP.Alpamayo);
  });
});

describe("WGSL shaders carry their compute entry points + bindings", () => {
  const shaders: Array<[string, string]> = [
    ["RAYTRACE_WGSL", (nv.kamiRt as { RAYTRACE_WGSL: string }).RAYTRACE_WGSL],
    ["PATHTRACE_WGSL", (nv.kamiRt as { PATHTRACE_WGSL: string }).PATHTRACE_WGSL],
  ];

  it.each(shaders)("%s declares @compute main + a framebuffer binding", (_name, src) => {
    expect(src.length).toBeGreaterThan(200);
    expect(src).toContain("@compute");
    expect(src).toContain("@workgroup_size(8, 8");
    expect(src).toMatch(/fn main\s*\(/);
    expect(src).toContain("fb"); // read_write framebuffer storage binding
    expect(src).toContain("@group(0)");
  });

  it("the path tracer shader includes the RNG + cosine-sampling helpers", () => {
    const src = (nv.kamiRt as { PATHTRACE_WGSL: string }).PATHTRACE_WGSL;
    expect(src).toContain("seedHash");
    expect(src).toContain("cosineSample");
    expect(src).toContain("traceClosest");
  });
});

describe("namespaced sub-exports resolve (spot check)", () => {
  it("kamiRt exposes the scene + trace API", () => {
    const k = nv.kamiRt as Record<string, unknown>;
    expect(typeof k.buildScene).toBe("function");
    expect(typeof k.traceImageCPU).toBe("function");
    expect(typeof k.pathTraceCPU).toBe("function");
  });

  it("alpamayo exposes the VLA model factory + horizon constants", () => {
    const a = nv.alpamayo as Record<string, unknown>;
    expect(typeof (a.AlpamayoR1 as { fromPretrained?: unknown }).fromPretrained).toBe("function");
    expect(a.TRAJECTORY_WAYPOINTS).toBe(64);
    expect(a.SAE_CEILING).toBe(4);
  });
});
