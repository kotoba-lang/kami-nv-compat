/**
 * nv-compat kami-usd parser error paths + geometry edge cases.
 *
 * Negative + boundary coverage for the USDA reader and the triangulation
 * bridge: malformed documents must throw with a clear message, and degenerate
 * geometry (empty meshes, n-gons, missing face counts, out-of-range indices)
 * must be handled without crashing.
 *
 *     pnpm exec vitest run test/nv-compat-usda-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 kami-usd.
 */

import { describe, it, expect } from "vitest";
import { parseUsda, flattenStage, usdaToPathScene } from "../src/kami-usd/index.js";
import { buildPathScene, material } from "../src/kami-rt/index.js";

describe("USDA parser error paths", () => {
  it("throws on non-specifier garbage at the top level", () => {
    expect(() => parseUsda("garbage \"x\" {}")).toThrow(/expected def\/over\/class/);
  });

  it("throws on an unterminated prim body (missing })", () => {
    expect(() => parseUsda(`def Xform "World" {\n  def Mesh "m" {`)).toThrow(/unexpected EOF/);
  });

  it("throws on an unbalanced metadata block", () => {
    expect(() => parseUsda(`def Xform "World" (\n  kind = "group"\n`)).toThrow(/unbalanced metadata/);
  });

  it("accepts over and class specifiers", () => {
    const roots = parseUsda(`over "World" {}\nclass "Proto" {}`);
    expect(roots.map((r) => r.specifier)).toEqual(["over", "class"]);
  });

  it("tolerates a #usda header, stage metadata, and comments", () => {
    const roots = parseUsda(`#usda 1.0\n(\n  defaultPrim = "World"\n)\n# a comment\ndef Xform "World" {}`);
    expect(roots).toHaveLength(1);
    expect(roots[0].name).toBe("World");
  });
});

describe("USDA value coercion", () => {
  it("parses ints, floats, negatives, scientific notation, bools, and nested arrays", () => {
    const roots = parseUsda(`def Scope "S" {
      int i = 42
      float neg = -1.5
      double sci = 1.0e2
      bool flag = true
      int[] xs = [1, 2, 3]
      float3 v = (-0.5, 0, 0.25)
    }`);
    const a = roots[0].attributes;
    expect(a.get("i")!.value).toBe(42);
    expect(a.get("neg")!.value).toBe(-1.5);
    expect(a.get("sci")!.value).toBe(100);
    expect(a.get("flag")!.value).toBe(true);
    expect(a.get("xs")!.value).toEqual([1, 2, 3]);
    expect(a.get("v")!.value).toEqual([-0.5, 0, 0.25]);
  });

  it("preserves the uniform flag on uniform attributes", () => {
    const roots = parseUsda(`def Scope "S" {\n  uniform token[] order = ["a", "b"]\n}`);
    expect(roots[0].attributes.get("order")!.uniform).toBe(true);
    expect(roots[0].attributes.get("order")!.value).toEqual(["a", "b"]);
  });
});

describe("geometry triangulation edge cases", () => {
  it("fan-triangulates an n-gon (pentagon → 3 triangles)", () => {
    const flat = flattenStage(parseUsda(`def Mesh "penta" {
      point3f[] points = [(0,0,0),(1,0,0),(1.5,1,0),(0.5,1.5,0),(-0.5,1,0)]
      int[] faceVertexCounts = [5]
      int[] faceVertexIndices = [0,1,2,3,4]
    }`));
    expect(flat.triangles).toHaveLength(3); // n-2 fan
  });

  it("treats indices as consecutive triangles when faceVertexCounts is absent", () => {
    const flat = flattenStage(parseUsda(`def Mesh "soup" {
      point3f[] points = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(2,0,0),(2,1,0)]
      int[] faceVertexIndices = [0,1,2,1,3,2]
    }`));
    expect(flat.triangles).toHaveLength(2);
  });

  it("yields no triangles for a mesh with no points or no indices", () => {
    expect(flattenStage(parseUsda(`def Mesh "empty" {}`)).triangles).toHaveLength(0);
    expect(
      flattenStage(parseUsda(`def Mesh "noidx" { point3f[] points = [(0,0,0),(1,0,0),(0,1,0)] }`)).triangles,
    ).toHaveLength(0);
  });

  it("skips faces with out-of-range index spans without crashing", () => {
    // faceVertexCounts says 4 but only 3 indices exist → the face is skipped.
    const flat = flattenStage(parseUsda(`def Mesh "bad" {
      point3f[] points = [(0,0,0),(1,0,0),(0,1,0)]
      int[] faceVertexCounts = [4]
      int[] faceVertexIndices = [0,1,2]
    }`));
    expect(flat.triangles).toHaveLength(0);
  });

  it("non-Mesh prims contribute no geometry", () => {
    const flat = flattenStage(parseUsda(`def Xform "World" { def Scope "grp" {} }`));
    expect(flat.triangles).toHaveLength(0);
  });
});

describe("usdaToPathScene material defaults", () => {
  it("falls back to a default grey albedo when no displayColor is present", () => {
    const scene = usdaToPathScene(`def Mesh "m" {
      point3f[] points = [(-1,-1,0),(1,-1,0),(0,1,0)]
      int[] faceVertexCounts = [3]
      int[] faceVertexIndices = [0,1,2]
    }`);
    expect(scene.soup.count).toBe(1);
    expect(scene.mats.albedo[0]).toBeCloseTo(0.8, 6);
  });

  it("buildPathScene rejects a triangle/material count mismatch", () => {
    expect(() => buildPathScene([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], [material([1, 0, 0]), material([0, 1, 0])])).toThrow(/1:1/);
  });
});
