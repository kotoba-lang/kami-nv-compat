/**
 * nv-compat kami-usd transform composition + material extraction.
 *
 * Coverage for the geometry bridge: hierarchical Xform composition (translate /
 * scale / rotateXYZ), xformOpOrder honoring + the default SRT order, material
 * extraction precedence (displayColor / diffuseColor / default + emissive), and
 * the Mat4 helpers.
 *
 *     pnpm exec vitest run test/nv-compat-kami-usd-xform.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 kami-usd.
 */

import { describe, it, expect } from "vitest";
import {
  flattenStage,
  identity4,
  mul4,
  parseUsda,
  transformPoint,
} from "../src/kami-usd/index.js";

/** First vertex of the first flattened triangle. */
function firstVertex(usda: string): [number, number, number] {
  return flattenStage(parseUsda(usda)).triangles[0][0] as [number, number, number];
}

const triBody = `point3f[] points = [(1,0,0),(0,1,0),(0,0,1)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0,1,2]`;

describe("Xform composition", () => {
  it("applies a translate to descendant geometry", () => {
    const v = firstVertex(`def Xform "W" {
      double3 xformOp:translate = (5, 0, 0)
      uniform token[] xformOpOrder = ["xformOp:translate"]
      def Mesh "m" { ${triBody} }
    }`);
    expect(v[0]).toBeCloseTo(6, 6); // (1,0,0) + (5,0,0)
  });

  it("applies a scale", () => {
    const v = firstVertex(`def Xform "W" {
      float3 xformOp:scale = (2, 2, 2)
      uniform token[] xformOpOrder = ["xformOp:scale"]
      def Mesh "m" { ${triBody} }
    }`);
    expect(v[0]).toBeCloseTo(2, 6); // (1,0,0) × 2
  });

  it("rotates 90° about +z (x → y)", () => {
    const v = firstVertex(`def Xform "W" {
      float3 xformOp:rotateXYZ = (0, 0, 90)
      uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
      def Mesh "m" { ${triBody} }
    }`);
    expect(v[0]).toBeCloseTo(0, 5);
    expect(v[1]).toBeCloseTo(1, 5); // (1,0,0) rotated 90° about z → (0,1,0)
  });

  it("honors xformOpOrder (scale∘translate ≠ translate∘scale)", () => {
    const scaleThenTranslate = firstVertex(`def Xform "W" {
      double3 xformOp:translate = (5, 0, 0)
      float3 xformOp:scale = (2, 2, 2)
      uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:translate"]
      def Mesh "m" { ${triBody} }
    }`);
    // M = S·T → S·(T·(1,0,0)) = S·(6,0,0) = (12,0,0)
    expect(scaleThenTranslate[0]).toBeCloseTo(12, 6);

    const translateThenScale = firstVertex(`def Xform "W" {
      double3 xformOp:translate = (5, 0, 0)
      float3 xformOp:scale = (2, 2, 2)
      uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
      def Mesh "m" { ${triBody} }
    }`);
    // M = T·S → T·(S·(1,0,0)) = T·(2,0,0) = (7,0,0)
    expect(translateThenScale[0]).toBeCloseTo(7, 6);
  });

  it("defaults to translate·rotate·scale when no xformOpOrder is given", () => {
    const v = firstVertex(`def Xform "W" {
      double3 xformOp:translate = (5, 0, 0)
      float3 xformOp:scale = (2, 2, 2)
      def Mesh "m" { ${triBody} }
    }`);
    expect(v[0]).toBeCloseTo(7, 6); // T·S → (1,0,0)→(2,0,0)→(7,0,0)
  });

  it("composes a parent Xform with a child Xform", () => {
    const v = firstVertex(`def Xform "Parent" {
      double3 xformOp:translate = (10, 0, 0)
      uniform token[] xformOpOrder = ["xformOp:translate"]
      def Xform "Child" {
        double3 xformOp:translate = (0, 3, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "m" { ${triBody} }
      }
    }`);
    expect(v[0]).toBeCloseTo(11, 6); // 1 + 10
    expect(v[1]).toBeCloseTo(3, 6); // 0 + 3
  });
});

describe("material extraction precedence", () => {
  function mat(usda: string) {
    return flattenStage(parseUsda(usda)).materials[0];
  }

  it("uses primvars:displayColor for albedo", () => {
    const m = mat(`def Mesh "m" { ${triBody}\n color3f[] primvars:displayColor = [(0.2, 0.7, 0.1)] }`);
    expect(m.albedo).toEqual([0.2, 0.7, 0.1]);
  });

  it("reads emissive colour into emission", () => {
    const m = mat(`def Mesh "m" { ${triBody}\n color3f[] primvars:emissiveColor = [(3, 3, 3)] }`);
    expect(m.emission).toEqual([3, 3, 3]);
  });

  it("falls back to inputs:diffuseColor then a default grey", () => {
    const diffuse = mat(`def Mesh "m" { ${triBody}\n color3f inputs:diffuseColor = (0.4, 0.4, 0.9) }`);
    expect(diffuse.albedo).toEqual([0.4, 0.4, 0.9]);
    const def = mat(`def Mesh "m" { ${triBody} }`);
    expect(def.albedo[0]).toBeCloseTo(0.8, 6);
    expect(def.emission).toEqual([0, 0, 0]);
  });
});

describe("Mat4 helpers", () => {
  it("identity leaves a point unchanged", () => {
    expect(transformPoint(identity4(), [3, -2, 5])).toEqual([3, -2, 5]);
  });

  it("A · I = A (multiplying by identity)", () => {
    const a = [2, 0, 0, 1, 0, 3, 0, 2, 0, 0, 4, 3, 0, 0, 0, 1];
    expect(mul4(a, identity4())).toEqual(a);
  });
});
