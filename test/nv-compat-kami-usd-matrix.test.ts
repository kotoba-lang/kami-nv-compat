/**
 * nv-compat kami-usd matrix4d + single-axis rotate xform ops.
 *
 * Coverage for the opMatrix branches not exercised by the xform-composition
 * test: xformOp:transform (flat-16 and nested-4×4 matrix4d), the single-axis
 * rotateX / rotateY / rotateZ ops, and the unknown-op → identity fallback.
 *
 *     pnpm exec vitest run test/nv-compat-kami-usd-matrix.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 kami-usd.
 */

import { describe, it, expect } from "vitest";
import { flattenStage, parseUsda } from "../src/kami-usd/index.js";

const triBody = `point3f[] points = [(1,0,0),(0,1,0),(0,0,1)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0,1,2]`;

/** The three transformed vertices of the first flattened triangle. */
function verts(usda: string): [number, number, number][] {
  return flattenStage(parseUsda(usda)).triangles[0] as [number, number, number][];
}
const close = (a: readonly number[], b: readonly number[], p = 5) =>
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i], p));

describe("xformOp:transform (matrix4d)", () => {
  it("applies a flat-16 row-major translation matrix", () => {
    // Row-major; translation in the 4th column (m[3], m[7], m[11]) → +5 in x.
    const v = verts(`def Xform "W" {
      matrix4d xformOp:transform = (1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1)
      uniform token[] xformOpOrder = ["xformOp:transform"]
      def Mesh "m" { ${triBody} }
    }`);
    close(v[0], [6, 0, 0]); // (1,0,0) + (5,0,0)
  });

  it("accepts a nested 4×4 matrix4d", () => {
    const v = verts(`def Xform "W" {
      matrix4d xformOp:transform = ( (1,0,0,5), (0,1,0,0), (0,0,1,0), (0,0,0,1) )
      uniform token[] xformOpOrder = ["xformOp:transform"]
      def Mesh "m" { ${triBody} }
    }`);
    close(v[0], [6, 0, 0]);
  });
});

describe("single-axis rotate ops", () => {
  it("rotateX 90° maps +y → +z", () => {
    const v = verts(`def Xform "W" {
      float xformOp:rotateX = 90
      uniform token[] xformOpOrder = ["xformOp:rotateX"]
      def Mesh "m" { ${triBody} }
    }`);
    close(v[1], [0, 0, 1]); // (0,1,0) → (0,0,1)
  });

  it("rotateY 90° maps +x → −z", () => {
    const v = verts(`def Xform "W" {
      float xformOp:rotateY = 90
      uniform token[] xformOpOrder = ["xformOp:rotateY"]
      def Mesh "m" { ${triBody} }
    }`);
    close(v[0], [0, 0, -1]); // (1,0,0) → (0,0,-1)
  });

  it("rotateZ 90° maps +x → +y", () => {
    const v = verts(`def Xform "W" {
      float xformOp:rotateZ = 90
      uniform token[] xformOpOrder = ["xformOp:rotateZ"]
      def Mesh "m" { ${triBody} }
    }`);
    close(v[0], [0, 1, 0]); // (1,0,0) → (0,1,0)
  });
});

describe("unknown op fallback", () => {
  it("an unrecognized xformOp resolves to identity (geometry unchanged)", () => {
    const v = verts(`def Xform "W" {
      float xformOp:somethingElse = 42
      uniform token[] xformOpOrder = ["xformOp:somethingElse"]
      def Mesh "m" { ${triBody} }
    }`);
    close(v[0], [1, 0, 0]); // unchanged
  });
});
