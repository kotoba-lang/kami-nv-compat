/**
 * nv-compat omni-usd pxr.Usd facade accessor coverage.
 *
 * Coverage for the facade surface not hit by the parser/geometry tests:
 * Stage.CreateInMemory, GetPseudoRoot, Prim accessors (name/specifier/children/
 * attribute names), Attribute validity, and the full UsdGeom.Mesh accessor set.
 *
 *     pnpm exec vitest run test/nv-compat-omni-usd-facade.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.4 omni-usd surface.
 */

import { describe, it, expect } from "vitest";
import { Stage, Usd, UsdGeom } from "../src/omni-usd.js";

const USDA = `#usda 1.0
def Xform "World" {
    def Scope "group" {
        def Mesh "quad" {
            point3f[] points = [(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0,1,2,3]
            color3f[] primvars:displayColor = [(0.5,0.5,0.5)]
        }
    }
}`;

describe("Stage.CreateInMemory (empty)", () => {
  it("has no prims to traverse and resolves no paths", () => {
    const s = Stage.CreateInMemory();
    expect([...s.Traverse()]).toHaveLength(0);
    expect(s.GetPrimAtPath("/anything")).toBeNull();
    expect(s.GetPseudoRoot()).toHaveLength(0);
  });
});

describe("Stage / Prim accessors", () => {
  const stage = Usd.Stage.Open(USDA);

  it("GetPseudoRoot returns the top-level prims", () => {
    const roots = stage.GetPseudoRoot();
    expect(roots).toHaveLength(1);
    expect(roots[0].GetName()).toBe("World");
  });

  it("Traverse is depth-first over the whole tree", () => {
    expect([...stage.Traverse()].map((p) => p.GetPath())).toEqual([
      "/World",
      "/World/group",
      "/World/group/quad",
    ]);
  });

  it("Prim exposes name / specifier / type / children", () => {
    const world = stage.GetPrimAtPath("/World")!;
    expect(world.IsValid()).toBe(true);
    expect(world.GetSpecifier()).toBe("def");
    expect(world.GetTypeName()).toBe("Xform");
    const children = world.GetChildren();
    expect(children).toHaveLength(1);
    expect(children[0].GetName()).toBe("group");
    expect(children[0].GetTypeName()).toBe("Scope");
  });

  it("Prim attribute introspection", () => {
    const quad = stage.GetPrimAtPath("/World/group/quad")!;
    expect(quad.HasAttribute("points")).toBe(true);
    expect(quad.HasAttribute("nope")).toBe(false);
    expect(quad.GetAttributeNames().sort()).toEqual(
      ["faceVertexCounts", "faceVertexIndices", "points", "primvars:displayColor"].sort(),
    );
    // node escape hatch exposes the parsed record.
    expect(quad.node.path).toBe("/World/group/quad");
  });
});

describe("Attribute validity + value", () => {
  const quad = Usd.Stage.Open(USDA).GetPrimAtPath("/World/group/quad")!;

  it("valid attribute reports type / name / value", () => {
    const pts = quad.GetAttribute("points");
    expect(pts.IsValid()).toBe(true);
    expect(pts.GetTypeName()).toBe("point3f[]");
    expect(pts.GetName()).toBe("points");
    expect((pts.Get() as number[][]).length).toBe(4);
  });

  it("missing attribute is invalid with a null value", () => {
    const a = quad.GetAttribute("missing");
    expect(a.IsValid()).toBe(false);
    expect(a.Get()).toBeNull();
    expect(a.GetTypeName()).toBe("");
  });
});

describe("UsdGeom.Mesh accessors", () => {
  it("exposes points / faceVertexIndices / faceVertexCounts / displayColor", () => {
    const quad = Stage.Open(USDA).GetPrimAtPath("/World/group/quad")!;
    const mesh = UsdGeom.Mesh.Get(quad);
    expect((mesh.GetPointsAttr().Get() as number[][]).length).toBe(4);
    expect(mesh.GetFaceVertexIndicesAttr().Get()).toEqual([0, 1, 2, 3]);
    expect(mesh.GetFaceVertexCountsAttr().Get()).toEqual([4]);
    expect(mesh.GetDisplayColorAttr().IsValid()).toBe(true);
  });
});
