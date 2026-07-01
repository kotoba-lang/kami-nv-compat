/**
 * nv-compat omni.usd / kami-usd validation.
 *
 * Parses a hand-authored USDA (ASCII USD) document through the clean-room
 * kami-usd reader and the pxr.Usd-shaped facade, then renders the resulting
 * geometry end-to-end through the OptiX (ray) and RTX (path) backends — so
 * the whole R1.2 + R1.4 chain (USD → kami-rt / kami-rtx) is exercised.
 *
 *     pnpm exec vitest run test/nv-compat-omni-usd.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.4 omni-usd surface.
 */

import { describe, it, expect } from "vitest";
import { parseUsda, flattenStage } from "../src/kami-usd/index.js";
import {
  Stage,
  Usd,
  UsdGeom,
  stageToScene,
  stageToPathScene,
} from "../src/omni-usd.js";
import { lookAt, traceClosest, traceImageCPU } from "../src/kami-rt/index.js";
import * as optix from "../src/optix.js";
import { createRenderer } from "../src/rtx-renderer.js";

// A single quad (2 triangles) at z=0, displaced +1 in x by an Xform, colored.
const QUAD_USDA = `#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "group" (
        kind = "group"
    )
    {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "quad"
        {
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            color3f[] primvars:displayColor = [(0.8, 0.1, 0.1)]
        }
    }
}
`;

describe("kami-usd USDA parser", () => {
  it("parses prims, types, paths, and nesting", () => {
    const roots = parseUsda(QUAD_USDA);
    expect(roots).toHaveLength(1);
    const world = roots[0];
    expect(world.specifier).toBe("def");
    expect(world.typeName).toBe("Xform");
    expect(world.name).toBe("World");
    expect(world.path).toBe("/World");
    const group = world.children[0];
    expect(group.path).toBe("/World/group");
    const quad = group.children[0];
    expect(quad.typeName).toBe("Mesh");
    expect(quad.path).toBe("/World/group/quad");
  });

  it("parses typed attribute values (points / indices / color)", () => {
    const roots = parseUsda(QUAD_USDA);
    const quad = roots[0].children[0].children[0];
    const pts = quad.attributes.get("points")!.value as number[][];
    expect(pts).toHaveLength(4);
    expect(pts[0]).toEqual([-1, -1, 0]);
    const idx = quad.attributes.get("faceVertexIndices")!.value as number[];
    expect(idx).toEqual([0, 1, 2, 3]);
    const counts = quad.attributes.get("faceVertexCounts")!.value as number[];
    expect(counts).toEqual([4]);
  });

  it("flattens a quad into 2 world-space triangles with the Xform applied", () => {
    const flat = flattenStage(parseUsda(QUAD_USDA));
    expect(flat.triangles).toHaveLength(2);
    expect(flat.materials).toHaveLength(2);
    // The Xform translated x by +1, so the quad spans x∈[0,2].
    const xs = flat.triangles.flat().map((v) => v[0]);
    expect(Math.min(...xs)).toBeCloseTo(0, 6);
    expect(Math.max(...xs)).toBeCloseTo(2, 6);
    // displayColor → albedo.
    expect(flat.materials[0].albedo[0]).toBeCloseTo(0.8, 6);
    expect(flat.materials[0].albedo[1]).toBeCloseTo(0.1, 6);
  });
});

describe("pxr.Usd-shaped facade", () => {
  it("Stage.Open + Traverse visits every prim depth-first", () => {
    const stage = Stage.Open(QUAD_USDA);
    const paths = [...stage.Traverse()].map((p) => p.GetPath());
    expect(paths).toEqual(["/World", "/World/group", "/World/group/quad"]);
  });

  it("GetPrimAtPath + GetAttribute round-trips through the facade", () => {
    const stage = Usd.Stage.Open(QUAD_USDA);
    const quad = stage.GetPrimAtPath("/World/group/quad");
    expect(quad).not.toBeNull();
    expect(quad!.GetTypeName()).toBe("Mesh");
    const pts = UsdGeom.Mesh.Get(quad!).GetPointsAttr();
    expect(pts.IsValid()).toBe(true);
    expect((pts.Get() as number[][]).length).toBe(4);
    expect(stage.GetPrimAtPath("/World/does/not/exist")).toBeNull();
  });
});

describe("USD → kami-rt / kami-rtx end-to-end", () => {
  it("ray-traces a USD stage: a ray through the displaced quad hits it", () => {
    const stage = Stage.Open(QUAD_USDA);
    const scene = stageToScene(stage);
    // Quad now centered at x=1, z=0. Shoot from (1,0,5) toward -z.
    const hit = traceClosest(scene.soup, scene.bvh, [1, 0, 5], [0, 0, -1]);
    expect(hit).not.toBeNull();
    expect(hit!.t).toBeCloseTo(5, 5);
    // A ray at the old (un-translated) center x=0 should now MISS the quad's
    // interior except its left edge — shoot well left to be sure.
    const miss = traceClosest(scene.soup, scene.bvh, [-3, 0, 5], [0, 0, -1]);
    expect(miss).toBeNull();
  });

  it("renders a USD stage through optixLaunch", () => {
    const stage = Stage.Open(QUAD_USDA);
    const scene = stageToScene(stage);
    const ctx = optix.optixDeviceContextCreate();
    const pco = optix.defaultPipelineCompileOptions();
    const mod = optix.optixModuleCreateFromWGSL(ctx, optix.defaultModuleCompileOptions(), pco, "");
    const pl = optix.optixPipelineCreate(ctx, pco, optix.optixProgramGroupCreate(ctx, [{ module: mod, kind: "raygen" }]));
    const cam = lookAt([1, 0, 4], [1, 0, 0], [0, 1, 0], 45, 1);
    const params: optix.OptixLaunchParams = { scene, camera: cam, width: 16, height: 16 };
    expect(optix.optixLaunch(pl, optix.optixShaderBindingTableCreate(), params)).toBe(
      optix.OptixResult.OPTIX_SUCCESS,
    );
    expect(params.framebuffer!.length).toBe(16 * 16 * 4);
  });

  it("path-traces a USD stage through the RTX renderer", () => {
    // Add an emitter so the path tracer has light; reuse the facade path scene.
    const lit = `#usda 1.0
def Xform "World"
{
    def Mesh "floor"
    {
        point3f[] points = [(-2,-1,-2),(2,-1,-2),(2,-1,2),(-2,-1,2)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0,1,2,3]
        color3f[] primvars:displayColor = [(0.7,0.7,0.7)]
    }
    def Mesh "light"
    {
        point3f[] points = [(-1,2,-1),(1,2,-1),(1,2,1),(-1,2,1)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0,1,2,3]
        color3f[] primvars:emissiveColor = [(8,8,8)]
    }
}
`;
    const pathScene = stageToPathScene(Stage.Open(lit));
    expect(pathScene.soup.count).toBe(4); // 2 quads → 4 triangles
    // Emissive triangles carry nonzero emission in the material soup.
    const totalEmission = pathScene.mats.emission.reduce((a, b) => a + b, 0);
    expect(totalEmission).toBeGreaterThan(0);

    const r = createRenderer({ samplesPerPixel: 8, maxBounces: 4 });
    // createScene rebuilds from meshes; here render the already-built scene by
    // wrapping it through the renderer's path. Use renderSync on a fresh scene
    // assembled from the same USDA via the facade bridge is simplest:
    const flatR = r.createScene(
      pathSceneTriangles(pathScene),
      pathSceneMaterials(pathScene),
    );
    const cam = lookAt([0, 0.5, 4], [0, 0, 0], [0, 1, 0], 50, 1);
    const out = r.renderSync(flatR, cam, 16, 16);
    expect(out.framebuffer.length).toBe(16 * 16 * 4);
    let lum = 0;
    for (let i = 0; i < out.framebuffer.length; i += 4) lum += out.framebuffer[i + 1];
    expect(lum).toBeGreaterThan(0); // the lit scene renders some radiance
  });
});

// Helpers to round-trip a built PathScene back into mesh/material lists for the
// renderer facade (the facade's createScene takes raw triangles + materials).
function pathSceneTriangles(s: ReturnType<typeof stageToPathScene>): [number, number, number][][] {
  const out: [number, number, number][][] = [];
  for (let t = 0; t < s.soup.count; t++) {
    const b = t * 9;
    const v = s.soup.verts;
    out.push([
      [v[b], v[b + 1], v[b + 2]],
      [v[b + 3], v[b + 4], v[b + 5]],
      [v[b + 6], v[b + 7], v[b + 8]],
    ]);
  }
  return out;
}
function pathSceneMaterials(s: ReturnType<typeof stageToPathScene>) {
  const out = [];
  for (let t = 0; t < s.soup.count; t++) {
    out.push({
      albedo: [s.mats.albedo[t * 3], s.mats.albedo[t * 3 + 1], s.mats.albedo[t * 3 + 2]] as [number, number, number],
      emission: [s.mats.emission[t * 3], s.mats.emission[t * 3 + 1], s.mats.emission[t * 3 + 2]] as [number, number, number],
    });
  }
  return out;
}
