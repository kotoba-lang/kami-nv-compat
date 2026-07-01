// @etzhayyim/kami-nv-compat/omni-usd
//
// Drop-in `pxr.Usd` / `UsdGeom` / `omni.usd` API-compat facade. Mirrors the
// documented public OpenUSD Python surface (Stage / Prim / Attribute +
// UsdGeom.Mesh / UsdGeom.Xformable) for the USDA geometry subset, so existing
// USD host code ports to KAMI via import-path-only changes — e.g.
//
//     import { Usd, UsdGeom } from "@etzhayyim/kami-nv-compat/omni-usd";
//
//     const stage = Usd.Stage.Open(usdaText);
//     for (const prim of stage.Traverse()) {
//       if (prim.GetTypeName() === "Mesh") { ... }
//     }
//     const scene = stageToPathScene(stage);   // → kami-rtx path scene
//
// Backed by the clean-room kami-usd reader (USDA parse + geometry flatten).
// Binary .usdc / .usdz are a later kami-usd milestone (tinyusdz WASM).
//
// Clean-room: this re-implements the public USD API *shape* (Google v. Oracle,
// 593 U.S. ___ (2021)). No USD / OpenUSD / tinyusdz source, headers, or SDK
// binaries are used. The canonical engine has a distinct name — `kami-usd`.
//
// Trademark: USD / OpenUSD are projects of Pixar / the Alliance for OpenUSD;
// "Omniverse" is a trademark of NVIDIA Corporation. Names here are API-compat
// identifiers only.
//
// ADR-2605261800 §D1/D6, R1.4 omni-usd surface.

import { type PathScene, type Scene, buildPathScene, buildScene } from "./kami-rt/index.js";
import {
  type FlatScene,
  type UsdAttribute,
  type UsdPrimNode,
  type UsdValue,
  flattenStage,
  parseUsda,
} from "./kami-usd/index.js";

export type { UsdValue } from "./kami-usd/index.js";

// ── pxr.Usd.Attribute ──────────────────────────────────────────────────────

export class Attribute {
  constructor(private readonly _attr: UsdAttribute | undefined) {}
  IsValid(): boolean {
    return this._attr !== undefined;
  }
  Get(): UsdValue | null {
    return this._attr?.value ?? null;
  }
  GetTypeName(): string {
    return this._attr?.typeName ?? "";
  }
  GetName(): string {
    return this._attr?.name ?? "";
  }
}

// ── pxr.Usd.Prim ────────────────────────────────────────────────────────────

export class Prim {
  constructor(private readonly _node: UsdPrimNode) {}
  IsValid(): boolean {
    return true;
  }
  GetPath(): string {
    return this._node.path;
  }
  GetName(): string {
    return this._node.name;
  }
  /** Schema type, e.g. "Mesh", "Xform", "Scope". */
  GetTypeName(): string {
    return this._node.typeName;
  }
  GetSpecifier(): string {
    return this._node.specifier;
  }
  GetAttribute(name: string): Attribute {
    return new Attribute(this._node.attributes.get(name));
  }
  HasAttribute(name: string): boolean {
    return this._node.attributes.has(name);
  }
  GetAttributeNames(): string[] {
    return [...this._node.attributes.keys()];
  }
  GetChildren(): Prim[] {
    return this._node.children.map((c) => new Prim(c));
  }
  /** Underlying parsed node (KAMI extension; escape hatch for the bridge). */
  get node(): UsdPrimNode {
    return this._node;
  }
}

// ── pxr.Usd.Stage ───────────────────────────────────────────────────────────

export class Stage {
  private readonly _index: Map<string, UsdPrimNode> = new Map();

  private constructor(private readonly _roots: UsdPrimNode[]) {
    const index = (n: UsdPrimNode): void => {
      this._index.set(n.path, n);
      for (const c of n.children) index(c);
    };
    for (const r of _roots) index(r);
  }

  /** `Usd.Stage.Open()` mirror. Accepts USDA text (a path-loading overload is
   *  a future kami-usd milestone; this build is text-in). */
  static Open(usdaText: string): Stage {
    return new Stage(parseUsda(usdaText));
  }

  /** `Usd.Stage.CreateInMemory()` mirror — an empty stage. */
  static CreateInMemory(): Stage {
    return new Stage([]);
  }

  GetPrimAtPath(path: string): Prim | null {
    const n = this._index.get(path);
    return n ? new Prim(n) : null;
  }

  GetPseudoRoot(): Prim[] {
    return this._roots.map((r) => new Prim(r));
  }

  /** `Usd.Stage.Traverse()` mirror — depth-first over every prim. */
  *Traverse(): Generator<Prim> {
    const stack = [...this._roots].reverse();
    while (stack.length) {
      const n = stack.pop() as UsdPrimNode;
      yield new Prim(n);
      for (let i = n.children.length - 1; i >= 0; i--) stack.push(n.children[i]);
    }
  }

  /** Underlying root nodes (KAMI extension; used by the scene bridge). */
  get roots(): readonly UsdPrimNode[] {
    return this._roots;
  }
}

// ── namespace-style aliases (pxr.Usd / pxr.UsdGeom) ──────────────────────────

export const Usd = { Stage, Prim, Attribute };

/** UsdGeom typed-schema helpers. `Mesh.Get(prim)` returns the mesh accessors
 *  for a prim whose type is "Mesh". */
export const UsdGeom = {
  Mesh: {
    Get(prim: Prim): {
      GetPointsAttr(): Attribute;
      GetFaceVertexIndicesAttr(): Attribute;
      GetFaceVertexCountsAttr(): Attribute;
      GetDisplayColorAttr(): Attribute;
    } {
      return {
        GetPointsAttr: () => prim.GetAttribute("points"),
        GetFaceVertexIndicesAttr: () => prim.GetAttribute("faceVertexIndices"),
        GetFaceVertexCountsAttr: () => prim.GetAttribute("faceVertexCounts"),
        GetDisplayColorAttr: () => prim.GetAttribute("primvars:displayColor"),
      };
    },
  },
};

// ── scene bridge (KAMI extension: USD → kami-rt / kami-rtx) ──────────────────

/** Flatten a stage to world-space triangles + materials. */
export function stageToFlatScene(stage: Stage): FlatScene {
  return flattenStage(stage.roots);
}

/** Build a kami-rt ray-trace {@link Scene} from a stage. */
export function stageToScene(stage: Stage): Scene {
  return buildScene(stageToFlatScene(stage).triangles);
}

/** Build a kami-rtx path-trace {@link PathScene} from a stage. */
export function stageToPathScene(stage: Stage): PathScene {
  const flat = stageToFlatScene(stage);
  return buildPathScene(flat.triangles, flat.materials);
}

export const KAMI_ENGINE = "kami-usd";
export const ADR = "ADR-2605261800";
