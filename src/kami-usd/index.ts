// @etzhayyim/kami-nv-compat/kami-usd
//
// Clean-room USD layer — the canonical KAMI engine behind the
// `nv-compat/omni-usd` API-compat facade. Parses the USDA (ASCII USD)
// geometry subset and flattens it into kami-rt triangles + kami-rtx
// materials.
//
// ADR-2605261800 §D6 / D10.4 kami-usd.

export {
  type UsdValue,
  type UsdAttribute,
  type UsdPrimNode,
  parseUsda,
} from "./usda.js";
export {
  type Mat4,
  type FlatScene,
  identity4,
  mul4,
  transformPoint,
  localTransform,
  triangulateMesh,
  meshMaterial,
  flattenStage,
} from "./geom.js";

import { type Scene, type PathScene, buildScene, buildPathScene } from "../kami-rt/index.js";
import { parseUsda } from "./usda.js";
import { type FlatScene, flattenStage } from "./geom.js";

/** Parse a USDA document and flatten it to world-space triangles + materials. */
export function usdaToFlatScene(text: string): FlatScene {
  return flattenStage(parseUsda(text));
}

/** Parse a USDA document into a kami-rt ray-trace {@link Scene}. */
export function usdaToScene(text: string): Scene {
  const flat = usdaToFlatScene(text);
  return buildScene(flat.triangles);
}

/** Parse a USDA document into a kami-rtx path-trace {@link PathScene}
 *  (geometry + materials). */
export function usdaToPathScene(text: string): PathScene {
  const flat = usdaToFlatScene(text);
  return buildPathScene(flat.triangles, flat.materials);
}
