// @etzhayyim/kami-nv-compat/kotoba-datomic-nucleus
//
// Clean-room Nucleus engine (kotoba-datomic-nucleus) — content-addressed,
// append-only, subscribable versioned store + an omni.client-style API. The
// canonical KAMI backend behind `nv-compat/omni-nucleus`; mirrors the kotoba
// Datom log content-addressing model.
//
// ADR-2605261800 §D6 / D10.4 kotoba-datomic-nucleus.

export {
  type Version,
  type ChangeEvent,
  type ChangeKind,
  NucleusStore,
  cidOf,
} from "./store.js";
export {
  type ListEntry,
  type StatInfo,
  Result,
  Client,
  parseUrl,
} from "./client.js";

import { type Client } from "./client.js";
import { Stage, stageToScene, stageToPathScene } from "../omni-usd.js";
import { type Scene, type PathScene } from "../kami-rt/index.js";

/** Read a USDA layer from Nucleus and parse it into a kami-rt ray scene. */
export function readSceneFromNucleus(client: Client, url: string): Scene | null {
  const { content } = client.readFile(url);
  return content ? stageToScene(Stage.Open(content)) : null;
}

/** Read a USDA layer from Nucleus and parse it into a kami-rtx path scene. */
export function readPathSceneFromNucleus(client: Client, url: string): PathScene | null {
  const { content } = client.readFile(url);
  return content ? stageToPathScene(Stage.Open(content)) : null;
}
