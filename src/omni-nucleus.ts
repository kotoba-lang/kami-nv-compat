// @etzhayyim/kami-nv-compat/omni-nucleus
//
// Drop-in `omni.client` / Omniverse Nucleus API-compat facade — the
// collaboration/data backend. Mirrors the documented client surface (stat /
// list / read_file / write_file / copy / delete / checkpoints / subscribe over
// `omniverse://` URLs) so Nucleus client code ports to KAMI via import-path-
// only changes — e.g.
//
//     import { Client, Result } from "@etzhayyim/kami-nv-compat/omni-nucleus";
//
//     const c = new Client();
//     c.writeFile("omniverse://kami/scenes/box.usda", usda);
//     const { content } = c.readFile("omniverse://kami/scenes/box.usda");
//     c.createCheckpoint("omniverse://kami/scenes/box.usda", "initial");
//     const unsub = c.subscribe("omniverse://kami/scenes/", (ev) => { ... });
//
// Backed by the clean-room kotoba-datomic-nucleus engine: a content-addressed,
// append-only, subscribable versioned store mirroring the kotoba Datom log
// model. No Nucleus source/binaries; from-spec reproduction (Google v. Oracle,
// 593 U.S. ___ (2021)). Canonical engine: kotoba-datomic-nucleus.
//
// Trademark: NVIDIA® / Omniverse® / Nucleus are trademarks of NVIDIA
// Corporation; API-compat identifiers only.
//
// ADR-2605261800 §D1/D6, R1.9 omni-nucleus surface.

export {
  type Version,
  type ChangeEvent,
  type ChangeKind,
  type ListEntry,
  type StatInfo,
  Result,
  Client,
  NucleusStore,
  parseUrl,
  cidOf,
  readSceneFromNucleus,
  readPathSceneFromNucleus,
} from "./kotoba-datomic-nucleus/index.js";

export const KAMI_ENGINE = "kotoba-datomic-nucleus";
export const ADR = "ADR-2605261800";
