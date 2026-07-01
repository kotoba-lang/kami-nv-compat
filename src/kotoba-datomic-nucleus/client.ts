// kotoba-datomic-nucleus — omni.client-style API over the versioned store.
//
// Mirrors the documented `omni.client` surface (stat / list / read_file /
// write_file / copy / delete / create_checkpoint / get_checkpoints /
// subscribe) used to talk to a Nucleus server, returning Result codes. URLs
// are `omniverse://<server>/<path>` (the server segment is a logical mount
// name; one process-wide store backs all servers here).
//
// ADR-2605261800 §D6 / D10.4 kotoba-datomic-nucleus.

import { type ChangeEvent, type Version, NucleusStore } from "./store.js";

export enum Result {
  OK = "OK",
  ERROR_NOT_FOUND = "ERROR_NOT_FOUND",
  ERROR_ACCESS_DENIED = "ERROR_ACCESS_DENIED",
  ERROR_ALREADY_EXISTS = "ERROR_ALREADY_EXISTS",
  ERROR_INVALID_URL = "ERROR_INVALID_URL",
}

export interface ListEntry {
  relativePath: string;
  cid: string;
  version: number;
}

export interface StatInfo {
  cid: string;
  version: number;
  size: number;
}

/** Parse `omniverse://server/path` → `{ server, path }`. A bare `/path` (no
 *  scheme) is accepted with server "" for convenience. */
export function parseUrl(url: string): { server: string; path: string } | null {
  if (url.startsWith("omniverse://")) {
    const rest = url.slice("omniverse://".length);
    const slash = rest.indexOf("/");
    if (slash < 0) return { server: rest, path: "/" };
    return { server: rest.slice(0, slash), path: rest.slice(slash) };
  }
  if (url.startsWith("/")) return { server: "", path: url };
  return null;
}

function key(url: string): string | null {
  const u = parseUrl(url);
  return u ? `${u.server}${u.path}` : null;
}

/** omni.client-compatible client over a {@link NucleusStore}. */
export class Client {
  constructor(readonly store: NucleusStore = new NucleusStore()) {}

  stat(url: string): { result: Result; info?: StatInfo } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL };
    const head = this.store.head(k);
    if (!head) return { result: Result.ERROR_NOT_FOUND };
    return { result: Result.OK, info: { cid: head.cid, version: head.index, size: head.content.length } };
  }

  readFile(url: string): { result: Result; content?: string } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL };
    const content = this.store.read(k);
    if (content === null) return { result: Result.ERROR_NOT_FOUND };
    return { result: Result.OK, content };
  }

  writeFile(url: string, content: string): { result: Result; version?: Version } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL };
    return { result: Result.OK, version: this.store.write(k, content) };
  }

  copy(srcUrl: string, dstUrl: string): { result: Result } {
    const s = key(srcUrl);
    const d = key(dstUrl);
    if (s === null || d === null) return { result: Result.ERROR_INVALID_URL };
    return { result: this.store.copy(s, d) ? Result.OK : Result.ERROR_NOT_FOUND };
  }

  delete(url: string): { result: Result } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL };
    return { result: this.store.delete(k) ? Result.OK : Result.ERROR_NOT_FOUND };
  }

  list(url: string): { result: Result; entries: ListEntry[] } {
    const u = parseUrl(url);
    if (!u) return { result: Result.ERROR_INVALID_URL, entries: [] };
    const prefix = `${u.server}${u.path}`;
    const entries: ListEntry[] = [];
    for (const path of this.store.list(prefix)) {
      const head = this.store.head(path);
      if (head) entries.push({ relativePath: path.slice(prefix.length), cid: head.cid, version: head.index });
    }
    return { result: Result.OK, entries };
  }

  // ── checkpoints (Nucleus versioning) ───────────────────────────────────────

  createCheckpoint(url: string, message: string): { result: Result; version?: Version } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL };
    const content = this.store.read(k);
    if (content === null) return { result: Result.ERROR_NOT_FOUND };
    // A checkpoint re-stamps the current content with a message (append-only).
    return { result: Result.OK, version: this.store.write(k, `${content}`, message) };
  }

  getCheckpoints(url: string): { result: Result; checkpoints: Version[] } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL, checkpoints: [] };
    const history = this.store.history(k);
    return { result: history.length ? Result.OK : Result.ERROR_NOT_FOUND, checkpoints: history };
  }

  restore(url: string, version: number): { result: Result; version?: Version } {
    const k = key(url);
    if (k === null) return { result: Result.ERROR_INVALID_URL };
    const v = this.store.restore(k, version);
    return v ? { result: Result.OK, version: v } : { result: Result.ERROR_NOT_FOUND };
  }

  /** Subscribe to changes at a URL (exact path, or `…/` prefix). */
  subscribe(url: string, cb: (ev: ChangeEvent) => void): () => void {
    const u = parseUrl(url);
    if (!u) return () => {};
    return this.store.subscribe(`${u.server}${u.path}`, cb);
  }
}
