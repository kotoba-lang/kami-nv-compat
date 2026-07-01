// kotoba-datomic-nucleus — clean-room content-addressed versioned store.
//
// The canonical KAMI implementation behind `nv-compat/omni-nucleus`. NVIDIA
// Omniverse Nucleus is the collaboration/data backend — a versioned USD asset
// store with checkpoints and live change notification. This module reproduces
// that behaviour on a content-addressed, append-only model that mirrors the
// kotoba Datom log ethos: every write hashes its content to a CID, history is
// append-only (a checkpoint chain), and nothing is destructively overwritten.
//
// Content addressing uses sha-256 (via @noble/hashes, already an SDK dep), so
// identical bytes always yield the same CID and a re-write of unchanged content
// is a no-op version.
//
// Clean-room: from-spec versioned store. No Nucleus source/binaries.
// ADR-2605261800 §D6 / D10.4 kotoba-datomic-nucleus.

import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils";

/** Content identifier — `sha2-256:` + hex digest of the content bytes. */
export function cidOf(content: string): string {
  return `sha2-256:${bytesToHex(sha256(utf8ToBytes(content)))}`;
}

export interface Version {
  cid: string;
  content: string;
  /** Monotonic version index for this path (0-based). */
  index: number;
  /** Optional checkpoint message. */
  message?: string;
}

interface Entry {
  /** Append-only version history (oldest first). */
  versions: Version[];
}

export type ChangeKind = "created" | "modified" | "deleted";

export interface ChangeEvent {
  path: string;
  kind: ChangeKind;
  cid?: string;
}

type Subscriber = (ev: ChangeEvent) => void;

/** A content-addressed, versioned, subscribable store keyed by path. */
export class NucleusStore {
  private readonly entries = new Map<string, Entry>();
  private readonly subscribers = new Map<string, Set<Subscriber>>();

  /** Write content to `path`, creating a new version. Re-writing identical
   *  content is a no-op (returns the existing head version). */
  write(path: string, content: string, message?: string): Version {
    const cid = cidOf(content);
    let entry = this.entries.get(path);
    const created = entry === undefined;
    if (!entry) {
      entry = { versions: [] };
      this.entries.set(path, entry);
    }
    const head = entry.versions[entry.versions.length - 1];
    // Unchanged content with no checkpoint message → no new version. A labeled
    // write (a checkpoint) always appends, even when the bytes are identical.
    if (head && head.cid === cid && message === undefined) return head;
    const version: Version = { cid, content, index: entry.versions.length, message };
    entry.versions.push(version);
    this.notify(path, { path, kind: created ? "created" : "modified", cid });
    return version;
  }

  /** Latest content at `path`, or null if absent/deleted. */
  read(path: string): string | null {
    const entry = this.entries.get(path);
    if (!entry || entry.versions.length === 0) return null;
    return entry.versions[entry.versions.length - 1].content;
  }

  /** Head version metadata for `path`. */
  head(path: string): Version | null {
    const entry = this.entries.get(path);
    if (!entry || entry.versions.length === 0) return null;
    return entry.versions[entry.versions.length - 1];
  }

  exists(path: string): boolean {
    return this.head(path) !== null;
  }

  /** Full append-only version history (oldest first). */
  history(path: string): Version[] {
    return [...(this.entries.get(path)?.versions ?? [])];
  }

  /** Restore `path` to a prior version index by appending it as a new head
   *  (history stays append-only). Returns the new head, or null if invalid. */
  restore(path: string, index: number, message = "restore"): Version | null {
    const entry = this.entries.get(path);
    if (!entry || index < 0 || index >= entry.versions.length) return null;
    return this.write(path, entry.versions[index].content, message);
  }

  /** Read a specific version by CID (across all paths). */
  readByCid(path: string, cid: string): string | null {
    const entry = this.entries.get(path);
    return entry?.versions.find((v) => v.cid === cid)?.content ?? null;
  }

  delete(path: string): boolean {
    if (!this.entries.has(path)) return false;
    this.entries.delete(path);
    this.notify(path, { path, kind: "deleted" });
    return true;
  }

  copy(from: string, to: string): Version | null {
    const content = this.read(from);
    if (content === null) return null;
    return this.write(to, content, `copy from ${from}`);
  }

  /** List paths under a prefix (folder-style). */
  list(prefix = ""): string[] {
    const out: string[] = [];
    for (const [path, entry] of this.entries) {
      if (entry.versions.length > 0 && path.startsWith(prefix)) out.push(path);
    }
    return out.sort();
  }

  // ── subscriptions ──────────────────────────────────────────────────────────

  /** Subscribe to changes at an exact path or any path under a `prefix/`.
   *  Returns an unsubscribe function. */
  subscribe(pathOrPrefix: string, cb: Subscriber): () => void {
    let set = this.subscribers.get(pathOrPrefix);
    if (!set) {
      set = new Set();
      this.subscribers.set(pathOrPrefix, set);
    }
    set.add(cb);
    return () => set?.delete(cb);
  }

  private notify(path: string, ev: ChangeEvent): void {
    for (const [key, set] of this.subscribers) {
      if (key === path || (key.endsWith("/") && path.startsWith(key))) {
        for (const cb of set) cb(ev);
      }
    }
  }
}
