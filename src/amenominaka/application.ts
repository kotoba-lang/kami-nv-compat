// amenominaka — clean-room Kit Application (extension lifecycle host).
//
// Mirrors `omni.kit.app.IApp` / `omni.kit.app.get_app()`: a singleton app shell
// that owns registered IExt instances and dispatches startup / update /
// shutdown in dependency-respecting order. Dependencies are declared in each
// extension's ExtensionToml; startup is a Kahn topological sort (parents
// before children), shutdown is reverse; a cycle throws.
//
// ADR-2605261800 §D6 / D10.4 amenominaka.

import { type ExtensionToml, IExt } from "./extension.js";

interface RegisteredExt {
  extId: string;
  instance: IExt;
  toml?: ExtensionToml;
  started: boolean;
}

export class Application {
  private readonly extensions = new Map<string, RegisteredExt>();
  private startupLog: string[] = [];
  private shutdownLog: string[] = [];

  registerExtension(extId: string, instance: IExt, toml?: ExtensionToml): void {
    this.extensions.set(extId, { extId, instance, toml, started: false });
  }

  unregisterExtension(extId: string): void {
    const ext = this.extensions.get(extId);
    if (!ext) return;
    if (ext.started) {
      ext.instance.onShutdown();
      ext.started = false;
    }
    this.extensions.delete(extId);
  }

  getExtension(extId: string): IExt | undefined {
    return this.extensions.get(extId)?.instance;
  }
  getExtensionIds(): string[] {
    return [...this.extensions.keys()];
  }
  numExtensions(): number {
    return this.extensions.size;
  }
  numStarted(): number {
    let n = 0;
    for (const e of this.extensions.values()) if (e.started) n++;
    return n;
  }

  /** Kahn topological order over the depends-on relation (only registered
   *  dependencies count). Throws on a cycle. Ties broken alphabetically for
   *  determinism. */
  private topologicalOrder(): string[] {
    const depsMap = new Map<string, string[]>();
    for (const [extId, ext] of this.extensions) {
      const deps: string[] = [];
      if (ext.toml) {
        for (const depId of Object.keys(ext.toml.dependencies)) {
          if (this.extensions.has(depId)) deps.push(depId);
        }
      }
      depsMap.set(extId, deps);
    }
    const inDegree = new Map<string, number>();
    for (const [eid, deps] of depsMap) inDegree.set(eid, deps.length);
    const ready = [...inDegree.entries()].filter(([, d]) => d === 0).map(([e]) => e).sort();
    const order: string[] = [];
    while (ready.length) {
      const eid = ready.shift() as string;
      order.push(eid);
      for (const [other, deps] of depsMap) {
        if (deps.includes(eid)) {
          const d = (inDegree.get(other) ?? 0) - 1;
          inDegree.set(other, d);
          if (d === 0) ready.push(other);
        }
      }
      ready.sort();
    }
    if (order.length !== this.extensions.size) {
      throw new Error("Cyclic dependency in extensions; cannot order startup");
    }
    return order;
  }

  /** Fire onStartup for all registered extensions in dependency order. */
  startupAll(): string[] {
    const order = this.topologicalOrder();
    this.startupLog = [];
    for (const eid of order) {
      const ext = this.extensions.get(eid) as RegisteredExt;
      if (!ext.started) {
        ext.instance.onStartup(eid);
        ext.started = true;
        this.startupLog.push(eid);
      }
    }
    return [...this.startupLog];
  }

  /** Tick all started extensions (registration-order) with dt seconds. */
  update(dt: number): void {
    for (const ext of this.extensions.values()) {
      if (ext.started) ext.instance.onUpdate(dt);
    }
  }

  /** Fire onShutdown for all started extensions in REVERSE dependency order. */
  shutdownAll(): string[] {
    let order: string[];
    try {
      order = this.topologicalOrder();
    } catch {
      order = [...this.extensions.keys()];
    }
    this.shutdownLog = [];
    for (let i = order.length - 1; i >= 0; i--) {
      const ext = this.extensions.get(order[i]) as RegisteredExt;
      if (ext.started) {
        ext.instance.onShutdown();
        ext.started = false;
        this.shutdownLog.push(order[i]);
      }
    }
    return [...this.shutdownLog];
  }
}

// ── global singleton ─────────────────────────────────────────────────────────

let _globalApp: Application | null = null;

/** Return the global Application singleton (mirrors `omni.kit.app.get_app()`). */
export function getApp(): Application {
  if (_globalApp === null) _globalApp = new Application();
  return _globalApp;
}

/** Reset the global app (test helper; not in upstream Kit). */
export function _resetApp(): void {
  _globalApp = null;
}
