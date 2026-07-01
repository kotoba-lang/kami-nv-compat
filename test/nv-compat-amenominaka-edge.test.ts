/**
 * nv-compat amenominaka (Kit app) parser + lifecycle + command-stack edges.
 *
 * Boundary coverage for the Kit application shell: extension.toml parsing
 * corner cases (# inside strings, arrays, inline tables, missing package),
 * Application register/unregister/update lifecycle, and the bounded undo/redo
 * command stack (history cap, clear, empty undo/redo).
 *
 *     pnpm exec vitest run test/nv-compat-amenominaka-edge.test.ts
 *
 * ADR-2605261800 §D6 / D10.4 amenominaka.
 */

import { describe, it, expect } from "vitest";
import {
  Application,
  CommandStack,
  IExt,
  SetAttributeCommand,
  parseExtensionToml,
} from "../src/amenominaka/index.js";

describe("extension.toml parser corner cases", () => {
  it("keeps a # that lives inside a string value", () => {
    const t = parseExtensionToml(`[package]\ntitle = "tag #1 release" # trailing comment`);
    expect(t.title).toBe("tag #1 release");
  });

  it("parses arrays, inline tables, and all package fields", () => {
    const t = parseExtensionToml(`
      [package]
      title = "Ext"
      version = "3.0.0"
      description = "d"
      category = "tools"
      keywords = ["a", "b", "c"]
      authors = ["x", "y"]
      repository = "https://example.test/repo"

      [dependencies]
      "omni.usd" = { version = "1.0", optional = true }
    `);
    expect(t.category).toBe("tools");
    expect(t.keywords).toEqual(["a", "b", "c"]);
    expect(t.authors).toEqual(["x", "y"]);
    expect(t.repository).toBe("https://example.test/repo");
    expect(t.dependencies["omni.usd"]).toEqual({ version: "1.0", optional: true });
  });

  it("applies defaults when [package] is absent", () => {
    const t = parseExtensionToml(`[dependencies]\n"omni.kit.uiapp" = {}`);
    expect(t.title).toBe("");
    expect(t.version).toBe("0.1.0");
    expect(t.keywords).toEqual([]);
    expect(Object.keys(t.dependencies)).toEqual(["omni.kit.uiapp"]);
  });

  it("collects repeated [[python.module]] tables in order", () => {
    const t = parseExtensionToml(`[[python.module]]\nname = "a"\n[[python.module]]\nname = "b"\nentry = "main"`);
    expect(t.pythonModules.map((m) => m.name)).toEqual(["a", "b"]);
    expect(t.pythonModules[1].entry).toBe("main");
  });
});

describe("Application register / unregister / update lifecycle", () => {
  class Rec extends IExt {
    started = false;
    ticks = 0;
    constructor(private readonly log: string[], readonly id: string) {
      super();
    }
    onStartup(): void {
      this.started = true;
      this.log.push(`up:${this.id}`);
    }
    onUpdate(): void {
      this.ticks++;
    }
    onShutdown(): void {
      this.started = false;
      this.log.push(`down:${this.id}`);
    }
  }

  it("re-registering an id replaces it (count stays 1)", () => {
    const a = new Application();
    a.registerExtension("e", new IExt());
    a.registerExtension("e", new IExt());
    expect(a.numExtensions()).toBe(1);
  });

  it("unregister shuts down a started extension and removes it", () => {
    const log: string[] = [];
    const a = new Application();
    const ext = new Rec(log, "e");
    a.registerExtension("e", ext);
    a.startupAll();
    expect(a.numStarted()).toBe(1);
    a.unregisterExtension("e");
    expect(log).toContain("down:e");
    expect(a.getExtension("e")).toBeUndefined();
    expect(a.numExtensions()).toBe(0);
    a.unregisterExtension("missing"); // no-op, no throw
  });

  it("update only ticks started extensions", () => {
    const log: string[] = [];
    const a = new Application();
    const ext = new Rec(log, "e");
    a.registerExtension("e", ext);
    a.update(0.1); // not started yet → no tick
    expect(ext.ticks).toBe(0);
    a.startupAll();
    a.update(0.1);
    a.update(0.1);
    expect(ext.ticks).toBe(2);
  });

  it("getExtensionIds reflects registration", () => {
    const a = new Application();
    a.registerExtension("x", new IExt());
    a.registerExtension("y", new IExt());
    expect(a.getExtensionIds().sort()).toEqual(["x", "y"]);
  });
});

describe("CommandStack bounded history + empty operations", () => {
  it("undo/redo on an empty stack return null", () => {
    const s = new CommandStack();
    expect(s.undo()).toBeNull();
    expect(s.redo()).toBeNull();
    expect(s.canUndo()).toBe(false);
    expect(s.canRedo()).toBe(false);
  });

  it("drops the oldest command past the history cap (no longer undoable)", () => {
    const target: Record<string, unknown> = { v: 0 };
    const s = new CommandStack(2); // cap = 2
    s.execute(new SetAttributeCommand(target, "v", 1));
    s.execute(new SetAttributeCommand(target, "v", 2));
    s.execute(new SetAttributeCommand(target, "v", 3)); // c1 dropped
    expect(target.v).toBe(3);
    expect(s.undo()).not.toBeNull(); // undo c3 → 2
    expect(target.v).toBe(2);
    expect(s.undo()).not.toBeNull(); // undo c2 → 1
    expect(target.v).toBe(1);
    expect(s.undo()).toBeNull(); // c1 was dropped → nothing left
    expect(target.v).toBe(1);
  });

  it("history() lists command names; clear() empties the stack", () => {
    const target: Record<string, unknown> = {};
    const s = new CommandStack();
    s.execute(new SetAttributeCommand(target, "a", 1));
    s.execute(new SetAttributeCommand(target, "b", 2));
    expect(s.history()).toEqual(["SetAttribute", "SetAttribute"]);
    s.clear();
    expect(s.canUndo()).toBe(false);
    expect(s.history()).toEqual([]);
  });
});
