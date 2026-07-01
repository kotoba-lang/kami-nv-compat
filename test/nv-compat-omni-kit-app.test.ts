/**
 * nv-compat omni.kit.app / amenominaka validation.
 *
 * Exercises the clean-room Kit application shell: the extension.toml parser,
 * the IExt lifecycle + dependency-ordered Application startup/shutdown, the
 * undoable command stack, and the KamiViewerExtension that hosts the
 * kami-usd → kami-rt pipeline as a Kit extension (the R1.4 integration).
 *
 *     pnpm exec vitest run test/nv-compat-omni-kit-app.test.ts
 *
 * ADR-2605261800 §D1/D6, R1.4 omni-kit-app surface.
 */

import { describe, it, expect } from "vitest";
import {
  Application,
  CommandStack,
  IExt,
  SetAttributeCommand,
  parseExtensionToml,
} from "../src/amenominaka/index.js";
import { app, commands, KamiViewerExtension } from "../src/omni-kit-app.js";

const EXT_TOML = `
[package]
title = "My Extension"
version = "1.2.3"
description = "demo"
keywords = ["kami", "kit"]

[dependencies]
"omni.usd" = {}
"omni.kit.uiapp" = { version = "1.0" }

[[python.module]]
name = "my_ext"

[[python.module]]
name = "my_ext.tools"
`;

describe("extension.toml parser", () => {
  it("parses package fields, dependencies, and python modules", () => {
    const toml = parseExtensionToml(EXT_TOML);
    expect(toml.title).toBe("My Extension");
    expect(toml.version).toBe("1.2.3");
    expect(toml.keywords).toEqual(["kami", "kit"]);
    expect(Object.keys(toml.dependencies).sort()).toEqual(["omni.kit.uiapp", "omni.usd"]);
    expect(toml.dependencies["omni.kit.uiapp"]).toEqual({ version: "1.0" });
    expect(toml.pythonModules.map((m) => m.name)).toEqual(["my_ext", "my_ext.tools"]);
  });

  it("tolerates comments and blank lines", () => {
    const toml = parseExtensionToml(`# header\n[package]\ntitle = "X" # trailing\n\nversion = "2.0.0"`);
    expect(toml.title).toBe("X");
    expect(toml.version).toBe("2.0.0");
  });
});

describe("Application lifecycle + dependency ordering", () => {
  class RecordingExt extends IExt {
    constructor(private readonly log: string[], private readonly id: string) {
      super();
    }
    onStartup(): void {
      this.log.push(`up:${this.id}`);
    }
    onShutdown(): void {
      this.log.push(`down:${this.id}`);
    }
  }

  it("starts parents before children and shuts down in reverse", () => {
    const log: string[] = [];
    const a = new Application();
    // child depends on base; ui depends on base.
    a.registerExtension("base", new RecordingExt(log, "base"));
    a.registerExtension("child", new RecordingExt(log, "child"), parseExtensionToml(`[dependencies]\n"base" = {}`));
    a.registerExtension("ui", new RecordingExt(log, "ui"), parseExtensionToml(`[dependencies]\n"base" = {}`));
    const order = a.startupAll();
    expect(order[0]).toBe("base"); // base first
    expect(order).toContain("child");
    expect(a.numStarted()).toBe(3);
    const down = a.shutdownAll();
    expect(down[down.length - 1]).toBe("base"); // base last to shut down
    expect(a.numStarted()).toBe(0);
  });

  it("throws on a dependency cycle", () => {
    const a = new Application();
    a.registerExtension("x", new IExt(), parseExtensionToml(`[dependencies]\n"y" = {}`));
    a.registerExtension("y", new IExt(), parseExtensionToml(`[dependencies]\n"x" = {}`));
    expect(() => a.startupAll()).toThrow(/Cyclic/);
  });

  it("update() ticks started extensions", () => {
    let ticks = 0;
    class Ticker extends IExt {
      onUpdate(): void {
        ticks++;
      }
    }
    const a = new Application();
    a.registerExtension("t", new Ticker());
    a.startupAll();
    a.update(0.016);
    a.update(0.016);
    expect(ticks).toBe(2);
  });

  it("getApp() returns a singleton", () => {
    expect(app.getApp()).toBe(app.getApp());
    app._resetApp();
  });
});

describe("command stack (undo / redo)", () => {
  it("executes, undoes, and redoes attribute mutations", () => {
    const target: Record<string, unknown> = { color: "red" };
    const stack = new CommandStack();
    stack.execute(new SetAttributeCommand(target, "color", "blue"));
    expect(target.color).toBe("blue");
    stack.undo();
    expect(target.color).toBe("red");
    stack.redo();
    expect(target.color).toBe("blue");
  });

  it("restores an absent key by removing it on undo", () => {
    const target: Record<string, unknown> = {};
    const stack = new CommandStack();
    stack.execute(new SetAttributeCommand(target, "added", 42));
    expect(target.added).toBe(42);
    stack.undo();
    expect("added" in target).toBe(false);
  });

  it("executing a new command clears the redo stack", () => {
    const target: Record<string, unknown> = { v: 0 };
    const stack = new CommandStack();
    stack.execute(new SetAttributeCommand(target, "v", 1));
    stack.undo();
    expect(stack.canRedo()).toBe(true);
    stack.execute(new SetAttributeCommand(target, "v", 2));
    expect(stack.canRedo()).toBe(false);
    expect(target.v).toBe(2);
  });

  it("the global commands namespace drives a shared stack", () => {
    commands._resetStack();
    const target: Record<string, unknown> = { n: 1 };
    commands.execute(new commands.SetAttributeCommand(target, "n", 9));
    expect(target.n).toBe(9);
    commands.undo();
    expect(target.n).toBe(1);
    commands._resetStack();
  });
});

describe("KamiViewerExtension (kami-usd → kami-rt hosted in a Kit app)", () => {
  const USDA = `#usda 1.0
def Xform "World" {
    def Mesh "tri" {
        point3f[] points = [(-1,-1,0),(1,-1,0),(0,1,0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
        color3f[] primvars:displayColor = [(0.8,0.2,0.2)]
    }
}`;

  it("loads a USD stage on startup and renders frames on update", () => {
    const a = new Application();
    const ext = new KamiViewerExtension(USDA, 32, 32);
    a.registerExtension("kami.viewer", ext);
    a.startupAll();
    expect(ext.scene).not.toBeNull();
    expect(ext.scene!.soup.count).toBe(1); // one triangle
    a.update(0.016);
    a.update(0.016);
    expect(ext.frameCount).toBe(2);
    expect(ext.lastFrame!.length).toBe(32 * 32 * 4);
    a.shutdownAll();
    expect(ext.scene).toBeNull();
  });
});
