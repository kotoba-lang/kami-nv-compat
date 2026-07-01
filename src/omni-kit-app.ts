// @etzhayyim/kami-nv-compat/omni-kit-app
//
// Drop-in `omni.kit.app` / `omni.kit.commands` API-compat facade — the Kit
// application framework. Mirrors the documented surface (the Application
// singleton via get_app(), the IExt extension lifecycle + extension.toml, and
// the undoable command stack), so Kit extensions port to KAMI via import-path-
// only changes — e.g.
//
//     import { app, commands } from "@etzhayyim/kami-nv-compat/omni-kit-app";
//
//     class MyExt extends app.IExt {
//       onStartup(id) { ... }
//       onShutdown() { ... }
//     }
//     app.getApp().registerExtension("my.ext", new MyExt(), toml);
//     app.getApp().startupAll();
//
// The bundled KamiViewerExtension wires the rest of the nv-compat stack
// (kami-usd → kami-rt) into a hosted Kit extension, demonstrating an end-to-end
// app: load a USD stage on startup, render a frame each update.
//
// Backed by the clean-room amenominaka engine. No Kit source/binaries; from-
// spec reproduction (Google v. Oracle, 593 U.S. ___ (2021)). Canonical engine:
// amenominaka.
//
// Trademark: NVIDIA® / Omniverse® / Kit are trademarks of NVIDIA Corporation;
// API-compat identifiers only.
//
// ADR-2605261800 §D1/D6, R1.4 omni-kit-app surface.

import {
  Application,
  Command,
  CommandStack,
  IExt,
  SetAttributeCommand,
  execute,
  getApp,
  parseExtensionToml,
  redo,
  undo,
  _resetApp,
  _resetStack,
} from "./amenominaka/index.js";
import { type Scene, type Camera, lookAt, traceImageCPU } from "./kami-rt/index.js";
import { Stage, stageToScene } from "./omni-usd.js";

export type { ExtensionToml } from "./amenominaka/index.js";

/** `omni.kit.app` namespace. */
export const app = {
  Application,
  IExt,
  getApp,
  parseExtensionToml,
  _resetApp,
};

/** `omni.kit.commands` namespace. */
export const commands = {
  Command,
  CommandStack,
  SetAttributeCommand,
  execute,
  undo,
  redo,
  _resetStack,
};

// ── KAMI viewer extension (end-to-end integration) ──────────────────────────

/** A Kit extension that loads a USD stage on startup and renders it on each
 *  update — the nv-compat stack (kami-usd → kami-rt) hosted as a Kit
 *  extension. Demonstrates the Application lifecycle driving the renderer. */
export class KamiViewerExtension extends IExt {
  scene: Scene | null = null;
  camera: Camera;
  lastFrame: Float32Array | null = null;
  frameCount = 0;

  constructor(
    private readonly usda: string,
    private readonly width = 64,
    private readonly height = 64,
    eye: [number, number, number] = [0, 0, 4],
    target: [number, number, number] = [0, 0, 0],
  ) {
    super();
    this.camera = lookAt(eye, target, [0, 1, 0], 45, width / height);
  }

  onStartup(_extId: string): void {
    this.scene = stageToScene(Stage.Open(this.usda));
  }

  onUpdate(_dt: number): void {
    if (!this.scene) return;
    this.lastFrame = traceImageCPU(this.scene, this.camera, this.width, this.height).framebuffer;
    this.frameCount++;
  }

  onShutdown(): void {
    this.scene = null;
    this.lastFrame = null;
  }
}

export const KAMI_ENGINE = "amenominaka";
export const ADR = "ADR-2605261800";
