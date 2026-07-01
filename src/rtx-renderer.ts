// @etzhayyim/kami-nv-compat/rtx-renderer
//
// Drop-in NVIDIA RTX Renderer API-compat facade. Mirrors the documented
// public shape of the Omniverse RTX path-traced renderer (a renderer object
// configured with render settings, fed a scene + camera, producing a
// framebuffer) so existing RTX/Hydra-render host code ports to KAMI via
// import-path-only changes — e.g.
//
//     import { createRenderer, RtxRenderMode } from "@etzhayyim/kami-nv-compat/rtx-renderer";
//
//     const r = createRenderer({ mode: RtxRenderMode.PATH_TRACED, samplesPerPixel: 64 });
//     const scene = r.createScene(meshes, materials);
//     const fb = await r.render(scene, camera, 512, 512);   // WebGPU, CPU fallback
//
// Backed by the clean-room kami-rtx Monte-Carlo path tracer (kami-rt/pathtrace)
// — the R1.2 `kami-rtx-native` path of ADR-2605261800 D10.4 (a from-scratch
// path tracer on kami-rt + WGSL, the fallback to the Mitsuba 3 wgpu upstream
// route D3).
//
// Clean-room: this re-implements the RTX Renderer *public API shape* (Google
// v. Oracle, 593 U.S. ___ (2021)). No RTX / OptiX / Mitsuba source, headers,
// or SDK binaries are used. The canonical engine has a distinct name —
// `kami-rtx` (see NV_COMPAT_MAP).
//
// Trademark: NVIDIA® and RTX® are trademarks of NVIDIA Corporation; this
// project is not affiliated with or endorsed by NVIDIA.
//
// ADR-2605261800 §D1/D6, R1.2 RTX Renderer surface.

import {
  type Camera,
  type Material,
  type PathScene,
  type PathSettings,
  type Vec3,
  buildPathScene,
  pathTrace,
  pathTraceCPU,
} from "./kami-rt/index.js";

export type { Camera, Material, Vec3 } from "./kami-rt/index.js";
export { material, lookAt, buildPathScene } from "./kami-rt/index.js";

/** Rendering mode. `REAL_TIME` simply caps samples/bounces for interactivity;
 *  both modes run the same unbiased path tracer (no separate raster path). */
export enum RtxRenderMode {
  PATH_TRACED = "path_traced",
  REAL_TIME = "real_time",
}

export interface RtxRenderSettings {
  mode: RtxRenderMode;
  /** Samples per pixel (accumulated per render call). */
  samplesPerPixel: number;
  /** Max path length (bounces). */
  maxBounces: number;
  /** Background / dome radiance returned when a ray escapes. */
  background: Vec3;
  /** Reserved: AI denoiser toggle. The kami-rtx native denoiser is not yet
   *  wired, so this is accepted for API parity and currently a no-op. */
  denoise: boolean;
}

export function defaultRenderSettings(): RtxRenderSettings {
  return {
    mode: RtxRenderMode.PATH_TRACED,
    samplesPerPixel: 64,
    maxBounces: 6,
    background: [0, 0, 0],
    denoise: false,
  };
}

/** Opaque scene handle (a built kami-rtx path scene). */
export interface RtxScene {
  readonly _scene: PathScene;
  readonly triangleCount: number;
}

export interface RtxRenderResult {
  framebuffer: Float32Array;
  backend: "webgpu" | "cpu";
  width: number;
  height: number;
  samplesPerPixel: number;
}

function toPathSettings(s: RtxRenderSettings): PathSettings {
  // REAL_TIME clamps the sample/bounce budget for interactivity.
  const rt = s.mode === RtxRenderMode.REAL_TIME;
  return {
    samplesPerPixel: rt ? Math.min(s.samplesPerPixel, 4) : s.samplesPerPixel,
    maxBounces: rt ? Math.min(s.maxBounces, 3) : s.maxBounces,
    background: s.background,
  };
}

/** RTX-Renderer-shaped object backed by kami-rtx. */
export class RtxRenderer {
  readonly settings: RtxRenderSettings;

  constructor(settings: Partial<RtxRenderSettings> = {}) {
    this.settings = { ...defaultRenderSettings(), ...settings };
  }

  /** Build a render scene from triangle meshes + a parallel material list.
   *  Each entry of `meshes` is a triangle `[v0, v1, v2]`. */
  createScene(meshes: readonly Vec3[][], materials: readonly Material[]): RtxScene {
    const scene = buildPathScene(meshes, materials);
    return { _scene: scene, triangleCount: scene.soup.count };
  }

  /** Render to a `width × height` RGBA-float framebuffer. Uses WebGPU when a
   *  device is available (or one is passed via `device`), else the CPU path
   *  tracer. */
  async render(
    scene: RtxScene,
    camera: Camera,
    width: number,
    height: number,
    device?: unknown,
  ): Promise<RtxRenderResult> {
    const settings = toPathSettings(this.settings);
    const res = await pathTrace(scene._scene, camera, width, height, { settings, device });
    return { ...res, samplesPerPixel: settings.samplesPerPixel };
  }

  /** Synchronous CPU render (deterministic; useful for tests / headless). */
  renderSync(scene: RtxScene, camera: Camera, width: number, height: number): RtxRenderResult {
    const settings = toPathSettings(this.settings);
    const res = pathTraceCPU(scene._scene, camera, width, height, settings);
    return { ...res, samplesPerPixel: settings.samplesPerPixel };
  }
}

export function createRenderer(settings: Partial<RtxRenderSettings> = {}): RtxRenderer {
  return new RtxRenderer(settings);
}

// ── compat-map metadata ───────────────────────────────────────────────────

/** Canonical KAMI engine name behind this facade (per NV_COMPAT_MAP). */
export const KAMI_ENGINE = "kami-rtx";
export const ADR = "ADR-2605261800";
