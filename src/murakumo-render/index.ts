// @etzhayyim/kami-nv-compat/murakumo-render
//
// Clean-room cloud render-farm engine (murakumo-render) — the canonical KAMI
// backend behind `nv-compat/omni-cloud`. Job queue over the kami-rt /
// kami-rtx renderers. Renders on the Murakumo fleet only (ADR-2605215000),
// never a commercial cloud.
//
// ADR-2605261800 §D6 / D10.4 murakumo-render.

export {
  type RenderMode,
  type RenderJobSpec,
  type JobStatus,
  type RenderJob,
  RenderFarm,
  turntableCameras,
} from "./farm.js";
