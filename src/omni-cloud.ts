// @etzhayyim/kami-nv-compat/omni-cloud
//
// Drop-in NVIDIA Omniverse Cloud API-compat facade — managed/streamed
// rendering of Omniverse scenes. Mirrors the documented job model (open a
// session, submit a render job, poll status, retrieve / stream frames) so
// cloud-render client code ports to KAMI via import-path-only changes — e.g.
//
//     import { OmniCloudSession } from "@etzhayyim/kami-nv-compat/omni-cloud";
//
//     const s = new OmniCloudSession();
//     const id = s.submitRender({ scene, cameras, width, height, mode: "rtx" });
//     s.process();                       // run queued jobs on the fleet
//     const frames = s.getResult(id);    // → Float32Array[] (one per camera)
//
// Backed by the clean-room murakumo-render farm over the kami renderers.
//
// Charter: per ADR-2605215000 the religious-corp compute path is the Murakumo
// fleet ONLY — NO commercial GPU rental / vendor cloud. This "cloud" renders on
// Murakumo (LiteLLM-gateway-adjacent fleet), never a third-party endpoint.
//
// Clean-room: from-spec session/job API. No Omniverse Cloud source/binaries
// (Google v. Oracle, 593 U.S. ___ (2021)). Canonical engine: murakumo-render.
//
// Trademark: NVIDIA® / Omniverse® are trademarks of NVIDIA Corporation;
// API-compat identifiers only.
//
// ADR-2605261800 §D1/D6, R1.7 omni-cloud surface.

import {
  type JobStatus,
  type RenderJob,
  type RenderJobSpec,
  RenderFarm,
} from "./murakumo-render/index.js";

export {
  type RenderMode,
  type RenderJobSpec,
  type JobStatus,
  type RenderJob,
  RenderFarm,
  turntableCameras,
} from "./murakumo-render/index.js";

/** Omniverse-Cloud-style session over the Murakumo render farm. */
export class OmniCloudSession {
  private readonly farm = new RenderFarm();

  /** Submit a render job; returns its id (status `queued`). */
  submitRender(spec: RenderJobSpec): string {
    return this.farm.submit(spec);
  }

  getStatus(id: string): JobStatus | undefined {
    return this.farm.status(id);
  }
  getResult(id: string): Float32Array[] | undefined {
    return this.farm.result(id);
  }
  getJob(id: string): RenderJob | undefined {
    return this.farm.get(id);
  }

  /** Run a single job, optionally streaming frames as they render. */
  render(id: string, onFrame?: (frame: Float32Array, index: number) => void): RenderJob {
    return this.farm.run(id, onFrame);
  }

  /** Process all queued jobs on the fleet. */
  process(): RenderJob[] {
    return this.farm.runAll();
  }
}

export const KAMI_ENGINE = "murakumo-render";
export const ADR = "ADR-2605261800";
