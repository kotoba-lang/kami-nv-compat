// murakumo-render — clean-room cloud render farm (Omniverse Cloud lineage).
//
// The canonical KAMI implementation behind `nv-compat/omni-cloud`. NVIDIA
// Omniverse Cloud offers managed/streamed rendering of Omniverse scenes; this
// module reproduces a render-FARM job model — submit a render job (a scene +
// one or more cameras + settings), the farm executes it via the kami
// renderers, and frames are retrieved or streamed back.
//
// Charter: per ADR-2605215000 the religious-corp compute path is the Murakumo
// fleet ONLY — no commercial GPU rental / vendor cloud. This farm executes
// locally (or on the Murakumo fleet), never a third-party cloud. The name
// `murakumo-render` is literal: it renders on Murakumo.
//
// Clean-room: from-spec job queue over kami-rt / kami-rtx. No Omniverse Cloud
// source/binaries. ADR-2605261800 §D6 / D10.4 murakumo-render.

import {
  type Camera,
  type PathScene,
  type PathSettings,
  type Scene,
  type ShadeParams,
  traceImageCPU,
  pathTraceCPU,
} from "../kami-rt/index.js";

export type RenderMode = "rtx" | "pathtrace";

export interface RenderJobSpec {
  /** Ray-trace scene (mode "rtx") or path-trace scene (mode "pathtrace"). */
  scene: Scene | PathScene;
  /** One frame is rendered per camera (batch / turntable / multi-view). */
  cameras: Camera[];
  width: number;
  height: number;
  mode: RenderMode;
  shade?: ShadeParams;
  pathSettings?: PathSettings;
  /** Optional label for status/UX. */
  label?: string;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface RenderJob {
  id: string;
  spec: RenderJobSpec;
  status: JobStatus;
  /** 0..1 over the camera batch. */
  progress: number;
  frames: Float32Array[];
  error?: string;
}

function isPathScene(s: Scene | PathScene): s is PathScene {
  return (s as PathScene).mats !== undefined;
}

/** Managed render farm: submit jobs, run them (synchronously here; on the
 *  Murakumo fleet in deployment), poll status, retrieve frames. */
export class RenderFarm {
  private readonly jobs = new Map<string, RenderJob>();
  private seq = 0;

  /** Submit a render job; returns its id. The job starts `queued`. */
  submit(spec: RenderJobSpec): string {
    const id = `job-${this.seq++}`;
    this.jobs.set(id, { id, spec, status: "queued", progress: 0, frames: [] });
    return id;
  }

  get(id: string): RenderJob | undefined {
    return this.jobs.get(id);
  }
  status(id: string): JobStatus | undefined {
    return this.jobs.get(id)?.status;
  }
  result(id: string): Float32Array[] | undefined {
    const job = this.jobs.get(id);
    return job?.status === "done" ? job.frames : undefined;
  }
  pending(): string[] {
    return [...this.jobs.values()].filter((j) => j.status === "queued").map((j) => j.id);
  }

  /** Render one job to completion. `onFrame` streams each frame as it lands. */
  run(id: string, onFrame?: (frame: Float32Array, index: number) => void): RenderJob {
    const job = this.jobs.get(id);
    if (!job) throw new Error(`unknown job ${id}`);
    job.status = "running";
    job.frames = [];
    try {
      const { spec } = job;
      spec.cameras.forEach((cam, i) => {
        let fb: Float32Array;
        if (spec.mode === "pathtrace") {
          if (!isPathScene(spec.scene)) throw new Error("pathtrace mode requires a PathScene");
          fb = pathTraceCPU(spec.scene, cam, spec.width, spec.height, spec.pathSettings).framebuffer;
        } else {
          if (isPathScene(spec.scene)) throw new Error("rtx mode requires a ray Scene");
          fb = traceImageCPU(spec.scene, cam, spec.width, spec.height, spec.shade).framebuffer;
        }
        job.frames.push(fb);
        job.progress = (i + 1) / spec.cameras.length;
        onFrame?.(fb, i);
      });
      job.status = "done";
    } catch (e) {
      job.status = "error";
      job.error = e instanceof Error ? e.message : String(e);
    }
    return job;
  }

  /** Run all queued jobs (FIFO). */
  runAll(): RenderJob[] {
    return this.pending().map((id) => this.run(id));
  }
}

/** Build a turntable of `n` cameras orbiting `target` at `radius` / `height`
 *  — a common cloud-render batch (one frame per camera). */
export function turntableCameras(
  makeCamera: (eye: [number, number, number], target: [number, number, number]) => Camera,
  target: [number, number, number],
  radius: number,
  height: number,
  n: number,
): Camera[] {
  const out: Camera[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    out.push(makeCamera([target[0] + radius * Math.cos(a), target[1] + height, target[2] + radius * Math.sin(a)], target));
  }
  return out;
}
