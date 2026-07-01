// utsushimi — Replicator writers (BasicWriter / CocoWriter / KittiWriter).
//
// Mirrors `omni.replicator.core.writers.*`. The Replicator G5 gate is
// "BasicWriter emits the same JSON schema as upstream (diff = 0)"; these
// writers reproduce the documented COCO-2017 / Kitti on-disk schemas exactly.
//
// Portability: instead of writing to a filesystem (which would not run in a
// browser / WASM host), each writer accumulates in memory and exposes
// `finalize()` (the structured dataset) + `toFiles()` (a path→content map a
// caller can persist with byte-identical layout). Per-object 2D bounding boxes
// come from the render bridge when present, else the upstream full-image
// placeholder.
//
// ADR-2605261800 §D6 / D10.4 utsushimi.

import { type PrimSpec, type Semantic } from "./randomize.js";

/** A scene primitive plus an optional real 2D bbox `[x, y, w, h]`
 *  (from the camera-projection render bridge). */
export interface AnnotatedPrim extends PrimSpec {
  bbox2d?: [number, number, number, number];
}

export interface FrameSample {
  frame: number;
  primitives: AnnotatedPrim[];
}

export interface WriterInit {
  outputDir?: string;
  rgb?: boolean;
  boundingBox2dTight?: boolean;
  semanticSegmentation?: boolean;
  imageWidth?: number;
  imageHeight?: number;
}

export interface Writer {
  initialize(init: WriterInit): void;
  attach(cameras: unknown[]): void;
  writeFrame(frameIndex: number, sample: FrameSample): void;
  finalize(): unknown;
  toFiles(): Record<string, string>;
}

function pad(n: number, width: number): string {
  return String(n).padStart(width, "0");
}

function classOf(prim: AnnotatedPrim): string | null {
  for (const s of prim.semantics ?? ([] as Semantic[])) {
    if (Array.isArray(s) && s.length === 2 && s[0] === "class") return s[1];
  }
  return null;
}

// ── BasicWriter ──────────────────────────────────────────────────────────────

interface BasicFrame {
  frame: number;
  cameras: unknown[];
  sample: FrameSample;
}

export class BasicWriter implements Writer {
  private outputDir = "_out";
  private cameras: unknown[] = [];
  private readonly frames: BasicFrame[] = [];

  initialize(init: WriterInit): void {
    this.outputDir = init.outputDir ?? "_out";
  }
  attach(cameras: unknown[]): void {
    this.cameras = [...cameras];
  }
  writeFrame(frameIndex: number, sample: FrameSample): void {
    this.frames.push({ frame: frameIndex, cameras: this.cameras, sample });
  }
  finalize(): { frames: BasicFrame[] } {
    return { frames: this.frames };
  }
  toFiles(): Record<string, string> {
    const files: Record<string, string> = {};
    for (const f of this.frames) {
      files[`${this.outputDir}/frame_${pad(f.frame, 4)}.json`] = JSON.stringify(
        { frame: f.frame, cameras: f.cameras, sample: f.sample },
        null,
        2,
      );
    }
    return files;
  }
}

// ── CocoWriter (COCO-2017 object-detection JSON) ─────────────────────────────

export interface CocoImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}
export interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: [number, number, number, number];
  area: number;
  iscrowd: 0;
}
export interface CocoCategory {
  id: number;
  name: string;
  supercategory: string;
}
export interface CocoDataset {
  info: { description: string; version: string; year: number };
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

export class CocoWriter implements Writer {
  private outputDir = "_out";
  private width = 640;
  private height = 480;
  private readonly images: CocoImage[] = [];
  private readonly annotations: CocoAnnotation[] = [];
  private readonly categories = new Map<string, number>();
  private readonly perFrame: Record<string, string> = {};
  private nextAnnId = 0;
  private nextCatId = 0;

  initialize(init: WriterInit): void {
    this.outputDir = init.outputDir ?? "_out";
    this.width = init.imageWidth ?? 640;
    this.height = init.imageHeight ?? 480;
  }
  attach(): void {
    /* cameras tracked by the orchestrator */
  }

  private categoryId(name: string): number {
    let id = this.categories.get(name);
    if (id === undefined) {
      id = this.nextCatId++;
      this.categories.set(name, id);
    }
    return id;
  }

  writeFrame(frameIndex: number, sample: FrameSample): void {
    this.images.push({
      id: frameIndex,
      file_name: `rgb_${pad(frameIndex, 4)}.png`,
      width: this.width,
      height: this.height,
    });
    this.perFrame[`${this.outputDir}/rgb_${pad(frameIndex, 4)}.json`] = JSON.stringify(
      { frame: frameIndex, sample },
      null,
      2,
    );
    for (const prim of sample.primitives) {
      const cls = classOf(prim);
      if (cls === null) continue;
      const catId = this.categoryId(cls);
      const bbox = prim.bbox2d ?? [0, 0, this.width, this.height];
      this.annotations.push({
        id: this.nextAnnId++,
        image_id: frameIndex,
        category_id: catId,
        bbox,
        area: bbox[2] * bbox[3],
        iscrowd: 0,
      });
    }
  }

  finalize(): CocoDataset {
    const categories: CocoCategory[] = [...this.categories.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([name, id]) => ({ id, name, supercategory: "object" }));
    return {
      info: { description: "utsushimi (nv-compat) COCO output", version: "1.0", year: 2026 },
      images: this.images,
      annotations: this.annotations,
      categories,
    };
  }

  toFiles(): Record<string, string> {
    return {
      ...this.perFrame,
      [`${this.outputDir}/annotations.json`]: JSON.stringify(this.finalize(), null, 2),
    };
  }
}

// ── KittiWriter (Kitti 3D object-detection label .txt) ───────────────────────

export class KittiWriter implements Writer {
  private outputDir = "_out";
  private width = 1242;
  private height = 375;
  private readonly labels: Record<string, string> = {};
  private readonly images: Record<string, string> = {};

  initialize(init: WriterInit): void {
    this.outputDir = init.outputDir ?? "_out";
    this.width = init.imageWidth ?? 1242;
    this.height = init.imageHeight ?? 375;
  }
  attach(): void {
    /* no-op */
  }

  private formatLine(prim: AnnotatedPrim): string | null {
    const cls = classOf(prim);
    if (cls === null) return null;
    const bbox = prim.bbox2d ?? [0, 0, this.width, this.height];
    const [h, w, l] = [1, 1, 1];
    const pos = prim.position ?? [0, 0, 10];
    const ry = prim.rotation_y ?? 0;
    return (
      `${cls} 0.00 0 0.00 ` +
      `${bbox[0].toFixed(2)} ${bbox[1].toFixed(2)} ${(bbox[0] + bbox[2]).toFixed(2)} ${(bbox[1] + bbox[3]).toFixed(2)} ` +
      `${h.toFixed(2)} ${w.toFixed(2)} ${l.toFixed(2)} ` +
      `${pos[0].toFixed(2)} ${pos[1].toFixed(2)} ${pos[2].toFixed(2)} ${ry.toFixed(2)}`
    );
  }

  writeFrame(frameIndex: number, sample: FrameSample): void {
    const lines: string[] = [];
    for (const prim of sample.primitives) {
      const line = this.formatLine(prim);
      if (line !== null) lines.push(line);
    }
    this.labels[`${this.outputDir}/label_2/${pad(frameIndex, 6)}.txt`] =
      lines.join("\n") + (lines.length ? "\n" : "");
    this.images[`${this.outputDir}/image_2/${pad(frameIndex, 6)}.json`] = JSON.stringify({
      frame: frameIndex,
      placeholder: true,
    });
  }

  finalize(): { labels: Record<string, string> } {
    return { labels: this.labels };
  }
  toFiles(): Record<string, string> {
    return { ...this.labels, ...this.images };
  }
}

// ── WriterRegistry ───────────────────────────────────────────────────────────

type WriterCtor = new () => Writer;

const _registry = new Map<string, WriterCtor>([
  ["BasicWriter", BasicWriter],
  ["CocoWriter", CocoWriter],
  ["KittiWriter", KittiWriter],
]);

export const WriterRegistry = {
  get(name: string): Writer {
    const ctor = _registry.get(name);
    if (!ctor) throw new Error(`unknown writer: ${name}`);
    return new ctor();
  },
  register(name: string, ctor: WriterCtor): void {
    _registry.set(name, ctor);
  },
};
