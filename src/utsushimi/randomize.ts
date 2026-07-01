// utsushimi — Replicator create / modify / randomize ops + resolve.
//
// Mirrors `omni.replicator.core.{create, modify, randomize}` and the `resolve`
// helper: scene-level domain-randomization ops are captured as tagged objects
// and materialized per-frame against a {@link Sampler}.
//
// ADR-2605261800 §D6 / D10.4 utsushimi.

import { type Dist, type Sampled, distribution, sample } from "./distribution.js";
import { type Sampler, globalSampler } from "./sampler.js";

// ── semantics + primitives ──────────────────────────────────────────────────

/** Replicator semantic tag: `["class", "cube"]`, `["color", "red"]`, … */
export type Semantic = [string, string];

export interface PrimSpec {
  _kind: "camera" | "light" | "cube" | "sphere";
  position?: number[];
  rotation?: number[];
  radius?: number;
  focal_length?: number;
  light_type?: string;
  intensity?: number;
  semantics?: Semantic[];
  /** Filled by scatter ops. */
  rotation_y?: number;
}

export const create = {
  camera(position: number[] = [0, 5, 0], rotation: number[] = [0, 0, 0], focal_length = 24): PrimSpec {
    return { _kind: "camera", position, rotation, focal_length };
  },
  light(rotation: number[] = [0, 0, 0], light_type = "distant", intensity = 1000): PrimSpec {
    return { _kind: "light", rotation, light_type, intensity };
  },
  cube(position: number[] = [0, 0, 0], semantics: Semantic[] = []): PrimSpec {
    return { _kind: "cube", position, semantics };
  },
  sphere(position: number[] = [0, 0, 0], radius = 1, semantics: Semantic[] = []): PrimSpec {
    return { _kind: "sphere", position, radius, semantics };
  },
};

export type ModifyOp =
  | { _op: "pose"; position: number[] | null; rotation: number[] | null }
  | { _op: "visibility"; visible: boolean };

export const modify = {
  pose(position: number[] | null = null, rotation: number[] | null = null): ModifyOp {
    return { _op: "pose", position, rotation };
  },
  visibility(visible = true): ModifyOp {
    return { _op: "visibility", visible };
  },
};

// ── randomize ops ────────────────────────────────────────────────────────────

export type RandomizeOp =
  | { _kind: "randomize_materials"; prims: PrimSpec[]; materials: Dist }
  | { _kind: "randomize_lights"; rotation: Dist; intensity: Dist; color: Dist }
  | { _kind: "scatter_2d"; prims: PrimSpec[]; plane: "xy" | "xz"; region: [number[], number[]]; rotation_z: Dist }
  | { _kind: "scatter_3d"; prims: PrimSpec[]; volume: [number[], number[]]; rotation: Dist }
  | { _kind: "randomize_physics"; prim: PrimSpec; mass: Dist; friction: Dist };

export const randomize = {
  materials(prims: PrimSpec[], materials: unknown[]): RandomizeOp {
    return { _kind: "randomize_materials", prims: [...prims], materials: distribution.choice(materials) };
  },
  lights(rotationDist?: Dist, intensityDist?: Dist, colorDist?: Dist): RandomizeOp {
    return {
      _kind: "randomize_lights",
      rotation: rotationDist ?? distribution.uniform([-90, -180, -180], [90, 180, 180]),
      intensity: intensityDist ?? distribution.uniform([500], [3000]),
      color: colorDist ?? distribution.uniform([0.7, 0.7, 0.7], [1, 1, 1]),
    };
  },
  scatter_2d(
    prims: PrimSpec[],
    plane: "xy" | "xz" = "xy",
    region: [number[], number[]] = [[-2, -2], [2, 2]],
    rotationZ?: Dist,
  ): RandomizeOp {
    return {
      _kind: "scatter_2d",
      prims: [...prims],
      plane,
      region: [[...region[0]], [...region[1]]],
      rotation_z: rotationZ ?? distribution.uniform([-180], [180]),
    };
  },
  scatter_3d(
    prims: PrimSpec[],
    volume: [number[], number[]] = [[-1, -1, 0], [1, 1, 2]],
    rotation?: Dist,
  ): RandomizeOp {
    return {
      _kind: "scatter_3d",
      prims: [...prims],
      volume: [[...volume[0]], [...volume[1]]],
      rotation: rotation ?? distribution.uniform([-180, -180, -180], [180, 180, 180]),
    };
  },
  physics_properties(prim: PrimSpec, massDist?: Dist, frictionDist?: Dist): RandomizeOp {
    return {
      _kind: "randomize_physics",
      prim,
      mass: massDist ?? distribution.uniform([0.5], [2]),
      friction: frictionDist ?? distribution.uniform([0.3], [0.9]),
    };
  },
};

// ── resolve (materialize a randomize op for a frame) ─────────────────────────

export interface ScatterPose {
  position: number[];
  rotation_z?: number;
  rotation?: number[];
}

export type ResolvedOp =
  | { kind: "randomize_materials"; prims: PrimSpec[]; material: unknown }
  | { kind: "randomize_lights"; rotation: number[]; intensity: number; color: number[] }
  | { kind: "scatter_2d" | "scatter_3d"; poses: ScatterPose[] }
  | { kind: "randomize_physics"; prim: PrimSpec; mass: number; friction: number };

function num1(v: Sampled): number {
  return Array.isArray(v) ? Number(v[0]) : Number(v);
}
function numArr(v: Sampled): number[] {
  return Array.isArray(v) ? v.map(Number) : [Number(v)];
}

/** Materialize a randomize op to a concrete scene operation for one frame. */
export function resolve(op: RandomizeOp, sampler?: Sampler): ResolvedOp {
  const s = sampler ?? globalSampler();
  switch (op._kind) {
    case "randomize_materials":
      return { kind: "randomize_materials", prims: op.prims, material: sample(op.materials, s) };
    case "randomize_lights":
      return {
        kind: "randomize_lights",
        rotation: numArr(sample(op.rotation, s)),
        intensity: num1(sample(op.intensity, s)),
        color: numArr(sample(op.color, s)),
      };
    case "scatter_2d": {
      const [rx0, ry0] = op.region[0];
      const [rx1, ry1] = op.region[1];
      const poses: ScatterPose[] = op.prims.map(() => {
        const x = s.nextUniform(rx0, rx1);
        const y = s.nextUniform(ry0, ry1);
        const rz = num1(sample(op.rotation_z, s));
        return op.plane === "xy"
          ? { position: [x, y, 0], rotation_z: rz }
          : { position: [x, 0, y], rotation_z: rz };
      });
      return { kind: "scatter_2d", poses };
    }
    case "scatter_3d": {
      const v0 = op.volume[0];
      const v1 = op.volume[1];
      const poses: ScatterPose[] = op.prims.map(() => ({
        position: [s.nextUniform(v0[0], v1[0]), s.nextUniform(v0[1], v1[1]), s.nextUniform(v0[2], v1[2])],
        rotation: numArr(sample(op.rotation, s)),
      }));
      return { kind: "scatter_3d", poses };
    }
    case "randomize_physics":
      return {
        kind: "randomize_physics",
        prim: op.prim,
        mass: num1(sample(op.mass, s)),
        friction: num1(sample(op.friction, s)),
      };
  }
}
