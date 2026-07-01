// wadachi-sim — clean-room AV simulation world (DriveSim lineage).
//
// The canonical KAMI engine behind the `nv-compat/drive-sim` facade. NVIDIA
// DRIVE Sim is the Omniverse-based sensor-realistic AV simulator; wadachi-sim
// reproduces its core loop — a scenario world (ego + traffic actors + static
// obstacles + ground) advanced over time, with camera / LiDAR / radar sensor
// models grounded in the kami-rt ray tracer.
//
// This module owns the world representation and per-tick scene construction:
// every actor / obstacle becomes an oriented box tessellated into triangles so
// the LiDAR and camera sensors can ray-trace it via the shared kami-rt BVH.
//
// World frame: +x forward, +y left, +z up; ego yaw about +z; ground at z=0.
//
// Clean-room: from-spec simulator. No DRIVE Sim source/binaries. Civilian,
// SAE-L4 ceiling, simulation-only (no actuation). ADR-2605261800 D1 (DriveSim
// → wadachi-sim); AV scope per ADR-2605242000 / ADR-2606010600.

import { type Scene, type Vec3, buildScene } from "../kami-rt/index.js";
import { type AgentKind } from "../kami-drive/index.js";

// ── world entities ──────────────────────────────────────────────────────────

export interface EgoState {
  x: number;
  y: number;
  /** Heading (rad), CCW from +x. */
  yaw: number;
  speed: number;
  /** Half-extents [length/2, width/2, height/2] (m). */
  extent: [number, number, number];
}

export interface Actor {
  id: string;
  kind: AgentKind;
  x: number;
  y: number;
  yaw: number;
  /** World-frame velocity (m/s). */
  vx: number;
  vy: number;
  extent: [number, number, number];
}

export interface StaticObstacle {
  id: string;
  kind: AgentKind;
  x: number;
  y: number;
  yaw: number;
  extent: [number, number, number];
}

export interface Scenario {
  ego: EgoState;
  actors: Actor[];
  obstacles: StaticObstacle[];
  /** Ground plane half-size (m); a flat quad centered at world origin. */
  groundHalfSize: number;
}

// ── oriented-box tessellation ────────────────────────────────────────────────

/** 12 triangles of a box centered at (cx,cy,cz_base+hz) with half-extents h,
 *  rotated `yaw` about +z. The box sits on the ground (base at z=0). */
export function boxTris(
  cx: number,
  cy: number,
  half: [number, number, number],
  yaw: number,
): Vec3[][] {
  const [hx, hy, hz] = half;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  // Local corners (z from 0 to 2*hz so the box rests on the ground).
  const local: Vec3[] = [
    [-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0],
    [-hx, -hy, 2 * hz], [hx, -hy, 2 * hz], [hx, hy, 2 * hz], [-hx, hy, 2 * hz],
  ];
  const v: Vec3[] = local.map(([lx, ly, lz]) => [cx + lx * c - ly * s, cy + lx * s + ly * c, lz]);
  const q = (a: number, b: number, cc: number, d: number): Vec3[][] => [
    [v[a], v[b], v[cc]], [v[a], v[cc], v[d]],
  ];
  return [
    ...q(0, 1, 2, 3), // bottom
    ...q(4, 5, 6, 7), // top
    ...q(0, 1, 5, 4), ...q(1, 2, 6, 5), ...q(2, 3, 7, 6), ...q(3, 0, 4, 7), // sides
  ];
}

function groundTris(half: number): Vec3[][] {
  const v: Vec3[] = [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]];
  return [[v[0], v[1], v[2]], [v[0], v[2], v[3]]];
}

/** Build the kami-rt scene for the current world (ground + actors + obstacles).
 *  The ego itself is not included (sensors are mounted on it). */
export function buildSensorScene(scenario: Scenario): Scene {
  const tris: Vec3[][] = [...groundTris(scenario.groundHalfSize)];
  for (const a of scenario.actors) tris.push(...boxTris(a.x, a.y, a.extent, a.yaw));
  for (const o of scenario.obstacles) tris.push(...boxTris(o.x, o.y, o.extent, o.yaw));
  return buildScene(tris);
}

// ── ground-truth object list ────────────────────────────────────────────────

export interface GtObject {
  id: string;
  kind: AgentKind;
  /** World-frame center of the box (z at half height). */
  center: Vec3;
  extent: [number, number, number];
  yaw: number;
}

/** Axis-aligned world bounds of an oriented box (for projection / collision). */
export function worldAabb(
  cx: number,
  cy: number,
  half: [number, number, number],
  yaw: number,
): { min: Vec3; max: Vec3 } {
  const [hx, hy, hz] = half;
  const c = Math.abs(Math.cos(yaw)), s = Math.abs(Math.sin(yaw));
  const ex = hx * c + hy * s;
  const ey = hx * s + hy * c;
  return { min: [cx - ex, cy - ey, 0], max: [cx + ex, cy + ey, 2 * hz] };
}

/** All annotatable objects (actors + obstacles) with world geometry. */
export function groundTruth(scenario: Scenario): GtObject[] {
  const out: GtObject[] = [];
  for (const a of scenario.actors) {
    out.push({ id: a.id, kind: a.kind, center: [a.x, a.y, a.extent[2]], extent: a.extent, yaw: a.yaw });
  }
  for (const o of scenario.obstacles) {
    out.push({ id: o.id, kind: o.kind, center: [o.x, o.y, o.extent[2]], extent: o.extent, yaw: o.yaw });
  }
  return out;
}
