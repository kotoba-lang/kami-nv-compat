// kami-drive — clean-room BEV unicycle kinematics (michibiki planner core).
//
// The canonical KAMI motion model behind the `nv-compat/alpamayo` VLA facade.
// NVIDIA Alpamayo represents trajectories with a "unicycle model in
// bird's-eye-view (BEV) space" driven by dynamic actions (acceleration +
// curvature); this module reproduces that motion model from textbook vehicle
// kinematics so a plan of {accel, curvature} actions rolls out to the same
// trajectory format Alpamayo emits (3D translation + 3×3 rotation per
// waypoint, in the ego frame, 0-yaw at t0).
//
// Clean-room: from-spec kinematics. No Alpamayo / Cosmos / DRIVE source,
// weights, or binaries are used. Civilian autonomous-mobility only, SAE-L4
// ceiling (ADR-2605242000 wadachi / ADR-2606010600 kami-autodrive).
//
// ADR-2605261800-adjacent (nv-compat namespace); AV scope per wadachi /
// kami-autodrive ADRs.

// ── state / action ──────────────────────────────────────────────────────────

/** Planar BEV vehicle state. `yaw` in radians, CCW from +x. */
export interface BevState {
  x: number;
  y: number;
  yaw: number;
  /** Longitudinal speed (m/s), ≥ 0. */
  speed: number;
}

/** Dynamic action: longitudinal acceleration (m/s²) + path curvature κ=1/R
 *  (1/m, +ve = left turn). This is Alpamayo's BEV action parameterization. */
export interface DynamicAction {
  accel: number;
  curvature: number;
}

/** A trajectory waypoint in the Alpamayo output format: ego-frame 3D
 *  translation + 3×3 rotation (row-major 9), plus the BEV scalars KAMI uses. */
export interface Waypoint {
  /** Seconds from t0. */
  t: number;
  /** [x, y, z] in ego coordinates (z = 0 on the BEV plane). */
  translation: [number, number, number];
  /** Row-major 3×3 rotation (yaw about +z). */
  rotation: [number, number, number, number, number, number, number, number, number];
  speed: number;
  accel: number;
  curvature: number;
}

/** Speed cap used when an action would otherwise integrate past it. */
export const DEFAULT_MAX_SPEED = 30; // m/s (~108 km/h; below any L4 urban need)

// ── integration ──────────────────────────────────────────────────────────────

/** Yaw → row-major 3×3 rotation about +z. */
export function yawToMat3(
  yaw: number,
): [number, number, number, number, number, number, number, number, number] {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

/** One forward-Euler unicycle step. `maxSpeed` clamps the integrated speed. */
export function stepUnicycle(
  state: BevState,
  action: DynamicAction,
  dt: number,
  maxSpeed = DEFAULT_MAX_SPEED,
): BevState {
  const x = state.x + state.speed * Math.cos(state.yaw) * dt;
  const y = state.y + state.speed * Math.sin(state.yaw) * dt;
  const yaw = state.yaw + state.speed * action.curvature * dt;
  const speed = Math.min(maxSpeed, Math.max(0, state.speed + action.accel * dt));
  return { x, y, yaw, speed };
}

/** Roll out an action sequence into Alpamayo-format waypoints. The first
 *  waypoint is t0 = the initial state; there are `actions.length` further
 *  waypoints at `dt` spacing. Planning starts in the ego frame (origin,
 *  0-yaw) per the Alpamayo trajectory convention. */
export function rolloutTrajectory(
  initial: BevState,
  actions: readonly DynamicAction[],
  dt: number,
  maxSpeed = DEFAULT_MAX_SPEED,
): Waypoint[] {
  const out: Waypoint[] = [];
  let s = initial;
  const emit = (t: number, a: DynamicAction): void => {
    out.push({
      t,
      translation: [s.x, s.y, 0],
      rotation: yawToMat3(s.yaw),
      speed: s.speed,
      accel: a.accel,
      curvature: a.curvature,
    });
  };
  emit(0, { accel: 0, curvature: 0 });
  for (let i = 0; i < actions.length; i++) {
    s = stepUnicycle(s, actions[i], dt, maxSpeed);
    emit((i + 1) * dt, actions[i]);
  }
  return out;
}

/** Arc length of a trajectory (sum of segment lengths in the ego/BEV plane). */
export function trajectoryLength(wps: readonly Waypoint[]): number {
  let d = 0;
  for (let i = 1; i < wps.length; i++) {
    const dx = wps[i].translation[0] - wps[i - 1].translation[0];
    const dy = wps[i].translation[1] - wps[i - 1].translation[1];
    d += Math.hypot(dx, dy);
  }
  return d;
}
