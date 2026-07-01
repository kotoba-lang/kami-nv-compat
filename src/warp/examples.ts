// Reference WGSL+JS kernel pairs that exercise the wgpuLaunch path.
//
// These are pedagogical examples — they prove the architectural
// pattern: a kernel author writes both a JS implementation (for the
// sync `launch` path and as the WebGPU fallback) and a WGSL compute
// shader (for actual GPU dispatch via `wgpuLaunch`). The runtime
// picks WGSL when navigator.gpu is available.

import { cos, sin, tid, type WpArray } from "./warp.js";
import { wgpuKernel, type WgpuKernel } from "./wgpu-backend.js";

/** Damping kernel: multiply each element of an array by a scalar.
 *
 *  Bindings:
 *    @group(0) @binding(0) = WpArray<number> (storage, read_write)
 *    @group(0) @binding(1) = scalar damping (uniform)
 *
 *  Workgroup size 64 matches the kami-cartpole-wasm precedent
 *  (per CLAUDE.md note: "kami-genesis/src/wgsl/cartpole_step.wgsl
 *  workgroup_size 64 WGSL kernel").
 */
export const dampingKernel: WgpuKernel = wgpuKernel({
  js: (arr: WpArray<number>, damping: number) => {
    const i = tid();
    arr.set(i, arr.get(i) * damping);
  },
  wgsl: `
struct DampingUniform {
  damping: f32,
  _pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> arr: array<f32>;
@group(0) @binding(1) var<uniform> uni: DampingUniform;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&arr)) {
    return;
  }
  arr[i] = arr[i] * uni.damping;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "uniform", inputIndex: 1 },
  ],
  workgroupSize: 64,
});

// ── Pendulum semi-implicit Euler step (env-parallel) ─────────────────────
//
// Mirrors the iter 71 single-pendulum integrator stepped across N envs in
// parallel. This is the canonical workload Warp / Isaac Lab were
// designed for — thousands of envs advancing one timestep in lock-step
// on the GPU.
//
// Per-env state: theta (angle), omega (angular velocity).
// Per-env input: tau (applied torque).
// Uniform params: dt (timestep), g (gravity magnitude), L (pendulum
// length, COM distance from pivot), mass.
//
// Dynamics: τ_total = τ_applied - m·g·L·sin(θ);  α = τ_total / (m·L²)
// Semi-implicit Euler:  ω' = ω + dt·α;  θ' = θ + dt·ω'
//
// At equilibrium (θ=0, ω=0, τ=0): α=0, ω' = 0, θ' = 0 (no drift).
// At θ=π/2, ω=0, τ=0: α = -g·L·sin(π/2)/(m·L²) = -g/L ≈ -9.81 (m=L=1)
// — matches Python iter 68 PASS 5 and TS iter 71 PASS 5.

/** State + input bindings (in this order):
 *    @group(0) @binding(0) theta  (storage read_write, N floats)
 *    @group(0) @binding(1) omega  (storage read_write, N floats)
 *    @group(0) @binding(2) tau    (storage read, N floats; writeback false)
 *    @group(0) @binding(3) dt     (uniform f32)
 *    @group(0) @binding(4) g      (uniform f32)
 *    @group(0) @binding(5) length (uniform f32)
 *    @group(0) @binding(6) mass   (uniform f32)
 */
export const pendulumStepKernel: WgpuKernel = wgpuKernel({
  js: (
    theta: WpArray<number>,
    omega: WpArray<number>,
    tau: WpArray<number>,
    dt: number,
    g: number,
    length: number,
    mass: number,
  ) => {
    const i = tid();
    const t = theta.get(i);
    const w = omega.get(i);
    const tor = tau.get(i);
    const alpha = (tor - mass * g * length * sin(t)) / (mass * length * length);
    const wNew = w + dt * alpha;
    const tNew = t + dt * wNew;
    omega.set(i, wNew);
    theta.set(i, tNew);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> theta:  array<f32>;
@group(0) @binding(1) var<storage, read_write> omega:  array<f32>;
@group(0) @binding(2) var<storage, read_write> tau:    array<f32>;
@group(0) @binding(3) var<uniform>             dt_u:     vec4<f32>;
@group(0) @binding(4) var<uniform>             g_u:      vec4<f32>;
@group(0) @binding(5) var<uniform>             length_u: vec4<f32>;
@group(0) @binding(6) var<uniform>             mass_u:   vec4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&theta)) {
    return;
  }
  let t = theta[i];
  let w = omega[i];
  let tor = tau[i];
  let dt = dt_u.x;
  let g = g_u.x;
  let L = length_u.x;
  let m = mass_u.x;
  let alpha = (tor - m * g * L * sin(t)) / (m * L * L);
  let w_new = w + dt * alpha;
  let t_new = t + dt * w_new;
  omega[i] = w_new;
  theta[i] = t_new;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "uniform", inputIndex: 3 },
    { binding: 4, kind: "uniform", inputIndex: 4 },
    { binding: 5, kind: "uniform", inputIndex: 5 },
    { binding: 6, kind: "uniform", inputIndex: 6 },
  ],
  workgroupSize: 64,
});

// ── Cartpole semi-implicit Euler step (env-parallel) ─────────────────────
//
// Mirrors the Python iter 68 _kernel.cartpole_step (Sutton & Barto /
// OpenAI Gym CartPole-v1 closed-form) stepped across N envs in parallel.
// 2-DoF coupled dynamics — revolute pole on a prismatic cart — closer
// to actual robot work than iter 78's single-pendulum.
//
// Per-env state: x (cart position), x_dot (cart velocity), theta
//   (pole angle from vertical, +θ = pole leans in +x direction), theta_dot.
// Per-env input: force (clamped to ±force_mag externally; this kernel
//   does NOT clamp — caller is responsible).
// Uniform params: dt, gravity, cart_mass, pole_mass, pole_half_length.
//
// Closed-form per Sutton & Barto:
//   temp = (force + m_pole·L·θ̇²·sin θ) / total_mass
//   θ̈   = (g·sin θ - cos θ · temp) / (L · (4/3 - m_pole·cos²θ / total_mass))
//   ẍ   = temp - m_pole·L·θ̈·cos θ / total_mass
//
// Semi-implicit Euler:
//   ẋ' = ẋ + dt·ẍ        x' = x + dt·ẋ'
//   θ̇' = θ̇ + dt·θ̈        θ' = θ + dt·θ̇'

/** Bindings:
 *    @group(0) @binding(0) x         (storage read_write)
 *    @group(0) @binding(1) x_dot     (storage read_write)
 *    @group(0) @binding(2) theta     (storage read_write)
 *    @group(0) @binding(3) theta_dot (storage read_write)
 *    @group(0) @binding(4) force     (storage read, writeback false)
 *    @group(0) @binding(5) dt        (uniform vec4<f32>.x)
 *    @group(0) @binding(6) gravity   (uniform vec4<f32>.x)
 *    @group(0) @binding(7) cart_mass (uniform vec4<f32>.x)
 *    @group(0) @binding(8) pole_mass (uniform vec4<f32>.x)
 *    @group(0) @binding(9) pole_half_length (uniform vec4<f32>.x)
 */
export const cartpoleStepKernel: WgpuKernel = wgpuKernel({
  js: (
    x: WpArray<number>,
    x_dot: WpArray<number>,
    theta: WpArray<number>,
    theta_dot: WpArray<number>,
    force: WpArray<number>,
    dt: number,
    gravity: number,
    cart_mass: number,
    pole_mass: number,
    pole_half_length: number,
  ) => {
    const i = tid();
    const t = theta.get(i);
    const td = theta_dot.get(i);
    const xd = x_dot.get(i);
    const f = force.get(i);
    const sinT = sin(t);
    const cosT = cos(t);
    const totalMass = cart_mass + pole_mass;
    const pml = pole_mass * pole_half_length;
    const temp = (f + pml * td * td * sinT) / totalMass;
    const thetaAcc =
      (gravity * sinT - cosT * temp) /
      (pole_half_length * (4 / 3 - pole_mass * cosT * cosT / totalMass));
    const xAcc = temp - pml * thetaAcc * cosT / totalMass;
    const xDotNew = xd + dt * xAcc;
    const xNew = x.get(i) + dt * xDotNew;
    const thetaDotNew = td + dt * thetaAcc;
    const thetaNew = t + dt * thetaDotNew;
    x_dot.set(i, xDotNew);
    x.set(i, xNew);
    theta_dot.set(i, thetaDotNew);
    theta.set(i, thetaNew);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> x:         array<f32>;
@group(0) @binding(1) var<storage, read_write> x_dot:     array<f32>;
@group(0) @binding(2) var<storage, read_write> theta:     array<f32>;
@group(0) @binding(3) var<storage, read_write> theta_dot: array<f32>;
@group(0) @binding(4) var<storage, read_write> force:     array<f32>;
@group(0) @binding(5) var<uniform>             dt_u:      vec4<f32>;
@group(0) @binding(6) var<uniform>             g_u:       vec4<f32>;
@group(0) @binding(7) var<uniform>             cm_u:      vec4<f32>;
@group(0) @binding(8) var<uniform>             pm_u:      vec4<f32>;
@group(0) @binding(9) var<uniform>             L_u:       vec4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&theta)) {
    return;
  }
  let t = theta[i];
  let td = theta_dot[i];
  let xd = x_dot[i];
  let f = force[i];
  let dt = dt_u.x;
  let g = g_u.x;
  let cart_mass = cm_u.x;
  let pole_mass = pm_u.x;
  let L = L_u.x;
  let sinT = sin(t);
  let cosT = cos(t);
  let totalMass = cart_mass + pole_mass;
  let pml = pole_mass * L;
  let temp = (f + pml * td * td * sinT) / totalMass;
  let theta_acc = (g * sinT - cosT * temp) /
                    (L * (4.0 / 3.0 - pole_mass * cosT * cosT / totalMass));
  let x_acc = temp - pml * theta_acc * cosT / totalMass;
  let xDotNew = xd + dt * x_acc;
  let xNew = x[i] + dt * xDotNew;
  let thetaDotNew = td + dt * theta_acc;
  let thetaNew = t + dt * thetaDotNew;
  x_dot[i] = xDotNew;
  x[i] = xNew;
  theta_dot[i] = thetaDotNew;
  theta[i] = thetaNew;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: true },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: true },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "uniform", inputIndex: 5 },
    { binding: 6, kind: "uniform", inputIndex: 6 },
    { binding: 7, kind: "uniform", inputIndex: 7 },
    { binding: 8, kind: "uniform", inputIndex: 8 },
    { binding: 9, kind: "uniform", inputIndex: 9 },
  ],
  workgroupSize: 64,
});

// ── Two-link arm step (env-parallel) ─────────────────────────────────────
//
// Closed-form 2-DoF planar arm dynamics — both joints revolute about
// world Y-axis, both links pendulum-like (gravity pulls toward -Z).
// Mirrors the iter 71 compound-pendulum reference (Python iter 68
// PASS 9-10 / TS iter 71 PASS 9-10) but generalised for arbitrary
// torques.
//
// Standard manipulator equation:
//
//   M(q) · q̈ + C(q, q̇) · q̇ + g(q) = τ
//
// For 2-link arm (link i has mass m_i, full length L_i, COM offset r_i
// from joint i, inertia I_i about COM):
//
//   M(q) = [[a + b + 2c·cosθ₂, b + c·cosθ₂],
//           [b + c·cosθ₂,       b           ]]
//
//   where a = m₁r₁² + I₁ + m₂L₁²
//         b = m₂r₂² + I₂
//         c = m₂·L₁·r₂
//
//   C(q,q̇)·q̇ = [[-c·sinθ₂·θ̇₂, -c·sinθ₂·(θ̇₁+θ̇₂)],     [θ̇₁]
//                [ c·sinθ₂·θ̇₁,  0                  ]] · [θ̇₂]
//
//   g(q) = [m₁·g·r₁·sin θ₁ + m₂·g·(L₁·sin θ₁ + r₂·sin(θ₁+θ₂)),
//           m₂·g·r₂·sin(θ₁+θ₂)]
//
// Invert M(q) by hand (2×2 closed form), solve for q̈, semi-implicit
// Euler integrate. Reference: Spong, Robot Modeling & Control, Ch. 7.

/** State + input + uniform bindings (11 total):
 *    @group(0) @binding(0)  theta1     (storage read_write)
 *    @group(0) @binding(1)  theta1_dot (storage read_write)
 *    @group(0) @binding(2)  theta2     (storage read_write)
 *    @group(0) @binding(3)  theta2_dot (storage read_write)
 *    @group(0) @binding(4)  tau1       (storage read, writeback false)
 *    @group(0) @binding(5)  tau2       (storage read, writeback false)
 *    @group(0) @binding(6)  dt         (uniform vec4<f32>.x)
 *    @group(0) @binding(7)  gravity    (uniform vec4<f32>.x)
 *    @group(0) @binding(8)  link1      (uniform vec4<f32>: m1, L1, r1, I1)
 *    @group(0) @binding(9)  link2      (uniform vec4<f32>: m2, L2, r2, I2)
 *                                       (L2 unused — arm-tip is at r2 + L2/2
 *                                        in conventional layout, but free
 *                                        for caller's interpretation)
 */
export const twoLinkArmStepKernel: WgpuKernel = wgpuKernel({
  js: (
    theta1:     WpArray<number>,
    theta1_dot: WpArray<number>,
    theta2:     WpArray<number>,
    theta2_dot: WpArray<number>,
    tau1:       WpArray<number>,
    tau2:       WpArray<number>,
    dt:         number,
    g:          number,
    m1: number, L1: number, r1: number, I1: number,
    m2: number, L2: number, r2: number, I2: number,
  ) => {
    const i = tid();
    const q1 = theta1.get(i);
    const q2 = theta2.get(i);
    const dq1 = theta1_dot.get(i);
    const dq2 = theta2_dot.get(i);
    const t1 = tau1.get(i);
    const t2 = tau2.get(i);
    void L2; // L2 reserved for future tip-frame variants
    const a = m1 * r1 * r1 + I1 + m2 * L1 * L1;
    const b = m2 * r2 * r2 + I2;
    const c = m2 * L1 * r2;
    const cosT2 = cos(q2);
    const sinT2 = sin(q2);
    // M(q)
    const M11 = a + b + 2 * c * cosT2;
    const M12 = b + c * cosT2;
    const M22 = b;
    // h = C·q̇ + g
    const h1 = -c * sinT2 * dq2 * dq1
                - c * sinT2 * (dq1 + dq2) * dq2
                + m1 * g * r1 * sin(q1)
                + m2 * g * (L1 * sin(q1) + r2 * sin(q1 + q2));
    const h2 =  c * sinT2 * dq1 * dq1
                + m2 * g * r2 * sin(q1 + q2);
    // Solve M·q̈ = τ - h via 2×2 inverse:
    //   det = M11·M22 - M12²
    //   q̈₁ = (M22·b₁ - M12·b₂) / det   where b = τ - h
    //   q̈₂ = (M11·b₂ - M12·b₁) / det
    const b1 = t1 - h1;
    const b2 = t2 - h2;
    const det = M11 * M22 - M12 * M12;
    const ddq1 = (M22 * b1 - M12 * b2) / det;
    const ddq2 = (M11 * b2 - M12 * b1) / det;
    // Semi-implicit Euler
    const dq1New = dq1 + dt * ddq1;
    const dq2New = dq2 + dt * ddq2;
    theta1_dot.set(i, dq1New);
    theta1.set(i, q1 + dt * dq1New);
    theta2_dot.set(i, dq2New);
    theta2.set(i, q2 + dt * dq2New);
  },
  wgsl: `
@group(0) @binding(0)  var<storage, read_write> theta1:     array<f32>;
@group(0) @binding(1)  var<storage, read_write> theta1_dot: array<f32>;
@group(0) @binding(2)  var<storage, read_write> theta2:     array<f32>;
@group(0) @binding(3)  var<storage, read_write> theta2_dot: array<f32>;
@group(0) @binding(4)  var<storage, read_write> tau1:       array<f32>;
@group(0) @binding(5)  var<storage, read_write> tau2:       array<f32>;
@group(0) @binding(6)  var<uniform>             dt_u:       vec4<f32>;
@group(0) @binding(7)  var<uniform>             g_u:        vec4<f32>;
@group(0) @binding(8)  var<uniform>             link1_u:    vec4<f32>;
@group(0) @binding(9)  var<uniform>             link2_u:    vec4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&theta1)) { return; }
  let q1 = theta1[i];
  let q2 = theta2[i];
  let dq1 = theta1_dot[i];
  let dq2 = theta2_dot[i];
  let t1 = tau1[i];
  let t2 = tau2[i];
  let dt = dt_u.x;
  let g  = g_u.x;
  let m1 = link1_u.x; let L1 = link1_u.y; let r1 = link1_u.z; let I1 = link1_u.w;
  let m2 = link2_u.x;                     let r2 = link2_u.z; let I2 = link2_u.w;
  let a = m1 * r1 * r1 + I1 + m2 * L1 * L1;
  let b = m2 * r2 * r2 + I2;
  let c = m2 * L1 * r2;
  let cosT2 = cos(q2);
  let sinT2 = sin(q2);
  let M11 = a + b + 2.0 * c * cosT2;
  let M12 = b + c * cosT2;
  let M22 = b;
  let h1 = -c * sinT2 * dq2 * dq1
           - c * sinT2 * (dq1 + dq2) * dq2
           + m1 * g * r1 * sin(q1)
           + m2 * g * (L1 * sin(q1) + r2 * sin(q1 + q2));
  let h2 =  c * sinT2 * dq1 * dq1
           + m2 * g * r2 * sin(q1 + q2);
  let b1 = t1 - h1;
  let b2 = t2 - h2;
  let det = M11 * M22 - M12 * M12;
  let ddq1 = (M22 * b1 - M12 * b2) / det;
  let ddq2 = (M11 * b2 - M12 * b1) / det;
  let dq1New = dq1 + dt * ddq1;
  let dq2New = dq2 + dt * ddq2;
  theta1_dot[i] = dq1New;
  theta1[i] = q1 + dt * dq1New;
  theta2_dot[i] = dq2New;
  theta2[i] = q2 + dt * dq2New;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: true },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: true },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "storage", inputIndex: 5, writeback: false },
    { binding: 6, kind: "uniform", inputIndex: 6 },
    { binding: 7, kind: "uniform", inputIndex: 7 },
    { binding: 8, kind: "uniform", inputIndex: 8 },
    { binding: 9, kind: "uniform", inputIndex: 9 },
  ],
  workgroupSize: 64,
});

// ── Franka 7-DoF forward kinematics (env-parallel) ───────────────────────
//
// Computes EE position for N envs in parallel from per-env q[7]. Real
// Franka FCI joint origins (iter 85). Foundation for future Jacobian
// + IK kernels — proves complex multi-joint FK with rpy frame rotations
// runs correctly on WebGPU.
//
// Per-env input: q[7]
// Per-env output: ee_pos[3]
//
// Storage layout (struct-of-arrays for coalesced GPU access):
//   q_in:    array<f32>  length 7*N  — q[i] = q_in[env*7 + i]
//   ee_out:  array<f32>  length 3*N  — ee[i] = ee_out[env*3 + i]
//
// Algorithm: 7 successive frame compositions. Each joint applies
//   R_origin (from URDF rpy)  ·  Rodrigues(axis_body_z, q_i)
// to the cumulative world-frame rotation, plus xyz translation.
// All inlined per-thread (~150 lines of WGSL).

/** Bindings:
 *    @group(0) @binding(0) q_in    (storage read,  N*7 floats; writeback false)
 *    @group(0) @binding(1) ee_out  (storage read_write, N*3 floats)
 */
export const frankaFkKernel: WgpuKernel = wgpuKernel({
  js: (qIn: WpArray<number>, eeOut: WpArray<number>) => {
    const env = tid();
    const q0 = qIn.get(env * 7 + 0);
    const q1 = qIn.get(env * 7 + 1);
    const q2 = qIn.get(env * 7 + 2);
    const q3 = qIn.get(env * 7 + 3);
    const q4 = qIn.get(env * 7 + 4);
    const q5 = qIn.get(env * 7 + 5);
    const q6 = qIn.get(env * 7 + 6);
    const ee = frankaFkInline([q0, q1, q2, q3, q4, q5, q6]);
    eeOut.set(env * 3 + 0, ee[0]);
    eeOut.set(env * 3 + 1, ee[1]);
    eeOut.set(env * 3 + 2, ee[2]);
  },
  wgsl: `
// Real Franka FCI joint origins (xyz triplet + rpy triplet per joint).
// Pre-computed: cos/sin of rpy values inlined as constants for speed.
// rpy values: (0,0,0), (-π/2,0,0), (π/2,0,0), (π/2,0,0), (-π/2,0,0), (π/2,0,0), (π/2,0,0)

@group(0) @binding(0) var<storage, read_write> q_in:    array<f32>;
@group(0) @binding(1) var<storage, read_write> ee_out:  array<f32>;

// Composed rotation R_world (3×3) stored row-major in 9 f32 locals.
// p_world stored in 3 f32 locals.
// Applies R_world ← R_world · R_origin · R_q (axis z, angle q[i])
// and p_world ← p_world + R_world_pre · xyz.

fn rot_rpy(r: f32, p: f32, y: f32) -> mat3x3<f32> {
  let cr = cos(r); let sr = sin(r);
  let cp = cos(p); let sp = sin(p);
  let cy = cos(y); let sy = sin(y);
  return mat3x3<f32>(
    vec3<f32>(cy*cp, sy*cp, -sp),
    vec3<f32>(cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr),
    vec3<f32>(cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr),
  );
}

fn rot_z(angle: f32) -> mat3x3<f32> {
  let c = cos(angle); let s = sin(angle);
  return mat3x3<f32>(
    vec3<f32>(c, s, 0.0),
    vec3<f32>(-s, c, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&ee_out) / 3u;
  if (env >= n_envs) { return; }

  let base = env * 7u;
  let q = array<f32, 7>(
    q_in[base + 0u], q_in[base + 1u], q_in[base + 2u], q_in[base + 3u],
    q_in[base + 4u], q_in[base + 5u], q_in[base + 6u],
  );

  let half_pi: f32 = 1.5707963267948966;
  // Joint origins: xyz vec3 + rpy.r (rpy.p, rpy.y are 0 for all Franka joints)
  let xyz = array<vec3<f32>, 7>(
    vec3<f32>(0.0,      0.0,     0.333),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.0,     -0.316,   0.0),
    vec3<f32>(0.0825,   0.0,     0.0),
    vec3<f32>(-0.0825,  0.384,   0.0),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.088,    0.0,     0.0),
  );
  let rpy_r = array<f32, 7>(
    0.0,
    -half_pi,
    half_pi,
    half_pi,
    -half_pi,
    half_pi,
    half_pi,
  );

  var R_world = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
  var p_world = vec3<f32>(0.0, 0.0, 0.0);

  for (var i = 0u; i < 7u; i = i + 1u) {
    let R_origin = rot_rpy(rpy_r[i], 0.0, 0.0);
    let R_q = rot_z(q[i]);
    let R_iInP = R_origin * R_q;
    // p contribution: rotated xyz, in current world frame (before this
    // joint's rotation).
    let rotated = R_world * xyz[i];
    p_world = p_world + rotated;
    R_world = R_world * R_iInP;
  }

  ee_out[env * 3u + 0u] = p_world.x;
  ee_out[env * 3u + 1u] = p_world.y;
  ee_out[env * 3u + 2u] = p_world.z;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
  ],
  workgroupSize: 64,
});

// ── JS reference impl (used by frankaFkKernel.js fallback) ──────────────

const _FRANKA_FK_HALF_PI = Math.PI / 2;
const _FRANKA_FK_XYZ: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 0.333],
  [0, 0, 0],
  [0, -0.316, 0],
  [0.0825, 0, 0],
  [-0.0825, 0.384, 0],
  [0, 0, 0],
  [0.088, 0, 0],
];
const _FRANKA_FK_RPY_R: ReadonlyArray<number> = [
  0, -_FRANKA_FK_HALF_PI, _FRANKA_FK_HALF_PI, _FRANKA_FK_HALF_PI,
  -_FRANKA_FK_HALF_PI, _FRANKA_FK_HALF_PI, _FRANKA_FK_HALF_PI,
];

function _rotRpy(r: number): number[][] {
  const cr = Math.cos(r), sr = Math.sin(r);
  return [[1, 0, 0], [0, cr, -sr], [0, sr, cr]];   // p=y=0, so only x-rotation
}

function _rotZ(angle: number): number[][] {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}

function _mat3MulSmall(a: number[][], b: number[][]): number[][] {
  const out: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
  }
  return out;
}

function _matVec3Small(m: number[][], v: readonly number[]): [number, number, number] {
  return [
    m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
    m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
    m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
  ];
}

/** Reference Franka 7-DoF FK in pure JS — used by the kernel's JS
 *  fallback AND callable directly for cross-validation.
 *  Returns the world-frame EE position from q[7].
 */
export function frankaFkInline(q: readonly number[]): [number, number, number] {
  let R_world: number[][] = [[1,0,0],[0,1,0],[0,0,1]];
  let p_world: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 7; i++) {
    const R_origin = _rotRpy(_FRANKA_FK_RPY_R[i]);
    const R_q = _rotZ(q[i]);
    const R_iInP = _mat3MulSmall(R_origin, R_q);
    const rotated = _matVec3Small(R_world, _FRANKA_FK_XYZ[i]);
    p_world = [p_world[0]+rotated[0], p_world[1]+rotated[1], p_world[2]+rotated[2]];
    R_world = _mat3MulSmall(R_world, R_iInP);
  }
  return p_world;
}

// ── Franka 7-DoF FK + linear Jacobian (env-parallel) ─────────────────────
//
// Extends frankaFkKernel (iter 88) with the 3×7 linear Jacobian. Each
// thread runs FK, stores all 7 joint poses in private memory, then
// computes per-joint Jacobian columns via axis_world_i × (p_ee - p_i).
//
// Per-env input: q[7]
// Per-env output: ee_pos[3] + J[3][7] = 24 floats
//
// Storage layout (struct-of-arrays):
//   q_in:    array<f32>  length 7*N
//   out_buf: array<f32>  length 24*N
//     env-i layout: [ee_x, ee_y, ee_z, J[0][0..6], J[1][0..6], J[2][0..6]]

/** Bindings:
 *    @group(0) @binding(0) q_in    (storage read,  N*7 floats)
 *    @group(0) @binding(1) out_buf (storage read_write, N*24 floats)
 */
export const frankaFkJacobianKernel: WgpuKernel = wgpuKernel({
  js: (qIn: WpArray<number>, outBuf: WpArray<number>) => {
    const env = tid();
    const q: number[] = [
      qIn.get(env * 7 + 0), qIn.get(env * 7 + 1), qIn.get(env * 7 + 2),
      qIn.get(env * 7 + 3), qIn.get(env * 7 + 4), qIn.get(env * 7 + 5),
      qIn.get(env * 7 + 6),
    ];
    const { ee, J } = frankaFkJacobianInline(q);
    const base = env * 24;
    outBuf.set(base + 0, ee[0]);
    outBuf.set(base + 1, ee[1]);
    outBuf.set(base + 2, ee[2]);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 7; c++) {
        outBuf.set(base + 3 + r * 7 + c, J[r][c]);
      }
    }
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q_in:    array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;

fn rot_rpy_x(r: f32) -> mat3x3<f32> {
  let cr = cos(r); let sr = sin(r);
  return mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, cr, sr),
    vec3<f32>(0.0, -sr, cr),
  );
}

fn rot_z(angle: f32) -> mat3x3<f32> {
  let c = cos(angle); let s = sin(angle);
  return mat3x3<f32>(
    vec3<f32>(c, s, 0.0),
    vec3<f32>(-s, c, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&out_buf) / 24u;
  if (env >= n_envs) { return; }

  let base_q = env * 7u;
  let q = array<f32, 7>(
    q_in[base_q + 0u], q_in[base_q + 1u], q_in[base_q + 2u], q_in[base_q + 3u],
    q_in[base_q + 4u], q_in[base_q + 5u], q_in[base_q + 6u],
  );

  let half_pi: f32 = 1.5707963267948966;
  let xyz = array<vec3<f32>, 7>(
    vec3<f32>(0.0,      0.0,     0.333),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.0,     -0.316,   0.0),
    vec3<f32>(0.0825,   0.0,     0.0),
    vec3<f32>(-0.0825,  0.384,   0.0),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.088,    0.0,     0.0),
  );
  let rpy_r = array<f32, 7>(0.0, -half_pi, half_pi, half_pi, -half_pi, half_pi, half_pi);

  // Pass 1: forward kinematics, store every joint's world-frame pose.
  var poses_R: array<mat3x3<f32>, 7>;
  var poses_p: array<vec3<f32>, 7>;
  var R_world = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
  var p_world = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0u; i < 7u; i = i + 1u) {
    let R_origin = rot_rpy_x(rpy_r[i]);
    let R_q = rot_z(q[i]);
    let R_iInP = R_origin * R_q;
    let rotated = R_world * xyz[i];
    p_world = p_world + rotated;
    R_world = R_world * R_iInP;
    poses_R[i] = R_world;
    poses_p[i] = p_world;
  }

  let ee_pos = poses_p[6];
  let base_out = env * 24u;
  out_buf[base_out + 0u] = ee_pos.x;
  out_buf[base_out + 1u] = ee_pos.y;
  out_buf[base_out + 2u] = ee_pos.z;

  // Pass 2: Jacobian columns J[:,i] = axis_world_i × (p_ee - p_i)
  // axis_world_i = R_world_i · (0,0,1) = third column of R_world_i
  for (var i = 0u; i < 7u; i = i + 1u) {
    let Ri = poses_R[i];
    let a_world = vec3<f32>(Ri[2].x, Ri[2].y, Ri[2].z);
    let dp = ee_pos - poses_p[i];
    let col = vec3<f32>(
      a_world.y * dp.z - a_world.z * dp.y,
      a_world.z * dp.x - a_world.x * dp.z,
      a_world.x * dp.y - a_world.y * dp.x,
    );
    out_buf[base_out + 3u + 0u * 7u + i] = col.x;
    out_buf[base_out + 3u + 1u * 7u + i] = col.y;
    out_buf[base_out + 3u + 2u * 7u + i] = col.z;
  }
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
  ],
  workgroupSize: 64,
});

/** Reference Franka FK + linear Jacobian in pure JS. Used by
 *  frankaFkJacobianKernel's JS fallback AND callable directly.
 *  Returns { ee, J } where ee is the EE world-frame position and
 *  J is the 3×7 linear Jacobian.
 */
export function frankaFkJacobianInline(q: readonly number[]): {
  ee: [number, number, number];
  J: number[][];
} {
  // Pass 1: store per-joint world poses.
  const poses_R: number[][][] = [];
  const poses_p: [number, number, number][] = [];
  let R_world: number[][] = [[1,0,0],[0,1,0],[0,0,1]];
  let p_world: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 7; i++) {
    const R_origin = _rotRpy(_FRANKA_FK_RPY_R[i]);
    const R_q = _rotZ(q[i]);
    const R_iInP = _mat3MulSmall(R_origin, R_q);
    const rotated = _matVec3Small(R_world, _FRANKA_FK_XYZ[i]);
    p_world = [p_world[0]+rotated[0], p_world[1]+rotated[1], p_world[2]+rotated[2]];
    R_world = _mat3MulSmall(R_world, R_iInP);
    poses_R.push(R_world.map((row) => [...row]));
    poses_p.push([...p_world]);
  }
  const ee = poses_p[6];
  // Pass 2: Jacobian columns.
  const J: number[][] = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]];
  for (let i = 0; i < 7; i++) {
    const Ri = poses_R[i];
    // axis_world = third column of R_world_i = R_i[:,2]
    const a: [number, number, number] = [Ri[0][2], Ri[1][2], Ri[2][2]];
    const dp: [number, number, number] = [ee[0] - poses_p[i][0], ee[1] - poses_p[i][1], ee[2] - poses_p[i][2]];
    J[0][i] = a[1] * dp[2] - a[2] * dp[1];
    J[1][i] = a[2] * dp[0] - a[0] * dp[2];
    J[2][i] = a[0] * dp[1] - a[1] * dp[0];
  }
  return { ee, J };
}

// ── Franka full DLS-IK reach (env-parallel) ──────────────────────────────
//
// All-in-one IK kernel — combines iter 88 FK + iter 89 Jacobian + the
// 3×3 cofactor DLS solve from iter 86's CPU demo. Per env, runs one
// DLS step entirely on GPU:
//
//   Pass 1: FK at q → store all 7 joint poses (R, p) in private memory
//   Pass 2: Linear Jacobian columns J[:,i] = axis_i × (p_ee - p_i)
//   Pass 3: err = target - p_ee
//   Pass 4: A = J·Jᵀ + λ²·I   (3×3)
//   Pass 5: A⁻¹ via cofactor expansion
//   Pass 6: y = A⁻¹ · err
//   Pass 7: Δq = Jᵀ · y       (7-vec)
//   Pass 8: q_new = q + α·Δq  (semi-implicit-equivalent step)
//
// Per-env input: q[7] + target[3] = 10 floats
// Per-env output: q_new[7] = 7 floats (in-place over q_in)
// Uniforms: lambda (DLS damping), alpha (step gain)
//
// Total per-thread temp memory: ~150 floats ≈ 600 bytes (well within
// per-invocation private memory limits for any WebGPU adapter).

/** Bindings:
 *    @group(0) @binding(0) q_inout    (storage read_write, N*7 floats — q is overwritten)
 *    @group(0) @binding(1) target_in  (storage read,       N*3 floats; writeback false)
 *    @group(0) @binding(2) lambda     (uniform vec4<f32>.x — DLS damping)
 *    @group(0) @binding(3) alpha      (uniform vec4<f32>.x — step gain)
 */
export const frankaReachKernel: WgpuKernel = wgpuKernel({
  js: (
    qInout: WpArray<number>,
    targetIn: WpArray<number>,
    lambda: number,
    alpha: number,
  ) => {
    const env = tid();
    const base = env * 7;
    const q = [
      qInout.get(base+0), qInout.get(base+1), qInout.get(base+2),
      qInout.get(base+3), qInout.get(base+4), qInout.get(base+5),
      qInout.get(base+6),
    ];
    const tbase = env * 3;
    const target: [number, number, number] = [
      targetIn.get(tbase+0), targetIn.get(tbase+1), targetIn.get(tbase+2),
    ];
    const qNew = frankaReachStepInline(q, target, lambda, alpha);
    for (let i = 0; i < 7; i++) qInout.set(base + i, qNew[i]);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q_inout:   array<f32>;
@group(0) @binding(1) var<storage, read_write> target_in: array<f32>;
@group(0) @binding(2) var<uniform>             lambda_u:  vec4<f32>;
@group(0) @binding(3) var<uniform>             alpha_u:   vec4<f32>;

fn rot_rpy_x(r: f32) -> mat3x3<f32> {
  let cr = cos(r); let sr = sin(r);
  return mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, cr, sr),
    vec3<f32>(0.0, -sr, cr),
  );
}

fn rot_z(angle: f32) -> mat3x3<f32> {
  let c = cos(angle); let s = sin(angle);
  return mat3x3<f32>(
    vec3<f32>(c, s, 0.0),
    vec3<f32>(-s, c, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&q_inout) / 7u;
  if (env >= n_envs) { return; }

  let base_q = env * 7u;
  let base_t = env * 3u;
  var q = array<f32, 7>(
    q_inout[base_q + 0u], q_inout[base_q + 1u], q_inout[base_q + 2u], q_inout[base_q + 3u],
    q_inout[base_q + 4u], q_inout[base_q + 5u], q_inout[base_q + 6u],
  );
  let target = vec3<f32>(target_in[base_t + 0u], target_in[base_t + 1u], target_in[base_t + 2u]);

  let half_pi: f32 = 1.5707963267948966;
  let xyz = array<vec3<f32>, 7>(
    vec3<f32>(0.0,      0.0,     0.333),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.0,     -0.316,   0.0),
    vec3<f32>(0.0825,   0.0,     0.0),
    vec3<f32>(-0.0825,  0.384,   0.0),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.088,    0.0,     0.0),
  );
  let rpy_r = array<f32, 7>(0.0, -half_pi, half_pi, half_pi, -half_pi, half_pi, half_pi);

  // ── Pass 1: FK with stored joint poses ──
  var poses_R: array<mat3x3<f32>, 7>;
  var poses_p: array<vec3<f32>, 7>;
  var R_world = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
  var p_world = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0u; i < 7u; i = i + 1u) {
    let R_origin = rot_rpy_x(rpy_r[i]);
    let R_q = rot_z(q[i]);
    let R_iInP = R_origin * R_q;
    let rotated = R_world * xyz[i];
    p_world = p_world + rotated;
    R_world = R_world * R_iInP;
    poses_R[i] = R_world;
    poses_p[i] = p_world;
  }
  let ee_pos = poses_p[6];

  // ── Pass 2: Linear Jacobian columns ──
  var J: array<vec3<f32>, 7>;   // 7 cols of (vx, vy, vz)
  for (var i = 0u; i < 7u; i = i + 1u) {
    let Ri = poses_R[i];
    let a_world = vec3<f32>(Ri[2].x, Ri[2].y, Ri[2].z);
    let dp = ee_pos - poses_p[i];
    J[i] = vec3<f32>(
      a_world.y * dp.z - a_world.z * dp.y,
      a_world.z * dp.x - a_world.x * dp.z,
      a_world.x * dp.y - a_world.y * dp.x,
    );
  }

  // ── Pass 3: error ──
  let err = target - ee_pos;

  // ── Pass 4: A = J·Jᵀ + λ²·I (3×3) ──
  let lam = lambda_u.x;
  let lam2 = lam * lam;
  var A00 = lam2; var A01 = 0.0; var A02 = 0.0;
  var A11 = lam2; var A12 = 0.0;
  var A22 = lam2;
  for (var i = 0u; i < 7u; i = i + 1u) {
    A00 = A00 + J[i].x * J[i].x;
    A01 = A01 + J[i].x * J[i].y;
    A02 = A02 + J[i].x * J[i].z;
    A11 = A11 + J[i].y * J[i].y;
    A12 = A12 + J[i].y * J[i].z;
    A22 = A22 + J[i].z * J[i].z;
  }
  // A is symmetric (A10 = A01, A20 = A02, A21 = A12)

  // ── Pass 5: A⁻¹ via cofactor expansion (3×3 closed form) ──
  let det = A00 * (A11 * A22 - A12 * A12)
          - A01 * (A01 * A22 - A12 * A02)
          + A02 * (A01 * A12 - A11 * A02);
  if (abs(det) < 1e-18) { return; }
  let invDet = 1.0 / det;
  let inv00 = (A11 * A22 - A12 * A12) * invDet;
  let inv01 = (A02 * A12 - A01 * A22) * invDet;
  let inv02 = (A01 * A12 - A02 * A11) * invDet;
  let inv11 = (A00 * A22 - A02 * A02) * invDet;
  let inv12 = (A02 * A01 - A00 * A12) * invDet;
  let inv22 = (A00 * A11 - A01 * A01) * invDet;

  // ── Pass 6: y = A⁻¹ · err (3-vec) ──
  let y = vec3<f32>(
    inv00 * err.x + inv01 * err.y + inv02 * err.z,
    inv01 * err.x + inv11 * err.y + inv12 * err.z,
    inv02 * err.x + inv12 * err.y + inv22 * err.z,
  );

  // ── Pass 7+8: Δq = Jᵀ · y; q_new = q + α·Δq ──
  let alpha = alpha_u.x;
  for (var i = 0u; i < 7u; i = i + 1u) {
    let dq_i = J[i].x * y.x + J[i].y * y.y + J[i].z * y.z;
    q_inout[base_q + i] = q[i] + alpha * dq_i;
  }
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "uniform", inputIndex: 2 },
    { binding: 3, kind: "uniform", inputIndex: 3 },
  ],
  workgroupSize: 64,
});

/** Reference Franka one-step DLS IK in pure JS — used by
 *  frankaReachKernel's JS fallback AND callable directly.
 *  Performs one DLS-IK step and returns the new q[7].
 */
export function frankaReachStepInline(
  q: readonly number[],
  target: readonly [number, number, number],
  lambda: number,
  alpha: number,
): number[] {
  const { ee, J } = frankaFkJacobianInline(q);
  const err: [number, number, number] = [target[0] - ee[0], target[1] - ee[1], target[2] - ee[2]];
  // A = J·Jᵀ + λ²I
  const lam2 = lambda * lambda;
  let A00 = lam2, A01 = 0, A02 = 0, A11 = lam2, A12 = 0, A22 = lam2;
  for (let i = 0; i < 7; i++) {
    A00 += J[0][i] * J[0][i];
    A01 += J[0][i] * J[1][i];
    A02 += J[0][i] * J[2][i];
    A11 += J[1][i] * J[1][i];
    A12 += J[1][i] * J[2][i];
    A22 += J[2][i] * J[2][i];
  }
  const det = A00 * (A11 * A22 - A12 * A12)
            - A01 * (A01 * A22 - A12 * A02)
            + A02 * (A01 * A12 - A11 * A02);
  if (Math.abs(det) < 1e-18) return [...q];
  const invDet = 1 / det;
  const inv00 = (A11 * A22 - A12 * A12) * invDet;
  const inv01 = (A02 * A12 - A01 * A22) * invDet;
  const inv02 = (A01 * A12 - A02 * A11) * invDet;
  const inv11 = (A00 * A22 - A02 * A02) * invDet;
  const inv12 = (A02 * A01 - A00 * A12) * invDet;
  const inv22 = (A00 * A11 - A01 * A01) * invDet;
  const y: [number, number, number] = [
    inv00 * err[0] + inv01 * err[1] + inv02 * err[2],
    inv01 * err[0] + inv11 * err[1] + inv12 * err[2],
    inv02 * err[0] + inv12 * err[1] + inv22 * err[2],
  ];
  const qNew: number[] = new Array(7);
  for (let i = 0; i < 7; i++) {
    const dq_i = J[0][i] * y[0] + J[1][i] * y[1] + J[2][i] * y[2];
    qNew[i] = q[i] + alpha * dq_i;
  }
  return qNew;
}

// ── Franka analytical gravity compensation (env-parallel) ────────────────
//
// Computes τ_g(q) — the joint torque vector that holds the Franka 7-DoF
// arm against gravity at configuration q. Closes the gap between iter
// 88-90's kinematics-only Franka WGSL and full dynamics.
//
// Algorithm (closed-form, no Featherstone needed):
//   1. Forward kinematics → per-joint world poses (R_i, p_i)
//   2. Per-link COM in world: com_world_k = R_world_k · com_local_k + p_world_k
//   3. For each joint i (revolute, axis world = R_world_i · z):
//      τ_i = a_world_i · Σ_{k≥i} [(com_world_k - p_world_i) × (m_k · g_world)]
//
// Real Franka link masses + COMs from iter 87 (franka_description URDF,
// Apache 2.0).
//
// Per-env input: q[7]
// Per-env output: tau[7]
// Uniform: gravity vec (typically (0, 0, -9.81))

const FRANKA_MASSES: readonly number[] = [2.74, 2.74, 2.38, 2.38, 2.74, 1.55, 0.54];
const FRANKA_COM_LOCAL: ReadonlyArray<readonly [number, number, number]> = [
  [0.003875, 0.002081, -0.04762],
  [-0.003141, -0.02872, 0.003495],
  [0.02785, 0.03094, -0.0961],
  [-0.05317, 0.1046, 0.02711],
  [-0.01121, 0.04123, -0.03825],
  [0.065, -0.016, -0.020],
  [0.010, 0.010, 0.045],
];

/** Bindings:
 *    @group(0) @binding(0) q_in       (storage read, N*7 floats)
 *    @group(0) @binding(1) tau_out    (storage read_write, N*7 floats)
 *    @group(0) @binding(2) gravity    (uniform vec4<f32>: gx, gy, gz, _)
 */
export const frankaGravCompKernel: WgpuKernel = wgpuKernel({
  js: (
    qIn: WpArray<number>,
    tauOut: WpArray<number>,
    gx: number, gy: number, gz: number,
  ) => {
    const env = tid();
    const q: number[] = [];
    for (let i = 0; i < 7; i++) q.push(qIn.get(env * 7 + i));
    const tau = frankaGravCompInline(q, [gx, gy, gz]);
    for (let i = 0; i < 7; i++) tauOut.set(env * 7 + i, tau[i]);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q_in:    array<f32>;
@group(0) @binding(1) var<storage, read_write> tau_out: array<f32>;
@group(0) @binding(2) var<uniform>             g_u:     vec4<f32>;

fn rot_rpy_x(r: f32) -> mat3x3<f32> {
  let cr = cos(r); let sr = sin(r);
  return mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, cr, sr),
    vec3<f32>(0.0, -sr, cr),
  );
}

fn rot_z(angle: f32) -> mat3x3<f32> {
  let c = cos(angle); let s = sin(angle);
  return mat3x3<f32>(
    vec3<f32>(c, s, 0.0),
    vec3<f32>(-s, c, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&tau_out) / 7u;
  if (env >= n_envs) { return; }

  let base = env * 7u;
  let q = array<f32, 7>(
    q_in[base + 0u], q_in[base + 1u], q_in[base + 2u], q_in[base + 3u],
    q_in[base + 4u], q_in[base + 5u], q_in[base + 6u],
  );

  let half_pi: f32 = 1.5707963267948966;
  let xyz = array<vec3<f32>, 7>(
    vec3<f32>(0.0,      0.0,     0.333),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.0,     -0.316,   0.0),
    vec3<f32>(0.0825,   0.0,     0.0),
    vec3<f32>(-0.0825,  0.384,   0.0),
    vec3<f32>(0.0,      0.0,     0.0),
    vec3<f32>(0.088,    0.0,     0.0),
  );
  let rpy_r = array<f32, 7>(0.0, -half_pi, half_pi, half_pi, -half_pi, half_pi, half_pi);

  // Real Franka link masses + COMs (iter 87)
  let masses = array<f32, 7>(2.74, 2.74, 2.38, 2.38, 2.74, 1.55, 0.54);
  let com_local = array<vec3<f32>, 7>(
    vec3<f32>(0.003875,   0.002081,   -0.04762),
    vec3<f32>(-0.003141,  -0.02872,    0.003495),
    vec3<f32>(0.02785,    0.03094,    -0.0961),
    vec3<f32>(-0.05317,   0.1046,      0.02711),
    vec3<f32>(-0.01121,   0.04123,    -0.03825),
    vec3<f32>(0.065,     -0.016,      -0.020),
    vec3<f32>(0.010,      0.010,       0.045),
  );

  let g = vec3<f32>(g_u.x, g_u.y, g_u.z);

  // Pass 1: forward kinematics, store per-joint pose + per-link COM world
  var poses_R: array<mat3x3<f32>, 7>;
  var poses_p: array<vec3<f32>, 7>;
  var com_world: array<vec3<f32>, 7>;
  var R_world = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
  var p_world = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0u; i < 7u; i = i + 1u) {
    let R_origin = rot_rpy_x(rpy_r[i]);
    let R_q = rot_z(q[i]);
    let R_iInP = R_origin * R_q;
    let rotated = R_world * xyz[i];
    p_world = p_world + rotated;
    R_world = R_world * R_iInP;
    poses_R[i] = R_world;
    poses_p[i] = p_world;
    com_world[i] = R_world * com_local[i] + p_world;
  }

  // Pass 2: per-joint gravity torque via cross-product accumulation
  // τ_i = a_world_i · Σ_{k≥i} (com_world_k - p_world_i) × (m_k · g_world)
  for (var i = 0u; i < 7u; i = i + 1u) {
    let Ri = poses_R[i];
    let a_world = vec3<f32>(Ri[2].x, Ri[2].y, Ri[2].z);
    var torque_sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = i; k < 7u; k = k + 1u) {
      let r_arm = com_world[k] - poses_p[i];
      let F = g * masses[k];
      // r_arm × F
      let cross_rF = vec3<f32>(
        r_arm.y * F.z - r_arm.z * F.y,
        r_arm.z * F.x - r_arm.x * F.z,
        r_arm.x * F.y - r_arm.y * F.x,
      );
      torque_sum = torque_sum + cross_rF;
    }
    // τ_i = a · torque_sum (dot product, scalar). Sign convention: τ_g compensates
    // gravity, so we negate (τ_g such that adding it cancels the gravity-induced motion).
    tau_out[base + i] = -(a_world.x * torque_sum.x + a_world.y * torque_sum.y + a_world.z * torque_sum.z);
  }
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
    { binding: 2, kind: "uniform", inputIndex: 2 },   // wrap gx into vec4<f32>.x; others ignored for now
  ],
  workgroupSize: 64,
});

// Note: for compactness, the WGSL binding 2 uses a vec4<f32> uniform with
// gravity in the first 3 components. The TS side only sends a single f32
// value through the uniform binding mechanism in iter 77, so for now the
// JS fallback path is the canonical one. To use the WGSL path, callers
// should write the full (gx, gy, gz) tuple to a uniform buffer manually.
// (Iter 77 wgpu-backend writes single scalar; full 3-vec uniform is
// straightforward inline in the iter 91 demo pattern.)

/** Reference Franka gravity-compensation torque in pure JS. */
export function frankaGravCompInline(
  q: readonly number[],
  gravity: readonly [number, number, number] = [0, 0, -9.81],
): number[] {
  // Pass 1: FK with stored joint poses + COM world.
  const poses_R: number[][][] = [];
  const poses_p: [number, number, number][] = [];
  const com_world: [number, number, number][] = [];
  let R_world: number[][] = [[1,0,0],[0,1,0],[0,0,1]];
  let p_world: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 7; i++) {
    const R_origin = _rotRpy(_FRANKA_FK_RPY_R[i]);
    const R_q = _rotZ(q[i]);
    const R_iInP = _mat3MulSmall(R_origin, R_q);
    const rotated = _matVec3Small(R_world, _FRANKA_FK_XYZ[i]);
    p_world = [p_world[0]+rotated[0], p_world[1]+rotated[1], p_world[2]+rotated[2]];
    R_world = _mat3MulSmall(R_world, R_iInP);
    poses_R.push(R_world.map((row) => [...row]));
    poses_p.push([...p_world]);
    // com_world = R_world · com_local + p_world
    const c_local = FRANKA_COM_LOCAL[i];
    const c_rot = _matVec3Small(R_world, c_local);
    com_world.push([p_world[0]+c_rot[0], p_world[1]+c_rot[1], p_world[2]+c_rot[2]]);
  }
  const tau: number[] = new Array(7);
  // Pass 2: per-joint torque.
  for (let i = 0; i < 7; i++) {
    const Ri = poses_R[i];
    const a: [number, number, number] = [Ri[0][2], Ri[1][2], Ri[2][2]];
    let tx = 0, ty = 0, tz = 0;
    for (let k = i; k < 7; k++) {
      const r0 = com_world[k][0] - poses_p[i][0];
      const r1 = com_world[k][1] - poses_p[i][1];
      const r2 = com_world[k][2] - poses_p[i][2];
      const m_k = FRANKA_MASSES[k];
      const fx = gravity[0] * m_k, fy = gravity[1] * m_k, fz = gravity[2] * m_k;
      tx += r1 * fz - r2 * fy;
      ty += r2 * fx - r0 * fz;
      tz += r0 * fy - r1 * fx;
    }
    tau[i] = -(a[0]*tx + a[1]*ty + a[2]*tz);
  }
  return tau;
}

// ── ANYmal C 12-DoF branched-chain FK (env-parallel) ─────────────────────
//
// First branched-chain WGSL kernel. ANYmal C has 4 legs (LF, LH, RF, RH)
// each with 3 joints (HAA, HFE, KFE) rooted at a common base. Each
// thread runs FK on all 4 legs independently and outputs 4 foot
// positions per env.
//
// Per-env input: q[12]   (LF_HAA, LF_HFE, LF_KFE, LH_HAA, LH_HFE, ...)
// Per-env output: feet[12] = 4 feet × 3 (xyz) per env
//
// Joint specs match iter 75 AnymalC asset:
//   per leg: HAA axis=(1,0,0), HFE axis=(0,1,0), KFE axis=(0,1,0)
//   origin xyz=(0,0,-0.15) downward stack
//   origin rpy=(0,0,0)
//
// Same closed-form per-joint frame composition as iter 88, but the
// pattern now runs 4× (once per leg).
//
// NOTE: iter 75 uses placeholder joint origins; real ANYmal has
// per-leg base offsets like LF_HAA at (0.277, 0.116, 0.043). A future
// iter could update iter 75 with real ANYbotics URDF offsets (similar
// to iter 85 for Franka). The kernel architecture supports it without
// changes — only the AnymalC asset needs updating.

const ANYMAL_LEG_NAMES = ["LF", "LH", "RF", "RH"] as const;
const ANYMAL_JOINT_AXES: readonly (readonly [number, number, number])[] = [
  [1, 0, 0], // HAA — hip ab/adduction
  [0, 1, 0], // HFE — hip flex/extension
  [0, 1, 0], // KFE — knee flex/extension
];

/** Bindings:
 *    @group(0) @binding(0) q_in    (storage read, N*12 floats; writeback false)
 *    @group(0) @binding(1) feet_out (storage read_write, N*12 floats — 4 feet × xyz)
 */
export const anymalFkKernel: WgpuKernel = wgpuKernel({
  js: (qIn: WpArray<number>, feetOut: WpArray<number>) => {
    const env = tid();
    const q: number[] = [];
    for (let i = 0; i < 12; i++) q.push(qIn.get(env * 12 + i));
    const feet = anymalFkInline(q);
    for (let leg = 0; leg < 4; leg++) {
      for (let k = 0; k < 3; k++) {
        feetOut.set(env * 12 + leg * 3 + k, feet[leg][k]);
      }
    }
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q_in:     array<f32>;
@group(0) @binding(1) var<storage, read_write> feet_out: array<f32>;

fn rot_axis(axis: vec3<f32>, angle: f32) -> mat3x3<f32> {
  let c = cos(angle); let s = sin(angle);
  let oc = 1.0 - c;
  let ax = axis.x; let ay = axis.y; let az = axis.z;
  return mat3x3<f32>(
    vec3<f32>(c + ax*ax*oc,        ay*ax*oc + az*s,    az*ax*oc - ay*s),
    vec3<f32>(ax*ay*oc - az*s,     c + ay*ay*oc,        az*ay*oc + ax*s),
    vec3<f32>(ax*az*oc + ay*s,     ay*az*oc - ax*s,    c + az*az*oc),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&feet_out) / 12u;
  if (env >= n_envs) { return; }

  let base_q = env * 12u;
  let base_out = env * 12u;

  // Real ANYbotics joint axes (HAA, HFE, KFE per leg).
  let axes = array<vec3<f32>, 3>(
    vec3<f32>(1.0, 0.0, 0.0),    // HAA
    vec3<f32>(0.0, 1.0, 0.0),    // HFE
    vec3<f32>(0.0, 1.0, 0.0),    // KFE
  );

  // Per-leg HAA base attachment (iter 94: real ANYbotics origins).
  // Order: LF, LH, RF, RH
  let haa_base = array<vec3<f32>, 4>(
    vec3<f32>( 0.277,  0.116, 0.0),  // LF
    vec3<f32>(-0.277,  0.116, 0.0),  // LH
    vec3<f32>( 0.277, -0.116, 0.0),  // RF
    vec3<f32>(-0.277, -0.116, 0.0),  // RH
  );

  // Canonical local joint origins (same for all legs).
  let hfe_local = vec3<f32>(0.0, 0.0635, 0.0);     // HFE in HAA frame
  let kfe_local = vec3<f32>(0.0, 0.041, -0.317);   // KFE in HFE frame
  let foot_local = vec3<f32>(0.0, 0.0, -0.317);    // foot in KFE frame

  // 4 legs, processed sequentially within this thread.
  for (var leg = 0u; leg < 4u; leg = leg + 1u) {
    // Start at per-leg HAA attachment point with identity frame.
    var R_world = mat3x3<f32>(
      vec3<f32>(1.0, 0.0, 0.0),
      vec3<f32>(0.0, 1.0, 0.0),
      vec3<f32>(0.0, 0.0, 1.0),
    );
    var p_world = haa_base[leg];
    // HAA joint
    R_world = R_world * rot_axis(axes[0], q_in[base_q + leg * 3u + 0u]);
    // HFE origin + joint
    p_world = p_world + R_world * hfe_local;
    R_world = R_world * rot_axis(axes[1], q_in[base_q + leg * 3u + 1u]);
    // KFE origin + joint
    p_world = p_world + R_world * kfe_local;
    R_world = R_world * rot_axis(axes[2], q_in[base_q + leg * 3u + 2u]);
    // Foot origin
    let foot = p_world + R_world * foot_local;
    feet_out[base_out + leg * 3u + 0u] = foot.x;
    feet_out[base_out + leg * 3u + 1u] = foot.y;
    feet_out[base_out + leg * 3u + 2u] = foot.z;
  }
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
  ],
  workgroupSize: 64,
});

function _rotAxis(axis: readonly number[], angle: number): number[][] {
  const c = Math.cos(angle), s = Math.sin(angle), oc = 1 - c;
  const [ax, ay, az] = axis;
  return [
    [c + ax*ax*oc,       ax*ay*oc - az*s,   ax*az*oc + ay*s],
    [ay*ax*oc + az*s,    c + ay*ay*oc,      ay*az*oc - ax*s],
    [az*ax*oc - ay*s,    az*ay*oc + ax*s,   c + az*az*oc],
  ];
}

/** Reference ANYmal C foot positions in pure JS — used by anymalFkKernel's
 *  JS fallback AND callable directly. Returns 4 foot positions, one per
 *  leg in LF, LH, RF, RH order.
 *  Real ANYbotics joint origins per iter 94.
 */
const ANYMAL_HAA_BASE: ReadonlyArray<readonly [number, number, number]> = [
  [ 0.277,  0.116, 0.0],  // LF
  [-0.277,  0.116, 0.0],  // LH
  [ 0.277, -0.116, 0.0],  // RF
  [-0.277, -0.116, 0.0],  // RH
];
const ANYMAL_HFE_LOCAL: readonly [number, number, number] = [0, 0.0635, 0];
const ANYMAL_KFE_LOCAL: readonly [number, number, number] = [0, 0.041, -0.317];
const ANYMAL_FOOT_LOCAL: readonly [number, number, number] = [0, 0, -0.317];

export function anymalFkInline(q: readonly number[]): [number, number, number][] {
  const feet: [number, number, number][] = [];
  for (let leg = 0; leg < 4; leg++) {
    let R_world: number[][] = [[1,0,0],[0,1,0],[0,0,1]];
    let p_world: [number, number, number] = [
      ANYMAL_HAA_BASE[leg][0],
      ANYMAL_HAA_BASE[leg][1],
      ANYMAL_HAA_BASE[leg][2],
    ];
    // HAA joint
    R_world = _mat3MulSmall(R_world, _rotAxis(ANYMAL_JOINT_AXES[0], q[leg * 3 + 0]));
    // HFE origin (in HAA frame) + joint
    const rH = _matVec3Small(R_world, ANYMAL_HFE_LOCAL);
    p_world = [p_world[0] + rH[0], p_world[1] + rH[1], p_world[2] + rH[2]];
    R_world = _mat3MulSmall(R_world, _rotAxis(ANYMAL_JOINT_AXES[1], q[leg * 3 + 1]));
    // KFE origin (in HFE frame) + joint
    const rK = _matVec3Small(R_world, ANYMAL_KFE_LOCAL);
    p_world = [p_world[0] + rK[0], p_world[1] + rK[1], p_world[2] + rK[2]];
    R_world = _mat3MulSmall(R_world, _rotAxis(ANYMAL_JOINT_AXES[2], q[leg * 3 + 2]));
    // Foot origin (in KFE frame)
    const rF = _matVec3Small(R_world, ANYMAL_FOOT_LOCAL);
    feet.push([p_world[0] + rF[0], p_world[1] + rF[1], p_world[2] + rF[2]]);
  }
  return feet;
}

// ── Generic serial-chain FK (env-parallel, arbitrary N ≤ 12) ─────────────
//
// Generalises iter 88's Franka-specific FK kernel to arbitrary
// serial-chain revolute/prismatic arms. Joint origins, rpy, and axes
// are runtime storage inputs (per-joint), so a single kernel handles
// Franka (N=7), UR10 (N=6), KUKA iiwa (N=7), Kinova Gen3 (N=7),
// Universal Robots UR5/UR16, etc.
//
// Max N=12 at compile time (covers all practical serial arms).
//
// Per-env input: q[N]
// Robot-config storage: xyz[N×3] + rpy[N×3] + axis[N×3] = 9·N floats
// Uniform: n (joint count)
// Per-env output: ee_pos[3]
//
// Algorithm: same R_origin · R_axis(q) frame composition as iter 88,
// but with axis as runtime input (so prismatic joints can be supported
// by setting axis to the translation direction and using prismatic
// joint formula — though current kernel only supports revolute via
// rot_axis). For prismatic support, a per-joint "kind" flag could be
// added later.

const MAX_N_SERIAL_FK = 12;

/** Bindings:
 *    @group(0) @binding(0) q_in        (storage read, envs × N floats)
 *    @group(0) @binding(1) joint_xyz   (storage read, N × 3 floats) — per-joint origin xyz
 *    @group(0) @binding(2) joint_rpy   (storage read, N × 3 floats) — per-joint origin rpy
 *    @group(0) @binding(3) joint_axis  (storage read, N × 3 floats) — per-joint axis (body frame)
 *    @group(0) @binding(4) ee_out      (storage read_write, envs × 3 floats)
 *    @group(0) @binding(5) n_uniform   (uniform vec4<u32>: x = N joint count)
 */
export const genericSerialFkKernel: WgpuKernel = wgpuKernel({
  js: (
    qIn: WpArray<number>,
    jointXyz: WpArray<number>,
    jointRpy: WpArray<number>,
    jointAxis: WpArray<number>,
    eeOut: WpArray<number>,
    n: number,
  ) => {
    const env = tid();
    const q: number[] = [];
    for (let i = 0; i < n; i++) q.push(qIn.get(env * n + i));
    const xyz: readonly [number, number, number][] = [];
    const rpy: readonly [number, number, number][] = [];
    const axis: readonly [number, number, number][] = [];
    const xyzArr = xyz as [number, number, number][];
    const rpyArr = rpy as [number, number, number][];
    const axisArr = axis as [number, number, number][];
    for (let i = 0; i < n; i++) {
      xyzArr.push([jointXyz.get(i*3+0), jointXyz.get(i*3+1), jointXyz.get(i*3+2)]);
      rpyArr.push([jointRpy.get(i*3+0), jointRpy.get(i*3+1), jointRpy.get(i*3+2)]);
      axisArr.push([jointAxis.get(i*3+0), jointAxis.get(i*3+1), jointAxis.get(i*3+2)]);
    }
    const ee = genericSerialFkInline(q, xyz, rpy, axis);
    eeOut.set(env * 3 + 0, ee[0]);
    eeOut.set(env * 3 + 1, ee[1]);
    eeOut.set(env * 3 + 2, ee[2]);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q_in:       array<f32>;
@group(0) @binding(1) var<storage, read_write> joint_xyz:  array<f32>;
@group(0) @binding(2) var<storage, read_write> joint_rpy:  array<f32>;
@group(0) @binding(3) var<storage, read_write> joint_axis: array<f32>;
@group(0) @binding(4) var<storage, read_write> ee_out:     array<f32>;
@group(0) @binding(5) var<uniform>             n_uniform:  vec4<u32>;

fn rot_rpy(r: f32, p: f32, y: f32) -> mat3x3<f32> {
  let cr = cos(r); let sr = sin(r);
  let cp = cos(p); let sp = sin(p);
  let cy = cos(y); let sy = sin(y);
  return mat3x3<f32>(
    vec3<f32>(cy*cp, sy*cp, -sp),
    vec3<f32>(cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr),
    vec3<f32>(cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr),
  );
}

fn rot_axis(axis: vec3<f32>, angle: f32) -> mat3x3<f32> {
  let c = cos(angle); let s = sin(angle);
  let oc = 1.0 - c;
  let ax = axis.x; let ay = axis.y; let az = axis.z;
  return mat3x3<f32>(
    vec3<f32>(c + ax*ax*oc,        ay*ax*oc + az*s,    az*ax*oc - ay*s),
    vec3<f32>(ax*ay*oc - az*s,     c + ay*ay*oc,        az*ay*oc + ax*s),
    vec3<f32>(ax*az*oc + ay*s,     ay*az*oc - ax*s,    c + az*az*oc),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n = n_uniform.x;
  let n_envs = arrayLength(&ee_out) / 3u;
  if (env >= n_envs) { return; }
  let base = env * n;

  var R_world = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
  );
  var p_world = vec3<f32>(0.0, 0.0, 0.0);

  // Loop bounded at MAX_N=12; runtime n controls early exit.
  for (var i = 0u; i < 12u; i = i + 1u) {
    if (i >= n) { break; }
    let xyz = vec3<f32>(joint_xyz[i*3u + 0u], joint_xyz[i*3u + 1u], joint_xyz[i*3u + 2u]);
    let R_origin = rot_rpy(joint_rpy[i*3u + 0u], joint_rpy[i*3u + 1u], joint_rpy[i*3u + 2u]);
    let axis = vec3<f32>(joint_axis[i*3u + 0u], joint_axis[i*3u + 1u], joint_axis[i*3u + 2u]);
    let R_q = rot_axis(axis, q_in[base + i]);
    let R_iInP = R_origin * R_q;
    let rotated = R_world * xyz;
    p_world = p_world + rotated;
    R_world = R_world * R_iInP;
  }
  ee_out[env * 3u + 0u] = p_world.x;
  ee_out[env * 3u + 1u] = p_world.y;
  ee_out[env * 3u + 2u] = p_world.z;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: false },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: true },
    { binding: 5, kind: "uniform", inputIndex: 5 },
  ],
  workgroupSize: 64,
});

/** Reference generic serial-chain FK in pure JS. */
export function genericSerialFkInline(
  q: readonly number[],
  jointXyz: ReadonlyArray<readonly [number, number, number]>,
  jointRpy: ReadonlyArray<readonly [number, number, number]>,
  jointAxis: ReadonlyArray<readonly [number, number, number]>,
): [number, number, number] {
  let R_world: number[][] = [[1,0,0],[0,1,0],[0,0,1]];
  let p_world: [number, number, number] = [0, 0, 0];
  const n = q.length;
  for (let i = 0; i < n; i++) {
    const R_origin = _rotRpyFull(jointRpy[i]);
    const R_q = _rotAxis(jointAxis[i], q[i]);
    const R_iInP = _mat3MulSmall(R_origin, R_q);
    const rotated = _matVec3Small(R_world, jointXyz[i]);
    p_world = [p_world[0]+rotated[0], p_world[1]+rotated[1], p_world[2]+rotated[2]];
    R_world = _mat3MulSmall(R_world, R_iInP);
  }
  return p_world;
}

function _rotRpyFull(rpy: readonly [number, number, number]): number[][] {
  const [r, p, y] = rpy;
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  return [
    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
    [-sp,   cp*sr,             cp*cr],
  ];
}

// ── PD joint controller (env-parallel) ───────────────────────────────────
//
// The most-used controller in Isaac Lab task definitions
// (`JointPositionPDController` / equivalent). One thread per (env, joint)
// pair computes:
//
//     τ_i = Kp_i · (q*_i − q_i) − Kd_i · q̇_i
//
// Per-joint Kp/Kd are shared across envs (storage), so a 1024-env ×
// 12-joint dispatch needs only 24 floats of joint config.
//
// This is the bridge between policy-net output (target q) and the
// τ-driven ABA forward dynamics already implemented at iter 68.

export const pdJointControllerKernel: WgpuKernel = wgpuKernel({
  js: (
    qActual: WpArray<number>,
    qdActual: WpArray<number>,
    qTarget: WpArray<number>,
    kp: WpArray<number>,
    kd: WpArray<number>,
    tauOut: WpArray<number>,
    n: number,
  ) => {
    const idx = tid();
    const j = idx % n;
    const qErr = qTarget.get(idx) - qActual.get(idx);
    const tau = kp.get(j) * qErr - kd.get(j) * qdActual.get(idx);
    tauOut.set(idx, tau);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q_actual:  array<f32>;
@group(0) @binding(1) var<storage, read_write> qd_actual: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_target:  array<f32>;
@group(0) @binding(3) var<storage, read_write> kp:        array<f32>;
@group(0) @binding(4) var<storage, read_write> kd:        array<f32>;
@group(0) @binding(5) var<storage, read_write> tau_out:   array<f32>;
@group(0) @binding(6) var<uniform>             n_uniform: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&tau_out);
  if (idx >= total) { return; }
  let n = n_uniform.x;
  let j = idx % n;
  let q_err = q_target[idx] - q_actual[idx];
  tau_out[idx] = kp[j] * q_err - kd[j] * qd_actual[idx];
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: false },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "storage", inputIndex: 5, writeback: true },
    { binding: 6, kind: "uniform", inputIndex: 6 },
  ],
  workgroupSize: 64,
});

/** Reference PD joint controller in pure JS.
 *  τ[i] = Kp[i % n] · (q*[i] − q[i]) − Kd[i % n] · q̇[i]
 */
export function pdJointControllerInline(
  qActual: readonly number[],
  qdActual: readonly number[],
  qTarget: readonly number[],
  kp: readonly number[],
  kd: readonly number[],
  n: number,
): number[] {
  const out = new Array(qActual.length);
  for (let i = 0; i < qActual.length; i++) {
    const j = i % n;
    out[i] = kp[j] * (qTarget[i] - qActual[i]) - kd[j] * qdActual[i];
  }
  return out;
}

// ── Action scale + clamp (env×joint parallel) ────────────────────────────
//
// Isaac Lab's `ActionManager` canonical pipeline:
//
//     q_target[i] = clamp(
//         action_scale[j] · action[i] + action_offset[j],
//         q_lower[j], q_upper[j])
//
// Sits between the policy net (action ∈ [-1, 1]^N typically tanh-squashed)
// and the PD controller (iter 100). Per-joint scale/offset/limits are
// shared across envs (storage), so 1024-env × 12-joint dispatch needs
// only 48 floats of joint config.

export const actionScaleClampKernel: WgpuKernel = wgpuKernel({
  js: (
    action: WpArray<number>,
    actionScale: WpArray<number>,
    actionOffset: WpArray<number>,
    qLower: WpArray<number>,
    qUpper: WpArray<number>,
    qTargetOut: WpArray<number>,
    n: number,
  ) => {
    const idx = tid();
    const j = idx % n;
    const raw = actionScale.get(j) * action.get(idx) + actionOffset.get(j);
    const lo = qLower.get(j);
    const hi = qUpper.get(j);
    const clamped = raw < lo ? lo : raw > hi ? hi : raw;
    qTargetOut.set(idx, clamped);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> action:       array<f32>;
@group(0) @binding(1) var<storage, read_write> action_scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> action_offset:array<f32>;
@group(0) @binding(3) var<storage, read_write> q_lower:      array<f32>;
@group(0) @binding(4) var<storage, read_write> q_upper:      array<f32>;
@group(0) @binding(5) var<storage, read_write> q_target_out: array<f32>;
@group(0) @binding(6) var<uniform>             n_uniform:    vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&q_target_out);
  if (idx >= total) { return; }
  let n = n_uniform.x;
  let j = idx % n;
  let raw = action_scale[j] * action[idx] + action_offset[j];
  q_target_out[idx] = clamp(raw, q_lower[j], q_upper[j]);
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: false },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "storage", inputIndex: 5, writeback: true },
    { binding: 6, kind: "uniform", inputIndex: 6 },
  ],
  workgroupSize: 64,
});

/** Reference action scale + clamp in pure JS. */
export function actionScaleClampInline(
  action: readonly number[],
  actionScale: readonly number[],
  actionOffset: readonly number[],
  qLower: readonly number[],
  qUpper: readonly number[],
  n: number,
): number[] {
  const out = new Array(action.length);
  for (let i = 0; i < action.length; i++) {
    const j = i % n;
    const raw = actionScale[j] * action[i] + actionOffset[j];
    out[i] = raw < qLower[j] ? qLower[j] : raw > qUpper[j] ? qUpper[j] : raw;
  }
  return out;
}

// ── Effort saturation (env×joint parallel, in-place) ─────────────────────
//
// Canonical safety layer between PD controller (iter 100) and the
// articulated-body forward dynamics (iter 68). Caps each joint's torque
// at its actuator effort limit, matching Isaac Lab's per-joint
// `effort_limit_sim` clamp.
//
//     τ_i = clamp(τ_i, −effort_limit_j, +effort_limit_j)
//
// In-place storage write; per-joint effort_limit shared across envs.

export const effortLimitKernel: WgpuKernel = wgpuKernel({
  js: (
    tau: WpArray<number>,
    effortLimit: WpArray<number>,
    n: number,
  ) => {
    const idx = tid();
    const j = idx % n;
    const lim = effortLimit.get(j);
    const t = tau.get(idx);
    tau.set(idx, t < -lim ? -lim : t > lim ? lim : t);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> tau:          array<f32>;
@group(0) @binding(1) var<storage, read_write> effort_limit: array<f32>;
@group(0) @binding(2) var<uniform>             n_uniform:    vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&tau);
  if (idx >= total) { return; }
  let n = n_uniform.x;
  let j = idx % n;
  let lim = effort_limit[j];
  tau[idx] = clamp(tau[idx], -lim, lim);
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "uniform", inputIndex: 2 },
  ],
  workgroupSize: 64,
});

/** Reference effort-saturation in pure JS (in-place mutation of tau). */
export function effortLimitInline(
  tau: number[],
  effortLimit: readonly number[],
  n: number,
): void {
  for (let i = 0; i < tau.length; i++) {
    const j = i % n;
    const lim = effortLimit[j];
    if (tau[i] < -lim) tau[i] = -lim;
    else if (tau[i] > lim) tau[i] = lim;
  }
}

// ── Observation normalize + noise + clamp (env×feature parallel) ─────────
//
// Isaac Lab's ObservationManager canonical pipeline:
//
//     obs[i] = clamp(
//         (raw[i] − mean[j]) / std[j] + noise[i],
//         clamp_low[j], clamp_high[j])
//
// Host pre-generates `noise[i]` (Gaussian or any; WGSL has no portable
// RNG), so the kernel itself is deterministic — the test path can pass
// zeros to verify the normalize+clamp path independently.
//
// Per-feature mean/std/clamp shared across envs (storage); per-(env,
// feature) noise. Mirror of actionScaleClampKernel but on the obs side.

export const observationNormalizeKernel: WgpuKernel = wgpuKernel({
  js: (
    obs: WpArray<number>,
    mean: WpArray<number>,
    std: WpArray<number>,
    noise: WpArray<number>,
    clampLow: WpArray<number>,
    clampHigh: WpArray<number>,
    n: number,
  ) => {
    const idx = tid();
    const j = idx % n;
    const sigma = Math.max(std.get(j), 1e-8);
    const normalized = (obs.get(idx) - mean.get(j)) / sigma + noise.get(idx);
    const lo = clampLow.get(j);
    const hi = clampHigh.get(j);
    obs.set(idx, normalized < lo ? lo : normalized > hi ? hi : normalized);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> obs:        array<f32>;
@group(0) @binding(1) var<storage, read_write> mean:       array<f32>;
@group(0) @binding(2) var<storage, read_write> std:        array<f32>;
@group(0) @binding(3) var<storage, read_write> noise:      array<f32>;
@group(0) @binding(4) var<storage, read_write> clamp_low:  array<f32>;
@group(0) @binding(5) var<storage, read_write> clamp_high: array<f32>;
@group(0) @binding(6) var<uniform>             n_uniform:  vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&obs);
  if (idx >= total) { return; }
  let n = n_uniform.x;
  let j = idx % n;
  let sigma = max(std[j], 1e-8);
  let normalized = (obs[idx] - mean[j]) / sigma + noise[idx];
  obs[idx] = clamp(normalized, clamp_low[j], clamp_high[j]);
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: false },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "storage", inputIndex: 5, writeback: false },
    { binding: 6, kind: "uniform", inputIndex: 6 },
  ],
  workgroupSize: 64,
});

/** Reference observation normalize + noise + clamp (in-place mutation of obs). */
export function observationNormalizeInline(
  obs: number[],
  mean: readonly number[],
  std: readonly number[],
  noise: readonly number[],
  clampLow: readonly number[],
  clampHigh: readonly number[],
  n: number,
): void {
  for (let i = 0; i < obs.length; i++) {
    const j = i % n;
    const sigma = Math.max(std[j], 1e-8);
    const v = (obs[i] - mean[j]) / sigma + noise[i];
    obs[i] = v < clampLow[j] ? clampLow[j] : v > clampHigh[j] ? clampHigh[j] : v;
  }
}

/** Host-side Marsaglia polar Gaussian sampler.
 *  Deterministic given a uniform RNG (e.g. mulberry32) so the obs noise
 *  path can be tested byte-for-byte across runs.
 */
export function gaussianMarsaglia(rng: () => number, count: number): number[] {
  const out: number[] = [];
  while (out.length < count) {
    let u: number, v: number, s: number;
    do {
      u = 2 * rng() - 1;
      v = 2 * rng() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt(-2 * Math.log(s) / s);
    out.push(u * f);
    if (out.length < count) out.push(v * f);
  }
  return out;
}

/** mulberry32 deterministic PRNG (good enough for test reproducibility). */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── L2-norm-squared per env (universal reward building block) ────────────
//
// Most Isaac Lab reward terms reduce to ||·||² of some per-env vector:
//   • action_l2          = ||a||²
//   • action_rate_l2     = ||a_t − a_{t-1}||²
//   • joint_vel_l2       = ||q̇||²
//   • joint_acc_l2       = ||q̈||²
//   • joint_torque_l2    = ||τ||²
//   • flat_orientation_l2 = ||up_vec − ẑ||²
//
// One thread per env iterates a bounded-d input (compile-bound 64; runtime
// d via uniform). Output: result[env] = Σ_i x[env*d + i]².
//
// Compose with combineWeightedRewards() (host helper) and trackVelExpInline()
// to reproduce full Isaac Lab RewardManager scoring per env.

export const l2NormSquaredKernel: WgpuKernel = wgpuKernel({
  js: (
    x: WpArray<number>,
    out: WpArray<number>,
    d: number,
  ) => {
    const env = tid();
    let s = 0;
    for (let i = 0; i < d; i++) {
      const v = x.get(env * d + i);
      s += v * v;
    }
    out.set(env, s);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> x:         array<f32>;
@group(0) @binding(1) var<storage, read_write> out:       array<f32>;
@group(0) @binding(2) var<uniform>             d_uniform: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&out);
  if (env >= n_envs) { return; }
  let d = d_uniform.x;
  var s: f32 = 0.0;
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (i >= d) { break; }
    let v = x[env * d + i];
    s = s + v * v;
  }
  out[env] = s;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: true },
    { binding: 2, kind: "uniform", inputIndex: 2 },
  ],
  workgroupSize: 64,
});

/** Reference per-env ||x||² in pure JS. */
export function l2NormSquaredInline(
  x: readonly number[],
  d: number,
): number[] {
  const nEnvs = x.length / d;
  const out = new Array(nEnvs);
  for (let env = 0; env < nEnvs; env++) {
    let s = 0;
    for (let i = 0; i < d; i++) {
      const v = x[env * d + i];
      s += v * v;
    }
    out[env] = s;
  }
  return out;
}

/** Isaac Lab `track_lin_vel_xy_exp`-style tracking reward (host-side scalar).
 *  reward = exp(−||v_target − v_actual||² / σ²)
 */
export function trackVelExpInline(
  vTarget: readonly number[],
  vActual: readonly number[],
  sigma: number,
  d: number,
): number[] {
  const nEnvs = vTarget.length / d;
  const sigma2 = sigma * sigma;
  const out = new Array(nEnvs);
  for (let env = 0; env < nEnvs; env++) {
    let s = 0;
    for (let i = 0; i < d; i++) {
      const dv = vTarget[env * d + i] - vActual[env * d + i];
      s += dv * dv;
    }
    out[env] = Math.exp(-s / sigma2);
  }
  return out;
}

/** Compose K per-env reward terms with per-term weights into total per-env reward.
 *  Mirror of Isaac Lab's RewardManager.compute(): weighted sum of term values.
 */
export function combineWeightedRewards(
  termValues: ReadonlyArray<readonly number[]>,
  weights: readonly number[],
): number[] {
  const nEnvs = termValues[0].length;
  const out = new Array(nEnvs).fill(0);
  for (let k = 0; k < termValues.length; k++) {
    const term = termValues[k];
    const w = weights[k];
    for (let env = 0; env < nEnvs; env++) {
      out[env] += w * term[env];
    }
  }
  return out;
}

// ── Episode terminations + truncations (env-parallel) ────────────────────
//
// Isaac Lab `TerminationManager` canonical pattern. Per-env booleans
// (encoded as f32 0.0/1.0) for:
//   terminated[env] = joint_limit_violation OR base_fall   (MDP termination)
//   truncated[env]  = step_count >= max_steps              (MDP truncation)
//
// Done = terminated OR truncated → host triggers `reset_env(env)`.
// One thread per env scans q[env*n..env*n+n] for limit violation +
// checks base_z + step_count.

const MAX_N_TERM = 32;

export const terminationsKernel: WgpuKernel = wgpuKernel({
  js: (
    q: WpArray<number>,
    qLower: WpArray<number>,
    qUpper: WpArray<number>,
    baseZ: WpArray<number>,
    step: WpArray<number>,
    terminated: WpArray<number>,
    truncated: WpArray<number>,
    n: number,
    minBaseZ: number,
    maxSteps: number,
  ) => {
    const env = tid();
    let limitViolated = 0;
    for (let i = 0; i < n; i++) {
      const v = q.get(env * n + i);
      if (v < qLower.get(i) || v > qUpper.get(i)) { limitViolated = 1; break; }
    }
    const fell = baseZ.get(env) < minBaseZ ? 1 : 0;
    const timedOut = step.get(env) >= maxSteps ? 1 : 0;
    terminated.set(env, (limitViolated || fell) ? 1 : 0);
    truncated.set(env, timedOut);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> q:          array<f32>;
@group(0) @binding(1) var<storage, read_write> q_lower:    array<f32>;
@group(0) @binding(2) var<storage, read_write> q_upper:    array<f32>;
@group(0) @binding(3) var<storage, read_write> base_z:     array<f32>;
@group(0) @binding(4) var<storage, read_write> step:       array<f32>;
@group(0) @binding(5) var<storage, read_write> terminated: array<f32>;
@group(0) @binding(6) var<storage, read_write> truncated:  array<f32>;
@group(0) @binding(7) var<uniform>             params:     vec4<f32>;
//                                                          .x = n_joints (cast u32)
//                                                          .y = min_base_z
//                                                          .z = max_steps

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let n_envs = arrayLength(&terminated);
  if (env >= n_envs) { return; }
  let n = u32(params.x);
  let min_base_z = params.y;
  let max_steps = params.z;

  var limit_violated: u32 = 0u;
  for (var i = 0u; i < 32u; i = i + 1u) {
    if (i >= n) { break; }
    let v = q[env * n + i];
    if (v < q_lower[i] || v > q_upper[i]) {
      limit_violated = 1u;
      break;
    }
  }
  let fell = select(0u, 1u, base_z[env] < min_base_z);
  let timed_out = select(0u, 1u, step[env] >= max_steps);
  terminated[env] = select(0.0, 1.0, (limit_violated | fell) != 0u);
  truncated[env] = select(0.0, 1.0, timed_out != 0u);
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: false },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "storage", inputIndex: 5, writeback: true },
    { binding: 6, kind: "storage", inputIndex: 6, writeback: true },
    { binding: 7, kind: "uniform", inputIndex: 7 },
  ],
  workgroupSize: 64,
});

/** Reference terminations + truncations per env. */
export function terminationsInline(
  q: readonly number[],
  qLower: readonly number[],
  qUpper: readonly number[],
  baseZ: readonly number[],
  step: readonly number[],
  n: number,
  minBaseZ: number,
  maxSteps: number,
): { terminated: number[]; truncated: number[] } {
  const nEnvs = baseZ.length;
  const terminated = new Array(nEnvs);
  const truncated = new Array(nEnvs);
  for (let env = 0; env < nEnvs; env++) {
    let limitViolated = 0;
    for (let i = 0; i < n; i++) {
      const v = q[env * n + i];
      if (v < qLower[i] || v > qUpper[i]) { limitViolated = 1; break; }
    }
    const fell = baseZ[env] < minBaseZ ? 1 : 0;
    const timedOut = step[env] >= maxSteps ? 1 : 0;
    terminated[env] = (limitViolated || fell) ? 1 : 0;
    truncated[env] = timedOut;
  }
  return { terminated, truncated };
}

// ── MLP policy-net forward (env-parallel) ────────────────────────────────
//
// Isaac Lab's PPO/SAC default policy: 2-layer MLP `Linear → ReLU →
// Linear → tanh`. Per-env forward pass:
//
//     hidden  = ReLU(W1 · obs + b1)        shape [hidden_dim]
//     action  = tanh(W2 · hidden + b2)     shape [action_dim]
//
// Weights are shared across envs (storage); per-env obs input + action
// output. Compile-bound MAX_HIDDEN=128 fits comfortably in 32 KB/
// workgroup → iPhone-12 WebGPU safe per ADR-2605241900 edge-target
// invariant.
//
// Closes the trained-policy inference gap: a checkpoint exported as
// (W1, b1, W2, b2) flat float32 arrays can now run in-browser
// alongside the full simulation loop.

const MAX_HIDDEN_MLP = 128;
const MAX_OBS_MLP    = 64;

export const mlpPolicyForwardKernel: WgpuKernel = wgpuKernel({
  js: (
    obs: WpArray<number>,
    W1: WpArray<number>,
    b1: WpArray<number>,
    W2: WpArray<number>,
    b2: WpArray<number>,
    actionOut: WpArray<number>,
    obsDim: number,
    hiddenDim: number,
    actionDim: number,
  ) => {
    const env = tid();
    const hidden = new Array(hiddenDim);
    for (let h = 0; h < hiddenDim; h++) {
      let s = b1.get(h);
      for (let o = 0; o < obsDim; o++) {
        s += W1.get(h * obsDim + o) * obs.get(env * obsDim + o);
      }
      hidden[h] = s > 0 ? s : 0;  // ReLU
    }
    for (let a = 0; a < actionDim; a++) {
      let s = b2.get(a);
      for (let h = 0; h < hiddenDim; h++) {
        s += W2.get(a * hiddenDim + h) * hidden[h];
      }
      actionOut.set(env * actionDim + a, Math.tanh(s));
    }
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> obs:        array<f32>;
@group(0) @binding(1) var<storage, read_write> W1:         array<f32>;
@group(0) @binding(2) var<storage, read_write> b1:         array<f32>;
@group(0) @binding(3) var<storage, read_write> W2:         array<f32>;
@group(0) @binding(4) var<storage, read_write> b2:         array<f32>;
@group(0) @binding(5) var<storage, read_write> action_out: array<f32>;
@group(0) @binding(6) var<uniform>             dims:       vec4<u32>;
//                                                          .x = obs_dim
//                                                          .y = hidden_dim
//                                                          .z = action_dim

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let env = gid.x;
  let obs_dim = dims.x;
  let hidden_dim = dims.y;
  let action_dim = dims.z;
  let n_envs = arrayLength(&action_out) / action_dim;
  if (env >= n_envs) { return; }

  var hidden: array<f32, 128>;
  for (var h = 0u; h < 128u; h = h + 1u) {
    if (h >= hidden_dim) { break; }
    var s: f32 = b1[h];
    for (var o = 0u; o < 64u; o = o + 1u) {
      if (o >= obs_dim) { break; }
      s = s + W1[h * obs_dim + o] * obs[env * obs_dim + o];
    }
    hidden[h] = max(0.0, s);
  }
  for (var a = 0u; a < action_dim; a = a + 1u) {
    var s: f32 = b2[a];
    for (var h = 0u; h < 128u; h = h + 1u) {
      if (h >= hidden_dim) { break; }
      s = s + W2[a * hidden_dim + h] * hidden[h];
    }
    action_out[env * action_dim + a] = tanh(s);
  }
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "storage", inputIndex: 3, writeback: false },
    { binding: 4, kind: "storage", inputIndex: 4, writeback: false },
    { binding: 5, kind: "storage", inputIndex: 5, writeback: true },
    { binding: 6, kind: "uniform", inputIndex: 6 },
  ],
  workgroupSize: 64,
});

/** Reference 2-layer MLP forward in pure JS. */
export function mlpPolicyForwardInline(
  obs: readonly number[],
  W1: readonly number[],
  b1: readonly number[],
  W2: readonly number[],
  b2: readonly number[],
  obsDim: number,
  hiddenDim: number,
  actionDim: number,
): number[] {
  const nEnvs = obs.length / obsDim;
  const out = new Array(nEnvs * actionDim);
  for (let env = 0; env < nEnvs; env++) {
    const hidden = new Array(hiddenDim);
    for (let h = 0; h < hiddenDim; h++) {
      let s = b1[h];
      for (let o = 0; o < obsDim; o++) {
        s += W1[h * obsDim + o] * obs[env * obsDim + o];
      }
      hidden[h] = s > 0 ? s : 0;
    }
    for (let a = 0; a < actionDim; a++) {
      let s = b2[a];
      for (let h = 0; h < hiddenDim; h++) {
        s += W2[a * hiddenDim + h] * hidden[h];
      }
      out[env * actionDim + a] = Math.tanh(s);
    }
  }
  return out;
}

// ── Conditional per-env reset (env×dim parallel) ─────────────────────────
//
// Closes the loop between TerminationManager (iter 105) and next-episode
// initial state. Per-(env, dim) thread: if done[env] == 1, copy
// resetState[env*d + i] → state[env*d + i]; else leave state untouched.
//
// One kernel handles any per-env array: joint state (d=N), velocity
// buffer (d=N), step counter (d=1), arbitrary observation history etc.
// Host invokes once per buffer with the appropriate d.

export const conditionalResetKernel: WgpuKernel = wgpuKernel({
  js: (
    state: WpArray<number>,
    resetState: WpArray<number>,
    done: WpArray<number>,
    d: number,
  ) => {
    const idx = tid();
    const env = Math.floor(idx / d);
    if (done.get(env) >= 0.5) {
      state.set(idx, resetState.get(idx));
    }
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> state:       array<f32>;
@group(0) @binding(1) var<storage, read_write> reset_state: array<f32>;
@group(0) @binding(2) var<storage, read_write> done:        array<f32>;
@group(0) @binding(3) var<uniform>             d_uniform:   vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&state);
  if (idx >= total) { return; }
  let d = d_uniform.x;
  let env = idx / d;
  if (done[env] >= 0.5) {
    state[idx] = reset_state[idx];
  }
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: true },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: false },
    { binding: 3, kind: "uniform", inputIndex: 3 },
  ],
  workgroupSize: 64,
});

/** Reference conditional reset (in-place mutation of state). */
export function conditionalResetInline(
  state: number[],
  resetState: readonly number[],
  done: readonly number[],
  d: number,
): void {
  for (let idx = 0; idx < state.length; idx++) {
    const env = Math.floor(idx / d);
    if (done[env] >= 0.5) state[idx] = resetState[idx];
  }
}

// ── Ground-plane contact (env×foot parallel, frictionless) ───────────────
//
// Spring-damper normal-force model — the minimal physical contact
// pattern that lets legged demos (ANYmal walking, biped standing)
// push against a ground plane. Per-foot:
//
//     penetration = ground_z − p_z
//     if penetration > 0:
//         F_z = max(0,  Kp · penetration − Kd · v_z)
//     else:
//         F_z = 0
//     F_x = F_y = 0  (frictionless; iter 110 will add Coulomb tangent)
//
// One thread per (env, foot) pair. ground_z, Kp, Kd, mu (unused R0)
// shared across all envs as scalars.
//
// Bridges from the foot-FK pipeline (iter 70, 88, 95 ANYmal) to a
// physically-driven simulation step — Fz can be summed into a base
// reaction force or projected through Jᵀ to per-joint contact torques.

export const groundContactKernel: WgpuKernel = wgpuKernel({
  js: (
    pWorld: WpArray<number>,
    vWorld: WpArray<number>,
    fOut: WpArray<number>,
    groundZ: number,
    kp: number,
    kd: number,
  ) => {
    const idx = tid();
    const base = idx * 3;
    const pz = pWorld.get(base + 2);
    const penetration = groundZ - pz;
    let fz = 0;
    if (penetration > 0) {
      const vz = vWorld.get(base + 2);
      const raw = kp * penetration - kd * vz;
      fz = raw > 0 ? raw : 0;
    }
    fOut.set(base + 0, 0);
    fOut.set(base + 1, 0);
    fOut.set(base + 2, fz);
  },
  wgsl: `
@group(0) @binding(0) var<storage, read_write> p_world: array<f32>;
@group(0) @binding(1) var<storage, read_write> v_world: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out:   array<f32>;
@group(0) @binding(3) var<uniform>             params:  vec4<f32>;
//                                                       .x = ground_z
//                                                       .y = Kp
//                                                       .z = Kd

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&f_out) / 3u;
  if (idx >= total) { return; }
  let base = idx * 3u;
  let pz = p_world[base + 2u];
  let penetration = params.x - pz;
  var fz: f32 = 0.0;
  if (penetration > 0.0) {
    let vz = v_world[base + 2u];
    fz = max(0.0, params.y * penetration - params.z * vz);
  }
  f_out[base + 0u] = 0.0;
  f_out[base + 1u] = 0.0;
  f_out[base + 2u] = fz;
}
`,
  bindings: [
    { binding: 0, kind: "storage", inputIndex: 0, writeback: false },
    { binding: 1, kind: "storage", inputIndex: 1, writeback: false },
    { binding: 2, kind: "storage", inputIndex: 2, writeback: true },
    { binding: 3, kind: "uniform", inputIndex: 3 },
  ],
  workgroupSize: 64,
});

/** Reference frictionless ground-plane contact per foot. */
export function groundContactInline(
  pWorld: readonly number[],
  vWorld: readonly number[],
  fOut: number[],
  groundZ: number,
  kp: number,
  kd: number,
): void {
  const total = fOut.length / 3;
  for (let i = 0; i < total; i++) {
    const base = i * 3;
    const pz = pWorld[base + 2];
    const penetration = groundZ - pz;
    let fz = 0;
    if (penetration > 0) {
      const vz = vWorld[base + 2];
      const raw = kp * penetration - kd * vz;
      fz = raw > 0 ? raw : 0;
    }
    fOut[base + 0] = 0;
    fOut[base + 1] = 0;
    fOut[base + 2] = fz;
  }
}
