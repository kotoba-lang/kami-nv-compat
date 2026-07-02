(ns kotoba.lang.kami-nv-compat.warp.examples
  "Reference WGSL+JS kernel pairs that exercise the wgpu-launch path —
  portable .cljc port of src/warp/examples.ts. These are pedagogical
  examples — they prove the architectural pattern: a kernel author writes
  both a Clojure implementation (for the sync `warp.warp/launch` path and as
  the WebGPU CPU fallback) and a WGSL compute shader (for actual GPU
  dispatch via `warp.wgpu-backend/wgpu-launch`). The runtime picks WGSL when
  a WebGPU device is available.

  Each kernel is `wgpu/wgpu-kernel {:js :wgsl :bindings :workgroup-size}` —
  the WGSL string is unchanged verbatim from the TS source (pure portable
  data, same treatment as kami-rt.wgsl-shaders); the :js function is an
  algorithm-for-algorithm port of the TS `js:` lambda using warp.warp's
  tid/wp-get/wp-set/sin/cos, with the exact same positional argument order
  (which the :bindings' :input-index fields also key off).

  Ported across multiple waves rather than one pass (~2500 LOC / 34 exports
  across 16 WgpuKernels + 18 CPU-reference \"Inline\" functions). Wave 45:
  the 3 self-contained physics-step kernels (damping, pendulum, cartpole)
  with no \"Inline\" companion. Wave 46: two-link-arm-step-kernel — a
  closed-form 2-DoF planar arm (M(q)*q_ddot + C*q_dot + g(q) = tau, 2x2
  matrix inverse by hand); also no \"Inline\" companion.

  Note: cartpole-step-kernel's semi-implicit-Euler integration order
  (compute the NEW velocity, then integrate position using that NEW
  velocity) is a genuinely different algorithm from e7m-shugyo.cartpole's
  cartpole-step (explicit Euler: integrate position using the OLD
  velocity) — these are separate reference implementations for separate
  purposes (a GPU-kernel exemplar vs. the Isaac Lab Cartpole RL env), not
  duplicates; verified by inspection, not unified/deduped."
  (:require [kotoba.lang.kami-nv-compat.warp.warp :as wp]
            [kotoba.lang.kami-nv-compat.warp.wgpu-backend :as wgpu]))

;; ── Damping kernel: multiply each element of an array by a scalar ────────
;;
;; Bindings:
;;   @group(0) @binding(0) = WpArray<number> (storage, read_write)
;;   @group(0) @binding(1) = scalar damping (uniform)
;; Workgroup size 64 matches the kami-cartpole-wasm precedent.

(def damping-kernel
  (wgpu/wgpu-kernel
    {:js (fn [arr damping]
           (let [i (wp/tid)]
             (wp/wp-set arr i (* (wp/wp-get arr i) damping))))
     :wgsl "
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
"
     :bindings [(wgpu/storage-binding 0 0)
                (wgpu/uniform-binding 1 1)]
     :workgroup-size 64}))

;; ── Pendulum semi-implicit Euler step (env-parallel) ──────────────────────
;;
;; Mirrors the iter 71 single-pendulum integrator stepped across N envs in
;; parallel. Per-env state: theta (angle), omega (angular velocity).
;; Per-env input: tau (applied torque). Uniform params: dt, g, length
;; (COM distance from pivot), mass.
;;
;; Dynamics: tau_total = tau_applied - m*g*L*sin(theta); alpha = tau_total / (m*L^2)
;; Semi-implicit Euler: omega' = omega + dt*alpha; theta' = theta + dt*omega'
;;
;; At equilibrium (theta=0, omega=0, tau=0): alpha=0, no drift.
;; At theta=pi/2, omega=0, tau=0: alpha = -g*L*sin(pi/2)/(m*L^2) = -g/L ~= -9.81 (m=L=1).
;;
;; Bindings (in order): theta, omega (storage read_write), tau (storage
;; read, writeback false), dt, g, length, mass (uniform).

(def pendulum-step-kernel
  (wgpu/wgpu-kernel
    {:js (fn [theta omega tau dt g length mass]
           (let [i     (wp/tid)
                 t     (wp/wp-get theta i)
                 w     (wp/wp-get omega i)
                 tor   (wp/wp-get tau i)
                 alpha (/ (- tor (* mass g length (wp/sin t))) (* mass length length))
                 w-new (+ w (* dt alpha))
                 t-new (+ t (* dt w-new))]
             (wp/wp-set omega i w-new)
             (wp/wp-set theta i t-new)))
     :wgsl "
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
"
     :bindings [(wgpu/storage-binding 0 0)
                (wgpu/storage-binding 1 1)
                (assoc (wgpu/storage-binding 2 2) :writeback false)
                (wgpu/uniform-binding 3 3)
                (wgpu/uniform-binding 4 4)
                (wgpu/uniform-binding 5 5)
                (wgpu/uniform-binding 6 6)]
     :workgroup-size 64}))

;; ── Cartpole semi-implicit Euler step (env-parallel) ──────────────────────
;;
;; Mirrors the Python iter 68 _kernel.cartpole_step (Sutton & Barto /
;; OpenAI Gym CartPole-v1 closed-form) stepped across N envs in parallel.
;; 2-DoF coupled dynamics — revolute pole on a prismatic cart.
;;
;; Per-env state: x, x_dot, theta (from vertical, +theta = pole leans +x),
;; theta_dot. Per-env input: force (caller clamps; this kernel does not).
;; Uniform params: dt, gravity, cart_mass, pole_mass, pole_half_length.
;;
;; Closed-form (Sutton & Barto):
;;   temp = (force + m_pole*L*theta_dot^2*sin(theta)) / total_mass
;;   theta_acc = (g*sin(theta) - cos(theta)*temp) / (L*(4/3 - m_pole*cos(theta)^2/total_mass))
;;   x_acc = temp - m_pole*L*theta_acc*cos(theta) / total_mass
;; Semi-implicit Euler: velocities update first, then positions integrate
;; using the NEW velocities.
;;
;; Bindings (10 total): x, x_dot, theta, theta_dot (storage read_write),
;; force (storage read, writeback false), dt, gravity, cart_mass, pole_mass,
;; pole_half_length (uniform).

(def cartpole-step-kernel
  (wgpu/wgpu-kernel
    {:js (fn [x x-dot theta theta-dot force dt gravity cart-mass pole-mass pole-half-length]
           (let [i          (wp/tid)
                 t          (wp/wp-get theta i)
                 td         (wp/wp-get theta-dot i)
                 xd         (wp/wp-get x-dot i)
                 f          (wp/wp-get force i)
                 sin-t      (wp/sin t)
                 cos-t      (wp/cos t)
                 total-mass (+ cart-mass pole-mass)
                 pml        (* pole-mass pole-half-length)
                 temp       (/ (+ f (* pml td td sin-t)) total-mass)
                 theta-acc  (/ (- (* gravity sin-t) (* cos-t temp))
                                (* pole-half-length (- (/ 4.0 3.0) (/ (* pole-mass cos-t cos-t) total-mass))))
                 x-acc      (- temp (/ (* pml theta-acc cos-t) total-mass))
                 x-dot-new  (+ xd (* dt x-acc))
                 x-new      (+ (wp/wp-get x i) (* dt x-dot-new))
                 theta-dot-new (+ td (* dt theta-acc))
                 theta-new  (+ t (* dt theta-dot-new))]
             (wp/wp-set x-dot i x-dot-new)
             (wp/wp-set x i x-new)
             (wp/wp-set theta-dot i theta-dot-new)
             (wp/wp-set theta i theta-new)))
     :wgsl "
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
"
     :bindings [(wgpu/storage-binding 0 0)
                (wgpu/storage-binding 1 1)
                (wgpu/storage-binding 2 2)
                (wgpu/storage-binding 3 3)
                (assoc (wgpu/storage-binding 4 4) :writeback false)
                (wgpu/uniform-binding 5 5)
                (wgpu/uniform-binding 6 6)
                (wgpu/uniform-binding 7 7)
                (wgpu/uniform-binding 8 8)
                (wgpu/uniform-binding 9 9)]
     :workgroup-size 64}))

;; ── Two-link arm step (env-parallel) ──────────────────────────────────────
;;
;; Closed-form 2-DoF planar arm dynamics — both joints revolute about world
;; Y-axis, both links pendulum-like (gravity pulls toward -Z). Standard
;; manipulator equation M(q)*q_ddot + C(q,q_dot)*q_dot + g(q) = tau, inverted
;; by hand (2x2 closed form). Reference: Spong, Robot Modeling & Control,
;; Ch. 7. link1/link2 = [m L r I] (mass, full length, COM offset from joint,
;; inertia about COM); link2's L is unused here (reserved for future
;; tip-frame variants — matches TS's `void L2`, so l2-mass/-r2/-I2 are read
;; but l2-len is simply never bound to a let).
;;
;; Bindings (10 total): theta1, theta1_dot, theta2, theta2_dot (storage
;; read_write), tau1, tau2 (storage read, writeback false), dt, gravity
;; (uniform), link1, link2 (uniform vec4: m, L, r, I).

(def two-link-arm-step-kernel
  (wgpu/wgpu-kernel
    {:js (fn [theta1 theta1-dot theta2 theta2-dot tau1 tau2 dt g
              m1 l1 r1 i1 m2 _l2 r2 i2]
           (let [idx    (wp/tid)
                 q1     (wp/wp-get theta1 idx)
                 q2     (wp/wp-get theta2 idx)
                 dq1    (wp/wp-get theta1-dot idx)
                 dq2    (wp/wp-get theta2-dot idx)
                 t1     (wp/wp-get tau1 idx)
                 t2     (wp/wp-get tau2 idx)
                 a      (+ (* m1 r1 r1) i1 (* m2 l1 l1))
                 b      (+ (* m2 r2 r2) i2)
                 c      (* m2 l1 r2)
                 cos-t2 (wp/cos q2)
                 sin-t2 (wp/sin q2)
                 m11    (+ a b (* 2 c cos-t2))
                 m12    (+ b (* c cos-t2))
                 m22    b
                 h1     (+ (- (* c sin-t2 dq2 dq1))
                           (- (* c sin-t2 (+ dq1 dq2) dq2))
                           (* m1 g r1 (wp/sin q1))
                           (* m2 g (+ (* l1 (wp/sin q1)) (* r2 (wp/sin (+ q1 q2))))))
                 h2     (+ (* c sin-t2 dq1 dq1) (* m2 g r2 (wp/sin (+ q1 q2))))
                 b1     (- t1 h1)
                 b2     (- t2 h2)
                 det    (- (* m11 m22) (* m12 m12))
                 ddq1   (/ (- (* m22 b1) (* m12 b2)) det)
                 ddq2   (/ (- (* m11 b2) (* m12 b1)) det)
                 dq1-new (+ dq1 (* dt ddq1))
                 dq2-new (+ dq2 (* dt ddq2))]
             (wp/wp-set theta1-dot idx dq1-new)
             (wp/wp-set theta1 idx (+ q1 (* dt dq1-new)))
             (wp/wp-set theta2-dot idx dq2-new)
             (wp/wp-set theta2 idx (+ q2 (* dt dq2-new)))))
     :wgsl "
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
"
     :bindings [(wgpu/storage-binding 0 0)
                (wgpu/storage-binding 1 1)
                (wgpu/storage-binding 2 2)
                (wgpu/storage-binding 3 3)
                (assoc (wgpu/storage-binding 4 4) :writeback false)
                (assoc (wgpu/storage-binding 5 5) :writeback false)
                (wgpu/uniform-binding 6 6)
                (wgpu/uniform-binding 7 7)
                (wgpu/uniform-binding 8 8)
                (wgpu/uniform-binding 9 9)]
     :workgroup-size 64}))
