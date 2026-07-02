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
  matrix inverse by hand); also no \"Inline\" companion. Wave 47:
  franka-fk-kernel + franka-fk-inline — the first of 4 Franka-arm kernels
  (FK is the foundation the Jacobian/reach/gravity-comp kernels build on,
  each its own later wave). franka-fk-inline is deliberately self-contained
  (own private rot-rpy/rot-z/mat3-mul-small/matvec3-small helpers) rather
  than routing through dynamics.articulated-dynamics — matches the TS
  source's own structure, which needs the same standalone-compilable-as-
  WGSL property every kernel here has; the joint xyz/rpy data matches
  assets.franka-panda's URDF origins (same canonical source).

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

;; ── Franka 7-DoF forward kinematics (env-parallel) ────────────────────────
;;
;; Computes EE position for N envs in parallel from per-env q[7]. Real
;; Franka FCI joint origins (matching assets/franka-panda's URDF xyz/rpy
;; data — same canonical source, but this is a lightweight, self-contained
;; FK loop rather than routing through the heavier dynamics.articulated-
;; dynamics machinery; algorithm-for-algorithm port of the TS kernel, which
;; is deliberately self-contained so it also compiles standalone as WGSL).
;;
;; Storage layout (struct-of-arrays for coalesced GPU access):
;;   q-in:   length 7*N — q[i] = q-in[env*7 + i]
;;   ee-out: length 3*N — ee[i] = ee-out[env*3 + i]
;;
;; Algorithm: 7 successive frame compositions. Each joint applies
;;   R_origin (from URDF rpy) . Rodrigues(axis_body_z, q_i)
;; to the cumulative world-frame rotation, plus xyz translation.
;;
;; Bindings: q-in (storage read, N*7 floats, writeback false), ee-out
;; (storage read_write, N*3 floats).

(def ^:private franka-fk-half-pi (/ Math/PI 2))

(def ^:private franka-fk-xyz
  [[0 0 0.333]
   [0 0 0]
   [0 -0.316 0]
   [0.0825 0 0]
   [-0.0825 0.384 0]
   [0 0 0]
   [0.088 0 0]])

(def ^:private franka-fk-rpy-r
  [0 (- franka-fk-half-pi) franka-fk-half-pi franka-fk-half-pi
   (- franka-fk-half-pi) franka-fk-half-pi franka-fk-half-pi])

(defn- rot-rpy
  "p=y=0 for every Franka joint origin, so only the x-rotation term survives."
  [r]
  (let [cr (wp/cos r) sr (wp/sin r)]
    [[1 0 0] [0 cr (- sr)] [0 sr cr]]))

(defn- rot-z [angle]
  (let [c (wp/cos angle) s (wp/sin angle)]
    [[c (- s) 0] [s c 0] [0 0 1]]))

(defn- mat3-mul-small [a b]
  (vec (for [i (range 3)]
         (vec (for [j (range 3)]
                (reduce + (for [k (range 3)] (* (get-in a [i k]) (get-in b [k j])))))))))

(defn- matvec3-small [m v]
  [(+ (* (get-in m [0 0]) (v 0)) (* (get-in m [0 1]) (v 1)) (* (get-in m [0 2]) (v 2)))
   (+ (* (get-in m [1 0]) (v 0)) (* (get-in m [1 1]) (v 1)) (* (get-in m [1 2]) (v 2)))
   (+ (* (get-in m [2 0]) (v 0)) (* (get-in m [2 1]) (v 1)) (* (get-in m [2 2]) (v 2)))])

(defn franka-fk-inline
  "Reference Franka 7-DoF FK in pure Clojure — used by the kernel's :js
  fallback AND callable directly for cross-validation. Returns the
  world-frame EE position [x y z] from q (a 7-element seq of joint angles)."
  [q]
  (loop [i 0 r-world [[1 0 0] [0 1 0] [0 0 1]] p-world [0.0 0.0 0.0]]
    (if (>= i 7)
      p-world
      (let [r-origin (rot-rpy (franka-fk-rpy-r i))
            r-q      (rot-z (nth q i))
            r-i-in-p (mat3-mul-small r-origin r-q)
            rotated  (matvec3-small r-world (franka-fk-xyz i))
            p-world' (vec (map + p-world rotated))
            r-world' (mat3-mul-small r-world r-i-in-p)]
        (recur (inc i) r-world' p-world')))))

(def franka-fk-kernel
  (wgpu/wgpu-kernel
    {:js (fn [q-in ee-out]
           (let [env (wp/tid)
                 q   (vec (for [j (range 7)] (wp/wp-get q-in (+ (* env 7) j))))
                 ee  (franka-fk-inline q)]
             (wp/wp-set ee-out (+ (* env 3) 0) (ee 0))
             (wp/wp-set ee-out (+ (* env 3) 1) (ee 1))
             (wp/wp-set ee-out (+ (* env 3) 2) (ee 2))))
     :wgsl "
// Real Franka FCI joint origins (xyz triplet + rpy triplet per joint).
// Pre-computed: cos/sin of rpy values inlined as constants for speed.
// rpy values: (0,0,0), (-pi/2,0,0), (pi/2,0,0), (pi/2,0,0), (-pi/2,0,0), (pi/2,0,0), (pi/2,0,0)

@group(0) @binding(0) var<storage, read_write> q_in:    array<f32>;
@group(0) @binding(1) var<storage, read_write> ee_out:  array<f32>;

// Composed rotation R_world (3x3) stored row-major in 9 f32 locals.
// p_world stored in 3 f32 locals.
// Applies R_world <- R_world . R_origin . R_q (axis z, angle q[i])
// and p_world <- p_world + R_world_pre . xyz.

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
"
     :bindings [(assoc (wgpu/storage-binding 0 0) :writeback false)
                (wgpu/storage-binding 1 1)]
     :workgroup-size 64}))

;; ── PD joint controller (env×joint parallel) ──────────────────────────────
;;
;; tau[i] = Kp[i % n] * (q*[i] - q[i]) - Kd[i % n] * qdot[i]
;;
;; Per-(env,joint) state: q_actual, qd_actual (measured); q_target (desired).
;; Per-joint gains kp, kd shared across envs (n = joints/env; idx % n picks
;; the joint). Canonical Isaac Lab / Isaac Gym PD actuator model — sits
;; between the action pipeline (action-scale-clamp-kernel below) and effort
;; saturation (effort-limit-kernel below).
;;
;; Bindings (7 total): q_actual, qd_actual, q_target, kp, kd (storage read,
;; writeback false), tau_out (storage read_write), n (uniform).

(def pd-joint-controller-kernel
  (wgpu/wgpu-kernel
    {:js (fn [q-actual qd-actual q-target kp kd tau-out n]
           (let [idx   (wp/tid)
                 j     (mod idx n)
                 q-err (- (wp/wp-get q-target idx) (wp/wp-get q-actual idx))
                 tau   (- (* (wp/wp-get kp j) q-err) (* (wp/wp-get kd j) (wp/wp-get qd-actual idx)))]
             (wp/wp-set tau-out idx tau)))
     :wgsl "
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
"
     :bindings [(assoc (wgpu/storage-binding 0 0) :writeback false)
                (assoc (wgpu/storage-binding 1 1) :writeback false)
                (assoc (wgpu/storage-binding 2 2) :writeback false)
                (assoc (wgpu/storage-binding 3 3) :writeback false)
                (assoc (wgpu/storage-binding 4 4) :writeback false)
                (wgpu/storage-binding 5 5)
                (wgpu/uniform-binding 6 6)]
     :workgroup-size 64}))

(defn pd-joint-controller-inline
  "Reference PD joint controller in pure Clojure.
  tau[i] = Kp[i % n] * (q*[i] - q[i]) - Kd[i % n] * qdot[i]"
  [q-actual qd-actual q-target kp kd n]
  (vec (for [i (range (count q-actual))]
         (let [j (mod i n)]
           (- (* (nth kp j) (- (nth q-target i) (nth q-actual i)))
              (* (nth kd j) (nth qd-actual i)))))))

;; ── Action scale + clamp (env×joint parallel) ────────────────────────────
;;
;; Isaac Lab's ActionManager canonical pipeline:
;;
;;     q_target[i] = clamp(
;;         action_scale[j] * action[i] + action_offset[j],
;;         q_lower[j], q_upper[j])
;;
;; Sits between the policy net (action in [-1, 1]^N typically tanh-squashed)
;; and the PD controller (pd-joint-controller-kernel above). Per-joint
;; scale/offset/limits are shared across envs (storage), so 1024-env x
;; 12-joint dispatch needs only 48 floats of joint config.
;;
;; Bindings (7 total): action, action_scale, action_offset, q_lower, q_upper
;; (storage read, writeback false), q_target_out (storage read_write), n
;; (uniform).

(def action-scale-clamp-kernel
  (wgpu/wgpu-kernel
    {:js (fn [action action-scale action-offset q-lower q-upper q-target-out n]
           (let [idx (wp/tid)
                 j   (mod idx n)
                 raw (+ (* (wp/wp-get action-scale j) (wp/wp-get action idx))
                        (wp/wp-get action-offset j))
                 lo  (wp/wp-get q-lower j)
                 hi  (wp/wp-get q-upper j)]
             (wp/wp-set q-target-out idx (wp/clamp raw lo hi))))
     :wgsl "
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
"
     :bindings [(assoc (wgpu/storage-binding 0 0) :writeback false)
                (assoc (wgpu/storage-binding 1 1) :writeback false)
                (assoc (wgpu/storage-binding 2 2) :writeback false)
                (assoc (wgpu/storage-binding 3 3) :writeback false)
                (assoc (wgpu/storage-binding 4 4) :writeback false)
                (wgpu/storage-binding 5 5)
                (wgpu/uniform-binding 6 6)]
     :workgroup-size 64}))

(defn action-scale-clamp-inline
  "Reference action scale + clamp in pure Clojure."
  [action action-scale action-offset q-lower q-upper n]
  (vec (for [i (range (count action))]
         (let [j   (mod i n)
               raw (+ (* (nth action-scale j) (nth action i)) (nth action-offset j))]
           (wp/clamp raw (nth q-lower j) (nth q-upper j))))))

;; ── Effort saturation (env×joint parallel, in-place) ─────────────────────
;;
;; Canonical safety layer between the PD controller (pd-joint-controller-
;; kernel above) and articulated-body forward dynamics. Caps each joint's
;; torque at its actuator effort limit, matching Isaac Lab's per-joint
;; effort_limit_sim clamp.
;;
;;     tau_i = clamp(tau_i, -effort_limit_j, +effort_limit_j)
;;
;; In-place storage write; per-joint effort_limit shared across envs.
;;
;; Bindings (3 total): tau (storage read_write), effort_limit (storage
;; read, writeback false), n (uniform).

(def effort-limit-kernel
  (wgpu/wgpu-kernel
    {:js (fn [tau effort-limit n]
           (let [idx (wp/tid)
                 j   (mod idx n)
                 lim (wp/wp-get effort-limit j)
                 t   (wp/wp-get tau idx)]
             (wp/wp-set tau idx (wp/clamp t (- lim) lim))))
     :wgsl "
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
"
     :bindings [(wgpu/storage-binding 0 0)
                (assoc (wgpu/storage-binding 1 1) :writeback false)
                (wgpu/uniform-binding 2 2)]
     :workgroup-size 64}))

(defn effort-limit-inline
  "Reference effort saturation in pure Clojure. TS mutates `tau` in place and
  returns void; Clojure vectors are immutable, so this returns the clamped
  vector instead (same functional result, idiomatic port)."
  [tau effort-limit n]
  (vec (for [i (range (count tau))]
         (let [j   (mod i n)
               lim (nth effort-limit j)]
           (wp/clamp (nth tau i) (- lim) lim)))))

;; ── Observation normalize + noise + clamp (env×feature parallel) ─────────
;;
;; Isaac Lab's ObservationManager canonical pipeline:
;;
;;     obs[i] = clamp(
;;         (raw[i] - mean[j]) / std[j] + noise[i],
;;         clamp_low[j], clamp_high[j])
;;
;; Host pre-generates noise[i] (Gaussian or any; WGSL has no portable RNG),
;; so the kernel itself is deterministic — the test path can pass zeros to
;; verify the normalize+clamp path independently.
;;
;; Per-feature mean/std/clamp shared across envs (storage); per-(env,
;; feature) noise. Mirror of action-scale-clamp-kernel but on the obs side.
;;
;; Bindings (7 total): obs (storage read_write), mean, std, noise,
;; clamp_low, clamp_high (storage read, writeback false), n (uniform).

(def observation-normalize-kernel
  (wgpu/wgpu-kernel
    {:js (fn [obs mean std noise clamp-low clamp-high n]
           (let [idx        (wp/tid)
                 j          (mod idx n)
                 sigma      (max (wp/wp-get std j) 1e-8)
                 normalized (+ (/ (- (wp/wp-get obs idx) (wp/wp-get mean j)) sigma)
                                (wp/wp-get noise idx))
                 lo         (wp/wp-get clamp-low j)
                 hi         (wp/wp-get clamp-high j)]
             (wp/wp-set obs idx (wp/clamp normalized lo hi))))
     :wgsl "
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
"
     :bindings [(wgpu/storage-binding 0 0)
                (assoc (wgpu/storage-binding 1 1) :writeback false)
                (assoc (wgpu/storage-binding 2 2) :writeback false)
                (assoc (wgpu/storage-binding 3 3) :writeback false)
                (assoc (wgpu/storage-binding 4 4) :writeback false)
                (assoc (wgpu/storage-binding 5 5) :writeback false)
                (wgpu/uniform-binding 6 6)]
     :workgroup-size 64}))

(defn observation-normalize-inline
  "Reference observation normalize + noise + clamp in pure Clojure. TS
  mutates `obs` in place and returns void; returns the new vector instead
  (same functional result, idiomatic port)."
  [obs mean std noise clamp-low clamp-high n]
  (vec (for [i (range (count obs))]
         (let [j     (mod i n)
               sigma (max (nth std j) 1e-8)
               v     (+ (/ (- (nth obs i) (nth mean j)) sigma) (nth noise i))]
           (wp/clamp v (nth clamp-low j) (nth clamp-high j))))))
