(ns kotoba.lang.kami-nv-compat.warp.examples-test
  "Coverage for warp.examples waves 1-3 (damping-kernel, pendulum-step-kernel,
  cartpole-step-kernel, two-link-arm-step-kernel, franka-fk-inline,
  franka-fk-kernel). Each kernel is exercised via warp.warp/launch, matching
  how a real caller would dispatch the sync/CPU-fallback path."
  (:require [clojure.test :refer [deftest is testing]]
            [kotoba.lang.kami-nv-compat.warp.examples :as ex]
            [kotoba.lang.kami-nv-compat.warp.warp :as wp]))

(defn- close? [a b tol] (< (Math/abs (- a b)) tol))

(deftest damping-kernel-scales-each-element
  (let [arr (wp/wp-array [10.0 20.0 30.0])]
    (wp/launch {:kernel-fn (:fn ex/damping-kernel) :dim 3 :inputs [arr 0.5]})
    (is (= [5.0 10.0 15.0] @arr))))

(deftest damping-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback true}
          {:binding 1 :kind :uniform :input-index 1 :writeback false}]
         (:bindings ex/damping-kernel)))
  (is (= 64 (:workgroup-size ex/damping-kernel))))

(deftest pendulum-step-equilibrium-has-no-drift
  (testing "theta=0, omega=0, tau=0 -> stays at rest"
    (let [theta (wp/wp-array [0.0]) omega (wp/wp-array [0.0]) tau (wp/wp-array [0.0])]
      (wp/launch {:kernel-fn (:fn ex/pendulum-step-kernel) :dim 1
                  :inputs [theta omega tau 0.01 9.81 1.0 1.0]})
      (is (= [0.0] @theta))
      (is (= [0.0] @omega)))))

(deftest pendulum-step-at-90-degrees-matches-hand-derived-alpha
  (testing "theta=pi/2, omega=0, tau=0, m=L=1 -> alpha = -g/L ~= -9.81,
            so omega after one dt step ~= -9.81*dt (matches the source's
            own worked example comment)"
    (let [dt 0.001
          theta (wp/wp-array [(/ Math/PI 2)])
          omega (wp/wp-array [0.0])
          tau   (wp/wp-array [0.0])]
      (wp/launch {:kernel-fn (:fn ex/pendulum-step-kernel) :dim 1
                  :inputs [theta omega tau dt 9.81 1.0 1.0]})
      (is (close? (first @omega) (* -9.81 dt) 1e-9)))))

(deftest pendulum-step-batches-independent-envs
  (testing "dim=N runs N independent envs (tid indexes each array)"
    (let [n 4
          theta (wp/wp-array (repeat n 0.0))
          omega (wp/wp-array (vec (range n)))          ; distinct initial omegas
          tau   (wp/wp-array (repeat n 0.0))]
      (wp/launch {:kernel-fn (:fn ex/pendulum-step-kernel) :dim n
                  :inputs [theta omega tau 0.01 9.81 1.0 1.0]})
      ;; at theta=0 the gravity torque term is 0, so each env's new theta is
      ;; just its own initial omega * dt (no cross-env interference)
      (is (= [0.0 0.01 0.02 0.03] @theta)))))

(deftest cartpole-step-at-rest-stays-at-rest
  (let [x (wp/wp-array [0.0]) x-dot (wp/wp-array [0.0])
        theta (wp/wp-array [0.0]) theta-dot (wp/wp-array [0.0])
        force (wp/wp-array [0.0])]
    (wp/launch {:kernel-fn (:fn ex/cartpole-step-kernel) :dim 1
                :inputs [x x-dot theta theta-dot force 0.02 9.8 1.0 0.1 0.5]})
    (is (= [0.0] @x))
    (is (= [0.0] @theta))))

(deftest cartpole-step-tilted-pole-accelerates-away-from-vertical
  (testing "a small positive tilt with no force -> theta-dot moves further
            positive (unstable inverted-pendulum equilibrium)"
    (let [x (wp/wp-array [0.0]) x-dot (wp/wp-array [0.0])
          theta (wp/wp-array [0.05]) theta-dot (wp/wp-array [0.0])
          force (wp/wp-array [0.0])]
      (wp/launch {:kernel-fn (:fn ex/cartpole-step-kernel) :dim 1
                  :inputs [x x-dot theta theta-dot force 0.02 9.8 1.0 0.1 0.5]})
      (is (pos? (first @theta-dot))))))

(deftest cartpole-step-symmetric-under-sign-flip
  (testing "flipping the initial tilt's sign flips theta-dot's sign (odd symmetry)"
    (let [run (fn [t0]
                (let [x (wp/wp-array [0.0]) x-dot (wp/wp-array [0.0])
                      theta (wp/wp-array [t0]) theta-dot (wp/wp-array [0.0])
                      force (wp/wp-array [0.0])]
                  (wp/launch {:kernel-fn (:fn ex/cartpole-step-kernel) :dim 1
                              :inputs [x x-dot theta theta-dot force 0.02 9.8 1.0 0.1 0.5]})
                  (first @theta-dot)))]
      (is (close? (run 0.05) (- (run -0.05)) 1e-12)))))

;; ── two-link-arm-step-kernel ──────────────────────────────────────────────

(def two-link-arm-params
  "m1 L1 r1 I1 m2 L2 r2 I2 — a concrete, non-degenerate parameter set (both
  links contribute mass/inertia so det(M) is never zero)."
  [1.0 1.0 0.5 0.1 1.0 1.0 0.5 0.1])

(defn- run-two-link-arm [q1 q2 dq1 dq2 t1 t2 dt]
  (let [theta1 (wp/wp-array [q1]) theta1-dot (wp/wp-array [dq1])
        theta2 (wp/wp-array [q2]) theta2-dot (wp/wp-array [dq2])
        tau1   (wp/wp-array [t1]) tau2       (wp/wp-array [t2])]
    (wp/launch {:kernel-fn (:fn ex/two-link-arm-step-kernel) :dim 1
                :inputs (into [theta1 theta1-dot theta2 theta2-dot tau1 tau2 dt 9.8]
                               two-link-arm-params)})
    {:theta1 (first @theta1) :theta1-dot (first @theta1-dot)
     :theta2 (first @theta2) :theta2-dot (first @theta2-dot)}))

(deftest two-link-arm-step-rest-at-zero-is-equilibrium
  (testing "q1=q2=0, zero velocity, zero torque -> both gravity terms vanish
            (sin(0)=0), stays at rest exactly"
    (let [r (run-two-link-arm 0.0 0.0 0.0 0.0 0.0 0.0 0.001)]
      (is (= 0.0 (:theta1 r)))
      (is (= 0.0 (:theta1-dot r)))
      (is (= 0.0 (:theta2 r)))
      (is (= 0.0 (:theta2-dot r))))))

(deftest two-link-arm-step-matches-independent-hand-derivation
  (testing "q1=0.1, q2=0, at rest, no torque: cross-checked against a raw
            (non-kernel) re-derivation of the same closed-form 2x2 solve"
    (let [m1 1.0 L1 1.0 r1 0.5 I1 0.1
          m2 1.0 r2 0.5 I2 0.1
          g 9.8 q1 0.1 q2 0.0 dt 0.001
          a (+ (* m1 r1 r1) I1 (* m2 L1 L1))
          b (+ (* m2 r2 r2) I2)
          c (* m2 L1 r2)
          cos-t2 (Math/cos q2)
          m11 (+ a b (* 2 c cos-t2)) m12 (+ b (* c cos-t2)) m22 b
          h1 (+ (* m1 g r1 (Math/sin q1)) (* m2 g (+ (* L1 (Math/sin q1)) (* r2 (Math/sin (+ q1 q2))))))
          h2 (* m2 g r2 (Math/sin (+ q1 q2)))
          det (- (* m11 m22) (* m12 m12))
          ddq1 (/ (- (* m22 (- h1)) (* m12 (- h2))) det)
          ddq2 (/ (- (* m11 (- h2)) (* m12 (- h1))) det)
          r (run-two-link-arm q1 q2 0.0 0.0 0.0 0.0 dt)]
      (is (close? (:theta1-dot r) (* dt ddq1) 1e-12))
      (is (close? (:theta2-dot r) (* dt ddq2) 1e-12)))))

(deftest two-link-arm-step-registers-correct-bindings
  (is (= 10 (count (:bindings ex/two-link-arm-step-kernel))))
  (is (every? false? (map :writeback (filter #(= (:kind %) :storage)
                                              (drop 4 (take 6 (:bindings ex/two-link-arm-step-kernel)))))))
  (is (= 64 (:workgroup-size ex/two-link-arm-step-kernel))))

;; ── franka-fk-inline / franka-fk-kernel ────────────────────────────────────

(defn- close3? [[ax ay az] [bx by bz] tol]
  (and (close? ax bx tol) (close? ay by tol) (close? az bz tol)))

(deftest franka-fk-inline-home-pose-matches-hand-derivation
  (testing "q=[0 0 0 0 0 0 0]: with every R_q = identity, the chain reduces
            to composing 7 fixed x-axis rotations by the URDF rpy angles —
            hand-traced through all 7 steps to [0.088, 0, 1.033], which also
            matches the commonly-cited Franka Panda home-pose EE height"
    (is (close3? [0.088 0.0 1.033] (ex/franka-fk-inline (repeat 7 0.0)) 1e-9))))

(deftest franka-fk-kernel-matches-franka-fk-inline
  (testing "dispatched through warp.warp/launch + WpArray plumbing, the
            kernel's :js path gives the identical result to calling
            franka-fk-inline directly"
    (let [q-in   (wp/wp-array (vec (repeat 7 0.0)))
          ee-out (wp/wp-array (vec (repeat 3 0.0)))]
      (wp/launch {:kernel-fn (:fn ex/franka-fk-kernel) :dim 1 :inputs [q-in ee-out]})
      (is (close3? (ex/franka-fk-inline (repeat 7 0.0)) @ee-out 1e-12)))))

(deftest franka-fk-kernel-batches-independent-envs
  (testing "dim=N runs N independent envs (env*7 / env*3 offsets don't
            collide across envs)"
    (let [envs   [[0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                  [0.5 -0.3 0.2 -1.5 0.1 1.8 0.7]]
          q-in   (wp/wp-array (vec (apply concat envs)))
          ee-out (wp/wp-array (vec (repeat 6 0.0)))]
      (wp/launch {:kernel-fn (:fn ex/franka-fk-kernel) :dim 2 :inputs [q-in ee-out]})
      (is (close3? (ex/franka-fk-inline (first envs)) (subvec @ee-out 0 3) 1e-12))
      (is (close3? (ex/franka-fk-inline (second envs)) (subvec @ee-out 3 6) 1e-12)))))

(deftest franka-fk-inline-finite-for-arbitrary-pose
  (is (every? #(#?(:clj Double/isFinite :cljs js/isFinite) %)
              (ex/franka-fk-inline [0.5 -0.3 0.2 -1.5 0.1 1.8 0.7]))))

(deftest franka-fk-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback false}
          {:binding 1 :kind :storage :input-index 1 :writeback true}]
         (:bindings ex/franka-fk-kernel)))
  (is (= 64 (:workgroup-size ex/franka-fk-kernel))))

;; ── pd-joint-controller-kernel / pd-joint-controller-inline ───────────────
;;
;; tau[i] = Kp[i % n] * (q*[i] - q[i]) - Kd[i % n] * qdot[i]

(deftest pd-joint-controller-kernel-matches-hand-derived-torque
  (testing "2 joints (n=2), 2 envs' worth of joint state packed flat: hand
            derived tau0 = 10*(2-1) - 2*0.5 = 9.0, tau1 = 5*(3-3) - 1*(-0.5) = 0.5"
    (let [q-actual  (wp/wp-array [1.0 3.0])
          qd-actual (wp/wp-array [0.5 -0.5])
          q-target  (wp/wp-array [2.0 3.0])
          kp        (wp/wp-array [10.0 5.0])
          kd        (wp/wp-array [2.0 1.0])
          tau-out   (wp/wp-array [0.0 0.0])]
      (wp/launch {:kernel-fn (:fn ex/pd-joint-controller-kernel) :dim 2
                  :inputs [q-actual qd-actual q-target kp kd tau-out 2]})
      (is (close? (first @tau-out) 9.0 1e-12))
      (is (close? (second @tau-out) 0.5 1e-12)))))

(deftest pd-joint-controller-kernel-matches-pd-joint-controller-inline
  (let [q-actual  [1.0 3.0] qd-actual [0.5 -0.5] q-target [2.0 3.0]
        kp [10.0 5.0] kd [2.0 1.0]
        tau-out (wp/wp-array [0.0 0.0])]
    (wp/launch {:kernel-fn (:fn ex/pd-joint-controller-kernel) :dim 2
                :inputs [(wp/wp-array q-actual) (wp/wp-array qd-actual) (wp/wp-array q-target)
                         (wp/wp-array kp) (wp/wp-array kd) tau-out 2]})
    (is (= @tau-out (ex/pd-joint-controller-inline q-actual qd-actual q-target kp kd 2)))))

(deftest pd-joint-controller-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback false}
          {:binding 1 :kind :storage :input-index 1 :writeback false}
          {:binding 2 :kind :storage :input-index 2 :writeback false}
          {:binding 3 :kind :storage :input-index 3 :writeback false}
          {:binding 4 :kind :storage :input-index 4 :writeback false}
          {:binding 5 :kind :storage :input-index 5 :writeback true}
          {:binding 6 :kind :uniform :input-index 6 :writeback false}]
         (:bindings ex/pd-joint-controller-kernel)))
  (is (= 64 (:workgroup-size ex/pd-joint-controller-kernel))))

;; ── action-scale-clamp-kernel / action-scale-clamp-inline ─────────────────

(deftest action-scale-clamp-kernel-clamps-at-both-bounds-and-passes-through
  (testing "raw = scale*action + offset; raw0=1.1 clamps to hi=1.0,
            raw1=0.3 passes through unclamped, raw2=-1.7 clamps to lo=-1.0"
    (let [action (wp/wp-array [0.5 0.1 -0.9])
          scale  (wp/wp-array [2.0]) offset (wp/wp-array [0.1])
          lo     (wp/wp-array [-1.0]) hi (wp/wp-array [1.0])
          out    (wp/wp-array [0.0 0.0 0.0])]
      (wp/launch {:kernel-fn (:fn ex/action-scale-clamp-kernel) :dim 3
                  :inputs [action scale offset lo hi out 1]})
      (is (close? (nth @out 0) 1.0 1e-12))
      (is (close? (nth @out 1) 0.3 1e-9))
      (is (close? (nth @out 2) -1.0 1e-12)))))

(deftest action-scale-clamp-kernel-matches-action-scale-clamp-inline
  (let [action [0.5 0.1 -0.9] scale [2.0] offset [0.1] lo [-1.0] hi [1.0]
        out (wp/wp-array [0.0 0.0 0.0])]
    (wp/launch {:kernel-fn (:fn ex/action-scale-clamp-kernel) :dim 3
                :inputs [(wp/wp-array action) (wp/wp-array scale) (wp/wp-array offset)
                         (wp/wp-array lo) (wp/wp-array hi) out 1]})
    (is (= @out (ex/action-scale-clamp-inline action scale offset lo hi 1)))))

(deftest action-scale-clamp-kernel-registers-correct-bindings
  (is (= 7 (count (:bindings ex/action-scale-clamp-kernel))))
  (is (every? false? (map :writeback (take 5 (:bindings ex/action-scale-clamp-kernel)))))
  (is (true? (:writeback (nth (:bindings ex/action-scale-clamp-kernel) 5))))
  (is (= 64 (:workgroup-size ex/action-scale-clamp-kernel))))

;; ── effort-limit-kernel / effort-limit-inline ──────────────────────────────

(deftest effort-limit-kernel-clamps-symmetrically-with-inclusive-boundary
  (testing "tau=15 -> 10 (over), tau=-15 -> -10 (under), tau=5 -> 5 (inside),
            tau=10 (exactly at limit) -> 10 (unchanged, inclusive boundary)"
    (let [tau (wp/wp-array [15.0 -15.0 5.0 10.0])
          lim (wp/wp-array [10.0])]
      (wp/launch {:kernel-fn (:fn ex/effort-limit-kernel) :dim 4 :inputs [tau lim 1]})
      (is (= [10.0 -10.0 5.0 10.0] @tau)))))

(deftest effort-limit-kernel-matches-effort-limit-inline
  (let [tau [15.0 -15.0 5.0 10.0] lim [10.0]
        tau-arr (wp/wp-array tau)]
    (wp/launch {:kernel-fn (:fn ex/effort-limit-kernel) :dim 4
                :inputs [tau-arr (wp/wp-array lim) 1]})
    (is (= @tau-arr (ex/effort-limit-inline tau lim 1)))))

(deftest effort-limit-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback true}
          {:binding 1 :kind :storage :input-index 1 :writeback false}
          {:binding 2 :kind :uniform :input-index 2 :writeback false}]
         (:bindings ex/effort-limit-kernel)))
  (is (= 64 (:workgroup-size ex/effort-limit-kernel))))

;; ── observation-normalize-kernel / observation-normalize-inline ───────────

(deftest observation-normalize-kernel-normalizes-clamps-and-floors-sigma
  (testing "feature 0: (5-2)/1.5 + 0 = 2.0, within [-10,10] -> 2.0 unchanged;
            feature 1: (100-0)/1.0 + 0 = 100, clamps to hi=1.0;
            feature 2: std=0 floors to sigma=1e-8, (1-0)/1e-8 = 1e8, within
            [-1e10,1e10] -> passes through unclamped"
    (let [obs   (wp/wp-array [5.0 100.0 1.0])
          mean  (wp/wp-array [2.0 0.0 0.0])
          std   (wp/wp-array [1.5 1.0 0.0])
          noise (wp/wp-array [0.0 0.0 0.0])
          clo   (wp/wp-array [-10.0 -1.0 -1.0e10])
          chi   (wp/wp-array [10.0 1.0 1.0e10])]
      (wp/launch {:kernel-fn (:fn ex/observation-normalize-kernel) :dim 3
                  :inputs [obs mean std noise clo chi 3]})
      (is (close? (nth @obs 0) 2.0 1e-12))
      (is (close? (nth @obs 1) 1.0 1e-12))
      (is (close? (nth @obs 2) 1.0e8 1.0)))))

(deftest observation-normalize-kernel-matches-observation-normalize-inline
  (let [obs [5.0 100.0 1.0] mean [2.0 0.0 0.0] std [1.5 1.0 0.0]
        noise [0.0 0.0 0.0] clo [-10.0 -1.0 -1.0e10] chi [10.0 1.0 1.0e10]
        obs-arr (wp/wp-array obs)]
    (wp/launch {:kernel-fn (:fn ex/observation-normalize-kernel) :dim 3
                :inputs [obs-arr (wp/wp-array mean) (wp/wp-array std) (wp/wp-array noise)
                         (wp/wp-array clo) (wp/wp-array chi) 3]})
    (is (= @obs-arr (ex/observation-normalize-inline obs mean std noise clo chi 3)))))

(deftest observation-normalize-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback true}
          {:binding 1 :kind :storage :input-index 1 :writeback false}
          {:binding 2 :kind :storage :input-index 2 :writeback false}
          {:binding 3 :kind :storage :input-index 3 :writeback false}
          {:binding 4 :kind :storage :input-index 4 :writeback false}
          {:binding 5 :kind :storage :input-index 5 :writeback false}
          {:binding 6 :kind :uniform :input-index 6 :writeback false}]
         (:bindings ex/observation-normalize-kernel)))
  (is (= 64 (:workgroup-size ex/observation-normalize-kernel))))

;; ── anymal-fk-inline / anymal-fk-kernel ─────────────────────────────────────

(deftest anymal-fk-inline-zero-q-matches-hand-derivation
  (testing "q=zeros for all 4 legs: every rot-axis(axis, 0) is the identity
            matrix, so each foot is simply the sum of that leg's
            haa-base + hfe-local + kfe-local + foot-local offsets — hand
            summed here independently of the R_world composition code path"
    (let [feet (ex/anymal-fk-inline (vec (repeat 12 0.0)))]
      (is (close3? [0.277 0.2205 -0.634] (feet 0) 1e-12))          ; LF
      (is (close3? [-0.277 0.2205 -0.634] (feet 1) 1e-12))         ; LH
      (is (close3? [0.277 -0.0115 -0.634] (feet 2) 1e-12))         ; RF
      (is (close3? [-0.277 -0.0115 -0.634] (feet 3) 1e-12)))))     ; RH

(deftest anymal-fk-kernel-matches-anymal-fk-inline
  (testing "dispatched through warp.warp/launch + WpArray plumbing, the
            kernel's :js path gives the identical result to calling
            anymal-fk-inline directly"
    (let [q-in     (wp/wp-array (vec (repeat 12 0.0)))
          feet-out (wp/wp-array (vec (repeat 12 0.0)))]
      (wp/launch {:kernel-fn (:fn ex/anymal-fk-kernel) :dim 1 :inputs [q-in feet-out]})
      (is (= (vec (apply concat (ex/anymal-fk-inline (vec (repeat 12 0.0)))))
             @feet-out)))))

(deftest anymal-fk-kernel-batches-independent-envs
  (testing "dim=N runs N independent envs (env*12 offsets don't collide
            across envs)"
    (let [env0     (vec (repeat 12 0.0))
          env1     [0.1 -0.2 0.3 0.05 -0.1 0.15 -0.05 0.2 -0.15 0.1 0.1 0.1]
          q-in     (wp/wp-array (vec (concat env0 env1)))
          feet-out (wp/wp-array (vec (repeat 24 0.0)))]
      (wp/launch {:kernel-fn (:fn ex/anymal-fk-kernel) :dim 2 :inputs [q-in feet-out]})
      (is (= (vec (apply concat (ex/anymal-fk-inline env0))) (subvec @feet-out 0 12)))
      (is (= (vec (apply concat (ex/anymal-fk-inline env1))) (subvec @feet-out 12 24))))))

(deftest anymal-fk-inline-finite-for-arbitrary-pose
  (is (every? #(#?(:clj Double/isFinite :cljs js/isFinite) %)
              (apply concat (ex/anymal-fk-inline [0.1 -0.2 0.3 0.05 -0.1 0.15
                                                   -0.05 0.2 -0.15 0.1 0.1 0.1])))))

(deftest anymal-fk-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback false}
          {:binding 1 :kind :storage :input-index 1 :writeback true}]
         (:bindings ex/anymal-fk-kernel)))
  (is (= 64 (:workgroup-size ex/anymal-fk-kernel))))

;; ── generic-serial-fk-inline / generic-serial-fk-kernel ─────────────────────
;;
;; Franka's own xyz/rpy/axis data (rpy = (r,0,0), axis = z for every joint)
;; is a genuine independent cross-check: generic-serial-fk-inline's
;; rot-rpy-full(r,0,0) reduces algebraically to wave 47's rot-rpy(r), and its
;; rot-axis([0,0,1], angle) reduces to wave 47's rot-z(angle) — both already
;; verified in franka-fk-*'s own tests — so feeding the generic kernel
;; Franka's config must reproduce franka-fk-inline's output exactly.

(def ^:private generic-fk-half-pi (/ Math/PI 2))

(def ^:private generic-fk-franka-xyz
  [[0 0 0.333] [0 0 0] [0 -0.316 0] [0.0825 0 0]
   [-0.0825 0.384 0] [0 0 0] [0.088 0 0]])

(def ^:private generic-fk-franka-rpy
  [[0 0 0] [(- generic-fk-half-pi) 0 0] [generic-fk-half-pi 0 0]
   [generic-fk-half-pi 0 0] [(- generic-fk-half-pi) 0 0]
   [generic-fk-half-pi 0 0] [generic-fk-half-pi 0 0]])

(def ^:private generic-fk-franka-axis (vec (repeat 7 [0 0 1])))

(deftest generic-serial-fk-inline-matches-franka-fk-inline-for-franka-config
  (testing "feeding generic-serial-fk-inline Franka's own joint xyz/rpy/axis
            data reproduces franka-fk-inline's output exactly, at both the
            home pose and an arbitrary pose"
    (is (close3? (ex/franka-fk-inline (repeat 7 0.0))
                 (ex/generic-serial-fk-inline (repeat 7 0.0) generic-fk-franka-xyz
                                               generic-fk-franka-rpy generic-fk-franka-axis)
                 1e-12))
    (let [q [0.5 -0.3 0.2 -1.5 0.1 1.8 0.7]]
      (is (close3? (ex/franka-fk-inline q)
                   (ex/generic-serial-fk-inline q generic-fk-franka-xyz
                                                 generic-fk-franka-rpy generic-fk-franka-axis)
                   1e-12)))))

(deftest generic-serial-fk-kernel-matches-generic-serial-fk-inline
  (testing "dispatched through warp.warp/launch + WpArray plumbing (including
            the plain-number n uniform), the kernel's :js path gives the
            identical result to calling generic-serial-fk-inline directly"
    (let [q          [0.5 -0.3 0.2 -1.5 0.1 1.8 0.7]
          q-in       (wp/wp-array (vec q))
          joint-xyz  (wp/wp-array (vec (apply concat generic-fk-franka-xyz)))
          joint-rpy  (wp/wp-array (vec (apply concat generic-fk-franka-rpy)))
          joint-axis (wp/wp-array (vec (apply concat generic-fk-franka-axis)))
          ee-out     (wp/wp-array (vec (repeat 3 0.0)))]
      (wp/launch {:kernel-fn (:fn ex/generic-serial-fk-kernel) :dim 1
                  :inputs [q-in joint-xyz joint-rpy joint-axis ee-out 7]})
      (is (close3? (ex/generic-serial-fk-inline q generic-fk-franka-xyz
                                                 generic-fk-franka-rpy generic-fk-franka-axis)
                   @ee-out 1e-12)))))

(deftest generic-serial-fk-kernel-batches-independent-envs
  (testing "dim=N runs N independent envs (env*n offsets don't collide
            across envs)"
    (let [envs       [[0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                       [0.5 -0.3 0.2 -1.5 0.1 1.8 0.7]]
          q-in       (wp/wp-array (vec (apply concat envs)))
          joint-xyz  (wp/wp-array (vec (apply concat generic-fk-franka-xyz)))
          joint-rpy  (wp/wp-array (vec (apply concat generic-fk-franka-rpy)))
          joint-axis (wp/wp-array (vec (apply concat generic-fk-franka-axis)))
          ee-out     (wp/wp-array (vec (repeat 6 0.0)))]
      (wp/launch {:kernel-fn (:fn ex/generic-serial-fk-kernel) :dim 2
                  :inputs [q-in joint-xyz joint-rpy joint-axis ee-out 7]})
      (is (close3? (ex/franka-fk-inline (first envs)) (subvec @ee-out 0 3) 1e-12))
      (is (close3? (ex/franka-fk-inline (second envs)) (subvec @ee-out 3 6) 1e-12)))))

(deftest generic-serial-fk-kernel-registers-correct-bindings
  (is (= [{:binding 0 :kind :storage :input-index 0 :writeback false}
          {:binding 1 :kind :storage :input-index 1 :writeback false}
          {:binding 2 :kind :storage :input-index 2 :writeback false}
          {:binding 3 :kind :storage :input-index 3 :writeback false}
          {:binding 4 :kind :storage :input-index 4 :writeback true}
          {:binding 5 :kind :uniform :input-index 5 :writeback false}]
         (:bindings ex/generic-serial-fk-kernel)))
  (is (= 64 (:workgroup-size ex/generic-serial-fk-kernel))))
