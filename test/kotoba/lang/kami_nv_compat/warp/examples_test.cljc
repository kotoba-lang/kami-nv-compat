(ns kotoba.lang.kami-nv-compat.warp.examples-test
  "Coverage for warp.examples waves 1-2 (damping-kernel, pendulum-step-kernel,
  cartpole-step-kernel, two-link-arm-step-kernel). Each kernel is exercised
  via warp.warp/launch, matching how a real caller would dispatch the
  sync/CPU-fallback path."
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
