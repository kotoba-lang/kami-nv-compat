(ns kotoba.lang.kami-nv-compat.warp.examples-test
  "Coverage for warp.examples wave 1 (damping-kernel, pendulum-step-kernel,
  cartpole-step-kernel). Each kernel is exercised via warp.warp/launch,
  matching how a real caller would dispatch the sync/CPU-fallback path."
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
