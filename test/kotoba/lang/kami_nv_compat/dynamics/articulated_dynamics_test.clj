(ns kotoba.lang.kami-nv-compat.dynamics.articulated-dynamics-test
  "articulated-dynamics FOUNDATION coverage (matrix + spatial-inertia helpers).
  buildArticulation / kinematics / solvers land with their waves."
  (:require [clojure.test :refer [deftest is testing]]
            [kotoba.lang.kami-nv-compat.dynamics.articulated-dynamics :as ad]))

(defn- mat-close? [a b tol]
  (let [diffs (for [i (range (count a)) j (range (count (a i)))]
                (- (get-in a [i j]) (get-in b [i j])))]
    (every? #(< (Math/abs %) tol) diffs)))

(def i3 [[1 0 0] [0 1 0] [0 0 1]])
(def i6 [[1 0 0 0 0 0] [0 1 0 0 0 0] [0 0 1 0 0 0]
         [0 0 0 1 0 0] [0 0 0 0 1 0] [0 0 0 0 0 1]])

(deftest skew3-test
  (is (mat-close? (ad/skew3 [1 2 3]) [[0 -3 2] [3 0 -1] [-2 1 0]] 1e-12))
  (is (mat-close? (ad/skew3 [0 0 0]) [[0 0 0] [0 0 0] [0 0 0]] 1e-12)))

(deftest mat3-algebra
  (testing "mat3-mul: identity is left/right identity"
    (let [m [[1 2 3] [4 5 6] [7 8 9]]]
      (is (mat-close? (ad/mat3-mul i3 m) m 1e-12))
      (is (mat-close? (ad/mat3-mul m i3) m 1e-12))))
  (testing "transpose of a skew-symmetric matrix is its negative"
    (is (mat-close? (ad/mat3-t (ad/skew3 [1 2 3]))
                    (ad/mat3-scale (ad/skew3 [1 2 3]) -1) 1e-12))))

(deftest rotation-from-rpy-and-rodrigues
  (testing "rot-from-rpy of zeros is identity"
    (is (mat-close? (ad/rot-from-rpy [0 0 0]) i3 1e-12)))
  (testing "rodrigues: zero angle is identity; 90 deg about z rotates x->y"
    (is (mat-close? (ad/rodrigues-rotation [0 0 1] 0) i3 1e-12))
    (let [r   (ad/rodrigues-rotation [0 0 1] (/ Math/PI 2))
          x-> (mapv #(reduce + (map * [1 0 0] %)) r)]   ; R · [1,0,0]
      (is (< (Math/abs (- (x-> 0) 0.0)) 1e-9))
      (is (< (Math/abs (- (x-> 1) 1.0)) 1e-9)))))       ; x maps to +y

(deftest spatial-inertia-point-mass-at-origin
  (testing "unit inertia + unit mass at origin -> 6x6 identity spatial inertia"
    (let [link {:inertia {:mass 1 :ixx 1 :iyy 1 :izz 1 :ixy 0 :ixz 0 :iyz 0
                          :com {:xyz [0 0 0] :rpy [0 0 0]}}}]
      (is (mat-close? (ad/spatial-inertia-from-link link) i6 1e-12)))))
