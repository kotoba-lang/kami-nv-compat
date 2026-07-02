(ns kotoba.lang.kami-nv-compat.dynamics.urdf-parser-test
  "URDF parser coverage (no dedicated TS test)."
  (:require [clojure.test :refer [deftest is testing]]
            [kotoba.lang.kami-nv-compat.dynamics.urdf-parser :as up]))

(def arm-urdf
  "<?xml version=\"1.0\"?>
<robot name=\"arm\">
  <link name=\"base\"><inertial><mass value=\"2.0\"/><inertia ixx=\"0.1\" iyy=\"0.2\" izz=\"0.3\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>
  <link name=\"link1\"/>
  <joint name=\"j0\" type=\"revolute\">
    <origin xyz=\"0 0 0.1\" rpy=\"0 0 0\"/>
    <parent link=\"base\"/>
    <child link=\"link1\"/>
    <axis xyz=\"0 0 1\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>
</robot>")

(deftest parse-urdf-basic
  (let [sys (up/parse-urdf arm-urdf)]
    (is (= "arm" (:name sys)))
    (is (= 2 (count (:links sys))))
    (is (= "base" (-> sys :links first :name)))
    (is (= 2.0 (-> sys :links first :inertia :mass)))
    (is (= 0.2 (-> sys :links first :inertia :iyy)))
    (is (= [0 0 0] (-> sys :links first :inertia :com :xyz)))     ; no <origin> in inertial
    (is (= "link1" (-> sys :links second :name)))
    (is (zero? (-> sys :links second :inertia :mass)))))          ; self-closing -> default inertia

(deftest parse-urdf-joint
  (let [j (first (:joints (up/parse-urdf arm-urdf)))]
    (is (= "j0" (:name j)))
    (is (= "revolute" (:kind j)))
    (is (= "base" (:parent j)))
    (is (= "link1" (:child j)))
    (is (= [0.0 0.0 0.1] (-> j :origin :xyz)))
    (is (= [0.0 0.0 1.0] (:axis j)))
    (is (= 0.5 (:damping j)))
    (is (= 0.1 (:friction j)))))

(deftest parse-urdf-validation
  (testing "unknown joint kind throws"
    (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs cljs.core/ExceptionInfo)
                 (up/parse-urdf "<robot name=\"r\"><joint name=\"j\" type=\"teleport\"><parent link=\"a\"/><child link=\"b\"/></joint></robot>"))))
  (testing "missing <parent> throws"
    (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs cljs.core/ExceptionInfo)
                 (up/parse-urdf "<robot name=\"r\"><joint name=\"j\" type=\"revolute\"><child link=\"b\"/></joint></robot>"))))
  (testing "no links/joints -> empty vectors, default name"
    (let [sys (up/parse-urdf "<robot></robot>")]
      (is (= "robot" (:name sys)))
      (is (empty? (:links sys)))
      (is (empty? (:joints sys))))))
