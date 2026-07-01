(ns kotoba.lang.kami-nv-compat.actions.action-term-test
  "ActionTerm + ActionManager coverage."
  (:require [clojure.test :refer [deftest is testing]]
            [kotoba.lang.kami-nv-compat.actions.action-term :as at]))

(defn mock-term [cfg]
  (let [state (at/make-action-term-state cfg)
        applied (atom nil)]
    (merge state
      {:applied applied}
      (reify at/IActionTerm
        (action-dim [_] (:action-dim state))
        (process-actions! [_ raw] (at/base-process-actions! state raw))
        (apply-actions! [_ env] (reset! applied (:processed-actions state)))
        (reset-term! [_] (at/base-reset! state))))))

(deftest action-term-process
  (let [t (mock-term {:joint-names [0 1 2] :scale 2.0 :offset 1.0})]
    (at/process-actions! t [1 2 3])
    (is (= [1 2 3] @(:raw-actions t)))
    (is (= [3.0 5.0 7.0] @(:processed-actions t)))))   ; 1*2+1, 2*2+1, 3*2+1

(deftest action-term-reset
  (let [t (mock-term {:joint-names [0 1]})]
    (at/process-actions! t [5 6])
    (at/reset-term! t)
    (is (= [0 0] @(:raw-actions t)))
    (is (= [0 0] @(:processed-actions t)))))

(deftest action-term-validation
  (is (thrown? clojure.lang.ExceptionInfo (at/make-action-term-state {:joint-names []})))
  (let [t (mock-term {:joint-names [0 1 2]})]
    (is (thrown? clojure.lang.ExceptionInfo (at/process-actions! t [1 2])))))

(deftest action-manager-compose
  (let [t1 (mock-term {:joint-names [0 1]})
        t2 (mock-term {:joint-names [2]})
        mgr (at/action-manager [t1 t2])]
    (is (= 3 (:total-action-dim mgr)))
    (at/manager-process-actions! mgr [10 20 30])
    (is (= [10 20] @(:processed-actions t1)))
    (is (= [30] @(:processed-actions t2)))
    (at/manager-apply-actions! mgr {})
    (is (= [10 20] @(:applied t1)))
    (is (= [30] @(:applied t2)))
    (at/manager-reset! mgr)
    (is (= [0 0] @(:raw-actions t1)))))

(deftest action-manager-validation
  (is (thrown? clojure.lang.ExceptionInfo (at/action-manager [])))
  (let [mgr (at/action-manager [(mock-term {:joint-names [0]})])]
    (is (thrown? clojure.lang.ExceptionInfo (at/manager-process-actions! mgr [1 2])))))
