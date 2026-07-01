(ns kotoba.lang.kami-nv-compat.dynamics.articulated-dynamics
  "Featherstone-1983 articulated-body dynamics — JVM port of
  src/dynamics/articulated-dynamics.ts (the Python iter 68-70 reference, ported
  algorithm-for-algorithm). Pure, zero runtime deps (the in-browser pure-JS
  fallback for the kami-engine Rust->wasm32 substrate).

  Spatial-vector convention (Featherstone 2008):
    v = [angular_x angular_y angular_z linear_x linear_y linear_z]
  6×6 Plücker transforms + spatial inertias are nested vectors.

  WAVE 13 (foundation): 3×3 matrix helpers + spatial-inertia. The 6×6
  spatial-algebra helpers (mat66-*, vec6-*, outer6, spatial-cross-*, plucker,
  joint-motion-*), buildArticulation, forward-kinematics, geometric-Jacobian,
  and the ABA/RNEA/CRBA solvers land in the waves that consume them. Pure data
  shapes (UrdfLink/Joint/System are plain maps; see
  kotoba.lang.kami-nv-compat.dynamics.urdf-parser). ADR-2607020130.")

;; ── 3-vector / 3×3 matrix helpers ─────────────────────────────────────────

(defn skew3 [v]
  [[0 (- (v 2)) (v 1)]
   [(v 2) 0 (- (v 0))]
   [(- (v 1)) (v 0) 0]])

(defn mat3-mul [a b]
  (vec (for [i (range 3)]
         (vec (for [j (range 3)]
                (loop [k 0 s 0]
                  (if (>= k 3) s
                    (recur (inc k) (+ s (* (get-in a [i k]) (get-in b [k j])))))))))))

(defn mat3-t [a]
  [[(get-in a [0 0]) (get-in a [1 0]) (get-in a [2 0])]
   [(get-in a [0 1]) (get-in a [1 1]) (get-in a [2 1])]
   [(get-in a [0 2]) (get-in a [1 2]) (get-in a [2 2])]])

(defn mat3-add [a b]
  (vec (for [i (range 3)] (vec (for [j (range 3)] (+ (get-in a [i j]) (get-in b [i j])))))))

(defn mat3-scale [a s]
  (vec (for [i (range 3)] (vec (for [j (range 3)] (* (get-in a [i j]) s))))))

(defn rot-from-rpy [rpy]
  (let [[r p y] rpy
        cr (Math/cos r) sr (Math/sin r)
        cp (Math/cos p) sp (Math/sin p)
        cy (Math/cos y) sy (Math/sin y)]
    [[(* cy cp) (- (* cy sp sr) (* sy cr)) (+ (* cy sp cr) (* sy sr))]
     [(* sy cp) (+ (* sy sp sr) (* cy cr)) (- (* sy sp cr) (* cy sr))]
     [(- sp) (* cp sr) (* cp cr)]]))

(defn rodrigues-rotation [axis angle]
  (let [[ax ay az] axis
        c (Math/cos angle) s (Math/sin angle) oc (- 1 c)]
    [[(+ c (* ax ax oc)) (- (* ax ay oc) (* az s)) (+ (* ax az oc) (* ay s))]
     [(+ (* ay ax oc) (* az s)) (+ c (* ay ay oc)) (- (* ay az oc) (* ax s))]
     [(- (* az ax oc) (* ay s)) (+ (* az ay oc) (* ax s)) (+ c (* az az oc))]]))

;; ── spatial inertia from a link ───────────────────────────────────────────

(defn- zeros66 [] (vec (repeat 6 (vec (repeat 6 0)))))

(defn spatial-inertia-from-link
  "6×6 spatial inertia of a link (mass, full inertia tensor, COM offset)."
  [link]
  (let [{:keys [mass ixx iyy izz ixy ixz iyz]} (:inertia link)
        c (:xyz (:com (:inertia link)))
        Ic [[ixx ixy ixz] [ixy iyy iyz] [ixz iyz izz]]
        skC (skew3 c)
        pseudo (mat3-scale (mat3-mul skC (mat3-t skC)) mass)
        upper-left (mat3-add Ic pseudo)
        mSkC (mat3-scale skC mass)
        mSkCt (mat3-t mSkC)
        I (atom (zeros66))]
    (doseq [i (range 3) j (range 3)]
      (swap! I (fn [m] (-> m
                           (assoc-in [i j]             (get-in upper-left [i j]))
                           (assoc-in [i (+ j 3)]       (get-in mSkC [i j]))
                           (assoc-in [(+ i 3) j]       (get-in mSkCt [i j]))
                           (assoc-in [(+ i 3) (+ j 3)] (if (= i j) mass 0))))))
    @I))

;; ── Plücker transform + joint motion subspace (build helpers) ─────────────

(defn- plucker-transform [rot-child-to-parent r]
  (let [Rt (mat3-t rot-child-to-parent)
        RtSkr (mat3-mul Rt (skew3 r))
        out (atom (zeros66))]
    (doseq [i (range 3) j (range 3)]
      (swap! out (fn [m] (-> m
                             (assoc-in [i j]             (get-in Rt [i j]))
                             (assoc-in [(+ i 3) j]       (- (get-in RtSkr [i j])))
                             (assoc-in [(+ i 3) (+ j 3)] (get-in Rt [i j]))))))
    @out))

(defn- joint-motion-subspace [joint]
  (let [[ax ay az] (:axis joint)
        n (Math/sqrt (+ (* ax ax) (* ay ay) (* az az)))]
    (if (< n 1e-12)
      [0 0 0 0 0 0]
      (let [ux (/ ax n) uy (/ ay n) uz (/ az n)]
        (condp = (:kind joint)
          "revolute"   [ux uy uz 0 0 0]
          "continuous" [ux uy uz 0 0 0]
          "prismatic"  [0 0 0 ux uy uz]
          [0 0 0 0 0 0])))))                       ; fixed

;; ── BuiltArticulation ─────────────────────────────────────────────────────

(defn build-articulation
  "Build the runtime articulation model from a parsed URDF system. Fixed joints
  are fused; :parent-joint is -1 for joints rooted at the base link."
  [sys]
  (let [children (set (map :child (:joints sys)))
        base-links (filter #(not (contains? children (:name %))) (:links sys))]
    (when (not= 1 (count base-links))
      (throw (ex-info (str "build-articulation: expected exactly 1 base link; found "
                           (count base-links)) {:base-links (mapv :name base-links)})))
    (let [base-name (:name (first base-links))
          link-by-name (into {} (for [l (:links sys)] [(:name l) l]))
          moving (vec (filter #(not= (:kind %) "fixed") (:joints sys)))
          n (count moving)]
      (if (zero? n)
        {:n 0 :joint-names [] :joint-kinds [] :parent-joint [] :motion-subspace []
         :fixed-origin-transform [] :child-link-inertia [] :joint-damping []
         :joint-friction [] :rpy-rotation-matrix [] :xyz-translation [] :joint-axis []}
        (let [link->parent-joint (into {} (map-indexed (fn [i j] [(:child j) i]) moving))
              parent-joint (vec (for [j moving]
                                  (loop [cursor (:parent j) pidx -1 guard 0]
                                    (cond
                                      (or (>= guard 1000) (= cursor base-name)) pidx
                                      (link->parent-joint cursor)                (link->parent-joint cursor)
                                      :else (if-let [upstream (first (filter #(= (:child %) cursor) (:joints sys)))]
                                              (recur (:parent upstream) pidx (inc guard))
                                              pidx)))))
              joint-axis (vec (for [j moving]
                                (let [[ax ay az] (:axis j)
                                      an (Math/sqrt (+ (* ax ax) (* ay ay) (* az az)))]
                                  (if (< an 1e-12) [0 0 1] [(/ ax an) (/ ay an) (/ az an)]))))]
          {:n n
           :joint-names  (mapv :name moving)
           :joint-kinds  (mapv :kind moving)
           :parent-joint parent-joint
           :motion-subspace     (mapv joint-motion-subspace moving)
           :fixed-origin-transform (vec (for [j moving]
                                          (plucker-transform (rot-from-rpy (get-in j [:origin :rpy]))
                                                             (get-in j [:origin :xyz]))))
           :child-link-inertia   (vec (for [j moving]
                                        (let [link (link-by-name (:child j))]
                                          (when-not link
                                            (throw (ex-info (str "build-articulation: missing child link " (:child j)) {})))
                                          (spatial-inertia-from-link link))))
           :joint-damping   (mapv #(get % :damping 0) moving)
           :joint-friction  (mapv #(get % :friction 0) moving)
           :rpy-rotation-matrix (mapv #(rot-from-rpy (get-in % [:origin :rpy])) moving)
           :xyz-translation     (mapv #(get-in % [:origin :xyz]) moving)
           :joint-axis          joint-axis})))))

(defn make-zero-state
  "Zero generalized coords/velocities/accelerations for an n-DoF articulation."
  [n]
  {:q (vec (repeat n 0)) :qdot (vec (repeat n 0)) :qddot (vec (repeat n 0))})
