(ns kotoba.lang.kami-nv-compat.dynamics.urdf-parser
  "URDF text -> UrdfArticulatedSystem parser — JVM port of
  src/dynamics/urdf-parser.ts. Pure regex / no XML-parser dep — handles the
  URDF subset emitted by the assets/urdf-builder (no namespaces, no CDATA).
  Robust for the Franka/ANYmal/UR10 assets + any URDF the SDK itself
  produces; strict (descriptive errors) outside that subset. Bridges an
  asset's :urdf-text directly into the articulated-dynamics build step.
  Wave 12 of ADR-2607020130.

  Returns plain maps: a system is {:name :links :joints}; a link is
  {:name :inertia}; a joint is {:name :kind :parent :child :origin :axis
  :damping :friction}; a pose is {:xyz :rpy}; inertia is
  {:mass :ixx :iyy :izz :ixy :ixz :iyz :com}.")

;; ── helpers ───────────────────────────────────────────────────────────────

(defn- extract-attr [text re]
  (when-let [m (re-find re text)] (nth m 1)))

(defn- parse-num [s] #?(:clj (Double/parseDouble s) :cljs (js/parseFloat s)))

(defn- extract-triplet
  ([text attr-re]
   (extract-triplet text attr-re [0 0 0]))
  ([text attr-re fallback]
   (if-let [m (re-find attr-re text)]
     (let [parts (vec (filter seq (.split ^String (nth m 1) "\\s+")))]
       (when (not= 3 (count parts))
         (throw (ex-info (str "URDF parse: expected 3-tuple, got '" (nth m 1) "'") {})))
       [(parse-num (parts 0)) (parse-num (parts 1)) (parse-num (parts 2))])
     fallback)))

(defn- parse-pose [block-text]
  (if (nil? block-text)
    {:xyz [0 0 0] :rpy [0 0 0]}
    {:xyz (extract-triplet block-text #"<origin[^>]*\sxyz=\"([^\"]+)\"")
     :rpy (extract-triplet block-text #"<origin[^>]*\srpy=\"([^\"]+)\"")}))

(defn- default-inertia []
  {:mass 0 :ixx 0 :iyy 0 :izz 0 :ixy 0 :ixz 0 :iyz 0
   :com {:xyz [0 0 0] :rpy [0 0 0]}})

(defn- parse-link [name body]
  (if-let [m (re-find #"<inertial>([\s\S]*?)</inertial>" body)]
    (let [inertial (nth m 1)
          attr     #(extract-attr inertial %)]
      {:name name
       :inertia {:mass (parse-num (or (attr #"<mass\s+value=\"([^\"]+)\"") "0"))
                 :ixx  (parse-num (or (attr #"<inertia[^>]*\sixx=\"([^\"]+)\"") "0"))
                 :iyy  (parse-num (or (attr #"<inertia[^>]*\siyy=\"([^\"]+)\"") "0"))
                 :izz  (parse-num (or (attr #"<inertia[^>]*\sizz=\"([^\"]+)\"") "0"))
                 :ixy  (parse-num (or (attr #"<inertia[^>]*\sixy=\"([^\"]+)\"") "0"))
                 :ixz  (parse-num (or (attr #"<inertia[^>]*\sixz=\"([^\"]+)\"") "0"))
                 :iyz  (parse-num (or (attr #"<inertia[^>]*\siyz=\"([^\"]+)\"") "0"))
                 :com  (parse-pose inertial)}})
    {:name name :inertia (default-inertia)}))

(def ^:private valid-joint-kinds #{"revolute" "continuous" "prismatic" "fixed"})

(defn- parse-joint [name kind body]
  (when-not (contains? valid-joint-kinds kind)
    (throw (ex-info (str "URDF parse: unknown joint type '" kind "' on joint '" name "'") {:kind kind})))
  (let [parent (extract-attr body #"<parent\s+link=\"([^\"]+)\"")
        _      (when-not parent (throw (ex-info (str "URDF parse: joint '" name "' has no <parent>") {})))
        child  (extract-attr body #"<child\s+link=\"([^\"]+)\"")
        _      (when-not child (throw (ex-info (str "URDF parse: joint '" name "' has no <child>") {})))]
    {:name name
     :kind kind
     :parent parent
     :child child
     :origin (parse-pose body)
     :axis (extract-triplet body #"<axis\s+xyz=\"([^\"]+)\"" [1 0 0])
     :damping (parse-num (or (extract-attr body #"<dynamics[^>]*\sdamping=\"([^\"]+)\"") "0"))
     :friction (parse-num (or (extract-attr body #"<dynamics[^>]*\sfriction=\"([^\"]+)\"") "0"))}))

;; ── public ────────────────────────────────────────────────────────────────

(defn parse-urdf
  "Parse a URDF text into an UrdfArticulatedSystem map {:name :links :joints}.
  Recognises <robot>, <link> (self-closing or with <inertial> body), <joint>
  (revolute/continuous/prismatic/fixed with origin/parent/child/axis/limit/
  dynamics). Ignores visual/collision/material."
  [text]
  (let [name (or (extract-attr text #"<robot\s+name=\"([^\"]+)\"") "robot")
        links (for [m (re-seq #"<link\s+name=\"([^\"]+)\"\s*(?:/>|>([\s\S]*?)</link>)" text)]
                (let [link-name (nth m 1) body (nth m 2)]
                  (if body (parse-link link-name body) {:name link-name :inertia (default-inertia)})))
        joints (for [m (re-seq #"<joint\s+name=\"([^\"]+)\"\s+type=\"([^\"]+)\"\s*>([\s\S]*?)</joint>" text)]
                 (parse-joint (nth m 1) (nth m 2) (nth m 3)))]
    {:name name :links (vec links) :joints (vec joints)}))
