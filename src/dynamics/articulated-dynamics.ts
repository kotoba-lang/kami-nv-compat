// TypeScript port of kotodama.nv_compat.dynamics.articulated_dynamics
//
// Featherstone-1983 Articulated-Body Algorithm + RNEA + CRBA + forward
// kinematics + geometric Jacobian. Mirrors the Python implementation
// (iter 68-70) algorithm-for-algorithm so the two impls can be cross-
// validated. Pure TypeScript, no external deps; ports straight to
// browsers + WASM (for kami-engine Rust→wasm32 substrate, the same
// algorithm runs natively; this TS path is the in-browser pure-JS
// fallback).
//
// ADR-2605261800 §D6+§D11 NV-compat namespace localization. "PhysX®"
// is a trademark of NVIDIA Corporation; this module is API namespace
// localization for forward dynamics interoperability per Google v.
// Oracle (2021).
//
// Spatial-vector convention (matches Featherstone 2008):
//   v = (angular_x, angular_y, angular_z, linear_x, linear_y, linear_z)
// 6×6 Plücker transforms are nested arrays.

// ── URDF-side types (mirror kotodama.nv_compat._kernel) ────────────────

export interface UrdfPose {
  xyz: [number, number, number];
  rpy: [number, number, number];
}

export interface UrdfInertia {
  mass: number;
  ixx: number;
  iyy: number;
  izz: number;
  ixy: number;
  ixz: number;
  iyz: number;
  com: UrdfPose;
}

export interface UrdfLink {
  name: string;
  inertia: UrdfInertia;
}

export type UrdfJointKind =
  | "revolute"
  | "continuous"
  | "prismatic"
  | "fixed";

export interface UrdfJoint {
  name: string;
  kind: UrdfJointKind;
  parent: string;
  child: string;
  origin: UrdfPose;
  axis: [number, number, number];
  damping?: number;
  friction?: number;
}

export interface UrdfArticulatedSystem {
  name: string;
  links: UrdfLink[];
  joints: UrdfJoint[];
}

// ── 3-vector / 3×3 matrix helpers ─────────────────────────────────────────

export function skew3(v: readonly number[]): number[][] {
  return [
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0],
  ];
}

export function mat3Mul(a: number[][], b: number[][]): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < 3; i++) {
    const row: number[] = [];
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[i][k] * b[k][j];
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

export function mat3T(a: number[][]): number[][] {
  return [
    [a[0][0], a[1][0], a[2][0]],
    [a[0][1], a[1][1], a[2][1]],
    [a[0][2], a[1][2], a[2][2]],
  ];
}

export function mat3Add(a: number[][], b: number[][]): number[][] {
  return [
    [a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
    [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
    [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]],
  ];
}

export function mat3Scale(a: number[][], s: number): number[][] {
  return [
    [a[0][0] * s, a[0][1] * s, a[0][2] * s],
    [a[1][0] * s, a[1][1] * s, a[1][2] * s],
    [a[2][0] * s, a[2][1] * s, a[2][2] * s],
  ];
}

export function rotFromRpy(rpy: readonly number[]): number[][] {
  const [r, p, y] = rpy;
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

export function rodriguesRotation(
  axis: readonly number[],
  angle: number,
): number[][] {
  const [ax, ay, az] = axis;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const oc = 1 - c;
  return [
    [c + ax * ax * oc, ax * ay * oc - az * s, ax * az * oc + ay * s],
    [ay * ax * oc + az * s, c + ay * ay * oc, ay * az * oc - ax * s],
    [az * ax * oc - ay * s, az * ay * oc + ax * s, c + az * az * oc],
  ];
}

// ── 6×6 spatial matrix helpers ────────────────────────────────────────────

function zeros66(): number[][] {
  return Array.from({ length: 6 }, () => new Array(6).fill(0));
}

function mat66Mul(a: number[][], b: number[][]): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) {
      let s = 0;
      for (let k = 0; k < 6; k++) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
  }
  return out;
}

function mat66T(a: number[][]): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) out[i][j] = a[j][i];
  }
  return out;
}

function mat66Vec(a: number[][], v: readonly number[]): number[] {
  const out = new Array<number>(6).fill(0);
  for (let i = 0; i < 6; i++) {
    let s = 0;
    for (let k = 0; k < 6; k++) s += a[i][k] * v[k];
    out[i] = s;
  }
  return out;
}

function vec6Add(a: readonly number[], b: readonly number[]): number[] {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5]];
}

function vec6Scale(a: readonly number[], s: number): number[] {
  return [a[0] * s, a[1] * s, a[2] * s, a[3] * s, a[4] * s, a[5] * s];
}

function vec6Dot(a: readonly number[], b: readonly number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
}

function outer6(a: readonly number[], b: readonly number[]): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) out[i][j] = a[i] * b[j];
  }
  return out;
}

function mat66Sub(a: number[][], b: number[][]): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) out[i][j] = a[i][j] - b[i][j];
  }
  return out;
}

function mat66Add(a: number[][], b: number[][]): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) out[i][j] = a[i][j] + b[i][j];
  }
  return out;
}

function mat66Scale(a: number[][], s: number): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) out[i][j] = a[i][j] * s;
  }
  return out;
}

// ── spatial cross products ────────────────────────────────────────────────

function spatialCrossMotion(v: readonly number[]): number[][] {
  const skw = skew3([v[0], v[1], v[2]]);
  const sku = skew3([v[3], v[4], v[5]]);
  const out = zeros66();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = skw[i][j];
      out[i + 3][j] = sku[i][j];
      out[i + 3][j + 3] = skw[i][j];
    }
  }
  return out;
}

function spatialCrossForce(v: readonly number[]): number[][] {
  const skw = skew3([v[0], v[1], v[2]]);
  const sku = skew3([v[3], v[4], v[5]]);
  const out = zeros66();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = skw[i][j];
      out[i][j + 3] = sku[i][j];
      out[i + 3][j + 3] = skw[i][j];
    }
  }
  return out;
}

// ── Plücker transform from translation + rotation ────────────────────────

function pluckerTransform(
  rotChildToParent: number[][],
  r: readonly number[],
): number[][] {
  const Rt = mat3T(rotChildToParent);
  const skR = skew3(r);
  const RtSkr = mat3Mul(Rt, skR);
  const out = zeros66();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = Rt[i][j];
      out[i + 3][j] = -RtSkr[i][j];
      out[i + 3][j + 3] = Rt[i][j];
    }
  }
  return out;
}

// ── spatial inertia from link ─────────────────────────────────────────────

export function spatialInertiaFromLink(link: UrdfLink): number[][] {
  const { mass: m, ixx, iyy, izz, ixy, ixz, iyz, com } = link.inertia;
  const c = com.xyz;
  const Ic: number[][] = [
    [ixx, ixy, ixz],
    [ixy, iyy, iyz],
    [ixz, iyz, izz],
  ];
  const skC = skew3(c);
  const skCt = mat3T(skC);
  let pseudo = mat3Mul(skC, skCt);
  pseudo = mat3Scale(pseudo, m);
  const upperLeft = mat3Add(Ic, pseudo);
  const mSkC = mat3Scale(skC, m);
  const mSkCt = mat3T(mSkC);
  const I = zeros66();
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      I[i][j] = upperLeft[i][j];
      I[i][j + 3] = mSkC[i][j];
      I[i + 3][j] = mSkCt[i][j];
      I[i + 3][j + 3] = i === j ? m : 0;
    }
  }
  return I;
}

// ── joint motion subspace ─────────────────────────────────────────────────

function jointMotionSubspace(joint: UrdfJoint): number[] {
  const [ax, ay, az] = joint.axis;
  const n = Math.sqrt(ax * ax + ay * ay + az * az);
  if (n < 1e-12) return [0, 0, 0, 0, 0, 0];
  const ux = ax / n, uy = ay / n, uz = az / n;
  if (joint.kind === "revolute" || joint.kind === "continuous") {
    return [ux, uy, uz, 0, 0, 0];
  }
  if (joint.kind === "prismatic") {
    return [0, 0, 0, ux, uy, uz];
  }
  return [0, 0, 0, 0, 0, 0];
}

// ── joint motion transform X_J(q) ─────────────────────────────────────────

function identity6(): number[][] {
  const out = zeros66();
  for (let i = 0; i < 6; i++) out[i][i] = 1;
  return out;
}

function identity3(): number[][] {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

function jointMotionTransform(
  kind: UrdfJointKind,
  axisUnit: readonly number[],
  q: number,
): number[][] {
  if (kind === "revolute" || kind === "continuous") {
    const rot = rodriguesRotation(axisUnit, q);
    return pluckerTransform(rot, [0, 0, 0]);
  }
  if (kind === "prismatic") {
    return pluckerTransform(identity3(), [q * axisUnit[0], q * axisUnit[1], q * axisUnit[2]]);
  }
  return identity6();
}

// ── BuiltArticulation ─────────────────────────────────────────────────────

export interface BuiltArticulation {
  n: number;
  jointNames: string[];
  jointKinds: UrdfJointKind[];
  parentJoint: number[];                    // -1 = base
  motionSubspace: number[][];               // n × 6
  fixedOriginTransform: number[][][];       // n × 6 × 6
  childLinkInertia: number[][][];           // n × 6 × 6
  jointDamping: number[];                   // n
  jointFriction: number[];                  // n
  rpyRotationMatrix: number[][][];          // n × 3 × 3
  xyzTranslation: number[][];               // n × 3
  jointAxis: number[][];                    // n × 3, unit-normalised
}

export function buildArticulation(sys: UrdfArticulatedSystem): BuiltArticulation {
  // Identify base link.
  const childrenSet = new Set<string>();
  for (const j of sys.joints) childrenSet.add(j.child);
  const baseLinks = sys.links.filter((l) => !childrenSet.has(l.name)).map((l) => l.name);
  if (baseLinks.length !== 1) {
    throw new Error(
      `buildArticulation: expected exactly 1 base link; found ${baseLinks.length}: ${JSON.stringify(baseLinks)}`,
    );
  }
  const baseLinkName = baseLinks[0];
  const linkByName = new Map<string, UrdfLink>();
  for (const l of sys.links) linkByName.set(l.name, l);
  const moving = sys.joints.filter((j) => j.kind !== "fixed");
  const n = moving.length;
  if (n === 0) {
    return {
      n: 0,
      jointNames: [],
      jointKinds: [],
      parentJoint: [],
      motionSubspace: [],
      fixedOriginTransform: [],
      childLinkInertia: [],
      jointDamping: [],
      jointFriction: [],
      rpyRotationMatrix: [],
      xyzTranslation: [],
      jointAxis: [],
    };
  }
  // Build link → parent moving-joint index map.
  const linkToParentJoint = new Map<string, number>();
  for (let i = 0; i < moving.length; i++) {
    linkToParentJoint.set(moving[i].child, i);
  }
  // For each moving joint, walk up the chain to find its parent moving joint.
  const parentJoint: number[] = [];
  for (const j of moving) {
    let cursor = j.parent;
    let pidx = -1;
    for (let guard = 0; guard < 1000 && cursor !== baseLinkName; guard++) {
      const mapped = linkToParentJoint.get(cursor);
      if (mapped !== undefined) {
        pidx = mapped;
        break;
      }
      const upstream = sys.joints.find((k) => k.child === cursor);
      if (!upstream) break;
      cursor = upstream.parent;
    }
    parentJoint.push(pidx);
  }
  // Per-joint primitives.
  const motionSubspace = moving.map(jointMotionSubspace);
  const fixedOriginTransform: number[][][] = [];
  const rpyRotationMatrix: number[][][] = [];
  const xyzTranslation: number[][] = [];
  const jointAxis: number[][] = [];
  for (const j of moving) {
    const rot = rotFromRpy(j.origin.rpy);
    const r: [number, number, number] = [...j.origin.xyz];
    fixedOriginTransform.push(pluckerTransform(rot, r));
    rpyRotationMatrix.push(rot);
    xyzTranslation.push(r);
    const [ax, ay, az] = j.axis;
    const an = Math.sqrt(ax * ax + ay * ay + az * az);
    if (an < 1e-12) {
      jointAxis.push([0, 0, 1]);
    } else {
      jointAxis.push([ax / an, ay / an, az / an]);
    }
  }
  const childLinkInertia = moving.map((j) => {
    const link = linkByName.get(j.child);
    if (!link) throw new Error(`buildArticulation: missing child link ${j.child}`);
    return spatialInertiaFromLink(link);
  });
  return {
    n,
    jointNames: moving.map((j) => j.name),
    jointKinds: moving.map((j) => j.kind),
    parentJoint,
    motionSubspace,
    fixedOriginTransform,
    childLinkInertia,
    jointDamping: moving.map((j) => j.damping ?? 0),
    jointFriction: moving.map((j) => j.friction ?? 0),
    rpyRotationMatrix,
    xyzTranslation,
    jointAxis,
  };
}

// ── ArticulatedState ──────────────────────────────────────────────────────

export interface ArticulatedState {
  q: number[];
  qdot: number[];
  qddot: number[];
}

export function makeZeroState(n: number): ArticulatedState {
  return {
    q: new Array<number>(n).fill(0),
    qdot: new Array<number>(n).fill(0),
    qddot: new Array<number>(n).fill(0),
  };
}

// ── shared joint-transform cache ──────────────────────────────────────────

function computeJointTransforms(built: BuiltArticulation, q: readonly number[]): number[][][] {
  const X: number[][][] = [];
  for (let i = 0; i < built.n; i++) {
    const XJ = jointMotionTransform(built.jointKinds[i], built.jointAxis[i], q[i]);
    X.push(mat66Mul(XJ, built.fixedOriginTransform[i]));
  }
  return X;
}

// ── ABA forward dynamics ──────────────────────────────────────────────────

export function abaForward(
  built: BuiltArticulation,
  q: readonly number[],
  qdot: readonly number[],
  tau: readonly number[],
  gravity: readonly [number, number, number] = [0, 0, -9.81],
): number[] {
  const n = built.n;
  if (q.length !== n || qdot.length !== n || tau.length !== n) {
    throw new Error(
      `abaForward: q/qdot/tau must all have length ${n}; got ${q.length}/${qdot.length}/${tau.length}`,
    );
  }
  if (n === 0) return [];
  const X = computeJointTransforms(built, q);
  // Pass 1.
  const v: number[][] = [];
  const c: number[][] = [];
  for (let i = 0; i < n; i++) {
    const Si = built.motionSubspace[i];
    const Sq = vec6Scale(Si, qdot[i]);
    const pidx = built.parentJoint[i];
    const vpInI = pidx < 0 ? [0, 0, 0, 0, 0, 0] : mat66Vec(X[i], v[pidx]);
    const vi = vec6Add(vpInI, Sq);
    v.push(vi);
    const crossM = spatialCrossMotion(vi);
    c.push(mat66Vec(crossM, Sq));
  }
  // Articulated inertia + bias.
  const Ia: number[][][] = built.childLinkInertia.map((m) => m.map((row) => [...row]));
  const pa: number[][] = [];
  for (let i = 0; i < n; i++) {
    const Iv = mat66Vec(Ia[i], v[i]);
    const cf = spatialCrossForce(v[i]);
    pa.push(mat66Vec(cf, Iv));
  }
  // Pass 2.
  const U: number[][] = new Array<number[]>(n);
  const D: number[] = new Array<number>(n).fill(0);
  const u: number[] = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    const Si = built.motionSubspace[i];
    U[i] = mat66Vec(Ia[i], Si);
    let d = vec6Dot(Si, U[i]) + built.jointDamping[i];
    if (Math.abs(d) < 1e-12) d = 1e-12;
    D[i] = d;
    u[i] = tau[i] - vec6Dot(Si, pa[i]);
    const pidx = built.parentJoint[i];
    if (pidx >= 0) {
      const Uo = mat66Scale(outer6(U[i], U[i]), 1 / D[i]);
      const inner = mat66Sub(Ia[i], Uo);
      const Xt = mat66T(X[i]);
      const tmp = mat66Mul(Xt, inner);
      const contribI = mat66Mul(tmp, X[i]);
      Ia[pidx] = mat66Add(Ia[pidx], contribI);
      const IaC = mat66Vec(Ia[i], c[i]);
      const Uu = vec6Scale(U[i], u[i] / D[i]);
      const sumTerm = vec6Add(vec6Add(pa[i], IaC), Uu);
      const contribP = mat66Vec(Xt, sumTerm);
      pa[pidx] = vec6Add(pa[pidx], contribP);
    }
  }
  // Pass 3.
  const aBase: number[] = [0, 0, 0, -gravity[0], -gravity[1], -gravity[2]];
  const a: number[][] = new Array<number[]>(n);
  const qddot = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const pidx = built.parentJoint[i];
    const apIn = pidx < 0 ? mat66Vec(X[i], aBase) : mat66Vec(X[i], a[pidx]);
    const aPrime = vec6Add(apIn, c[i]);
    qddot[i] = (u[i] - vec6Dot(U[i], aPrime)) / D[i];
    a[i] = vec6Add(aPrime, vec6Scale(built.motionSubspace[i], qddot[i]));
  }
  return qddot;
}

export function articulatedStep(
  built: BuiltArticulation,
  state: ArticulatedState,
  tau: readonly number[],
  dt: number,
  gravity: readonly [number, number, number] = [0, 0, -9.81],
): void {
  const qddot = abaForward(built, state.q, state.qdot, tau, gravity);
  state.qddot = qddot;
  for (let i = 0; i < built.n; i++) {
    state.qdot[i] += dt * qddot[i];
    state.q[i] += dt * state.qdot[i];
  }
}

// ── RNEA inverse dynamics ─────────────────────────────────────────────────

export function rneaInverseDynamics(
  built: BuiltArticulation,
  q: readonly number[],
  qdot: readonly number[],
  qddot: readonly number[],
  gravity: readonly [number, number, number] = [0, 0, -9.81],
): number[] {
  const n = built.n;
  if (q.length !== n || qdot.length !== n || qddot.length !== n) {
    throw new Error(
      `rneaInverseDynamics: q/qdot/qddot must all have length ${n}`,
    );
  }
  if (n === 0) return [];
  const X = computeJointTransforms(built, q);
  const v: number[][] = [];
  const a: number[][] = [];
  const f: number[][] = [];
  const aBase: number[] = [0, 0, 0, -gravity[0], -gravity[1], -gravity[2]];
  for (let i = 0; i < n; i++) {
    const Si = built.motionSubspace[i];
    const Sq = vec6Scale(Si, qdot[i]);
    const Sqd = vec6Scale(Si, qddot[i]);
    const pidx = built.parentJoint[i];
    const vpIn = pidx < 0 ? [0, 0, 0, 0, 0, 0] : mat66Vec(X[i], v[pidx]);
    const apIn = pidx < 0 ? mat66Vec(X[i], aBase) : mat66Vec(X[i], a[pidx]);
    const vi = vec6Add(vpIn, Sq);
    v.push(vi);
    const crossM = spatialCrossMotion(vi);
    const cor = mat66Vec(crossM, Sq);
    a.push(vec6Add(vec6Add(apIn, Sqd), cor));
    const Ii = built.childLinkInertia[i];
    const Ia = mat66Vec(Ii, a[i]);
    const Iv = mat66Vec(Ii, vi);
    const cf = spatialCrossForce(vi);
    f.push(vec6Add(Ia, mat66Vec(cf, Iv)));
  }
  const tau = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    const Si = built.motionSubspace[i];
    tau[i] = vec6Dot(Si, f[i]) + built.jointDamping[i] * qdot[i];
    const pidx = built.parentJoint[i];
    if (pidx >= 0) {
      const Xt = mat66T(X[i]);
      f[pidx] = vec6Add(f[pidx], mat66Vec(Xt, f[i]));
    }
  }
  return tau;
}

export function coriolisGravityVector(
  built: BuiltArticulation,
  q: readonly number[],
  qdot: readonly number[],
  gravity: readonly [number, number, number] = [0, 0, -9.81],
): number[] {
  return rneaInverseDynamics(built, q, qdot, new Array<number>(built.n).fill(0), gravity);
}

// ── CRBA: joint-space inertia matrix ─────────────────────────────────────

export function crbaMassMatrix(
  built: BuiltArticulation,
  q: readonly number[],
): number[][] {
  const n = built.n;
  if (q.length !== n) {
    throw new Error(`crbaMassMatrix: q length must be ${n}; got ${q.length}`);
  }
  if (n === 0) return [];
  const X = computeJointTransforms(built, q);
  const Ic: number[][][] = built.childLinkInertia.map((m) => m.map((row) => [...row]));
  for (let i = n - 1; i >= 0; i--) {
    const pidx = built.parentJoint[i];
    if (pidx >= 0) {
      const Xt = mat66T(X[i]);
      const tmp = mat66Mul(Xt, Ic[i]);
      const contrib = mat66Mul(tmp, X[i]);
      Ic[pidx] = mat66Add(Ic[pidx], contrib);
    }
  }
  const M: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const Si = built.motionSubspace[i];
    let F = mat66Vec(Ic[i], Si);
    M[i][i] = vec6Dot(Si, F);
    let j = i;
    while (built.parentJoint[j] >= 0) {
      const Xt = mat66T(X[j]);
      F = mat66Vec(Xt, F);
      j = built.parentJoint[j];
      M[i][j] = vec6Dot(built.motionSubspace[j], F);
      M[j][i] = M[i][j];
    }
  }
  return M;
}

export function kineticEnergy(
  built: BuiltArticulation,
  q: readonly number[],
  qdot: readonly number[],
): number {
  const M = crbaMassMatrix(built, q);
  let s = 0;
  for (let i = 0; i < built.n; i++) {
    const row = M[i];
    const qi = qdot[i];
    for (let j = 0; j < built.n; j++) s += qi * row[j] * qdot[j];
  }
  return 0.5 * s;
}

// ── Forward kinematics ────────────────────────────────────────────────────

export interface JointWorldPose {
  R: number[][]; // 3×3
  p: number[];   // 3
}

export function forwardKinematics(
  built: BuiltArticulation,
  q: readonly number[],
): JointWorldPose[] {
  if (q.length !== built.n) {
    throw new Error(`forwardKinematics: q length must be ${built.n}; got ${q.length}`);
  }
  const out: JointWorldPose[] = [];
  for (let i = 0; i < built.n; i++) {
    const Rorigin = built.rpyRotationMatrix[i];
    const porigin = built.xyzTranslation[i];
    const axis = built.jointAxis[i];
    const kind = built.jointKinds[i];
    let RiInParent: number[][];
    let piInParent: number[];
    if (kind === "revolute" || kind === "continuous") {
      const Rq = rodriguesRotation(axis, q[i]);
      RiInParent = mat3Mul(Rorigin, Rq);
      piInParent = [...porigin];
    } else if (kind === "prismatic") {
      RiInParent = Rorigin.map((row) => [...row]);
      const delta = [q[i] * axis[0], q[i] * axis[1], q[i] * axis[2]];
      const RoriginDelta = [
        Rorigin[0][0] * delta[0] + Rorigin[0][1] * delta[1] + Rorigin[0][2] * delta[2],
        Rorigin[1][0] * delta[0] + Rorigin[1][1] * delta[1] + Rorigin[1][2] * delta[2],
        Rorigin[2][0] * delta[0] + Rorigin[2][1] * delta[1] + Rorigin[2][2] * delta[2],
      ];
      piInParent = [
        porigin[0] + RoriginDelta[0],
        porigin[1] + RoriginDelta[1],
        porigin[2] + RoriginDelta[2],
      ];
    } else {
      RiInParent = Rorigin.map((row) => [...row]);
      piInParent = [...porigin];
    }
    const pidx = built.parentJoint[i];
    let Rworld: number[][];
    let pworld: number[];
    if (pidx < 0) {
      Rworld = RiInParent;
      pworld = piInParent;
    } else {
      const Rparent = out[pidx].R;
      const pparent = out[pidx].p;
      Rworld = mat3Mul(Rparent, RiInParent);
      const rotated = [
        Rparent[0][0] * piInParent[0] + Rparent[0][1] * piInParent[1] + Rparent[0][2] * piInParent[2],
        Rparent[1][0] * piInParent[0] + Rparent[1][1] * piInParent[1] + Rparent[1][2] * piInParent[2],
        Rparent[2][0] * piInParent[0] + Rparent[2][1] * piInParent[1] + Rparent[2][2] * piInParent[2],
      ];
      pworld = [rotated[0] + pparent[0], rotated[1] + pparent[1], rotated[2] + pparent[2]];
    }
    out.push({ R: Rworld, p: pworld });
  }
  return out;
}

// ── Geometric Jacobian ────────────────────────────────────────────────────

function isAncestor(built: BuiltArticulation, ancestor: number, descendant: number): boolean {
  let cur = descendant;
  while (cur >= 0) {
    if (cur === ancestor) return true;
    cur = built.parentJoint[cur];
  }
  return false;
}

export function geometricJacobian(
  built: BuiltArticulation,
  q: readonly number[],
  targetJointIdx: number,
  pointOffsetBody?: readonly number[],
): number[][] {
  const n = built.n;
  if (targetJointIdx < 0 || targetJointIdx >= n) {
    throw new Error(
      `geometricJacobian: targetJointIdx=${targetJointIdx} out of range [0, ${n})`,
    );
  }
  const poses = forwardKinematics(built, q);
  const { R: Rtarget, p: pTargetOrig } = poses[targetJointIdx];
  let pTarget: number[];
  if (pointOffsetBody !== undefined) {
    if (pointOffsetBody.length !== 3) {
      throw new Error(`pointOffsetBody must be 3-vec; got length ${pointOffsetBody.length}`);
    }
    const offsetWorld = [
      Rtarget[0][0] * pointOffsetBody[0] + Rtarget[0][1] * pointOffsetBody[1] + Rtarget[0][2] * pointOffsetBody[2],
      Rtarget[1][0] * pointOffsetBody[0] + Rtarget[1][1] * pointOffsetBody[1] + Rtarget[1][2] * pointOffsetBody[2],
      Rtarget[2][0] * pointOffsetBody[0] + Rtarget[2][1] * pointOffsetBody[1] + Rtarget[2][2] * pointOffsetBody[2],
    ];
    pTarget = [
      pTargetOrig[0] + offsetWorld[0],
      pTargetOrig[1] + offsetWorld[1],
      pTargetOrig[2] + offsetWorld[2],
    ];
  } else {
    pTarget = [...pTargetOrig];
  }
  const J: number[][] = Array.from({ length: 6 }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    if (!isAncestor(built, i, targetJointIdx)) continue;
    const { R: Rworld, p: pworld } = poses[i];
    const axisBody = built.jointAxis[i];
    const aWorld = [
      Rworld[0][0] * axisBody[0] + Rworld[0][1] * axisBody[1] + Rworld[0][2] * axisBody[2],
      Rworld[1][0] * axisBody[0] + Rworld[1][1] * axisBody[1] + Rworld[1][2] * axisBody[2],
      Rworld[2][0] * axisBody[0] + Rworld[2][1] * axisBody[1] + Rworld[2][2] * axisBody[2],
    ];
    const kind = built.jointKinds[i];
    if (kind === "revolute" || kind === "continuous") {
      const dp = [pTarget[0] - pworld[0], pTarget[1] - pworld[1], pTarget[2] - pworld[2]];
      const linear = [
        aWorld[1] * dp[2] - aWorld[2] * dp[1],
        aWorld[2] * dp[0] - aWorld[0] * dp[2],
        aWorld[0] * dp[1] - aWorld[1] * dp[0],
      ];
      for (let k = 0; k < 3; k++) {
        J[k][i] = aWorld[k];
        J[k + 3][i] = linear[k];
      }
    } else if (kind === "prismatic") {
      for (let k = 0; k < 3; k++) {
        J[k][i] = 0;
        J[k + 3][i] = aWorld[k];
      }
    }
  }
  return J;
}
