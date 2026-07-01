/**
 * nv-compat dynamics (Featherstone) physical invariants.
 *
 * Cross-checks the high-level articulated-dynamics functions against each
 * other on a 2-link revolute chain: CRBA mass-matrix symmetry/PD, kinetic
 * energy, the gravity bias, the RNEA↔CRBA linearity in qddot, and the
 * ABA↔RNEA forward/inverse round-trip. These algebraic identities are the
 * strongest correctness guard for the solver.
 *
 *     pnpm exec vitest run test/nv-compat-dynamics-invariants.test.ts
 *
 * ADR-2605261800 §D6/§D11.
 */

import { describe, it, expect } from "vitest";
import {
  abaForward,
  buildArticulation,
  coriolisGravityVector,
  crbaMassMatrix,
  forwardKinematics,
  kineticEnergy,
  makeZeroState,
  parseUrdf,
  rneaInverseDynamics,
} from "../src/dynamics/index.js";
import { buildSerialChainUrdf } from "../src/assets/index.js";

const built = buildArticulation(
  parseUrdf(
    buildSerialChainUrdf("arm", [
      { name: "j0", type: "revolute", axis: [0, 0, 1] },
      { name: "j1", type: "revolute", axis: [0, 0, 1] },
    ]),
  ),
);
const n = built.n;
const q = [0.3, -0.5];
const qdot = [0.7, 0.2];
const zeros = new Array<number>(n).fill(0);

const matVec = (M: number[][], v: readonly number[]): number[] => M.map((row) => row.reduce((s, m, j) => s + m * v[j], 0));

describe("CRBA mass matrix", () => {
  const M = crbaMassMatrix(built, q);
  it("is square (n×n) with positive diagonal", () => {
    expect(M).toHaveLength(n);
    for (let i = 0; i < n; i++) {
      expect(M[i]).toHaveLength(n);
      expect(M[i][i]).toBeGreaterThan(0);
    }
  });
  it("is symmetric", () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) expect(M[i][j]).toBeCloseTo(M[j][i], 9);
  });
});

describe("kinetic energy", () => {
  it("is zero at rest and equals ½·qdotᵀ·M·qdot otherwise", () => {
    expect(kineticEnergy(built, q, zeros)).toBeCloseTo(0, 12);
    const M = crbaMassMatrix(built, q);
    const expected = 0.5 * qdot.reduce((s, qi, i) => s + qi * matVec(M, qdot)[i], 0);
    expect(kineticEnergy(built, q, qdot)).toBeCloseTo(expected, 9);
    expect(kineticEnergy(built, q, qdot)).toBeGreaterThan(0);
  });
});

describe("gravity bias", () => {
  it("coriolisGravityVector at rest equals the RNEA bias torque", () => {
    const bias = coriolisGravityVector(built, q, zeros);
    const rnea = rneaInverseDynamics(built, q, zeros, zeros);
    bias.forEach((b, i) => expect(b).toBeCloseTo(rnea[i], 9));
  });
});

describe("RNEA ↔ CRBA linearity in qddot", () => {
  it("τ(q,0,qddot) − τ(q,0,0) ≈ M(q)·qddot", () => {
    const qddot = [0.4, -0.9];
    const withAccel = rneaInverseDynamics(built, q, zeros, qddot);
    const bias = rneaInverseDynamics(built, q, zeros, zeros);
    const delta = withAccel.map((t, i) => t - bias[i]);
    const Mqddot = matVec(crbaMassMatrix(built, q), qddot);
    delta.forEach((d, i) => expect(d).toBeCloseTo(Mqddot[i], 6));
  });
});

describe("ABA ↔ RNEA forward/inverse round-trip", () => {
  it("rnea(q, qdot, abaForward(q, qdot, τ)) ≈ τ", () => {
    const tau = [1.2, -0.6];
    const qddot = abaForward(built, q, qdot, tau);
    const recovered = rneaInverseDynamics(built, q, qdot, qddot);
    recovered.forEach((t, i) => expect(t).toBeCloseTo(tau[i], 5));
  });
});

describe("forward kinematics + zero state", () => {
  it("returns a finite world pose per joint", () => {
    const poses = forwardKinematics(built, q);
    expect(poses).toHaveLength(n);
    for (const p of poses) {
      expect(p.p).toHaveLength(3);
      p.p.forEach((c) => expect(Number.isFinite(c)).toBe(true));
    }
  });
  it("makeZeroState allocates n-length q/qdot/qddot", () => {
    const s = makeZeroState(n);
    expect(s.q).toHaveLength(n);
    expect(s.qdot).toHaveLength(n);
    expect(s.qddot).toHaveLength(n);
  });
});
