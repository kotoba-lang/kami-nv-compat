// TypeScript port of kotodama.nv_compat.isaaclab.utils.warp
//
// NVIDIA Warp® kernel-API parity stubs — single-threaded JS execution.
//
// Mirror of `warp` (Warp 1.x) — the SIMT kernel framework Isaac Lab
// uses heavily for parallel MDP event / reward / observation functions.
// Real Warp JIT-compiles `@wp.kernel`-decorated Python functions to
// CUDA / Vulkan / WebGPU and dispatches them as N-way parallel grids.
//
// This TS port provides the SAME PUBLIC API SURFACE so Isaac Lab task
// definitions that reference `wp.kernel` / `wp.launch` / `wp.array` /
// `wp.vec3` / `wp.quat` / `wp.transform` / atomic ops can be authored
// in TypeScript and run in browsers + Node. Execution is sequential
// JavaScript (`for (let i = 0; i < dim; i++) kernel(...inputs)`) —
// performance bad, semantics correct. A future iter swaps in a WebGPU
// compute-shader backend behind the same `launch` entry point.
//
// Trademark: "NVIDIA®" and "Warp®" are trademarks of NVIDIA Corporation.
// Per ADR-2605261800 §D6 this is API namespace localization for
// interoperability purposes only (Google v. Oracle 2021 — API fair use).
// The canonical religious-corp equivalent will live as kami-warp.
//
// API surface covered (mirrors Python iter 66's 58 names):
//
//   Decorators / launch:
//     kernel(fn) — mark a function as a Warp kernel (no-op shim)
//     func(fn) — mark a helper function (no-op shim)
//     launch(opts) — sequential N-way "grid" dispatch
//     tid() — thread index inside a kernel body
//     init() — module init (no-op)
//     config — config namespace (mode / verifyCuda — no-op)
//
//   Scalar dtype sentinels (type-marker objects):
//     float32, float64, int32, int64, uint32, uint8, boolDtype
//
//   Container dtypes:
//     Array<T>, zeros, empty, fromTypedArray, indexOf
//
//   Linear-algebra value types (small classes):
//     Vec3, Vec4, Quat, Mat33, Transform
//
//   Math (scalar): sin, cos, tan, atan2, sqrt, abs, min, max, clamp,
//                  floor, ceil, exp, log, pi
//   Math (vec / quat / transform):
//     length, lengthSq, normalize, dot, cross
//     quatIdentity, quatFromAxisAngle, quatInverse, quatMul,
//     quatRotate, quatRotateInv
//     transformIdentity, transformPoint, transformVector,
//     transformGetTranslation, transformGetRotation, transformMultiply
//
//   Atomic ops (single-threaded under stub — semantically correct):
//     atomicAdd, atomicSub, atomicMax, atomicMin

// ── module constants ──────────────────────────────────────────────────────

export const pi = Math.PI;

// ── thread-local kernel state ─────────────────────────────────────────────

let _currentTid: number | null = null;

/** Thread index inside an in-flight kernel launch.
 *  Throws when called outside `launch`. */
export function tid(): number {
  if (_currentTid === null) {
    throw new Error(
      "wp.tid() called outside of an active wp.launch — kernel functions " +
        "must run via wp.launch({ kernel, dim, inputs })",
    );
  }
  return _currentTid;
}

// ── dtype sentinels ──────────────────────────────────────────────────────

export interface DtypeMarker {
  readonly name: string;
  (value?: unknown): number | boolean;
}

function makeDtype(name: string): DtypeMarker {
  const fn = ((value: unknown = 0): number | boolean => {
    if (name === "float32" || name === "float64") return Number(value);
    if (name === "int32" || name === "int64" || name === "uint32" || name === "uint8") {
      const n = Number(value);
      return Math.trunc(n);
    }
    if (name === "bool") return Boolean(value);
    return Number(value);
  }) as DtypeMarker;
  Object.defineProperty(fn, "name", { value: `wp.${name}` });
  return fn;
}

export const float32 = makeDtype("float32");
export const float64 = makeDtype("float64");
export const int32 = makeDtype("int32");
export const int64 = makeDtype("int64");
export const uint32 = makeDtype("uint32");
export const uint8 = makeDtype("uint8");
/** Bool dtype — `boolDtype` to avoid shadowing JS `Boolean` constructor. */
export const boolDtype = makeDtype("bool");

// ── config namespace ─────────────────────────────────────────────────────

export const config = {
  mode: "release" as "release" | "debug",
  verifyCuda: false,
  verifyFp: false,
  printLaunches: false,
  cacheKernels: true,
};

export function init(): void {
  /* no-op stub */
}

// ── Linear-algebra value types ───────────────────────────────────────────

export class Vec3 {
  constructor(public x: number = 0, public y: number = 0, public z: number = 0) {}
  add(o: Vec3 | readonly number[]): Vec3 {
    return new Vec3(this.x + asNum(o, 0), this.y + asNum(o, 1), this.z + asNum(o, 2));
  }
  sub(o: Vec3 | readonly number[]): Vec3 {
    return new Vec3(this.x - asNum(o, 0), this.y - asNum(o, 1), this.z - asNum(o, 2));
  }
  mul(o: number | Vec3 | readonly number[]): Vec3 {
    if (typeof o === "number") return new Vec3(this.x * o, this.y * o, this.z * o);
    return new Vec3(this.x * asNum(o, 0), this.y * asNum(o, 1), this.z * asNum(o, 2));
  }
  neg(): Vec3 {
    return new Vec3(-this.x, -this.y, -this.z);
  }
  get length(): number {
    return 3;
  }
  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }
}

export class Vec4 {
  constructor(
    public x: number = 0,
    public y: number = 0,
    public z: number = 0,
    public w: number = 0,
  ) {}
  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }
}

export class Quat {
  constructor(
    public x: number = 0,
    public y: number = 0,
    public z: number = 0,
    public w: number = 1,
  ) {}
  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }
}

/** 3×3 row-major matrix. Construct from 9 scalars / single 9-flat / 3×3 nested
 *  / zero-arg (zeros). */
export class Mat33 {
  rows: number[][];
  constructor(...args: number[] | [number[]] | [number[][]]) {
    if (args.length === 0) {
      this.rows = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    } else if (args.length === 1 && Array.isArray(args[0]) && args[0].length === 9) {
      const f = args[0] as number[];
      this.rows = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
    } else if (args.length === 1 && Array.isArray(args[0])) {
      const nested = args[0] as number[][];
      this.rows = nested.map((r) => [...r]);
    } else if (args.length === 9) {
      const f = args as unknown as number[];
      this.rows = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
    } else {
      throw new Error(`mat33(): expected 0 / 1 / 9 args; got ${args.length}`);
    }
  }
  get(i: number, j?: number): number | number[] {
    if (j === undefined) return this.rows[i];
    return this.rows[i][j];
  }
  set(i: number, j: number, v: number): void {
    this.rows[i][j] = v;
  }
}

export class Transform {
  constructor(
    public translation: Vec3 = new Vec3(0, 0, 0),
    public rotation: Quat = new Quat(0, 0, 0, 1),
  ) {}
}

function asNum(o: Vec3 | readonly number[], i: number): number {
  if (o instanceof Vec3) {
    return [o.x, o.y, o.z][i] ?? 0;
  }
  return o[i] ?? 0;
}

function _coerceVec(x: unknown, dim: 3 | 4): number[] {
  let c: number[];
  if (x instanceof Vec3) c = [x.x, x.y, x.z];
  else if (x instanceof Vec4 || x instanceof Quat) c = [x.x, x.y, x.z, x.w];
  else if (Array.isArray(x)) c = x.map(Number);
  else if (typeof x === "number") c = new Array(dim).fill(x);
  else c = [];
  while (c.length < dim) c.push(0);
  return c.slice(0, dim);
}

// ── Array container ──────────────────────────────────────────────────────

export class WpArray<T = number> {
  data: T[];
  dtype: DtypeMarker;
  shape: readonly number[];
  device: string;

  constructor(opts: {
    data?: T[] | readonly T[];
    shape?: number | readonly number[];
    dtype?: DtypeMarker;
    device?: string;
  } = {}) {
    this.dtype = opts.dtype ?? (float32 as DtypeMarker);
    this.device = opts.device ?? "cpu";
    if (opts.data !== undefined) {
      this.data = [...opts.data];
      this.shape = [this.data.length];
    } else if (opts.shape !== undefined) {
      const n = typeof opts.shape === "number" ? opts.shape : opts.shape[0];
      this.data = new Array<T>(n).fill(0 as unknown as T);
      this.shape = [n];
    } else {
      this.data = [];
      this.shape = [0];
    }
  }

  get(i: number): T { return this.data[i]; }
  set(i: number, v: T): void { this.data[i] = v; }
  get length(): number { return this.data.length; }
  /** In-place fill (matches `wp.array.fill_`). */
  fill(value: T): WpArray<T> {
    for (let i = 0; i < this.data.length; i++) this.data[i] = value;
    return this;
  }
  /** Replace data in-place. */
  assign(src: readonly T[]): WpArray<T> {
    if (src.length !== this.data.length) {
      throw new Error(`Array.assign: length mismatch (have ${this.data.length}, got ${src.length})`);
    }
    this.data = [...src];
    return this;
  }
  toArray(): T[] { return [...this.data]; }
}

export function zeros<T = number>(
  shape: number | readonly number[],
  opts: { dtype?: DtypeMarker; device?: string } = {},
): WpArray<T> {
  return new WpArray<T>({ shape, dtype: opts.dtype, device: opts.device });
}

export function empty<T = number>(
  shape: number | readonly number[],
  opts: { dtype?: DtypeMarker; device?: string } = {},
): WpArray<T> {
  return new WpArray<T>({ shape, dtype: opts.dtype, device: opts.device });
}

export function fromTypedArray<T = number>(
  data: readonly T[],
  opts: { dtype?: DtypeMarker; device?: string } = {},
): WpArray<T> {
  return new WpArray<T>({ data, dtype: opts.dtype, device: opts.device });
}

export function indexOf<T>(arr: WpArray<T>, i: number): T {
  return arr.get(i);
}

// ── Kernel + launch ──────────────────────────────────────────────────────

// A kernel registry holds functions of heterogeneous, kernel-specific
// signatures (e.g. (arr: WpArray<number>, damping: number) => void). Under
// strictFunctionTypes those specific signatures are not assignable to a
// `(...args: unknown[])` parameter (contravariance), so the registry element
// type uses `any[]` — the conventional shape for a varied-signature callback
// table. Callers go through `launch(kernel, ...args)` which is checked at the
// kernel's own definition site.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type KernelFn = (...args: any[]) => void;

/** Wrapper around a kernel function. Real Warp JIT-compiles to GPU
 *  bytecode; the stub just stores the function. Decorator-style use
 *  in TS: `const myKernel = wpKernel((args) => { ... })`. */
export interface Kernel {
  readonly name: string;
  fn: KernelFn;
  (...args: unknown[]): void;
}

export function kernel(fn: KernelFn): Kernel {
  const wrapper = ((...args: unknown[]) => fn(...args)) as Kernel;
  Object.defineProperty(wrapper, "name", { value: fn.name || "kernel" });
  (wrapper as { fn: KernelFn }).fn = fn;
  return wrapper;
}

/** `@wp.func` — annotates a helper fn callable from inside a kernel.
 *  No-op under the stub. */
export function func<T extends (...args: unknown[]) => unknown>(fn: T): T {
  return fn;
}

/** Execute `kernel` sequentially across the launch grid.
 *
 * Semantics match real Warp: the kernel runs `prod(dim)` times with
 * `tid()` returning the linearised thread index 0..N-1. Real Warp runs
 * these in parallel on GPU; this stub runs them sequentially in JS.
 */
export function launch(opts: {
  kernel: Kernel;
  dim: number | readonly number[];
  inputs?: unknown[];
  outputs?: unknown[];
  device?: string;
  stream?: unknown;
}): void {
  const dim = opts.dim;
  let total: number;
  if (typeof dim === "number") {
    total = dim;
  } else {
    total = 1;
    for (const d of dim) total *= Math.trunc(d);
  }
  if (total < 0) throw new Error(`launch: dim must be ≥ 0; got ${JSON.stringify(dim)}`);
  const args: unknown[] = [...(opts.inputs ?? []), ...(opts.outputs ?? [])];
  const prevTid = _currentTid;
  try {
    for (let i = 0; i < total; i++) {
      _currentTid = i;
      opts.kernel(...args);
    }
  } finally {
    _currentTid = prevTid;
  }
}

// ── Scalar math (parity with Math.* so kernels can use wp.* uniformly) ──

export const sin = Math.sin;
export const cos = Math.cos;
export const tan = Math.tan;
export const atan2 = Math.atan2;
export const sqrt = Math.sqrt;
export const exp = Math.exp;
export const log = Math.log;
export const floor = Math.floor;
export const ceil = Math.ceil;

export function abs(x: number): number {
  return Math.abs(x);
}

export function min(a: number | Vec3, b: number | Vec3 | readonly number[]): number | Vec3 {
  if (a instanceof Vec3) {
    const bc = _coerceVec(b, 3);
    return new Vec3(Math.min(a.x, bc[0]), Math.min(a.y, bc[1]), Math.min(a.z, bc[2]));
  }
  return Math.min(a as number, b as number);
}

export function max(a: number | Vec3, b: number | Vec3 | readonly number[]): number | Vec3 {
  if (a instanceof Vec3) {
    const bc = _coerceVec(b, 3);
    return new Vec3(Math.max(a.x, bc[0]), Math.max(a.y, bc[1]), Math.max(a.z, bc[2]));
  }
  return Math.max(a as number, b as number);
}

export function clamp(x: number, low: number, high: number): number {
  if (x < low) return low;
  if (x > high) return high;
  return x;
}

// ── Vector math ──────────────────────────────────────────────────────────

export function length(v: Vec3 | Vec4 | Quat | readonly number[]): number {
  const c =
    v instanceof Vec3
      ? [v.x, v.y, v.z]
      : v instanceof Vec4 || v instanceof Quat
        ? [v.x, v.y, v.z, v.w]
        : [...(v as readonly number[])];
  let s = 0;
  for (const x of c) s += x * x;
  return Math.sqrt(s);
}

export function lengthSq(v: Vec3 | Vec4 | Quat | readonly number[]): number {
  const l = length(v);
  return l * l;
}

export function normalize(v: Vec3): Vec3;
export function normalize(v: Vec4): Vec4;
export function normalize(v: Quat): Quat;
export function normalize(v: Vec3 | Vec4 | Quat): Vec3 | Vec4 | Quat {
  const n = length(v);
  if (n < 1e-12) {
    if (v instanceof Vec3) return new Vec3(0, 0, 0);
    if (v instanceof Vec4) return new Vec4(0, 0, 0, 0);
    return new Quat(0, 0, 0, 1);
  }
  if (v instanceof Vec3) return new Vec3(v.x / n, v.y / n, v.z / n);
  if (v instanceof Vec4) return new Vec4(v.x / n, v.y / n, v.z / n, v.w / n);
  return new Quat(v.x / n, v.y / n, v.z / n, v.w / n);
}

export function dot(a: Vec3 | readonly number[], b: Vec3 | readonly number[]): number {
  const ac = _coerceVec(a, 3);
  const bc = _coerceVec(b, 3);
  return ac[0] * bc[0] + ac[1] * bc[1] + ac[2] * bc[2];
}

export function cross(a: Vec3 | readonly number[], b: Vec3 | readonly number[]): Vec3 {
  const ac = _coerceVec(a, 3);
  const bc = _coerceVec(b, 3);
  return new Vec3(
    ac[1] * bc[2] - ac[2] * bc[1],
    ac[2] * bc[0] - ac[0] * bc[2],
    ac[0] * bc[1] - ac[1] * bc[0],
  );
}

// ── Quaternion math (Hamilton, [x,y,z,w]) ────────────────────────────────

export function quatIdentity(): Quat {
  return new Quat(0, 0, 0, 1);
}

export function quatFromAxisAngle(axis: Vec3 | readonly number[], angle: number): Quat {
  const ax = _coerceVec(axis, 3);
  const n = Math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2);
  if (n < 1e-12) return new Quat(0, 0, 0, 1);
  const u = [ax[0] / n, ax[1] / n, ax[2] / n];
  const h = angle * 0.5;
  const s = Math.sin(h);
  return new Quat(u[0] * s, u[1] * s, u[2] * s, Math.cos(h));
}

export function quatInverse(q: Quat): Quat {
  return new Quat(-q.x, -q.y, -q.z, q.w);
}

export function quatMul(a: Quat, b: Quat): Quat {
  return new Quat(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  );
}

export function quatRotate(q: Quat, v: Vec3 | readonly number[]): Vec3 {
  const vc = _coerceVec(v, 3);
  const tx = q.y * vc[2] - q.z * vc[1] + q.w * vc[0];
  const ty = q.z * vc[0] - q.x * vc[2] + q.w * vc[1];
  const tz = q.x * vc[1] - q.y * vc[0] + q.w * vc[2];
  return new Vec3(
    vc[0] + 2.0 * (q.y * tz - q.z * ty),
    vc[1] + 2.0 * (q.z * tx - q.x * tz),
    vc[2] + 2.0 * (q.x * ty - q.y * tx),
  );
}

export function quatRotateInv(q: Quat, v: Vec3 | readonly number[]): Vec3 {
  return quatRotate(quatInverse(q), v);
}

// ── Transform math ───────────────────────────────────────────────────────

export function transformIdentity(): Transform {
  return new Transform(new Vec3(0, 0, 0), new Quat(0, 0, 0, 1));
}

export function transformPoint(t: Transform, p: Vec3 | readonly number[]): Vec3 {
  const rotated = quatRotate(t.rotation, p);
  return rotated.add(t.translation);
}

export function transformVector(t: Transform, v: Vec3 | readonly number[]): Vec3 {
  return quatRotate(t.rotation, v);
}

export function transformGetTranslation(t: Transform): Vec3 {
  return t.translation;
}

export function transformGetRotation(t: Transform): Quat {
  return t.rotation;
}

export function transformMultiply(a: Transform, b: Transform): Transform {
  const rot = quatMul(a.rotation, b.rotation);
  const trans = quatRotate(a.rotation, b.translation).add(a.translation);
  return new Transform(trans, rot);
}

// ── Atomic ops (single-threaded under stub — semantically correct) ──────

export function atomicAdd<T>(arr: WpArray<T>, i: number, v: T): T {
  const old = arr.get(i);
  arr.set(i, ((old as unknown as number) + (v as unknown as number)) as unknown as T);
  return old;
}

export function atomicSub<T>(arr: WpArray<T>, i: number, v: T): T {
  const old = arr.get(i);
  arr.set(i, ((old as unknown as number) - (v as unknown as number)) as unknown as T);
  return old;
}

export function atomicMax<T>(arr: WpArray<T>, i: number, v: T): T {
  const old = arr.get(i);
  const next = (old as unknown as number) >= (v as unknown as number) ? old : v;
  arr.set(i, next);
  return old;
}

export function atomicMin<T>(arr: WpArray<T>, i: number, v: T): T {
  const old = arr.get(i);
  const next = (old as unknown as number) <= (v as unknown as number) ? old : v;
  arr.set(i, next);
  return old;
}
