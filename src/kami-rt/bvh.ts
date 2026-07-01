// kami-rt — clean-room ray-tracing core (CPU reference path).
//
// This is the canonical KAMI implementation that `nv-compat/optix.ts`
// routes to. NVIDIA OptiX® is the NVIDIA Corporation GPU ray-tracing
// API; kami-rt reproduces the *behaviour* of an OptiX launch (raygen →
// BVH traversal → closest-hit / miss → framebuffer) on WebGPU + WGSL,
// with this module providing the byte-compatible CPU fallback that runs
// everywhere (Node + browsers without a GPU).
//
// Nothing here is derived from OptiX source, headers, or the binary SDK
// — it is a from-spec re-implementation of textbook ray tracing (BVH
// median split, Möller–Trumbore, pinhole camera). The OptiX *names*
// appear only in optix.ts as API-compat identifiers (Google v. Oracle,
// 593 U.S. ___ (2021)).
//
// ADR-2605261800 §D6 nv-compat namespace localization; D10.4 kami-rt
// fallback path (WGSL software BVH, LBVH-class), R1.2 surface.

// ── small vector helpers (plain number triples; no allocation classes) ────

export type Vec3 = readonly [number, number, number];

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

// ── scene representation ──────────────────────────────────────────────────

/** A triangle soup. `verts` is a flat Float32Array, 9 floats per triangle
 *  (v0.xyz, v1.xyz, v2.xyz). This is the same buffer uploaded verbatim to
 *  the WGSL `tris` storage binding. */
export interface TriangleSoup {
  /** 9 floats per triangle. */
  verts: Float32Array;
  /** Triangle count (`verts.length / 9`). */
  count: number;
}

export function triangleSoup(triangles: readonly Vec3[][]): TriangleSoup {
  const verts = new Float32Array(triangles.length * 9);
  triangles.forEach((tri, t) => {
    for (let v = 0; v < 3; v++) {
      verts[t * 9 + v * 3 + 0] = tri[v][0];
      verts[t * 9 + v * 3 + 1] = tri[v][1];
      verts[t * 9 + v * 3 + 2] = tri[v][2];
    }
  });
  return { verts, count: triangles.length };
}

// ── BVH ───────────────────────────────────────────────────────────────────

// Flattened node layout, 8 floats/node, identical on CPU and in WGSL:
//   [0..2] aabbMin.xyz
//   [3]    leftFirst  — internal: index of left child node;
//                        leaf: first triangle index in `triIndex`
//   [4..6] aabbMax.xyz
//   [7]    count      — 0 ⇒ internal node; >0 ⇒ leaf with `count` tris
export const NODE_STRIDE = 8;

export interface Bvh {
  /** Flattened nodes, NODE_STRIDE floats each. */
  nodes: Float32Array;
  /** Triangle permutation referenced by leaf `leftFirst..leftFirst+count`. */
  triIndex: Uint32Array;
  nodeCount: number;
}

interface BuildNode {
  min: Vec3;
  max: Vec3;
  leftFirst: number;
  count: number;
}

function triBounds(soup: TriangleSoup, tri: number): { min: Vec3; max: Vec3; centroid: Vec3 } {
  const b = tri * 9;
  const v = soup.verts;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < 3; k++) {
    for (let c = 0; c < 3; c++) {
      const x = v[b + k * 3 + c];
      if (x < min[c]) min[c] = x;
      if (x > max[c]) max[c] = x;
    }
  }
  const centroid: Vec3 = [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5,
  ];
  return { min, max, centroid };
}

/** Build a binary BVH by recursive median split on the longest axis of the
 *  centroid bounds. Deterministic — same input ⇒ same tree, so the CPU and
 *  GPU traversals visit nodes identically. */
export function buildBvh(soup: TriangleSoup): Bvh {
  const n = soup.count;
  const triIndex = new Uint32Array(n);
  for (let i = 0; i < n; i++) triIndex[i] = i;

  // Empty soup → empty BVH. `traceClosest` guards on `nodeCount === 0`, so
  // an empty scene renders as all-background instead of crashing. (Without
  // this, the `n === 0` fallback node would make `computeEnd` recurse on
  // itself indefinitely.)
  if (n === 0) return { nodes: new Float32Array(0), triIndex, nodeCount: 0 };

  // Precompute per-triangle bounds + centroids.
  const tmin: Vec3[] = new Array(n);
  const tmax: Vec3[] = new Array(n);
  const cen: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = triBounds(soup, i);
    tmin[i] = b.min;
    tmax[i] = b.max;
    cen[i] = b.centroid;
  }

  const out: BuildNode[] = [];

  // Recursion over [start, end) of triIndex. Returns the emitted node index.
  const build = (start: number, end: number): number => {
    let mn: [number, number, number] = [Infinity, Infinity, Infinity];
    let mx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    let cmn: [number, number, number] = [Infinity, Infinity, Infinity];
    let cmx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const t = triIndex[i];
      for (let c = 0; c < 3; c++) {
        if (tmin[t][c] < mn[c]) mn[c] = tmin[t][c];
        if (tmax[t][c] > mx[c]) mx[c] = tmax[t][c];
        if (cen[t][c] < cmn[c]) cmn[c] = cen[t][c];
        if (cen[t][c] > cmx[c]) cmx[c] = cen[t][c];
      }
    }

    const nodeIdx = out.length;
    out.push({ min: mn, max: mx, leftFirst: 0, count: 0 });

    const span = end - start;
    // Leaf cutoff: ≤2 tris, or degenerate centroid spread.
    const ext: Vec3 = [cmx[0] - cmn[0], cmx[1] - cmn[1], cmx[2] - cmn[2]];
    const axis = ext[0] >= ext[1] && ext[0] >= ext[2] ? 0 : ext[1] >= ext[2] ? 1 : 2;
    if (span <= 2 || ext[axis] < 1e-9) {
      out[nodeIdx].leftFirst = start;
      out[nodeIdx].count = span;
      return nodeIdx;
    }

    // Median split on `axis`.
    const mid = start + (span >> 1);
    const slice = Array.from(triIndex.subarray(start, end));
    slice.sort((a, b) => cen[a][axis] - cen[b][axis]);
    for (let i = start; i < end; i++) triIndex[i] = slice[i - start];

    const left = build(start, mid);
    build(mid, end); // right is always left+subtree; we store left index only
    out[nodeIdx].leftFirst = left;
    out[nodeIdx].count = 0;
    return nodeIdx;
  };

  if (n > 0) build(0, n);
  else out.push({ min: [0, 0, 0], max: [0, 0, 0], leftFirst: 0, count: 0 });

  // Flatten to the shared node encoding. Nodes are emitted in DFS order, so
  // build(left) occupies a contiguous slot range followed immediately by
  // build(right). The right child index is therefore `lastNodeOf(left) + 1`.
  // We compute that once per node (memoized) and store it explicitly so the
  // WGSL traversal needs no recomputation.
  //
  // Encoding per node (count slot distinguishes the two kinds):
  //   leaf     → count = +numTris ,  leftFirst = firstTriangleIndex
  //   internal → count = -rightIdx (always ≤ -1),  leftFirst = leftChildIdx
  const subtreeEnd = new Int32Array(out.length).fill(-1);
  const computeEnd = (idx: number): number => {
    if (subtreeEnd[idx] >= 0) return subtreeEnd[idx];
    if (out[idx].count > 0) return (subtreeEnd[idx] = idx);
    const leftEnd = computeEnd(out[idx].leftFirst);
    return (subtreeEnd[idx] = computeEnd(leftEnd + 1));
  };
  if (out.length > 0 && out[0].count === 0) computeEnd(0);

  const nodes = new Float32Array(out.length * NODE_STRIDE);
  for (let i = 0; i < out.length; i++) {
    const o = out[i];
    const base = i * NODE_STRIDE;
    nodes[base + 0] = o.min[0];
    nodes[base + 1] = o.min[1];
    nodes[base + 2] = o.min[2];
    nodes[base + 4] = o.max[0];
    nodes[base + 5] = o.max[1];
    nodes[base + 6] = o.max[2];
    if (o.count > 0) {
      nodes[base + 3] = o.leftFirst; // first triangle index
      nodes[base + 7] = o.count; // +count ⇒ leaf
    } else {
      nodes[base + 3] = o.leftFirst; // left child index
      nodes[base + 7] = -(computeEnd(o.leftFirst) + 1); // -rightIdx ⇒ internal
    }
  }

  return { nodes, triIndex, nodeCount: out.length };
}

// ── pinhole camera ──────────────────────────────────────────────────────

export interface Camera {
  origin: Vec3;
  lowerLeft: Vec3;
  horizontal: Vec3;
  vertical: Vec3;
}

/** Build a pinhole camera looking from `eye` to `target`, vertical FOV in
 *  degrees, image aspect `width/height`. */
export function lookAt(
  eye: Vec3,
  target: Vec3,
  up: Vec3,
  vfovDeg: number,
  aspect: number,
): Camera {
  const theta = (vfovDeg * Math.PI) / 180;
  const h = Math.tan(theta / 2);
  const viewH = 2 * h;
  const viewW = aspect * viewH;
  const w = norm(sub(eye, target)); // points back toward eye
  const u = norm(cross(up, w));
  const v = cross(w, u);
  const horizontal: Vec3 = [u[0] * viewW, u[1] * viewW, u[2] * viewW];
  const vertical: Vec3 = [v[0] * viewH, v[1] * viewH, v[2] * viewH];
  const lowerLeft: Vec3 = [
    eye[0] - horizontal[0] / 2 - vertical[0] / 2 - w[0],
    eye[1] - horizontal[1] / 2 - vertical[1] / 2 - w[1],
    eye[2] - horizontal[2] / 2 - vertical[2] / 2 - w[2],
  ];
  return { origin: eye, lowerLeft, horizontal, vertical };
}

// ── ray / triangle intersection (Möller–Trumbore) ─────────────────────────

export interface Hit {
  t: number;
  tri: number;
  /** barycentric (u, v); w = 1 - u - v. */
  u: number;
  v: number;
}

const EPS = 1e-7;

function intersectTri(
  soup: TriangleSoup,
  tri: number,
  ro: Vec3,
  rd: Vec3,
  tMax: number,
): Hit | null {
  const b = tri * 9;
  const v = soup.verts;
  const v0: Vec3 = [v[b], v[b + 1], v[b + 2]];
  const v1: Vec3 = [v[b + 3], v[b + 4], v[b + 5]];
  const v2: Vec3 = [v[b + 6], v[b + 7], v[b + 8]];
  const e1 = sub(v1, v0);
  const e2 = sub(v2, v0);
  const p = cross(rd, e2);
  const det = dot(e1, p);
  if (det > -EPS && det < EPS) return null; // parallel
  const inv = 1 / det;
  const tvec = sub(ro, v0);
  const u = dot(tvec, p) * inv;
  if (u < 0 || u > 1) return null;
  const q = cross(tvec, e1);
  const vv = dot(rd, q) * inv;
  if (vv < 0 || u + vv > 1) return null;
  const t = dot(e2, q) * inv;
  if (t < EPS || t > tMax) return null;
  return { t, tri, u, v: vv };
}

function slabHit(
  node: Float32Array,
  base: number,
  ro: Vec3,
  invD: Vec3,
  tMax: number,
): boolean {
  let tmin = 0;
  let tmax = tMax;
  for (let c = 0; c < 3; c++) {
    const lo = (node[base + c] - ro[c]) * invD[c];
    const hi = (node[base + 4 + c] - ro[c]) * invD[c];
    const t0 = Math.min(lo, hi);
    const t1 = Math.max(lo, hi);
    tmin = Math.max(tmin, t0);
    tmax = Math.min(tmax, t1);
    if (tmax < tmin) return false;
  }
  return true;
}

/** Closest-hit traversal over the BVH. JS reference; the WGSL kernel mirrors
 *  this exactly (same node encoding, same Möller–Trumbore, same stack). */
export function traceClosest(
  soup: TriangleSoup,
  bvh: Bvh,
  ro: Vec3,
  rd: Vec3,
  tMax = Infinity,
): Hit | null {
  if (bvh.nodeCount === 0) return null;
  const invD: Vec3 = [1 / rd[0], 1 / rd[1], 1 / rd[2]];
  let best: Hit | null = null;
  let bestT = tMax;

  const stack = new Int32Array(64);
  let sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const ni = stack[--sp];
    const base = ni * NODE_STRIDE;
    if (!slabHit(bvh.nodes, base, ro, invD, bestT)) continue;
    const count = bvh.nodes[base + 7];
    if (count > 0) {
      // leaf
      const first = bvh.nodes[base + 3];
      for (let i = 0; i < count; i++) {
        const tri = bvh.triIndex[first + i];
        const h = intersectTri(soup, tri, ro, rd, bestT);
        if (h && h.t < bestT) {
          best = h;
          bestT = h.t;
        }
      }
    } else {
      const left = bvh.nodes[base + 3];
      const right = -bvh.nodes[base + 7];
      stack[sp++] = left;
      stack[sp++] = right;
    }
  }
  return best;
}

// ── geometric normal of a hit triangle ────────────────────────────────────

export function triNormal(soup: TriangleSoup, tri: number): Vec3 {
  const b = tri * 9;
  const v = soup.verts;
  const v0: Vec3 = [v[b], v[b + 1], v[b + 2]];
  const v1: Vec3 = [v[b + 3], v[b + 4], v[b + 5]];
  const v2: Vec3 = [v[b + 6], v[b + 7], v[b + 8]];
  return norm(cross(sub(v1, v0), sub(v2, v0)));
}

// ── JS software image trace (closest-hit + Lambert shade) ─────────────────

export interface ShadeParams {
  /** Normalized directional light (points *from* light toward scene). */
  lightDir: Vec3;
  albedo: Vec3;
  bgTop: Vec3;
  bgBottom: Vec3;
  ambient: number;
}

export const DEFAULT_SHADE: ShadeParams = {
  lightDir: norm([-0.5, -1, -0.3]),
  albedo: [0.82, 0.82, 0.88],
  bgTop: [0.5, 0.7, 1.0],
  bgBottom: [1.0, 1.0, 1.0],
  ambient: 0.2,
};

/** Render `width × height` RGBA float framebuffer on the CPU. Identical
 *  output to the WGSL path (within f32 rounding). Pixel (0,0) is the
 *  lower-left of the image, matching the pinhole `lowerLeft` basis. */
export function traceImageSync(
  soup: TriangleSoup,
  bvh: Bvh,
  cam: Camera,
  width: number,
  height: number,
  shade: ShadeParams = DEFAULT_SHADE,
): Float32Array {
  const fb = new Float32Array(width * height * 4);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const s = (px + 0.5) / width;
      const tparam = (py + 0.5) / height;
      const dir = norm([
        cam.lowerLeft[0] + s * cam.horizontal[0] + tparam * cam.vertical[0] - cam.origin[0],
        cam.lowerLeft[1] + s * cam.horizontal[1] + tparam * cam.vertical[1] - cam.origin[1],
        cam.lowerLeft[2] + s * cam.horizontal[2] + tparam * cam.vertical[2] - cam.origin[2],
      ]);
      const hit = traceClosest(soup, bvh, cam.origin, dir);
      let r: number, g: number, bch: number;
      if (hit) {
        let nrm = triNormal(soup, hit.tri);
        if (dot(nrm, dir) > 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];
        const diff = Math.max(0, dot(nrm, [-shade.lightDir[0], -shade.lightDir[1], -shade.lightDir[2]]));
        const lit = shade.ambient + (1 - shade.ambient) * diff;
        r = shade.albedo[0] * lit;
        g = shade.albedo[1] * lit;
        bch = shade.albedo[2] * lit;
      } else {
        const a = tparam;
        r = (1 - a) * shade.bgBottom[0] + a * shade.bgTop[0];
        g = (1 - a) * shade.bgBottom[1] + a * shade.bgTop[1];
        bch = (1 - a) * shade.bgBottom[2] + a * shade.bgTop[2];
      }
      const o = (py * width + px) * 4;
      fb[o] = r;
      fb[o + 1] = g;
      fb[o + 2] = bch;
      fb[o + 3] = 1;
    }
  }
  return fb;
}
