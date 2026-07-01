// kami-rt WGSL ray tracer.
//
// This is the GPU reproduction of an OptiX launch: one thread per pixel,
// pinhole raygen, stack-based BVH traversal, Möller–Trumbore closest-hit,
// Lambert shade / sky miss. The node encoding, intersection math and stack
// semantics mirror `bvh.ts` line-for-line so the GPU framebuffer matches the
// CPU reference within f32 rounding.
//
// Layout contract (must match bvh.ts):
//   tris  : array<f32>, 9 floats/triangle (v0.xyz, v1.xyz, v2.xyz)
//   nodes : array<f32>, 8 floats/node
//             [0..2] aabbMin   [3] leftFirst
//             [4..6] aabbMax    [7] count   (+leaf / -rightIdx internal)
//   triIdx: array<u32>, leaf triangle permutation
//   fb    : array<f32>, 4 floats/pixel RGBA, row-major, (0,0) lower-left
//
// ADR-2605261800 §D6 / D10.4 kami-rt.

export const RAYTRACE_WGSL = /* wgsl */ `
struct Params {
  origin     : vec4<f32>,  // xyz origin,      w = width
  lowerLeft  : vec4<f32>,  // xyz,             w = height
  horizontal : vec4<f32>,  // xyz,             w = numTris (unused)
  vertical   : vec4<f32>,  // xyz,             w unused
  light      : vec4<f32>,  // xyz lightDir,    w = ambient
  albedo     : vec4<f32>,  // xyz albedo
  bgTop      : vec4<f32>,  // xyz
  bgBottom   : vec4<f32>,  // xyz
};

@group(0) @binding(0) var<storage, read>        tris   : array<f32>;
@group(0) @binding(1) var<storage, read>        nodes  : array<f32>;
@group(0) @binding(2) var<storage, read>        triIdx : array<u32>;
@group(0) @binding(3) var<storage, read_write>  fb     : array<f32>;
@group(0) @binding(4) var<uniform>              P      : Params;

const EPS : f32 = 1e-7;
const NODE_STRIDE : u32 = 8u;

struct Hit { t: f32, tri: i32, u: f32, v: f32 };

fn triVert(tri: u32, k: u32) -> vec3<f32> {
  let b = tri * 9u + k * 3u;
  return vec3<f32>(tris[b], tris[b + 1u], tris[b + 2u]);
}

fn intersectTri(tri: u32, ro: vec3<f32>, rd: vec3<f32>, tMax: f32) -> Hit {
  var h : Hit;
  h.t = -1.0;
  let v0 = triVert(tri, 0u);
  let v1 = triVert(tri, 1u);
  let v2 = triVert(tri, 2u);
  let e1 = v1 - v0;
  let e2 = v2 - v0;
  let p = cross(rd, e2);
  let det = dot(e1, p);
  if (det > -EPS && det < EPS) { return h; }
  let inv = 1.0 / det;
  let tvec = ro - v0;
  let u = dot(tvec, p) * inv;
  if (u < 0.0 || u > 1.0) { return h; }
  let q = cross(tvec, e1);
  let vv = dot(rd, q) * inv;
  if (vv < 0.0 || u + vv > 1.0) { return h; }
  let t = dot(e2, q) * inv;
  if (t < EPS || t > tMax) { return h; }
  h.t = t; h.tri = i32(tri); h.u = u; h.v = vv;
  return h;
}

fn slabHit(ni: u32, ro: vec3<f32>, invD: vec3<f32>, tMax: f32) -> bool {
  let base = ni * NODE_STRIDE;
  let mn = vec3<f32>(nodes[base], nodes[base + 1u], nodes[base + 2u]);
  let mx = vec3<f32>(nodes[base + 4u], nodes[base + 5u], nodes[base + 6u]);
  let t0 = (mn - ro) * invD;
  let t1 = (mx - ro) * invD;
  let lo = min(t0, t1);
  let hi = max(t0, t1);
  let tmin = max(max(lo.x, lo.y), max(lo.z, 0.0));
  let tmax = min(min(hi.x, hi.y), min(hi.z, tMax));
  return tmax >= tmin;
}

fn traceClosest(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
  var best : Hit;
  best.t = -1.0;
  var bestT = 1e30;
  let invD = vec3<f32>(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);

  var stack : array<i32, 64>;
  var sp : i32 = 0;
  stack[sp] = 0; sp = sp + 1;

  loop {
    if (sp <= 0) { break; }
    sp = sp - 1;
    let ni = u32(stack[sp]);
    if (!slabHit(ni, ro, invD, bestT)) { continue; }
    let base = ni * NODE_STRIDE;
    let count = nodes[base + 7u];
    if (count > 0.0) {
      let first = u32(nodes[base + 3u]);
      let n = u32(count);
      for (var i: u32 = 0u; i < n; i = i + 1u) {
        let tri = triIdx[first + i];
        let h = intersectTri(tri, ro, rd, bestT);
        if (h.t > 0.0 && h.t < bestT) {
          best = h;
          bestT = h.t;
        }
      }
    } else {
      let left = i32(nodes[base + 3u]);
      let right = -i32(count);
      if (sp < 62) {
        stack[sp] = left;  sp = sp + 1;
        stack[sp] = right; sp = sp + 1;
      }
    }
  }
  return best;
}

fn triNormal(tri: u32) -> vec3<f32> {
  let v0 = triVert(tri, 0u);
  let v1 = triVert(tri, 1u);
  let v2 = triVert(tri, 2u);
  return normalize(cross(v1 - v0, v2 - v0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width  = u32(P.origin.w);
  let height = u32(P.lowerLeft.w);
  if (gid.x >= width || gid.y >= height) { return; }

  let s = (f32(gid.x) + 0.5) / f32(width);
  let tp = (f32(gid.y) + 0.5) / f32(height);
  let ro = P.origin.xyz;
  let target = P.lowerLeft.xyz + s * P.horizontal.xyz + tp * P.vertical.xyz;
  let rd = normalize(target - ro);

  let hit = traceClosest(ro, rd);

  var col : vec3<f32>;
  if (hit.t > 0.0) {
    var nrm = triNormal(u32(hit.tri));
    if (dot(nrm, rd) > 0.0) { nrm = -nrm; }
    let ambient = P.light.w;
    let diff = max(0.0, dot(nrm, -P.light.xyz));
    let lit = ambient + (1.0 - ambient) * diff;
    col = P.albedo.xyz * lit;
  } else {
    col = mix(P.bgBottom.xyz, P.bgTop.xyz, tp);
  }

  let o = (gid.y * width + gid.x) * 4u;
  fb[o] = col.x;
  fb[o + 1u] = col.y;
  fb[o + 2u] = col.z;
  fb[o + 3u] = 1.0;
}
`;
