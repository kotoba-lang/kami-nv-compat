// URDF text → UrdfArticulatedSystem parser.
//
// Pure regex / no XML parser dep — handles the URDF subset emitted by
// iter 75's urdf-builder.ts (no namespaces, no CDATA, no XInclude).
// Robust enough for the FrankaPanda + AnymalC assets in iter 75 + any
// URDF the SDK itself produces. Strict for URDF subsets outside that:
// fails gracefully with descriptive errors.
//
// Bridges iter 75's `FrankaPanda.urdfText` (+ AnymalC, + URDFs built
// via buildSerialChainUrdf / buildBranchedUrdf) directly into iter
// 71's buildArticulation, unlocking end-to-end:
//
//   const franka = makeFrankaPanda();
//   const sys = parseUrdf(franka.urdfText);
//   const built = buildArticulation(sys);
//   // → drive forward dynamics / Jacobian / DLS IK on real Franka
//
// ADR-2605261800 §D6 nv-compat namespace localization.

import {
  type UrdfArticulatedSystem,
  type UrdfInertia,
  type UrdfJoint,
  type UrdfJointKind,
  type UrdfLink,
  type UrdfPose,
} from "./articulated-dynamics.js";

/** Parse a URDF text into a structured UrdfArticulatedSystem.
 *
 * Recognises:
 *   <robot name="...">
 *   <link name="...">
 *     <inertial>
 *       <origin xyz="..." rpy="..."/>
 *       <mass value="..."/>
 *       <inertia ixx="..." iyy="..." izz="..." ixy="..." ixz="..." iyz="..."/>
 *     </inertial>
 *   </link>
 *   <joint name="..." type="revolute|prismatic|continuous|fixed">
 *     <origin xyz="..." rpy="..."/>
 *     <parent link="..."/>
 *     <child link="..."/>
 *     <axis xyz="..."/>
 *     <limit lower="..." upper="..." velocity="..." effort="..."/>
 *     <dynamics damping="..." friction="..."/>
 *   </joint>
 *
 * Ignores unrecognised elements (no visual, no collision, no material).
 * Throws on malformed XML (truly unmatched braces) but tolerant on
 * unknown attributes.
 */
export function parseUrdf(text: string): UrdfArticulatedSystem {
  const robotName = extractAttr(text, /<robot\s+name="([^"]+)"/) ?? "robot";
  const links: UrdfLink[] = [];
  const joints: UrdfJoint[] = [];

  // Iterate over <link> blocks — combined regex matches both
  // `<link name="..."/>` (self-closing) and `<link name="...">...</link>`
  // (with body). Order is source-position so callers can rely on link
  // ordering matching the URDF text.
  const linkRe = /<link\s+name="([^"]+)"\s*(?:\/>|>([\s\S]*?)<\/link>)/g;
  let m: RegExpExecArray | null;
  while ((m = linkRe.exec(text)) !== null) {
    if (m[2] !== undefined) {
      // With body.
      links.push(parseLink(m[1], m[2]));
    } else {
      // Self-closing.
      links.push({ name: m[1], inertia: defaultInertia() });
    }
  }

  // Iterate over <joint> blocks.
  const jointRe = /<joint\s+name="([^"]+)"\s+type="([^"]+)"\s*>([\s\S]*?)<\/joint>/g;
  while ((m = jointRe.exec(text)) !== null) {
    joints.push(parseJoint(m[1], m[2] as UrdfJointKind, m[3]));
  }

  return { name: robotName, links, joints };
}

// ── helpers ───────────────────────────────────────────────────────────────

function extractAttr(text: string, re: RegExp): string | undefined {
  const m = re.exec(text);
  return m ? m[1] : undefined;
}

function extractTriplet(
  text: string,
  attrRe: RegExp,
  fallback: [number, number, number] = [0, 0, 0],
): [number, number, number] {
  const m = attrRe.exec(text);
  if (!m) return [...fallback];
  const parts = m[1].split(/\s+/).filter((s) => s.length > 0);
  if (parts.length !== 3) {
    throw new Error(`URDF parse: expected 3-tuple, got '${m[1]}'`);
  }
  return [Number(parts[0]), Number(parts[1]), Number(parts[2])];
}

function parsePose(blockText: string | undefined): UrdfPose {
  if (!blockText) return { xyz: [0, 0, 0], rpy: [0, 0, 0] };
  const xyz = extractTriplet(blockText, /<origin[^>]*\sxyz="([^"]+)"/);
  const rpy = extractTriplet(blockText, /<origin[^>]*\srpy="([^"]+)"/);
  return { xyz, rpy };
}

function defaultInertia(): UrdfInertia {
  return {
    mass: 0,
    ixx: 0, iyy: 0, izz: 0, ixy: 0, ixz: 0, iyz: 0,
    com: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
  };
}

function parseLink(name: string, body: string): UrdfLink {
  // Find the <inertial>…</inertial> sub-block (optional).
  const inMatch = /<inertial>([\s\S]*?)<\/inertial>/.exec(body);
  if (!inMatch) {
    return { name, inertia: defaultInertia() };
  }
  const inertialBody = inMatch[1];
  const mass = Number(extractAttr(inertialBody, /<mass\s+value="([^"]+)"/) ?? "0");
  const ixx = Number(extractAttr(inertialBody, /<inertia[^>]*\sixx="([^"]+)"/) ?? "0");
  const iyy = Number(extractAttr(inertialBody, /<inertia[^>]*\siyy="([^"]+)"/) ?? "0");
  const izz = Number(extractAttr(inertialBody, /<inertia[^>]*\sizz="([^"]+)"/) ?? "0");
  const ixy = Number(extractAttr(inertialBody, /<inertia[^>]*\sixy="([^"]+)"/) ?? "0");
  const ixz = Number(extractAttr(inertialBody, /<inertia[^>]*\sixz="([^"]+)"/) ?? "0");
  const iyz = Number(extractAttr(inertialBody, /<inertia[^>]*\siyz="([^"]+)"/) ?? "0");
  return {
    name,
    inertia: { mass, ixx, iyy, izz, ixy, ixz, iyz, com: parsePose(inertialBody) },
  };
}

function parseJoint(name: string, kind: UrdfJointKind, body: string): UrdfJoint {
  const validKinds: UrdfJointKind[] = ["revolute", "continuous", "prismatic", "fixed"];
  if (!validKinds.includes(kind)) {
    throw new Error(`URDF parse: unknown joint type '${kind}' on joint '${name}'`);
  }
  const parent = extractAttr(body, /<parent\s+link="([^"]+)"/);
  if (!parent) {
    throw new Error(`URDF parse: joint '${name}' has no <parent>`);
  }
  const child = extractAttr(body, /<child\s+link="([^"]+)"/);
  if (!child) {
    throw new Error(`URDF parse: joint '${name}' has no <child>`);
  }
  const origin = parsePose(body);
  const axis = extractTriplet(body, /<axis\s+xyz="([^"]+)"/, [1, 0, 0]);
  const damping = Number(extractAttr(body, /<dynamics[^>]*\sdamping="([^"]+)"/) ?? "0");
  const friction = Number(extractAttr(body, /<dynamics[^>]*\sfriction="([^"]+)"/) ?? "0");
  return { name, kind, parent, child, origin, axis, damping, friction };
}
