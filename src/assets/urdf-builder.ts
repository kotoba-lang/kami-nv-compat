// TypeScript port of kotodama.nv_compat.isaacsim.assets.urdf_builder
//
// URDF builder helpers — programmatically construct minimal URDFs from
// joint specifications. Used by the Franka Panda + ANYmal C wrappers
// when a vendored mesh-bearing URDF isn't available. Generates valid
// URDFs (parseable by standard ROS urdfdom + iter 71 buildArticulation
// after a sister parse pass) with:
//
//   - serial-chain link structure (link0 → link1 → ... → link_N)
//   - revolute / prismatic / continuous joints with axes + limits
//   - placeholder unit-mass inertias (no mesh refs; no visual/collision)
//
// Branched chains (e.g. quadruped legs all rooted at a single base) are
// supported via buildBranchedUrdf which takes a tree of joint specs.
//
// ADR-2605261800 §D6 nv-compat namespace localization.

export type JointType = "revolute" | "prismatic" | "continuous" | "fixed";

export interface UrdfJointSpec {
  name: string;
  type: JointType;
  /** Joint axis in joint-body frame. Defaults to (0, 0, 1). */
  axis?: readonly [number, number, number];
  /** Joint position lower limit (revolute / prismatic). */
  lower?: number;
  /** Joint position upper limit. */
  upper?: number;
  /** Joint velocity limit (rad/s for revolute, m/s for prismatic). */
  velocity?: number;
  /** Joint effort limit (N·m for revolute, N for prismatic). */
  effort?: number;
  /** Origin xyz in parent link frame. Defaults to (0, 0, 0.1). */
  originXyz?: readonly [number, number, number];
  /** Origin rpy in parent link frame. Defaults to (0, 0, 0). */
  originRpy?: readonly [number, number, number];
}

/** Default placeholder inertial block (unit-mass, diagonal inertia 0.1). */
function unitInertia(): string {
  return (
    '<inertial>' +
    '<mass value="1.0"/>' +
    '<inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>' +
    '</inertial>'
  );
}

function originXml(
  xyz: readonly number[] = [0, 0, 0],
  rpy: readonly number[] = [0, 0, 0],
): string {
  return `<origin xyz="${xyz[0]} ${xyz[1]} ${xyz[2]}" rpy="${rpy[0]} ${rpy[1]} ${rpy[2]}"/>`;
}

function linkXml(name: string): string {
  return `<link name="${name}">${unitInertia()}</link>`;
}

function jointXml(
  joint: UrdfJointSpec,
  parentLink: string,
  childLink: string,
): string {
  const axis = joint.axis ?? [0, 0, 1];
  const originXyz = joint.originXyz ?? [0, 0, 0.1];
  const originRpy = joint.originRpy ?? [0, 0, 0];
  const parts: string[] = [
    `<joint name="${joint.name}" type="${joint.type}">`,
    originXml(originXyz, originRpy),
    `<parent link="${parentLink}"/>`,
    `<child link="${childLink}"/>`,
    `<axis xyz="${axis[0]} ${axis[1]} ${axis[2]}"/>`,
  ];
  if (joint.type === "revolute" || joint.type === "prismatic") {
    const lower = joint.lower ?? -3.14159;
    const upper = joint.upper ?? 3.14159;
    const velocity = joint.velocity ?? 1.0;
    const effort = joint.effort ?? 100.0;
    parts.push(
      `<limit lower="${lower}" upper="${upper}" velocity="${velocity}" effort="${effort}"/>`,
    );
  }
  parts.push(`</joint>`);
  return parts.join("");
}

// ── Public builders ──────────────────────────────────────────────────────

/** Build a serial-chain URDF from a list of joint specs.
 *
 * Each joint connects link_i → link_(i+1). Link names follow
 * `<robotName>_link<i>` (i = 0..joints.length).
 */
export function buildSerialChainUrdf(robotName: string, joints: UrdfJointSpec[]): string {
  const parts: string[] = [`<?xml version="1.0"?>`, `<robot name="${robotName}">`];
  // Base link
  parts.push(linkXml(`${robotName}_link0`));
  // Joints + child links
  for (let i = 0; i < joints.length; i++) {
    const parent = `${robotName}_link${i}`;
    const child = `${robotName}_link${i + 1}`;
    parts.push(jointXml(joints[i], parent, child));
    parts.push(linkXml(child));
  }
  parts.push(`</robot>`);
  return parts.join("");
}

/** Build a URDF with a common base link and multiple serial branches
 *  (e.g. quadruped — base + 4 legs).
 *
 * @param robotName name attribute on <robot>
 * @param baseLink name of the common root link
 * @param branches per-branch joint-spec lists; each branch is a serial
 *                 chain rooted at baseLink
 * @param branchLinkPrefixes optional per-branch link prefix
 */
export function buildBranchedUrdf(
  robotName: string,
  baseLink: string,
  branches: UrdfJointSpec[][],
  branchLinkPrefixes?: string[],
): string {
  const parts: string[] = [
    `<?xml version="1.0"?>`,
    `<robot name="${robotName}">`,
    linkXml(baseLink),
  ];
  for (let b = 0; b < branches.length; b++) {
    const prefix =
      branchLinkPrefixes !== undefined && b < branchLinkPrefixes.length
        ? branchLinkPrefixes[b]
        : `branch${b}_link`;
    const branch = branches[b];
    // First joint: baseLink → prefix0
    const firstChild = `${prefix}0`;
    parts.push(jointXml(branch[0], baseLink, firstChild));
    parts.push(linkXml(firstChild));
    for (let i = 1; i < branch.length; i++) {
      const parent = `${prefix}${i - 1}`;
      const child = `${prefix}${i}`;
      parts.push(jointXml(branch[i], parent, child));
      parts.push(linkXml(child));
    }
  }
  parts.push(`</robot>`);
  return parts.join("");
}

// ── Parsing utilities ────────────────────────────────────────────────────

/** Count non-fixed joints in URDF text via regex.
 *
 * Lightweight — avoids a full XML parser dep. Sufficient for the URDFs
 * this builder emits (no namespace, no nested CDATA).
 */
export function countJoints(urdfText: string): number {
  const matches = urdfText.match(/<joint\s+name="[^"]+"\s+type="([^"]+)"/g) ?? [];
  let n = 0;
  for (const m of matches) {
    const typeMatch = m.match(/type="([^"]+)"/);
    if (typeMatch && typeMatch[1] !== "fixed") n++;
  }
  return n;
}

/** Return joint names in URDF order (excludes type='fixed'). */
export function jointNames(urdfText: string): string[] {
  const re = /<joint\s+name="([^"]+)"\s+type="([^"]+)"/g;
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(urdfText)) !== null) {
    if (m[2] !== "fixed") out.push(m[1]);
  }
  return out;
}
