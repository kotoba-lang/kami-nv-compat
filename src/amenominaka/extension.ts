// amenominaka — clean-room Kit extension model (IExt + extension.toml).
//
// The canonical KAMI implementation behind `nv-compat/omni-kit-app`. NVIDIA
// Omniverse Kit hosts an app as a graph of extensions, each declared by an
// `extension.toml` manifest and implementing the `omni.ext.IExt` lifecycle
// (on_startup / on_shutdown). This module reproduces that contract and a
// minimal-but-faithful TOML manifest parser.
//
// Clean-room: from-spec IExt + a hand-written TOML subset parser (the subset
// Kit manifests actually use). No Kit source/binaries. ADR-2605261800 §D6 /
// D10.4 amenominaka.

// ── IExt lifecycle ───────────────────────────────────────────────────────────

/** Base class for Kit extensions (mirrors `omni.ext.IExt`). Override the
 *  lifecycle hooks; the Application invokes them in dependency order. */
export abstract class IExt {
  /** Called when the Application loads this extension. */
  onStartup(_extId: string): void {}
  /** Called once per app tick (KAMI extension to the Kit IExt surface). */
  onUpdate(_dt: number): void {}
  /** Called when the Application unloads this extension. */
  onShutdown(): void {}
}

// ── extension.toml model ─────────────────────────────────────────────────────

export interface ExtensionToml {
  title: string;
  version: string;
  description: string;
  category: string;
  keywords: string[];
  authors: string[];
  repository: string;
  /** Dependency id → inline-table options. */
  dependencies: Record<string, Record<string, unknown>>;
  /** `[[python.module]]` entries. */
  pythonModules: Array<Record<string, unknown>>;
  /** All parsed tables keyed by dotted name (escape hatch). */
  rawTables: Record<string, unknown>;
}

type TomlValue = string | number | boolean | TomlValue[] | { [k: string]: TomlValue };

function stripComment(line: string): string {
  let inString = false;
  let out = "";
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"' && (i === 0 || line[i - 1] !== "\\")) inString = !inString;
    if (c === "#" && !inString) break;
    out += c;
  }
  return out;
}

function parseValue(raw: string): TomlValue {
  const s = raw.trim();
  if (s.startsWith('"') && s.endsWith('"')) return s.slice(1, -1);
  if (s.startsWith("[") && s.endsWith("]")) {
    const inner = s.slice(1, -1).trim();
    if (!inner) return [];
    return splitTopLevel(inner).map((p) => parseValue(p));
  }
  if (s.startsWith("{") && s.endsWith("}")) {
    const inner = s.slice(1, -1).trim();
    const obj: { [k: string]: TomlValue } = {};
    if (!inner) return obj;
    for (const part of splitTopLevel(inner)) {
      const eq = part.indexOf("=");
      if (eq < 0) continue;
      obj[part.slice(0, eq).trim().replace(/^"|"$/g, "")] = parseValue(part.slice(eq + 1));
    }
    return obj;
  }
  if (s === "true" || s === "false") return s === "true";
  const n = Number(s);
  if (s.length > 0 && !Number.isNaN(n)) return s.includes(".") ? n : Math.trunc(n);
  return s;
}

/** Split a comma list respecting nested [] / {} depth. */
function splitTopLevel(inner: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = "";
  for (const ch of inner) {
    if (ch === "[" || ch === "{") depth++;
    else if (ch === "]" || ch === "}") depth--;
    if (ch === "," && depth === 0) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur);
  return out;
}

function ensureTable(root: Record<string, TomlValue>, path: string[]): Record<string, TomlValue> {
  let node = root;
  for (const key of path) {
    if (typeof node[key] !== "object" || node[key] === null || Array.isArray(node[key])) {
      node[key] = {};
    }
    node = node[key] as Record<string, TomlValue>;
  }
  return node;
}

function ensureArrayOfTables(root: Record<string, TomlValue>, path: string[]): Record<string, TomlValue> {
  const parent = ensureTable(root, path.slice(0, -1));
  const key = path[path.length - 1];
  if (!Array.isArray(parent[key])) parent[key] = [];
  const arr = parent[key] as TomlValue[];
  const entry: Record<string, TomlValue> = {};
  arr.push(entry);
  return entry;
}

/** Parse the subset of TOML used by Kit extension manifests. */
export function parseExtensionToml(text: string): ExtensionToml {
  const root: Record<string, TomlValue> = {};
  let current = root;
  for (const rawLine of text.split("\n")) {
    const line = stripComment(rawLine).trim();
    if (!line) continue;
    if (line.startsWith("[[") && line.includes("]]")) {
      const name = line.slice(2, line.indexOf("]]")).trim();
      current = ensureArrayOfTables(root, name.split("."));
    } else if (line.startsWith("[") && line.includes("]")) {
      const name = line.slice(1, line.indexOf("]")).trim();
      current = ensureTable(root, name.split("."));
    } else {
      const eq = line.indexOf("=");
      if (eq < 0) continue;
      const key = line.slice(0, eq).trim().replace(/^"|"$/g, "");
      current[key] = parseValue(line.slice(eq + 1));
    }
  }

  const pkg = (root.package as Record<string, TomlValue>) ?? {};
  const py = root.python as Record<string, TomlValue> | undefined;
  const modules = (py?.module as TomlValue[] | undefined) ?? [];
  return {
    title: String(pkg.title ?? ""),
    version: String(pkg.version ?? "0.1.0"),
    description: String(pkg.description ?? ""),
    category: String(pkg.category ?? ""),
    keywords: (pkg.keywords as string[]) ?? [],
    authors: (pkg.authors as string[]) ?? [],
    repository: String(pkg.repository ?? ""),
    dependencies: (root.dependencies as Record<string, Record<string, unknown>>) ?? {},
    pythonModules: modules as Array<Record<string, unknown>>,
    rawTables: root,
  };
}
