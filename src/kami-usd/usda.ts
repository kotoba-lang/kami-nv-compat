// kami-usd — clean-room USDA (ASCII USD) reader.
//
// The canonical KAMI implementation behind `nv-compat/omni-usd.ts`. Pixar's
// USD is the Universal Scene Description format; this module parses the
// documented ASCII (.usda) crate grammar for the geometry subset KAMI needs
// (UsdGeomMesh / Xform prims + their attributes) and exposes a small,
// pxr.Usd-shaped tree. Binary .usdc / .usdz are a later kami-usd milestone
// (tinyusdz WASM, ADR-2605261800 D6/D10.4).
//
// Clean-room: this is a from-spec re-implementation of the USDA text grammar.
// No USD / OpenUSD / tinyusdz source, headers, or binaries are used. The
// parser is a small recursive-descent scanner over the public crate syntax.
//
// ADR-2605261800 §D6 / D10.4 kami-usd.

// ── value model ────────────────────────────────────────────────────────────

/** A parsed USDA attribute value: scalar, tuple, or (possibly nested) array. */
export type UsdValue = number | string | boolean | UsdValue[];

export interface UsdAttribute {
  /** Declared type, e.g. "point3f[]", "float3", "token", "color3f[]". */
  typeName: string;
  name: string;
  /** Parsed value, or null for a declaration with no `= ...`. */
  value: UsdValue | null;
  /** True for `uniform` attributes. */
  uniform: boolean;
}

export interface UsdPrimNode {
  /** Specifier: "def" | "over" | "class". */
  specifier: string;
  /** Schema type token after the specifier, e.g. "Mesh", "Xform"; "" if none. */
  typeName: string;
  name: string;
  /** Absolute path, e.g. "/World/box". */
  path: string;
  attributes: Map<string, UsdAttribute>;
  children: UsdPrimNode[];
}

// ── scanner ────────────────────────────────────────────────────────────────

class Scanner {
  private i = 0;
  constructor(private readonly s: string) {}

  eof(): boolean {
    this.skip();
    return this.i >= this.s.length;
  }

  /** Skip whitespace and `#` line comments. */
  skip(): void {
    for (;;) {
      const c = this.s[this.i];
      if (c === " " || c === "\t" || c === "\r" || c === "\n") {
        this.i++;
      } else if (c === "#") {
        while (this.i < this.s.length && this.s[this.i] !== "\n") this.i++;
      } else {
        break;
      }
    }
  }

  peek(): string {
    this.skip();
    return this.s[this.i] ?? "";
  }

  /** Consume the next char regardless of whitespace handling. */
  take(): string {
    return this.s[this.i++] ?? "";
  }

  expect(ch: string): void {
    this.skip();
    if (this.s[this.i] !== ch) {
      throw new Error(`USDA parse: expected '${ch}' at offset ${this.i}, got '${this.s[this.i] ?? "<eof>"}'`);
    }
    this.i++;
  }

  tryConsume(ch: string): boolean {
    this.skip();
    if (this.s[this.i] === ch) {
      this.i++;
      return true;
    }
    return false;
  }

  /** A bareword token: identifiers, base type names, namespaced attribute
   *  names (with `:`), numbers, signs, dots. Note `[` / `]` are NOT token
   *  chars (they delimit arrays and the `[]` type suffix is read separately),
   *  so a scalar like `4` in `[4]` does not swallow the closing bracket. */
  token(): string {
    this.skip();
    const start = this.i;
    while (this.i < this.s.length) {
      const c = this.s[this.i];
      if (
        (c >= "a" && c <= "z") ||
        (c >= "A" && c <= "Z") ||
        (c >= "0" && c <= "9") ||
        c === "_" || c === ":" ||
        c === "." || c === "-" || c === "+" || c === "e" || c === "E"
      ) {
        this.i++;
      } else {
        break;
      }
    }
    return this.s.slice(start, this.i);
  }

  /** An optional empty `[]` array-type suffix, or "" if the next token is not
   *  one. Leaves a non-empty `[ ... ]` (an array value) untouched. */
  arraySuffix(): string {
    if (this.peek() === "[") {
      const save = this.i;
      this.expect("[");
      if (this.tryConsume("]")) return "[]";
      this.i = save; // a real array value, not a type suffix
    }
    return "";
  }

  /** Read a base type name token plus an optional `[]` array suffix
   *  (e.g. `point3f` → `point3f`, `point3f[]` → `point3f[]`). */
  typeToken(): string {
    return this.token() + this.arraySuffix();
  }

  /** A double-quoted string ("..." or '...'). */
  quoted(): string {
    this.skip();
    const q = this.s[this.i];
    if (q !== '"' && q !== "'") throw new Error(`USDA parse: expected string at offset ${this.i}`);
    this.i++;
    let out = "";
    while (this.i < this.s.length && this.s[this.i] !== q) {
      if (this.s[this.i] === "\\") {
        this.i++;
        out += this.s[this.i] ?? "";
      } else {
        out += this.s[this.i];
      }
      this.i++;
    }
    this.i++; // closing quote
    return out;
  }
}

// ── value parsing ────────────────────────────────────────────────────────────

function parseScalar(raw: string): UsdValue {
  if (raw === "true") return true;
  if (raw === "false") return false;
  const n = Number(raw);
  if (raw.length > 0 && !Number.isNaN(n)) return n;
  return raw; // token / enum
}

function parseValue(sc: Scanner): UsdValue {
  const c = sc.peek();
  if (c === "[") return parseArray(sc);
  if (c === "(") return parseTuple(sc);
  if (c === '"' || c === "'") return sc.quoted();
  return parseScalar(sc.token());
}

function parseArray(sc: Scanner): UsdValue[] {
  sc.expect("[");
  const out: UsdValue[] = [];
  if (sc.tryConsume("]")) return out;
  for (;;) {
    out.push(parseValue(sc));
    if (sc.tryConsume(",")) {
      if (sc.peek() === "]") {
        sc.expect("]");
        break;
      }
      continue;
    }
    sc.expect("]");
    break;
  }
  return out;
}

function parseTuple(sc: Scanner): UsdValue[] {
  sc.expect("(");
  const out: UsdValue[] = [];
  if (sc.tryConsume(")")) return out;
  for (;;) {
    out.push(parseValue(sc));
    if (sc.tryConsume(",")) {
      if (sc.peek() === ")") {
        sc.expect(")");
        break;
      }
      continue;
    }
    sc.expect(")");
    break;
  }
  return out;
}

// ── prim / attribute parsing ─────────────────────────────────────────────────

const SPECIFIERS = new Set(["def", "over", "class"]);

function isAttrTypeStart(tok: string): boolean {
  // Attribute type tokens are lower-cased schema value types; prim specifiers
  // are def/over/class (handled separately). Anything else starting a line in
  // a prim body that is not a specifier is an attribute declaration.
  return tok.length > 0 && !SPECIFIERS.has(tok);
}

function parsePrimBody(sc: Scanner, node: UsdPrimNode): void {
  sc.expect("{");
  for (;;) {
    if (sc.tryConsume("}")) return;
    if (sc.eof()) throw new Error("USDA parse: unexpected EOF inside prim body");
    const tok = sc.token();
    if (SPECIFIERS.has(tok)) {
      node.children.push(parsePrim(sc, tok, node.path));
      continue;
    }
    // Attribute: `[custom] [uniform] <type>[\[\]] <name> [= value] [( meta )]`.
    let typeName: string;
    let uniform = false;
    if (tok === "uniform") {
      uniform = true;
      typeName = sc.typeToken();
    } else if (tok === "custom") {
      typeName = sc.typeToken();
      if (typeName === "uniform") {
        uniform = true;
        typeName = sc.typeToken();
      }
    } else {
      typeName = tok + sc.arraySuffix();
    }
    if (!isAttrTypeStart(typeName)) {
      throw new Error(`USDA parse: malformed attribute near '${typeName}' (path ${node.path})`);
    }
    const name = sc.token();
    let value: UsdValue | null = null;
    if (sc.tryConsume("=")) {
      value = parseValue(sc);
    }
    // Optional attribute metadata block ( ... ) — skipped.
    if (sc.peek() === "(") skipBalanced(sc, "(", ")");
    node.attributes.set(name, { typeName, name, value, uniform });
  }
}

function skipBalanced(sc: Scanner, open: string, close: string): void {
  sc.expect(open);
  let depth = 1;
  while (depth > 0) {
    if (sc.eof()) throw new Error("USDA parse: unbalanced metadata block");
    const c = sc.take();
    if (c === open) depth++;
    else if (c === close) depth--;
  }
}

function parsePrim(sc: Scanner, specifier: string, parentPath: string): UsdPrimNode {
  // After the specifier token: optional schema type, then quoted name.
  let typeName = "";
  let name: string;
  const nxt = sc.peek();
  if (nxt === '"' || nxt === "'") {
    name = sc.quoted();
  } else {
    typeName = sc.token();
    name = sc.quoted();
  }
  // Optional prim metadata ( ... ).
  if (sc.peek() === "(") skipBalanced(sc, "(", ")");
  const path = parentPath === "/" ? `/${name}` : `${parentPath}/${name}`;
  const node: UsdPrimNode = {
    specifier,
    typeName,
    name,
    path,
    attributes: new Map(),
    children: [],
  };
  parsePrimBody(sc, node);
  return node;
}

/** Parse a USDA document into a list of root prims. An optional leading
 *  `#usda 1.0` header line and stage metadata `( ... )` are tolerated. */
export function parseUsda(text: string): UsdPrimNode[] {
  const sc = new Scanner(text);
  const roots: UsdPrimNode[] = [];
  // Optional stage metadata right after the header comment.
  if (sc.peek() === "(") skipBalanced(sc, "(", ")");
  while (!sc.eof()) {
    const tok = sc.token();
    if (tok === "") break;
    if (!SPECIFIERS.has(tok)) {
      throw new Error(`USDA parse: expected def/over/class at top level, got '${tok}'`);
    }
    roots.push(parsePrim(sc, tok, "/"));
  }
  return roots;
}
