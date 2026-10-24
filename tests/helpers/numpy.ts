import { spawnSync } from "child_process";
import { expect } from "vitest";
import { np } from "../../src/index";

/**
 * Differential testing helper: runs the same numpy expression in real python/numpy
 * and compares the result with ndarray-js.
 * Set NDARRAYJS_PYTHON to a python with numpy installed (default: python3).
 */
export const PYTHON = process.env.NDARRAYJS_PYTHON || "python3";

export const HAS_NUMPY = (() => {
  try {
    const r = spawnSync(PYTHON, ["-c", "import numpy"]);
    return r.status === 0;
  } catch (e) { return false; }
})();

function to_python_literal(value: any): string {
  if (value instanceof np.NDArray) {
    const dtype = value.dtype;
    const py = JSON.stringify(np.tolist(value), (_k, v) =>
      typeof v === "number" && !Number.isFinite(v) ? `__${v}__` : v)
      .replace(/"__NaN__"/g, "np.nan").replace(/"__Infinity__"/g, "np.inf").replace(/"__-Infinity__"/g, "-np.inf")
      .replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None");
    const pyDtype = { bool: "bool", int32: "np.int32", float32: "np.float32", float64: "np.float64", object: "object" }[dtype];
    return `np.array(${py}, dtype=${pyDtype})`;
  }
  if (typeof value === "number") {
    if (Number.isNaN(value)) return "np.nan";
    if (value === Infinity) return "np.inf";
    if (value === -Infinity) return "(-np.inf)";
    return JSON.stringify(value);
  }
  if (typeof value === "boolean") return value ? "True" : "False";
  if (value === null || value === undefined) return "None";
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(to_python_literal).join(", ")}]`;
  if (typeof value === "object") return `{${Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${to_python_literal(v)}`).join(", ")}}`;
  throw new Error(`Cannot convert to python: ${value}`);
}

const PY_PROGRAM = (code: string) => `
import numpy as np
import json, math
def enc(o):
    if isinstance(o, (np.ndarray,)):
        return {"__nd__": True, "dtype": str(o.dtype), "shape": list(o.shape), "data": enc(o.tolist())}
    if isinstance(o, (np.generic,)):
        return enc(o.item())
    if isinstance(o, (list, tuple)):
        return [enc(x) for x in o]
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, float):
        if math.isnan(o): return "__nan__"
        if o == math.inf: return "__inf__"
        if o == -math.inf: return "__-inf__"
    return o
out = (${code})
print(json.dumps(enc(out)), flush=True)
`;

function dec(o: any): any {
  if (o === "__nan__") return NaN;
  if (o === "__inf__") return Infinity;
  if (o === "__-inf__") return -Infinity;
  if (Array.isArray(o)) return o.map(dec);
  if (o && typeof o === "object") {
    if (o.__nd__) return { nd: true, dtype: o.dtype, shape: o.shape, data: dec(o.data) };
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, dec(v)]));
  }
  return o;
}

/** Evaluate a python/numpy expression. Interpolated values are converted to python literals. */
export function numpy(template: TemplateStringsArray, ...vars: any[]): any {
  let code = template[0];
  vars.forEach((v, i) => { code += to_python_literal(v) + template[i + 1]; });
  const r = spawnSync(PYTHON, ["-c", PY_PROGRAM(code)], { maxBuffer: 1 << 28 });
  if (r.status !== 0) throw new Error(`numpy failed on: ${code}\n${r.stderr.toString()}`);
  return dec(JSON.parse(r.stdout.toString()));
}

/** True if python raises for this expression */
export function numpy_throws(template: TemplateStringsArray, ...vars: any[]): boolean {
  let code = template[0];
  vars.forEach((v, i) => { code += to_python_literal(v) + template[i + 1]; });
  const r = spawnSync(PYTHON, ["-c", PY_PROGRAM(code)], { maxBuffer: 1 << 28 });
  return r.status !== 0;
}

function close(a: any, b: any, rtol: number, atol: number, path: string): string | null {
  if (typeof a === "boolean" || typeof b === "boolean") return a === b ? null : `${path}: ${a} !== ${b}`;
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isNaN(a) && Number.isNaN(b)) return null;
    if (a === b) return null;
    if (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= atol + rtol * Math.abs(b)) return null;
    return `${path}: ${a} !~ ${b}`;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${path}: length ${a.length} !== ${b.length}`;
    for (let i = 0; i < a.length; i++) { const e = close(a[i], b[i], rtol, atol, `${path}[${i}]`); if (e) return e; }
    return null;
  }
  return a === b ? null : `${path}: ${JSON.stringify(a)} !== ${JSON.stringify(b)}`;
}

const KIND = (d: string) => d.startsWith("bool") ? "bool" : d.startsWith("int") || d.startsWith("uint") ? "int" : d.startsWith("float") ? "float" : "other";

/**
 * Compare an ndarray-js result with a numpy expected value (from `numpy`).
 * Compares values and shape; dtype *kind* is compared when check_dtype is true (bool/int/float).
 */
export function expect_same(got: any, want: any, { rtol = 1e-7, atol = 1e-9, check_dtype = true } = {}) {
  if (want && want.nd && want.shape.length === 0 && !(got instanceof np.NDArray) && !Array.isArray(got)) {
    // 0-d arrays are returned as plain JS scalars
    const err = close(got, want.data, rtol, atol, "");
    expect(err, err || "").toBeNull();
  } else if (want && want.nd) {
    expect(got instanceof np.NDArray || Array.isArray(got), `expected an array, got ${typeof got}: ${got}`).toBe(true);
    const arr = np.asarray(got);
    expect(arr.shape, "shape").toEqual(want.shape);
    if (check_dtype) expect(KIND(arr.dtype), `dtype ${arr.dtype} vs numpy ${want.dtype}`).toBe(KIND(want.dtype));
    const err = close(np.tolist(arr), want.data, rtol, atol, "");
    expect(err, err || "").toBeNull();
  } else if (Array.isArray(want)) {
    expect(Array.isArray(got), `expected list, got ${got}`).toBe(true);
    expect(got.length).toBe(want.length);
    want.forEach((w, i) => expect_same(got[i], w, { rtol, atol, check_dtype }));
  } else if (want && typeof want === "object") {
    for (const k of Object.keys(want)) expect_same(got[k], want[k], { rtol, atol, check_dtype });
  } else {
    if (got instanceof np.NDArray) {
      expect(got.shape, "expected a scalar").toEqual([]);
      got = got.item();
    }
    const err = close(got, want, rtol, atol, "");
    expect(err, err || "").toBeNull();
  }
}


// ======================================================================
//  Batch mode: evaluate many numpy expressions in one python process
// ======================================================================

export type NumpyResult = { ok: true, value: any } | { ok: false, error: string };

const BATCH_PROGRAM = `
import sys, json, math, warnings
import numpy as np
warnings.simplefilter("ignore")
def enc(o):
    if isinstance(o, np.ndarray):
        return {"__nd__": True, "dtype": str(o.dtype), "shape": list(o.shape), "data": enc(o.tolist())}
    if isinstance(o, np.generic):
        return enc(o.item())
    if isinstance(o, (list, tuple)):
        return [enc(x) for x in o]
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, float):
        if math.isnan(o): return "__nan__"
        if o == math.inf: return "__inf__"
        if o == -math.inf: return "__-inf__"
    return o
codes = json.load(sys.stdin)
results = []
for code in codes:
    try:
        results.append({"ok": True, "value": enc(eval(code))})
    except Exception as e:
        results.append({"ok": False, "error": type(e).__name__ + ": " + str(e)})
print(json.dumps(results))
`;

/** Evaluates python/numpy expressions (strings) all at once. Returns one result per expression. */
export function numpy_batch(codes: string[]): NumpyResult[] {
  if (!HAS_NUMPY) return codes.map(() => ({ ok: false, error: "numpy not available" }));
  const r = spawnSync(PYTHON, ["-c", BATCH_PROGRAM], { input: JSON.stringify(codes), maxBuffer: 1 << 28 });
  if (r.status !== 0) throw new Error(`numpy batch failed:\n${r.stderr.toString()}`);
  return (JSON.parse(r.stdout.toString()) as any[]).map(res => res.ok ? { ok: true, value: dec(res.value) } : res);
}

/** Evaluates a numpy expression with ndarray-js using its own numpy-syntax parser */
export function js_eval(code: string): any {
  return (np as any)(Object.assign([code], { raw: [code] }));
}
