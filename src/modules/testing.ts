//@ts-check
/** numpy.testing: assertion helpers. They throw an AssertionError (an Error named "AssertionError") when the check fails. */
import { NDArray, asarray, isarray, broadcast_shapes } from "../NDArray";
import { broadcast_to } from "./manipulation";
import { float32, float64, dtype_is_float } from "../dtypes";
import { nextafter_scalar } from "../array/elementwise";

export class AssertionError extends Error {
  constructor(message: string) { super(message); this.name = "AssertionError"; }
}

const fail = (msg: string, err_msg: string, detail = ""): never => {
  throw new AssertionError(`\n${msg}${detail ? "\n" + detail : ""}${err_msg ? "\n" + err_msg : ""}`);
};
const flat_of = (a: NDArray) => a.dtype === "bool" ? Array.from(a.flat as any, (x: any) => !!x) : Array.from(a.flat as any) as any[];
const is_nan = (x: any) => typeof x === "number" && Number.isNaN(x);

function compare(actual: any, desired: any, name: string, err_msg: string, ok: (a: number, d: number) => boolean, strict_shape = true, show = (a: any, d: any) => "") {
  const A = asarray(actual), D = asarray(desired);
  let a = A, d = D;
  if (A.shape.length != D.shape.length || A.shape.some((v, i) => v != D.shape[i])) {
    // scalars broadcast; otherwise the shapes must match
    if (A.shape.length == 0 || D.shape.length == 0) { const [shape] = broadcast_shapes(A.shape, D.shape); a = broadcast_to(A, shape); d = broadcast_to(D, shape); }
    else fail(`${name}\n(shapes (${A.shape}), (${D.shape}) mismatch)`, err_msg, ` ACTUAL: shape (${A.shape})\n DESIRED: shape (${D.shape})`);
  }
  const fa = flat_of(a), fd = flat_of(d);
  const bad: number[] = [];
  for (let i = 0; i < fa.length; i++) if (!ok(fa[i], fd[i])) bad.push(i);
  if (bad.length) {
    const i = bad[0];
    fail(`${name}\nMismatched elements: ${bad.length} / ${fa.length} (${(100 * bad.length / fa.length).toFixed(3).replace(/\.?0+$/, '')}%)`, err_msg,
      ` ACTUAL: ${fa[i]}\n DESIRED: ${fd[i]}${show(fa[i], fd[i])}`);
  }
}

const same = (a: any, d: any) => a === d || (is_nan(a) && is_nan(d)) || (typeof a === "number" && typeof d === "number" && a == d);

export function assert_(val: any, msg: string | (() => string) = ""): void {
  if (!val) throw new AssertionError(typeof msg === "function" ? msg() : msg);
}

export function assert_equal(actual: any, desired: any, err_msg = "", verbose = true): void {
  if (Array.isArray(desired) && Array.isArray(actual) && !desired.every(x => typeof x === "number" || typeof x === "boolean") ) {
    if (actual.length != desired.length) fail(`Lists differ in length: ${actual.length} != ${desired.length}`, err_msg);
    desired.forEach((d, i) => assert_equal(actual[i], d, err_msg + (err_msg ? " " : "") + `item=${i}`));
    return;
  }
  if (isarray(actual) || isarray(desired) || Array.isArray(actual) || Array.isArray(desired)) return assert_array_equal(actual, desired, err_msg, verbose);
  if (!same(actual, desired)) fail(`Items are not equal:`, err_msg, ` ACTUAL: ${actual}\n DESIRED: ${desired}`);
}

export function assert_array_equal(x: any, y: any, err_msg = "", verbose = true, strict = false): void {
  const X = asarray(x), Y = asarray(y);
  if (strict && (X.dtype !== Y.dtype || X.shape.length != Y.shape.length || X.shape.some((v, i) => v != Y.shape[i]))) fail(`Arrays are not equal (strict mode: dtype or shape mismatch)`, err_msg, ` ACTUAL: ${X.dtype} ${X.shape}\n DESIRED: ${Y.dtype} ${Y.shape}`);
  compare(x, y, "Arrays are not equal", err_msg, same);
}

export function assert_allclose(actual: any, desired: any, rtol = 1e-7, atol = 0, equal_nan = true, err_msg = "", verbose = true, strict = false): void {
  const ok = (a: number, d: number) => {
    if (is_nan(a) || is_nan(d)) return equal_nan && is_nan(a) && is_nan(d);
    if (a === d) return true;
    return Math.abs(a - d) <= atol + rtol * Math.abs(d);
  };
  compare(actual, desired, `Not equal to tolerance rtol=${rtol}, atol=${atol}`, err_msg, ok, true,
    (a, d) => `\nMax abs difference: ${Math.abs(a - d)}`);
}

export function assert_almost_equal(actual: any, desired: any, decimal = 7, err_msg = "", verbose = true): void {
  if (isarray(actual) || isarray(desired) || Array.isArray(actual) || Array.isArray(desired)) return assert_array_almost_equal(actual, desired, decimal, err_msg);
  const ok = (a: number, d: number) => same(a, d) || Math.abs(d - a) < 1.5 * Math.pow(10, -decimal);
  if (!ok(+actual, +desired)) fail(`Arrays are not almost equal to ${decimal} decimals`, err_msg, ` ACTUAL: ${actual}\n DESIRED: ${desired}`);
}

export function assert_array_almost_equal(x: any, y: any, decimal = 6, err_msg = "", verbose = true): void {
  compare(x, y, `Arrays are not almost equal to ${decimal} decimals`, err_msg, (a, d) => same(a, d) || Math.abs(d - a) < 1.5 * Math.pow(10, -decimal),
    true, (a, d) => `\nMax absolute difference among violations: ${Math.abs(a - d)}`);
}

export function assert_approx_equal(actual: number, desired: number, significant = 7, err_msg = "", verbose = true): void {
  if (same(actual, desired)) return;
  const scale = 0.5 * (Math.abs(desired) + Math.abs(actual));
  const sc = Math.pow(10, Math.floor(Math.log10(scale)));
  if (!(Math.abs(desired / sc - actual / sc) < Math.pow(10, -(significant - 1)))) fail(`Items are not equal to ${significant} significant digits:`, err_msg, ` ACTUAL: ${actual}\n DESIRED: ${desired}`);
}

export function assert_array_less(x: any, y: any, err_msg = "", verbose = true, strict = false): void {
  compare(x, y, "Arrays are not strictly ordered `x < y`", err_msg, (a, d) => !is_nan(a) && !is_nan(d) && a < d || (is_nan(a) && is_nan(d)));
}

export function assert_string_equal(actual: string, desired: string): void {
  if (typeof actual !== "string" || typeof desired !== "string") throw new AssertionError(`Both arguments must be strings`);
  if (actual !== desired) fail(`Strings are not equal:`, "", ` ACTUAL: ${JSON.stringify(actual)}\n DESIRED: ${JSON.stringify(desired)}`);
}

export function print_assert_equal(test_string: string, actual: any, desired: any): void {
  try { assert_equal(actual, desired); }
  catch (e) { console.log(`${test_string}\n  ACTUAL: ${actual}\n  DESIRED: ${desired}`); throw e; }
}

/** Distance in representable floating point numbers between a and b (capped at 1e6) */
function ulp_distance(a: number, b: number, dtype: string): number {
  if (a === b) return 0;
  if (is_nan(a) || is_nan(b)) return Infinity;
  let lo = Math.min(a, b);
  const hi = Math.max(a, b);
  let n = 0;
  while (lo < hi && n < 1e6) { lo = nextafter_scalar(lo, hi, dtype === "float32" ? float32 : float64); n++; }
  return lo < hi ? Infinity : n;
}

export function assert_array_almost_equal_nulp(x: any, y: any, nulp = 1): void {
  const X = asarray(x), Y = asarray(y);
  compare(X, Y, `Arrays are not almost equal to ${nulp} ULP`, "", (a, d) => {
    if (same(a, d)) return true;
    const ref = Math.max(Math.abs(a), Math.abs(d));
    const spacing = Math.abs(nextafter_scalar(ref, Infinity, X.dtype === "float32" ? float32 : float64) - ref);
    return Math.abs(a - d) <= nulp * spacing;
  });
}

export function assert_array_max_ulp(a: any, b: any, maxulp = 1, dtype: any = null): number {
  const A = asarray(a), B = asarray(b);
  const dt = dtype || (A.dtype === "float32" ? "float32" : "float64");
  const fa = flat_of(A), fb = flat_of(B.size == 1 && A.size > 1 ? broadcast_to(B, A.shape) : B);
  let worst = 0;
  fa.forEach((x, i) => { worst = Math.max(worst, ulp_distance(+x, +fb[i], dt)); });
  if (worst > maxulp) throw new AssertionError(`Arrays are not almost equal up to ${maxulp} ULP (max difference is ${worst} ULP)`);
  return worst;
}

function matches(e: any, exception: any) {
  if (typeof exception === "function") return e instanceof exception;
  return true;
}

/** assert_raises(ErrorClass, fn, ...args): checks that fn(...args) throws */
export function assert_raises(exception: any, fn?: (...args: any[]) => any, ...args: any[]): any {
  const check = (f: (...a: any[]) => any, a: any[]) => {
    try { f(...a); }
    catch (e) { if (matches(e, exception)) return e; throw e; }
    throw new AssertionError(`${typeof exception === "function" ? exception.name : exception} not raised`);
  };
  if (fn === undefined) return { run: (f: (...a: any[]) => any, ...a: any[]) => check(f, a) };
  return check(fn, args);
}

export function assert_raises_regex(exception: any, expected_regexp: string | RegExp, fn: (...args: any[]) => any, ...args: any[]): any {
  const re = typeof expected_regexp === "string" ? new RegExp(expected_regexp) : expected_regexp;
  const e = assert_raises(exception, fn, ...args);
  if (!re.test(String(e?.message ?? e))) throw new AssertionError(`"${re}" does not match "${e?.message}"`);
  return e;
}

/** Runs fn and fails if console.warn was not called (JS has no warning categories) */
export function assert_warns(warning_class: any, fn: (...args: any[]) => any, ...args: any[]): any {
  const original = console.warn;
  let warned = 0;
  console.warn = () => { warned++; };
  let result: any;
  try { result = fn(...args); } finally { console.warn = original; }
  if (!warned) throw new AssertionError(`No warning raised`);
  return result;
}

export function assert_no_warnings(fn: (...args: any[]) => any, ...args: any[]): any {
  const original = console.warn;
  let warned = 0;
  console.warn = () => { warned++; };
  let result: any;
  try { result = fn(...args); } finally { console.warn = original; }
  if (warned) throw new AssertionError(`Got warnings`);
  return result;
}

/** There is no reference counting to check in JS: runs the function */
export function assert_no_gc_cycles(fn: (...args: any[]) => any, ...args: any[]): any { return fn(...args); }

/** Applies `decorator` to the methods of a class whose names match `testmatch` (default: /(?:^|[\\b_\\.%-])[Tt]est/) */
export function decorate_methods(cls: any, decorator: (f: any) => any, testmatch: RegExp | string | null = null): void {
  const re = testmatch === null ? /(?:^|[\b_.%-])[Tt]est/ : (typeof testmatch === "string" ? new RegExp(testmatch) : testmatch);
  const proto = cls.prototype || cls;
  for (const name of Object.getOwnPropertyNames(proto)) {
    if (name === "constructor" || !re.test(name)) continue;
    if (typeof proto[name] === "function") proto[name] = decorator(proto[name]);
  }
}
