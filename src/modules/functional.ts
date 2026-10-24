//@ts-check
import { NDArray, asarray, isarray, broadcast_n_shapes, broadcast_strides } from "../NDArray";
import { DType, bool, int32, float64, object, new_buffer, convert_buffer, fix_buffer, dtype_max, same_out, float_out, sum_out, truthy } from "../dtypes";
import { normalize_axis } from "../array/transform";
import { expand_dims, broadcast_to, reindex } from "./manipulation";
import { map_broadcast, finish_output, n_ary_elementwise } from "../array/operators";
import { ravel, reshape } from "../array/shape_operations";
import * as sets from "./sets";
import { Complex } from "../complex";
import { complex128 } from "../dtypes";
import { applyFuncToArray } from "../array/elementwise";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export function apply_over_axes(func: (a: NDArray, axis: number) => NDArray, a: NDArray, axes: number | number[]): NDArray {
  let val = asarray(a);
  const N = val.shape.length;
  for (let axis of (Array.isArray(axes) ? axes : [axes])) {
    if (axis < 0) axis = N + axis;
    let res = asarray(func(val, axis));
    if (res.shape.length != val.shape.length) res = expand_dims(res, axis);
    if (res.shape.length != val.shape.length) throw new Error(`function is not returning an array of the correct shape`);
    val = res;
  }
  return val;
}

export interface VectorizeOptions { otypes?: DType[] | string | null, excluded?: (number | string)[] | null, doc?: string | null, cache?: boolean, signature?: string | null }

/**
 * Generalized function class: returns a function that applies `pyfunc` elementwise (with broadcasting).
 * Arguments listed in `excluded` (by position) are passed as they are.
 */
export function vectorize(pyfunc: (...args: any[]) => any, options: VectorizeOptions = {}) {
  if (options.signature) throw new Error(`vectorize with a gufunc signature is not supported`);
  const excluded = new Set(options.excluded || []);
  const otypes = typeof options.otypes === "string" ? options.otypes.split("") as any : options.otypes;
  const vectorized = (...args: any[]) => {
    const positions = args.map((_, i) => i).filter(i => !excluded.has(i));
    const arrays = positions.map(i => asarray(args[i]));
    if (!arrays.length) return pyfunc(...args);
    const [, shape] = broadcast_n_shapes(...arrays.map(a => a.shape));
    const shp = shape as number[];
    const results: any[] = new Array(prod(shp));
    const strides = arrays.map(a => broadcast_strides(shp, a.shape));
    const call = (...vals: any[]) => {
      const full = args.slice();
      positions.forEach((p, k) => { full[p] = vals[k]; });
      return pyfunc(...full);
    };
    map_broadcast(shp, strides, arrays.map(a => (a.dtype === bool ? Array.from(a.flat as any, (x: any) => !!x) : a.flat) as any), call, results);
    if (results.length == 0 && !otypes) throw new Error(`cannot call \`vectorize\` on size 0 inputs unless \`otypes\` is set`);
    const out = asarray(results.length ? results.map(r => (isarray(r) ? r.item() : r)) : [], otypes ? otypes[0] : undefined);
    return reshape(out, shp);
  };
  return vectorized;
}

/** Takes an arbitrary function and returns a function that applies it elementwise, returning object arrays */
export function frompyfunc(func: (...args: any[]) => any, nin: number, nout: number, identity: any = null) {
  const f = (...args: any[]) => {
    if (args.length != nin) throw new Error(`function takes exactly ${nin} arguments (${args.length} given)`);
    const arrays = args.map(a => asarray(a));
    const [, shape] = broadcast_n_shapes(...arrays.map(a => a.shape));
    const shp = shape as number[];
    const results: any[] = new Array(prod(shp));
    map_broadcast(shp, arrays.map(a => broadcast_strides(shp, a.shape)), arrays.map(a => (a.dtype === bool ? Array.from(a.flat as any, (x: any) => !!x) : a.flat) as any), func, results);
    if (nout == 1) return new NDArray(results, shp, object);
    return range(nout).map(k => new NDArray(results.map(r => r[k]), shp, object));
  };
  return Object.assign(f, { nin, nout, identity });
}

/** Evaluates a list of functions (or constant values) on the elements of x selected by the boolean arrays in condlist */
export function piecewise(x: NDArray, condlist: any[], funclist: any[], args: any[], kw: { [k: string]: any }): NDArray {
  x = asarray(x);
  let conds: NDArray[] = (isarray(condlist) && condlist.dtype === bool && condlist.shape.length == x.shape.length) ? [condlist] : (condlist as any[]).map(c => asarray(c));
  if (x.shape.length > 0 && conds.length > 0 && conds.every(c => c.shape.length == 0)) conds = conds.map(c => broadcast_to(c, x.shape));
  if (conds.length == 1 && x.shape.length > 0 && conds[0].dtype === bool && conds[0].shape.length == x.shape.length + 1) conds = range(conds[0].shape[0]).map(i => asarray(conds[0].index(i)));
  const n = conds.length;
  let funcs = funclist.slice();
  if (funcs.length == n + 1) {
    const otherwise = conds.reduce((acc, c) => { const f = c.flat as any; return acc.map((v, i) => v && !truthy(f[i])); }, new Array(x.size).fill(true));
    conds = [...conds, new NDArray(new_buffer(otherwise.map(Number), bool), x.shape, bool)];
  } else if (funcs.length != n) throw new Error(`with ${n} condition(s), either ${n} or ${n + 1} functions are expected`);
  const out = new NDArray(new_buffer(x.size, x.dtype), x.shape, x.dtype);
  const xf = x.flat as any;
  const buffer = out._flat as any;
  conds.forEach((c, k) => {
    const cf = c.flat as any;
    const pos: number[] = [];
    for (let i = 0; i < cf.length; i++) if (truthy(cf[i])) pos.push(i);
    if (!pos.length) return;
    const f = funcs[k];
    if (typeof f === "function") {
      const sel = new NDArray(new_buffer(pos.map(i => xf[i]), x.dtype), [pos.length], x.dtype);
      const r = asarray(f(sel, ...args, ...(Object.keys(kw).length ? [kw] : [])));
      const rv = r.flat as any;
      pos.forEach((i, j) => { buffer[i] = rv[r.size == 1 ? 0 : j]; });
    } else {
      const v = asarray(f).item();
      pos.forEach(i => { buffer[i] = v; });
    }
  });
  fix_buffer(buffer, x.dtype);
  return out;
}

// ==============================
//   Iteration
// ==============================

/** Iterator over all the indices of an array of the given shape (C order): yields number[] */
export function* ndindex(shape: number[]): Generator<number[]> {
  const n = prod(shape);
  const idx = new Array(shape.length).fill(0);
  for (let i = 0; i < n; i++) {
    yield idx.slice();
    for (let ax = shape.length - 1; ax >= 0; ax--) { if (++idx[ax] < shape[ax]) break; idx[ax] = 0; }
  }
}

/** Yields [index, value] for every element of the array */
export function* ndenumerate(arr: NDArray): Generator<[number[], any]> {
  arr = asarray(arr);
  const flat = arr.flat as any;
  let i = 0;
  for (const idx of ndindex(arr.shape)) { yield [idx, arr.dtype === bool ? !!flat[i] : flat[i]]; i++; }
}

/** Iterator over the elements of an array (like arr.flat in numpy) */
export class flatiter {
  base: NDArray;
  index = 0;
  private values: any;
  constructor(arr: NDArray) {
    this.base = asarray(arr);
    this.values = this.base.flat;
  }
  get length() { return this.base.size; }
  get coords(): number[] {
    const out: number[] = []; let v = this.index;
    for (let d = this.base.shape.length - 1; d >= 0; d--) { out[d] = v % this.base.shape[d]; v = Math.floor(v / this.base.shape[d]); }
    return out;
  }
  get(i: number) { const n = this.length; if (i < -n || i >= n) throw new Error(`index ${i} is out of bounds for size ${n}`); const v = this.values[i < 0 ? i + n : i]; return this.base.dtype === bool ? !!v : v; }
  set(i: number, value: any) { const n = this.length; if (i < -n || i >= n) throw new Error(`index ${i} is out of bounds for size ${n}`); this.values[i < 0 ? i + n : i] = +value; this.base.flat = this.values; }
  next(): IteratorResult<any> {
    if (this.index >= this.length) return { done: true, value: undefined };
    return { done: false, value: this.get(this.index++) };
  }
  copy(): NDArray { return ravel(new NDArray(this.values.slice(), this.base.shape, this.base.dtype)); }
  [Symbol.iterator]() { this.index = 0; return this as any; }
}

/** Iterates over the (broadcast) elements of one or several arrays in C order. Yields a value, or a list of values for several operands. */
export class nditer {
  shape: number[];
  ndim: number;
  itersize: number;
  nop: number;
  private values: any[][];
  constructor(op: NDArray | NDArray[], flags: string[] | null = null) {
    const ops = (Array.isArray(op) ? op : [op]).map(o => asarray(o));
    const [, shape] = broadcast_n_shapes(...ops.map(o => o.shape));
    this.shape = shape as number[];
    this.ndim = this.shape.length;
    this.itersize = prod(this.shape);
    this.nop = ops.length;
    this.values = ops.map(o => {
      const b = broadcast_to(o, this.shape);
      const f = b.flat as any;
      return o.dtype === bool ? Array.from(f, (x: any) => !!x) : Array.from(f);
    });
    this.single = !Array.isArray(op);
  }
  private single: boolean;
  *[Symbol.iterator]() {
    for (let i = 0; i < this.itersize; i++) yield this.single ? this.values[0][i] : this.values.map(v => v[i]);
  }
}

/** Nested iterators: nested_iters(arr, [[0], [1]]) iterates axis 0 in the outer iterator and axis 1 in the inner one. */
export function nested_iters(op: NDArray, axes: number[][]): any[] {
  const arr = asarray(op);
  const nd = arr.shape.length;
  const norm = axes.map(g => g.map(a => normalize_axis(a, nd)));
  const flat = norm.flat();
  if (new Set(flat).size != flat.length) throw new Error(`axes must be unique`);
  const level = (fixed: { [axis: number]: number }, depth: number): any => {
    const group = norm[depth];
    const sizes = group.map(a => arr.shape[a]);
    return {
      *[Symbol.iterator]() {
        for (const idx of ndindex(sizes)) {
          const next = { ...fixed };
          group.forEach((a, k) => { next[a] = idx[k]; });
          if (depth == norm.length - 1) {
            const where = arr.shape.map((_, a) => (a in next) ? next[a] : ':');
            const v = arr.index(...where);
            yield isarray(v) ? v : v;
          } else yield level(next, depth + 1);
        }
      }
    };
  };
  return [level({}, 0)];
}

export function iterable(y: any): boolean {
  if (y === null || y === undefined) return false;
  try { return typeof (Object(y) as any)[Symbol.iterator] === "function"; } catch { return false; }
}

// ==============================
//   Domain-restricted math (numpy.emath)
// ==============================
// numpy.emath returns complex numbers outside of the real domain: the result is real when every input is
// in the domain of the real function, complex otherwise.

function emath(domain: (x: number) => boolean, f: (x: number) => number, cf: (z: Complex) => Complex) {
  return (x: any) => {
    const a = asarray(x);
    if (a.dtype === complex128) return applyFuncToArray(float_out, f, a, null, cf);
    const flat = a.flat as any;
    let inDomain = true;
    for (let i = 0; i < flat.length; i++) if (!domain(+flat[i]) && !Number.isNaN(+flat[i])) { inDomain = false; break; }
    return applyFuncToArray(inDomain ? float_out : (() => complex128), f, a, null, cf);
  };
}
const ln = (z: Complex) => z.log();
export const emath_sqrt = emath(x => x >= 0, Math.sqrt, z => z.sqrt());
export const emath_log = emath(x => x >= 0, Math.log, ln);
export const emath_log2 = emath(x => x >= 0, Math.log2, z => ln(z).div(new Complex(Math.LN2, 0)));
export const emath_log10 = emath(x => x >= 0, Math.log10, z => ln(z).div(new Complex(Math.LN10, 0)));
export const emath_arccos = emath(x => Math.abs(x) <= 1, Math.acos, z => z.acos());
export const emath_arcsin = emath(x => Math.abs(x) <= 1, Math.asin, z => z.asin());
export const emath_arctanh = emath(x => Math.abs(x) <= 1, Math.atanh, z => z.atanh());
export function emath_logn(n: any, x: any) {
  const nn = asarray(n), xx = asarray(x);
  const bad = Array.from(nn.flat as any).some((v: any) => v < 0) || Array.from(xx.flat as any).some((v: any) => v < 0) || nn.dtype === complex128 || xx.dtype === complex128;
  if (!bad) return n_ary_elementwise(float_out, (v: number, b: number) => Math.log(v) / Math.log(b), [xx, nn]);
  return n_ary_elementwise(() => complex128, (v: any, b: any) => Complex.from(v).log().div(Complex.from(b).log()), [xx, nn]);
}
export function emath_power(x: any, p: any) {
  const xx = asarray(x), pp = asarray(p);
  const bad = xx.dtype === complex128 || pp.dtype === complex128 || Array.from(xx.flat as any).some((v: any) => v < 0);
  if (!bad) return n_ary_elementwise(float_out, (v: number, e: number) => Math.pow(v, e), [xx, pp]);
  return n_ary_elementwise(() => complex128, (v: any, e: any) => Complex.from(v).pow(Complex.from(e)), [xx, pp]);
}
