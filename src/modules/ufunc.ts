//@ts-check
/**
 * Methods of the binary ufuncs: np.add.reduce, np.add.accumulate, np.add.outer, np.add.reduceat and np.add.at.
 * They are built on top of the (vectorised) binary functions, folding slice by slice along the reduced axis.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { np } from "../np_circular_import";
import { TYPE_BY_NAME, DType, bool } from "../dtypes";
import { template_mode } from "../kwargs/kwargs";

const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

type Op = (a: any, b: any) => any;

function norm_axes(axis: any, nd: number): number[] {
  if (axis === null || axis === undefined) return Array.from({ length: nd }, (_, i) => i);
  const list = Array.isArray(axis) ? axis : [axis];
  const out = list.map((a: number) => { const x = a < 0 ? a + nd : a; if (x < 0 || x >= nd) throw new Error(`axis ${a} is out of bounds for array of dimension ${nd}`); return x; });
  if (new Set(out).size !== out.length) throw new Error("duplicate value in 'axis'");
  return out;
}
/** add and multiply promote small integers (and bool) to the platform integer */
function promote(a: NDArray, name: string, dtype: DType | null): NDArray {
  if (dtype) return np.astype(a, dtype);
  if (name !== 'add' && name !== 'multiply') return a;
  const t = TYPE_BY_NAME[a.dtype];
  if (t.kind === 'b' || (t.kind === 'i' && t.bits < 64)) return np.astype(a, 'int64');
  if (t.kind === 'u' && t.bits < 64) return np.astype(a, 'uint64');
  return a;
}
const arr0 = (x: any) => asarray(x);
/** a.take(i, axis) as an array (also for 0-d results, keeping the dtype) */
function pick(a: NDArray, i: number, axis: number): NDArray {
  const r = np.take(a, [i], axis) as NDArray;
  const shape = a.shape.filter((_, k) => k !== (axis < 0 ? axis + a.shape.length : axis));
  return new NDArray(r.flat as any, shape, r.dtype);
}
const COMPARISONS = ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'];
const REORDERABLE = ['add', 'multiply', 'maximum', 'minimum', 'fmax', 'fmin', 'logical_and', 'logical_or', 'logical_xor', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'gcd', 'hypot', 'logaddexp', 'logaddexp2', ...COMPARISONS];

export function make_methods(name: string, op: Op, identity: any) {
  /** the dtype of op(x, x): the first element of every fold is converted to it (e.g. ints -> floats for hypot, -> bool for logical_and) */
  function first_cast(a: NDArray): NDArray {
    if (COMPARISONS.includes(name) && a.dtype !== bool) throw new TypeError(`No loop matching the specified signature and casting was found for ufunc ${name}`);
    const one = np.take(np.ravel(a), [0]), probe = asarray(op(one, one)); // shape [1]: a 0-d result would come back as a JS number
    return probe.dtype === a.dtype ? a : np.astype(a, probe.dtype);
  }
  function reduce(array: any, axis: any = 0, dtype: DType | null = null, out: any = null, keepdims: boolean = false, initial: any = null, where: any = true) {
    if (where !== true) throw new Error("the `where` argument of ufunc.reduce is not supported");
    let a = asarray(array);
    const nd = a.shape.length;
    if (nd === 0) return a.item();
    a = promote(a, name, dtype);
    if (a.size > 0) a = first_cast(a);
    const axes = norm_axes(axis, nd);
    if (axes.length > 1 && !REORDERABLE.includes(name)) throw new Error(`reduction operation '${name}' is not reorderable, so at most one axis may be specified`);
    const keep = Array.from({ length: nd }, (_, i) => i).filter(i => !axes.includes(i));
    const rest0 = keep.map(i => a.shape[i]), N = prod(axes.map(i => a.shape[i]));
    const rest = rest0.length ? rest0 : [1]; // folds work on arrays (a 0-d result would lose its dtype)
    const moved = np.reshape(np.transpose(a, [...keep, ...axes]), [...rest, N]);
    let acc: any, start = 0;
    if (initial !== null && initial !== undefined) acc = np.full(rest, initial, undefined);
    else if (N > 0) { acc = pick(moved, 0, -1); start = 1; }
    else if (identity !== undefined) acc = np.full(rest, identity, (name.startsWith('logical') ? 'bool' : ['hypot', 'logaddexp', 'logaddexp2'].includes(name) ? 'float64' : a.dtype));
    else throw new Error(`zero-size array to reduction operation ${name} which has no identity`);
    if (initial !== null && initial !== undefined && !isarray(acc)) acc = arr0(acc);
    for (let i = start; i < N; i++) acc = op(acc, pick(moved, i, -1));
    let res = arr0(acc);
    res = rest0.length ? np.reshape(res, rest0) : new NDArray(res.flat as any, [], res.dtype);
    if (keepdims) { const shape = a.shape.map((d, i) => axes.includes(i) ? 1 : d); res = np.reshape(res, shape); }
    if (out !== null && out !== undefined) { np.copyto(out, res); return out; }
    if (res.shape.length !== 0 || template_mode.depth > 0) return res;
    return res.dtype === 'bool' ? !!res.item() : res.item();
  }
  function accumulate(array: any, axis: number = 0, dtype: DType | null = null, out: any = null) {
    let a = promote(asarray(array), name, dtype);
    if (a.shape.length === 0) throw new Error("cannot accumulate on a scalar");
    if (a.size > 0) a = first_cast(a);
    const ax = norm_axes(axis, a.shape.length)[0];
    const n = a.shape[ax];
    if (n === 0) return a;
    if (a.shape.length === 1) { // fold over [n, 1] so that every step is an array
      const r = accumulate(np.reshape(a, [n, 1]), 0, null, null);
      const flat = np.reshape(r, [n]);
      if (out !== null && out !== undefined) { np.copyto(out, flat); return out; }
      return flat;
    }
    const parts: any[] = [];
    let acc: any = pick(a, 0, ax);
    parts.push(acc);
    for (let i = 1; i < n; i++) { acc = op(acc, pick(a, i, ax)); parts.push(acc); }
    const res = np.stack(parts.map(arr0), ax);
    if (out !== null && out !== undefined) { np.copyto(out, res); return out; }
    return res;
  }
  function outer(A: any, B: any, out: any = null) {
    const a = asarray(A), b = asarray(B);
    const res = op(np.reshape(a, [...a.shape, ...b.shape.map(() => 1)]), b);
    if (out !== null && out !== undefined) { np.copyto(out, res); return out; }
    return res;
  }
  function reduceat(array: any, indices: any, axis: number = 0, dtype: DType | null = null, out: any = null) {
    let a = promote(asarray(array), name, dtype);
    if (a.size > 0) a = first_cast(a);
    const ax = norm_axes(axis, a.shape.length)[0];
    const idx = Array.from(asarray(indices).flat as any, (v: any) => +v);
    const n = a.shape[ax];
    if (a.shape.length === 1) {
      const r = reduceat(np.reshape(a, [n, 1]), indices, 0, null, null);
      const flat = np.reshape(r, [idx.length]);
      if (out !== null && out !== undefined) { np.copyto(out, flat); return out; }
      return flat;
    }
    const parts = idx.map((s, k) => {
      if (s < 0 || s >= n) throw new Error(`index ${s} out-of-bounds in ${name}.reduceat [0, ${n})`);
      const e = k < idx.length - 1 ? idx[k + 1] : n;
      if (s >= e) return pick(a, s, ax);
      const sl = np.take(a, np.arange(s, e), ax);
      return reduce(sl, ax, null, null, false);
    });
    const res = np.stack(parts.map(arr0), ax);
    if (out !== null && out !== undefined) { np.copyto(out, res); return out; }
    return res;
  }
  function at(a: any, indices: any, b: any = null) {
    if (!isarray(a)) throw new TypeError("first operand must be an array");
    const idx = Array.isArray(indices) || isarray(indices) ? Array.from(asarray(indices).flat as any, (v: any) => +v) : [+indices];
    const vals = b === null ? null : asarray(b);
    const n = a.shape[0];
    idx.forEach((i, k) => {
      const j = i < 0 ? i + n : i;
      if (j < 0 || j >= n) throw new Error(`index ${i} is out of bounds for axis 0 with size ${n}`);
      const cur = pick(a, j, 0);
      const v = vals === null ? null : (vals.shape.length === 0 ? vals : np.take(vals, k % vals.shape[0], 0));
      const res = v === null ? op(cur, undefined) : op(cur, v);
      np.copyto(np.take(a, [j], 0), np.reshape(arr0(res), [1, ...a.shape.slice(1)]));
    });
  }
  return { reduce, accumulate, outer, reduceat, at };
}

/** Name -> identity (undefined when there is none) */
export const IDENTITIES: { [name: string]: any } = {
  add: 0, subtract: undefined, multiply: 1, divide: undefined, true_divide: undefined, floor_divide: undefined, power: undefined, mod: undefined, remainder: undefined, fmod: undefined,
  maximum: undefined, minimum: undefined, fmax: undefined, fmin: undefined, hypot: 0, arctan2: undefined, copysign: undefined, nextafter: undefined, logaddexp: -Infinity, logaddexp2: -Infinity,
  logical_and: true, logical_or: false, logical_xor: false, bitwise_and: -1, bitwise_or: 0, bitwise_xor: 0, left_shift: undefined, right_shift: undefined, gcd: 0, lcm: undefined,
  equal: undefined, not_equal: undefined, less: undefined, less_equal: undefined, greater: undefined, greater_equal: undefined, heaviside: undefined, float_power: undefined, ldexp: undefined,
};
