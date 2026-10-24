//@ts-check
import { NDArray, asarray, isarray, shape_shifts, broadcast_shapes, broadcast_n_shapes, broadcast_strides } from "../NDArray";
import { DType, bool, int32, int64, float64, new_buffer, convert_buffer, fix_buffer, dtype_max, dtype_is_integer, truthy, same_out, weak_dtype } from "../dtypes";
import { normalize_axis, transpose, swapAxes, concatenate } from "../array/transform";
import { ravel, reshape } from "../array/shape_operations";
import { n_ary_elementwise, finish_output, map_broadcast } from "../array/operators";
import { reindex, tri, expand_dims, broadcast_to } from "./manipulation";
import { nonzero } from "./indexing";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const flat_list = (v: any): number[] => Array.from(asarray(v).flat as any).map(x => +(x as any));

export type IndexMode = 'raise' | 'wrap' | 'clip';

function fix_index(i: number, n: number, mode: IndexMode, what = 'index'): number {
  if (!Number.isInteger(i)) throw new Error(`${what} ${i} is not an integer`);
  if (mode === 'wrap') return n == 0 ? 0 : ((i % n) + n) % n;
  if (mode === 'clip') return Math.min(Math.max(i, 0), n - 1);
  if (mode !== 'raise') throw new Error(`clipmode must be one of 'clip', 'raise', or 'wrap' (got '${mode}')`);
  if (i < -n || i >= n) throw new Error(`${what} ${i} is out of bounds for axis with size ${n}`);
  return i < 0 ? i + n : i;
}

// ==============================
//   take and friends
// ==============================

/** Takes elements from an array along an axis (axis=null: from the flattened array). The result has the shape of `indices` in place of the axis. */
export function take(a: NDArray, indices: NDArray, axis: number | null, out: NDArray | null, mode: IndexMode): NDArray {
  a = asarray(a);
  const idx = asarray(indices);
  if (axis === null) { a = ravel(a); axis = 0; }
  if (a.shape.length == 0) throw new Error(`cannot take from a 0-d array`);
  axis = normalize_axis(axis, a.shape.length);
  const n = a.shape[axis];
  const outer = prod(a.shape.slice(0, axis)), inner = prod(a.shape.slice(axis + 1));
  const ind = flat_list(idx).map(i => fix_index(i, n, mode));
  if (n == 0 && ind.length > 0) throw new Error(`cannot do a non-empty take from an empty axes.`);
  const m = ind.length;
  const flat = a.flat as any;
  const buffer = new_buffer(outer * m * inner, a.dtype);
  for (let o = 0; o < outer; o++) for (let j = 0; j < m; j++) {
    const src = (o * n + ind[j]) * inner, dst = (o * m + j) * inner;
    for (let k = 0; k < inner; k++) buffer[dst + k] = flat[src + k];
  }
  return finish_output(buffer, [...a.shape.slice(0, axis), ...idx.shape, ...a.shape.slice(axis + 1)], a.dtype, out);
}

function odometer(shape: number[], cb: (i: number, idx: number[]) => void) {
  const n = prod(shape);
  const idx = new Array(shape.length).fill(0);
  for (let i = 0; i < n; i++) {
    cb(i, idx);
    for (let ax = shape.length - 1; ax >= 0; ax--) { if (++idx[ax] < shape[ax]) break; idx[ax] = 0; }
  }
}

/** Shapes and strides to address arr[..., indices, ...] along an axis (other axes broadcast between arr and indices) */
function along_axis_setup(arr: NDArray, indices: NDArray, axis: number) {
  if (arr.shape.length != indices.shape.length) throw new Error(`\`indices\` and \`arr\` must have the same number of dimensions`);
  const nd = arr.shape.length;
  const outShape: number[] = [];
  for (let d = 0; d < nd; d++) {
    if (d == axis) { outShape.push(indices.shape[d]); continue; }
    const a = arr.shape[d], b = indices.shape[d];
    if (a !== b && a !== 1 && b !== 1) throw new Error(`shape mismatch: indexing arrays could not be broadcast together with shapes (${arr.shape}) (${indices.shape})`);
    outShape.push(a === 1 ? b : a);
  }
  const sa = shape_shifts(arr.shape), si = shape_shifts(indices.shape);
  const strideA = arr.shape.map((d, i) => i == axis ? sa[i] : (d == 1 ? 0 : sa[i]));
  const strideI = indices.shape.map((d, i) => i == axis ? si[i] : (d == 1 ? 0 : si[i]));
  return { outShape, strideA, strideI, axisStride: sa[axis] };
}

export function take_along_axis(arr: NDArray, indices: NDArray, axis: number | null): NDArray {
  arr = asarray(arr); indices = asarray(indices);
  if (axis === null) {
    if (indices.shape.length != 1) throw new Error(`when axis=None, \`indices\` must have a single dimension.`);
    arr = ravel(arr); axis = 0;
  }
  if (!dtype_is_integer(indices.dtype)) throw new Error(`\`indices\` must be an integer array`);
  axis = normalize_axis(axis, arr.shape.length);
  const { outShape, strideA, strideI, axisStride } = along_axis_setup(arr, indices, axis);
  const n = arr.shape[axis];
  const flat = arr.flat as any, ind = indices.flat as any;
  const buffer = new_buffer(prod(outShape), arr.dtype);
  odometer(outShape, (i, idx) => {
    let pa = 0, pi = 0;
    for (let d = 0; d < idx.length; d++) {
      if (d != axis) pa += idx[d] * strideA[d];
      pi += idx[d] * strideI[d];
    }
    buffer[i] = flat[pa + fix_index(ind[pi], n, 'raise') * axisStride];
  });
  return new NDArray(buffer, outShape, arr.dtype);
}

export function put_along_axis(arr: NDArray, indices: NDArray, values: NDArray, axis: number | null): NDArray {
  indices = asarray(indices); values = asarray(values);
  let target = arr;
  if (axis === null) {
    if (indices.shape.length != 1) throw new Error(`when axis=None, \`indices\` must have a single dimension.`);
    target = ravel(arr); axis = 0;
  }
  axis = normalize_axis(axis, target.shape.length);
  const { outShape, strideA, strideI, axisStride } = along_axis_setup(target, indices, axis);
  const n = target.shape[axis];
  const flat = target.flat as any, ind = indices.flat as any;
  const vals = broadcast_to(values, outShape).flat as any;
  odometer(outShape, (i, idx) => {
    let pa = 0, pi = 0;
    for (let d = 0; d < idx.length; d++) {
      if (d != axis) pa += idx[d] * strideA[d];
      pi += idx[d] * strideI[d];
    }
    flat[pa + fix_index(ind[pi], n, 'raise') * axisStride] = vals[i];
  });
  fix_buffer(flat, target.dtype);
  arr.flat = flat;
  return arr;
}

export function choose(a: NDArray, choices: NDArray[] | NDArray, out: NDArray | null, mode: IndexMode): NDArray {
  a = asarray(a);
  const ch: NDArray[] = isarray(choices) ? range(choices.shape[0]).map(i => asarray(choices.index(i))) : (choices as any[]).map(c => asarray(c));
  if (!ch.length) throw new Error(`0-length sequence.`);
  const n = ch.length;
  return n_ary_elementwise((dtypes) => dtype_max(...dtypes.slice(1)), (idx: number, ...vals: any[]) => vals[fix_index(idx, n, mode, 'invalid entry in choice array at index')], [a, ...ch], out);
}

export function compress(condition: NDArray, a: NDArray, axis: number | null, out: NDArray | null): NDArray {
  condition = asarray(condition); a = asarray(a);
  if (condition.shape.length != 1) throw new Error(`condition must be a 1-d array`);
  if (axis === null) { a = ravel(a); axis = 0; }
  axis = normalize_axis(axis, a.shape.length);
  const c = condition.flat as any;
  const keep = range(Math.min(c.length, a.shape[axis])).filter(i => truthy(c[i]));
  const res = reindex(a, a.shape.map((_, ax) => ax == axis ? keep : null));
  return finish_output(res.flat, res.shape, a.dtype, out);
}

export function diagonal(a: NDArray, offset: number, axis1: number, axis2: number): NDArray {
  a = asarray(a);
  const nd = a.shape.length;
  if (nd < 2) throw new Error(`diag requires an array of at least two dimensions`);
  axis1 = normalize_axis(axis1, nd); axis2 = normalize_axis(axis2, nd);
  if (axis1 == axis2) throw new Error(`axis1 and axis2 cannot be the same`);
  const others = range(nd).filter(i => i != axis1 && i != axis2);
  const t = transpose(a, [...others, axis1, axis2]);
  const [rows, cols] = t.shape.slice(-2);
  const r0 = offset >= 0 ? 0 : -offset, c0 = offset >= 0 ? offset : 0;
  const len = Math.max(0, Math.min(rows - r0, cols - c0));
  const lead = prod(t.shape.slice(0, -2));
  const flat = t.flat as any;
  const buffer = new_buffer(lead * len, a.dtype);
  for (let l = 0; l < lead; l++) for (let i = 0; i < len; i++) buffer[l * len + i] = flat[l * rows * cols + (r0 + i) * cols + (c0 + i)];
  return new NDArray(buffer, [...t.shape.slice(0, -2), len], a.dtype);
}

export function select(condlist: NDArray[], choicelist: NDArray[], dflt: any): NDArray {
  if (condlist.length != choicelist.length) throw new Error(`list of cases must be same length as list of conditions`);
  if (!condlist.length) throw new Error(`select with an empty condition list is not possible`);
  const conds = condlist.map(c => { c = asarray(c); if (c.dtype !== bool) throw new Error(`invalid entry ${condlist.indexOf(c)} in condlist: should be boolean ndarray`); return c; });
  const choices = choicelist.map(c => asarray(c));
  const d = asarray(dflt);
  const k = conds.length;
  return n_ary_elementwise(
    (dtypes) => dtype_max(...dtypes.slice(k)),
    (...vals: any[]) => { for (let i = 0; i < k; i++) if (vals[i]) return vals[k + i]; return vals[2 * k]; },
    [...conds, ...choices, d]);
}

// ==============================
//   In place scatter operations
// ==============================

function write_flat(arr: NDArray, fn: (flat: any) => void) {
  const flat = arr.flat as any;
  fn(flat);
  fix_buffer(flat, arr.dtype);
  arr.flat = flat;
}

export function place(arr: NDArray, mask: NDArray, vals: NDArray): void {
  mask = asarray(mask); vals = asarray(vals);
  if (mask.size != arr.size) throw new Error(`place: mask and data must be the same size`);
  const v = vals.flat as any, m = mask.flat as any;
  if (v.length == 0) { if (m.some((x: any) => truthy(x))) throw new Error(`Cannot insert from an empty array!`); return; }
  write_flat(arr, (flat) => { let k = 0; for (let i = 0; i < flat.length; i++) if (truthy(m[i])) flat[i] = v[k++ % v.length]; });
}

export function put(a: NDArray, ind: NDArray, v: NDArray, mode: IndexMode): void {
  const idx = flat_list(ind), vals = asarray(v).flat as any;
  if (!idx.length) return;
  if (vals.length == 0) throw new Error(`cannot put from an empty array`);
  write_flat(a, (flat) => { idx.forEach((i, k) => { flat[fix_index(i, flat.length, mode)] = vals[k % vals.length]; }); });
}

export function putmask(a: NDArray, mask: NDArray, values: NDArray): void {
  mask = asarray(mask);
  if (mask.size != a.size) throw new Error(`putmask: mask and data must be the same size`);
  const v = asarray(values).flat as any, m = mask.flat as any;
  if (v.length == 0) return;
  write_flat(a, (flat) => { for (let i = 0; i < flat.length; i++) if (truthy(m[i])) flat[i] = v[i % v.length]; });
}

export function fill_diagonal(a: NDArray, val: NDArray, wrap: boolean): void {
  if (a.shape.length < 2) throw new Error(`array must be at least 2-d`);
  const nd = a.shape.length;
  if (nd > 2 && a.shape.some(d => d != a.shape[0])) throw new Error(`All dimensions of input must be of equal length`);
  const strides = shape_shifts(a.shape);
  const step = strides.reduce((s, x) => s + x, 0);
  const v = asarray(val).flat as any;
  const end = (nd == 2 && !wrap) ? Math.min(a.size, a.shape[1] * a.shape[1]) : a.size;
  write_flat(a, (flat) => { let k = 0; for (let i = 0; i < end; i += step) flat[i] = v[k++ % v.length]; });
}

// ==============================
//   Index generators
// ==============================

export function indices(dimensions: number[], dtype: DType, sparse: boolean): NDArray | NDArray[] {
  const nd = dimensions.length;
  const make = (k: number, shape: number[]) => {
    const buffer = new_buffer(prod(shape), dtype);
    const strides = shape.map((_, i) => i == k ? 1 : 0);
    const base = new_buffer(dimensions[k], dtype);
    for (let i = 0; i < base.length; i++) base[i] = i;
    map_broadcast(shape, [strides], [base], (x) => x, buffer);
    return new NDArray(buffer, shape, dtype);
  };
  if (sparse) return range(nd).map(k => { const s = dimensions.map((_, i) => i == k ? dimensions[k] : 1); return make(k, s); });
  const grids = range(nd).map(k => make(k, dimensions));
  return concatenate(grids.map(g => reshape(g, [1, ...dimensions])), 0);
}

export function ix_(args: NDArray[]): NDArray[] {
  const nd = args.length;
  return args.map((a, k) => {
    let arr = asarray(a);
    if (arr.shape.length != 1 && arr.size > 0 || arr.shape.length > 1) throw new Error(`Cross index must be 1 dimensional`);
    if (arr.size == 0) arr = new NDArray(new_buffer(0, int64), [0], int64);
    if (arr.dtype === bool) arr = nonzero(arr)[0];
    return reshape(arr, range(nd).map(i => i == k ? arr.size : 1));
  });
}

export function unravel_index(idx: NDArray, shape: number[] | number, order: 'C' | 'F'): NDArray[] {
  const dims = typeof shape === "number" ? [shape] : Array.from(shape as any) as number[];
  const arr = asarray(idx);
  if (!dtype_is_integer(arr.dtype) && arr.dtype !== bool && arr.size > 0) throw new Error(`only int indices permitted`);
  const total = prod(dims);
  const f = arr.flat as any;
  const outs = dims.map(() => new_buffer(f.length, int64));
  const order_dims = order === 'C' ? range(dims.length).reverse() : range(dims.length);
  for (let i = 0; i < f.length; i++) {
    let v = f[i];
    if (v < 0 || v >= total) throw new Error(`index ${v} is out of bounds for array with size ${total}`);
    for (const d of order_dims) { outs[d][i] = v % dims[d]; v = Math.floor(v / dims[d]); }
  }
  return outs.map(b => new NDArray(b, arr.shape, int64));
}

export function ravel_multi_index(multi_index: NDArray[], dims: number[] | number, mode: IndexMode | IndexMode[], order: 'C' | 'F'): NDArray {
  const shape = typeof dims === "number" ? [dims] : Array.from(dims as any) as number[];
  const list: any[] = isarray(multi_index) ? range((multi_index as any).shape[0]).map(i => (multi_index as any).index(i)) : multi_index;
  const arrs = list.map(m => asarray(m));
  if (arrs.length != shape.length) throw new Error(`parameter multi_index must be a sequence of length ${shape.length}`);
  const [, bshape] = broadcast_n_shapes(...arrs.map(a => a.shape));
  const bs = bshape as number[];
  const modes = Array.isArray(mode) ? mode : shape.map(() => mode);
  const strides = new Array(shape.length);
  let acc = 1;
  for (const d of (order === 'C' ? range(shape.length).reverse() : range(shape.length))) { strides[d] = acc; acc *= shape[d]; }
  const flats = arrs.map(a => broadcast_to(a, bs).flat as any);
  const n = prod(bs);
  const out = new_buffer(n, int64);
  for (let i = 0; i < n; i++) {
    let pos = 0;
    for (let d = 0; d < shape.length; d++) {
      const v = flats[d][i];
      if (!Number.isInteger(v)) throw new Error(`multi_index entries must be integers`);
      pos += fix_index_ravel(v, shape[d], modes[d]) * strides[d];
    }
    out[i] = pos;
  }
  return new NDArray(out, bs, int64);
}
function fix_index_ravel(v: number, n: number, mode: IndexMode): number {
  if (mode === 'raise') { if (v < 0 || v >= n) throw new Error(`invalid entry in coordinates array`); return v; }
  return fix_index(v, n, mode);
}

export function diag_indices(n: number, ndim: number): NDArray[] {
  return range(ndim).map(() => new NDArray(new_buffer(range(n), int64), [n], int64));
}
export function diag_indices_from(arr: NDArray): NDArray[] {
  arr = asarray(arr);
  if (arr.shape.length < 2) throw new Error(`input array must be at least 2-d`);
  if (arr.shape.some(d => d != arr.shape[0])) throw new Error(`All dimensions of input must be of equal length`);
  return diag_indices(arr.shape[0], arr.shape.length);
}
export function mask_indices(n: number, mask_func: (m: NDArray, k: number) => NDArray, k: number): NDArray[] {
  const ones = new NDArray(new_buffer(n * n, int32).fill(1), [n, n], int32);
  return nonzero(asarray(mask_func(ones, k)));
}
export const tril_indices = (n: number, k: number, m: number | null) => nonzero(tri(n, m, k, bool));
export const triu_indices = (n: number, k: number, m: number | null) => {
  const lower = tri(n, m, k - 1, bool);
  return nonzero(new NDArray(lower.flat.map((x: number) => x ? 0 : 1), lower.shape, bool));
};
export function tril_indices_from(arr: NDArray, k: number) {
  arr = asarray(arr);
  if (arr.shape.length != 2) throw new Error(`input array must be 2-d`);
  return tril_indices(arr.shape[0], k, arr.shape[1]);
}
export function triu_indices_from(arr: NDArray, k: number) {
  arr = asarray(arr);
  if (arr.shape.length != 2) throw new Error(`input array must be 2-d`);
  return triu_indices(arr.shape[0], k, arr.shape[1]);
}
