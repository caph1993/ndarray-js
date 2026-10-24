//@ts-check
import { asarray, NDArray } from '../NDArray';
import { bool, bool_out, DType, DtypeResolver, float_out, int32, new_buffer, same_out, sum_out, argmax_out, HasDType, truthy, convert_buffer } from '../dtypes';
import { finish_output } from './operators';
import { normalize_axis, transpose } from './transform';

export type AxisArg = number | number[] | null;

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Normalizes an axis argument to a sorted list of unique valid axes (all axes for null) */
export function normalize_axes(axis: AxisArg, ndim: number): number[] {
  if (axis == null) return Array.from({ length: ndim }, (_, i) => i);
  const axes = (Array.isArray(axis) ? axis : [axis]).map(ax => normalize_axis(ax, ndim));
  if (new Set(axes).size != axes.length) throw new Error(`duplicate value in 'axis'`);
  return axes.sort((a, b) => a - b);
}

/**
 * Reduces an array along one or several axes. `lane_fn` receives each lane (1D slice with the reduced
 * elements, row-major order) and returns the reduced value.
 */
export function reduce_lanes(
  arr: NDArray,
  axis: AxisArg,
  keepdims: boolean,
  dtype: DType,
  lane_fn: (lane: any) => any,
  out: NDArray | DType | null = null,
): NDArray {
  arr = asarray(arr);
  const nd = arr.shape.length;
  const axes = normalize_axes(axis, nd);
  const kept: number[] = [];
  for (let i = 0; i < nd; i++) if (!axes.includes(i)) kept.push(i);
  const perm = [...kept, ...axes];
  const identity = perm.every((v, i) => v === i);
  const t = identity ? arr : transpose(arr, perm);
  const nLane = prod(axes.map(ax => arr.shape[ax]));
  const keptShape = kept.map(ax => arr.shape[ax]);
  const nOut = prod(keptShape);
  const flat = t.flat;
  const buffer = new_buffer(nOut, dtype);
  if (dtype === bool) for (let o = 0; o < nOut; o++) buffer[o] = lane_fn(flat.slice(o * nLane, (o + 1) * nLane)) ? 1 : 0;
  else for (let o = 0; o < nOut; o++) buffer[o] = lane_fn(flat.slice(o * nLane, (o + 1) * nLane));
  const shape = keepdims ? arr.shape.map((d, i) => axes.includes(i) ? 1 : d) : keptShape;
  return finish_output(buffer, shape, dtype, out);
}

// ==============================
//       Lane functions
// ==============================

export function lane_sum(lane: any) { let s = 0; for (let i = 0; i < lane.length; i++) s += +lane[i]; return s; }
export function lane_prod(lane: any) { let s = 1; for (let i = 0; i < lane.length; i++) s *= +lane[i]; return s; }
export function lane_nansum(lane: any) { let s = 0; for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v) s += v; } return s; }
export function lane_nanprod(lane: any) { let s = 1; for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v) s *= v; } return s; }
export function lane_mean(lane: any) { return lane_sum(lane) / lane.length; }
export function lane_nanmean(lane: any) {
  let s = 0, n = 0;
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v) { s += v; n++; } }
  return s / n;
}
export function lane_var(lane: any, ddof: number, skip_nan = false) {
  let s = 0, n = 0;
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (skip_nan && v !== v) continue; s += v; n++; }
  const mean = s / n;
  let ss = 0;
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (skip_nan && v !== v) continue; ss += (v - mean) * (v - mean); }
  return ss / Math.max(n - ddof, 0);
}

function empty_error(name: string) {
  return new Error(`zero-size array to reduction operation ${name} which has no identity`);
}
export function lane_max(lane: any) {
  if (lane.length == 0) throw empty_error("maximum");
  let m = +lane[0];
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v !== v) return NaN; if (v > m) m = v; }
  return m;
}
export function lane_min(lane: any) {
  if (lane.length == 0) throw empty_error("minimum");
  let m = +lane[0];
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v !== v) return NaN; if (v < m) m = v; }
  return m;
}
export function lane_nanmax(lane: any) {
  let m = NaN;
  if (lane.length == 0) throw empty_error("fmax");
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v && !(v <= m)) m = v; }
  return m;
}
export function lane_nanmin(lane: any) {
  let m = NaN;
  if (lane.length == 0) throw empty_error("fmin");
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v && !(v >= m)) m = v; }
  return m;
}
export function lane_argmax(lane: any) {
  if (lane.length == 0) throw new Error("attempt to get argmax of an empty sequence");
  let best = 0, m = +lane[0];
  if (m !== m) return 0;
  for (let i = 1; i < lane.length; i++) { const v = +lane[i]; if (v !== v) return i; if (v > m) { m = v; best = i; } }
  return best;
}
export function lane_argmin(lane: any) {
  if (lane.length == 0) throw new Error("attempt to get argmin of an empty sequence");
  let best = 0, m = +lane[0];
  if (m !== m) return 0;
  for (let i = 1; i < lane.length; i++) { const v = +lane[i]; if (v !== v) return i; if (v < m) { m = v; best = i; } }
  return best;
}
export function lane_nanargmax(lane: any) {
  let best = -1, m = -Infinity;
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v && (best == -1 || v > m)) { m = v; best = i; } }
  if (best == -1) throw new Error("All-NaN slice encountered");
  return best;
}
export function lane_nanargmin(lane: any) {
  let best = -1, m = Infinity;
  for (let i = 0; i < lane.length; i++) { const v = +lane[i]; if (v === v && (best == -1 || v < m)) { m = v; best = i; } }
  if (best == -1) throw new Error("All-NaN slice encountered");
  return best;
}
export function lane_any(lane: any) { for (let i = 0; i < lane.length; i++) if (truthy(lane[i])) return true; return false; }
export function lane_all(lane: any) { for (let i = 0; i < lane.length; i++) if (!truthy(lane[i])) return false; return true; }
export function lane_count_nonzero(lane: any) { let n = 0; for (let i = 0; i < lane.length; i++) if (truthy(lane[i])) n++; return n; }

// ==============================
//       Reducing functions
// ==============================

type ReducerDtype = (arr: NDArray, axis: AxisArg, keepdims: boolean, dtype?: DType | null, out?: NDArray | null) => NDArray;
type Reducer = (arr: NDArray, axis: AxisArg, keepdims: boolean, out?: NDArray | null) => NDArray;

/** Reducer that supports a `dtype` argument: (arr, axis, keepdims, dtype, out) */
function mk_reducer_dtype(dtype_resolver: DtypeResolver, lane_fn: (lane: any) => any): ReducerDtype {
  return (arr, axis, keepdims, dtype = null, out = null) => {
    arr = asarray(arr);
    // As numpy, the input is cast to dtype before reducing
    if (dtype && dtype !== arr.dtype) arr = new NDArray(convert_buffer(arr.flat, dtype), arr.shape, dtype);
    const resolved = dtype ? dtype : dtype_resolver([arr.dtype], out);
    return reduce_lanes(arr, axis, keepdims, resolved, lane_fn, out);
  }
}

/** Reducer without dtype argument: (arr, axis, keepdims, out) */
function mk_reducer(dtype_resolver: DtypeResolver, lane_fn: (lane: any) => any): Reducer {
  return (arr, axis, keepdims, out = null) => {
    arr = asarray(arr);
    const resolved = dtype_resolver([arr.dtype], out);
    return reduce_lanes(arr, axis, keepdims, resolved, lane_fn, out);
  }
}

export const sum = mk_reducer_dtype(sum_out, lane_sum);
export const product = mk_reducer_dtype(sum_out, lane_prod);
export const nansum = mk_reducer_dtype(sum_out, lane_nansum);
export const nanprod = mk_reducer_dtype(sum_out, lane_nanprod);
export const mean = mk_reducer_dtype(float_out, lane_mean);
export const nanmean = mk_reducer_dtype(float_out, lane_nanmean);
export const max = mk_reducer(same_out, lane_max);
export const min = mk_reducer(same_out, lane_min);
export const nanmax = mk_reducer(same_out, lane_nanmax);
export const nanmin = mk_reducer(same_out, lane_nanmin);
export const argmax = mk_reducer(argmax_out, lane_argmax);
export const argmin = mk_reducer(argmax_out, lane_argmin);
export const nanargmax = mk_reducer(argmax_out, lane_nanargmax);
export const nanargmin = mk_reducer(argmax_out, lane_nanargmin);
export const len = mk_reducer(argmax_out, (lane: any) => lane.length);
export const any = mk_reducer(bool_out, lane_any);
export const all = mk_reducer(bool_out, lane_all);
export const count_nonzero = mk_reducer(argmax_out, lane_count_nonzero);
export const ptp = (arr: NDArray, axis: AxisArg, keepdims: boolean, out: NDArray | null = null): NDArray => {
  arr = asarray(arr);
  return reduce_lanes(arr, axis, keepdims, same_out([arr.dtype], out), (lane) => lane_max(lane) - lane_min(lane), out);
};

export const norm = (arr: NDArray, ord: number, axis: AxisArg, keepdims: boolean) => {
  arr = asarray(arr);
  const dtype = float_out([arr.dtype]);
  let fn: (lane: any) => number;
  if (ord == Infinity) fn = (lane) => { let m = 0; for (let i = 0; i < lane.length; i++) { const v = Math.abs(lane[i]); if (v > m || v !== v) m = v; } return m; };
  else if (ord == 1) fn = (lane) => { let s = 0; for (let i = 0; i < lane.length; i++) s += Math.abs(lane[i]); return s; };
  else if (ord == 2) fn = (lane) => { let s = 0; for (let i = 0; i < lane.length; i++) s += lane[i] * lane[i]; return Math.sqrt(s); };
  else fn = (lane) => { let s = 0; for (let i = 0; i < lane.length; i++) s += Math.pow(Math.abs(lane[i]), ord); return Math.pow(s, 1 / ord); };
  return reduce_lanes(arr, axis, keepdims, dtype, fn);
}

export const variance = (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean, dtype: DType | null = null, out: NDArray | null = null) => {
  arr = asarray(arr);
  return reduce_lanes(arr, axis, keepdims, float_out([dtype || arr.dtype], out || dtype), (lane) => lane_var(lane, ddof), out);
}
export const std = (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean, dtype: DType | null = null, out: NDArray | null = null) => {
  arr = asarray(arr);
  return reduce_lanes(arr, axis, keepdims, float_out([dtype || arr.dtype], out || dtype), (lane) => Math.sqrt(lane_var(lane, ddof)), out);
}
export const nanvar = (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean, dtype: DType | null = null, out: NDArray | null = null) => {
  arr = asarray(arr);
  return reduce_lanes(arr, axis, keepdims, float_out([dtype || arr.dtype], out || dtype), (lane) => lane_var(lane, ddof, true), out);
}
export const nanstd = (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean, dtype: DType | null = null, out: NDArray | null = null) => {
  arr = asarray(arr);
  return reduce_lanes(arr, axis, keepdims, float_out([dtype || arr.dtype], out || dtype), (lane) => Math.sqrt(lane_var(lane, ddof, true)), out);
}
