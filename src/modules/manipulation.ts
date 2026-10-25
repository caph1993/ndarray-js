import { is_cx, dtype_max as promote_types_ } from '../dtypes';
import { Complex } from '../complex';
//@ts-check
import { NDArray, asarray, isarray, shape_shifts, broadcast_shapes, broadcast_n_shapes, broadcast_strides, empty } from "../NDArray";
import {
  DType, bool, int32, float64, new_buffer, convert_buffer, fix_buffer, dtype_max, dtype_is_integer, truthy
} from "../dtypes";
import { normalize_axis, transpose, swapAxes, concatenate, apply_along_axis } from "../array/transform";
import { ravel, reshape } from "../array/shape_operations";
import { map_broadcast, apply_binary_operation, op_assign } from "../array/operators";
import { same_out } from "../dtypes";
import { reduce_lanes } from "../array/reduce";
import { rint_scalar } from "../array/elementwise";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const as_list = (v: any): number[] => {
  if (v === null || v === undefined) return [];
  if (typeof v === "number") return [v];
  if (isarray(v)) return Array.from(v.flat as any).map(x => +(x as any));
  return Array.from(v).map(x => +(x as any));
};

// ==============================
//   Generic reindexing
// ==============================

/**
 * Builds a new array where, along each axis, output position i takes the source position maps[axis][i]
 * (null keeps the axis as it is). A negative source position produces `fill` (default 0).
 */
export function reindex(arr: NDArray, maps: (ArrayLike<number> | null)[], fill: any = 0): NDArray {
  const nd = arr.shape.length;
  const strides = shape_shifts(arr.shape);
  const offs = maps.map((m, ax) => m
    ? Array.from(m, i => i < 0 ? -1 : i * strides[ax])
    : range(arr.shape[ax]).map(i => i * strides[ax]));
  const shape = offs.map(o => o.length);
  const n = prod(shape);
  const src = arr.flat as any;
  const buffer = new_buffer(n, arr.dtype);
  if (n > 0) {
    if (nd == 0) buffer[0] = src[0];
    else {
      const idx = new Array(nd).fill(0);
      for (let i = 0; i < n; i++) {
        let pos = 0, missing = false;
        for (let ax = 0; ax < nd; ax++) {
          const o = offs[ax][idx[ax]];
          if (o < 0) { missing = true; break; }
          pos += o;
        }
        buffer[i] = missing ? fill : src[pos];
        for (let ax = nd - 1; ax >= 0; ax--) {
          if (++idx[ax] < shape[ax]) break;
          idx[ax] = 0;
        }
      }
    }
  }
  fix_buffer(buffer, arr.dtype);
  return new NDArray(buffer, shape, arr.dtype);
}

// ==============================
//   Constructors
// ==============================

export function eye(N: number, M: number | null, k: number, dtype: DType): NDArray {
  const cols = M === null ? N : M;
  const buffer = new_buffer(N * cols, dtype);
  for (let i = 0; i < N; i++) { const j = i + k; if (j >= 0 && j < cols) buffer[i * cols + j] = 1; }
  return new NDArray(buffer, [N, cols], dtype);
}
export function identity(n: number, dtype: DType): NDArray { return eye(n, null, 0, dtype); }

export function tri(N: number, M: number | null, k: number, dtype: DType): NDArray {
  const cols = M === null ? N : M;
  const buffer = new_buffer(N * cols, dtype);
  for (let i = 0; i < N; i++) for (let j = 0; j < cols; j++) if (j <= i + k) buffer[i * cols + j] = 1;
  return new NDArray(buffer, [N, cols], dtype);
}

function triangle(m: NDArray, k: number, keep: (i: number, j: number, k: number) => boolean): NDArray {
  m = asarray(m);
  if (m.shape.length == 0) throw new Error(`input array must be at least 1-d`);
  if (m.shape.length == 1) m = broadcast_to(m, [m.shape[0], m.shape[0]]);
  const [rows, cols] = m.shape.slice(-2);
  const flat = m.flat as any;
  const buffer = new_buffer(flat.length, m.dtype);
  for (let i = 0; i < flat.length; i++) {
    const r = Math.floor(i / cols) % rows, c = i % cols;
    buffer[i] = keep(r, c, k) ? flat[i] : 0;
  }
  return new NDArray(buffer, m.shape, m.dtype);
}
export const tril = (m: NDArray, k: number) => triangle(m, k, (i, j, k) => j <= i + k);
export const triu = (m: NDArray, k: number) => triangle(m, k, (i, j, k) => j >= i + k);

export function diag(v: NDArray, k: number): NDArray {
  v = asarray(v);
  const flat = v.flat as any;
  if (v.shape.length == 1) {
    const n = v.shape[0] + Math.abs(k);
    const buffer = new_buffer(n * n, v.dtype);
    for (let i = 0; i < v.shape[0]; i++) {
      const r = k >= 0 ? i : i - k, c = k >= 0 ? i + k : i;
      buffer[r * n + c] = flat[i];
    }
    return new NDArray(buffer, [n, n], v.dtype);
  }
  if (v.shape.length == 2) {
    const [rows, cols] = v.shape;
    const out: any[] = [];
    for (let i = 0; i < rows; i++) { const j = i + k; if (j >= 0 && j < cols) out.push(flat[i * cols + j]); }
    return new NDArray(new_buffer(out, v.dtype), [out.length], v.dtype);
  }
  throw new Error(`Input must be 1- or 2-d.`);
}
export const diagflat = (v: NDArray, k: number) => diag(ravel(asarray(v)), k);

export function full(shape: number[] | number, fill_value: any, dtype: DType | null): NDArray {
  const shp = typeof shape === "number" ? [shape] : Array.from(shape as any) as number[];
  const fv = asarray(fill_value);
  const dt = dtype || fv.dtype;
  const buffer = new_buffer(prod(shp), dt);
  const [bshape] = broadcast_shapes(shp, fv.shape);
  if (bshape.length != shp.length || bshape.some((d, i) => d != shp[i])) throw new Error(`could not broadcast input array from shape [${fv.shape}] into shape [${shp}]`);
  map_broadcast(shp, [broadcast_strides(shp, fv.shape)], [convert_buffer(fv.flat, dt)], (x) => x, buffer);
  return new NDArray(fix_buffer(buffer, dt), shp, dt);
}
export function full_like(a: NDArray, fill_value: any, dtype: DType | null, shape: number[] | null): NDArray {
  a = asarray(a);
  return full(shape || a.shape, fill_value, dtype || a.dtype);
}
export const zeros_like = (a: NDArray, dtype: DType | null, shape: number[] | null) => full_like(a, 0, dtype, shape);
export const ones_like = (a: NDArray, dtype: DType | null, shape: number[] | null) => full_like(a, 1, dtype, shape);
export const empty_like = (a: NDArray, dtype: DType | null, shape: number[] | null) => full_like(a, 0, dtype, shape);

export function logspace(start: number, stop: number, num: number, endpoint: boolean, base: number, dtype: DType | null): NDArray {
  const div = endpoint ? num - 1 : num;
  const buffer = new_buffer(num, float64);
  for (let i = 0; i < num; i++) {
    const e = div > 0 ? start + i * (stop - start) / div : start;
    buffer[i] = Math.pow(base, (endpoint && num > 1 && i == num - 1) ? stop : e);
  }
  let out = new NDArray(buffer, [num], float64);
  if (dtype && dtype !== float64) out = astype(out, dtype, true);
  return out;
}

export function meshgrid(xi: NDArray[], indexing: string, sparse: boolean, copy: boolean): NDArray[] {
  if (indexing !== 'xy' && indexing !== 'ij') throw new Error(`Valid values for \`indexing\` are 'xy' and 'ij'.`);
  const xs = xi.map(x => ravel(asarray(x)));
  const nd = xs.length;
  const lens = xs.map(x => x.size);
  const swap = indexing == 'xy' && nd > 1;
  const shape = lens.slice();
  if (swap) { shape[0] = lens[1]; shape[1] = lens[0]; }
  return xs.map((x, k) => {
    const axis = swap ? (k == 0 ? 1 : k == 1 ? 0 : k) : k;
    const own = shape.map((_, i) => i == axis ? x.size : 1);
    const target = sparse ? own : shape;
    const buffer = new_buffer(prod(target), x.dtype);
    const strides = own.map((d, i) => 0);
    const srcStrides = target.map((_, i) => i == axis ? 1 : 0);
    map_broadcast(target, [srcStrides], [x.flat as any], (v) => v, buffer);
    return new NDArray(buffer, target, x.dtype);
  });
}

function parse_grid_spec(spec: string | [number, number, number] | number): { start: number, stop: number, num: number, floatStep: boolean } {
  if (typeof spec === "number") return { start: 0, stop: spec, num: spec, floatStep: false };
  let start: number, stop: number, step = 1, count = false;
  if (typeof spec === "string") {
    const parts = spec.split(':').map(s => s.trim());
    if (parts.length < 2 || parts.length > 3) throw new Error(`Invalid grid specification '${spec}'. Use 'start:stop' or 'start:stop:step' ('start:stop:5j' for 5 points)`);
    start = parts[0] === '' ? 0 : parseFloat(parts[0]);
    stop = parseFloat(parts[1]);
    if (parts[2] !== undefined && parts[2] !== '') {
      if (parts[2].endsWith('j')) { count = true; step = parseFloat(parts[2].slice(0, -1)); }
      else step = parseFloat(parts[2]);
    }
  } else [start, stop, step] = spec;
  if (count) return { start, stop, num: step, floatStep: true };
  const num = Math.max(0, Math.ceil((stop - start) / step));
  return { start, stop: start + num * step, num, floatStep: !(Number.isInteger(start) && Number.isInteger(step)) };
}

function grid_axis(spec: any) {
  const { start, stop, num, floatStep } = parse_grid_spec(spec);
  const buffer = new_buffer(num, floatStep ? float64 : int32);
  const step = typeof spec === "string" && spec.split(':')[2]?.endsWith('j')
    ? (num > 1 ? (stop - start) / (num - 1) : 0)
    : (num > 0 ? (stop - start) / num : 1);
  for (let i = 0; i < num; i++) buffer[i] = start + i * step;
  return new NDArray(buffer, [num], floatStep ? float64 : int32);
}

/** Dense multi-dimensional "meshgrid" with 'ij' indexing from slice strings: mgrid('0:3', '0:2') */
export function mgrid(...specs: any[]): NDArray | NDArray[] {
  const axes = specs.map(grid_axis);
  const grids = meshgrid(axes, 'ij', false, true);
  if (grids.length == 1) return grids[0];
  return stack_new(grids);
}
/** Open (sparse) mesh-grid: ogrid('0:3', '0:2') returns arrays of shapes (3,1) and (1,2) */
export function ogrid(...specs: any[]): NDArray | NDArray[] {
  const axes = specs.map(grid_axis);
  const grids = meshgrid(axes, 'ij', true, true);
  return grids.length == 1 ? grids[0] : grids;
}
function stack_new(arrays: NDArray[]): NDArray {
  return concatenate(arrays.map(a => reshape(a, [1, ...a.shape])), 0);
}

// ==============================
//   Attributes and conversions
// ==============================

export const ndim = (a: NDArray) => asarray(a).shape.length;
export const shape = (a: NDArray) => asarray(a).shape.slice();
export function size(a: NDArray, axis: number | null): number {
  a = asarray(a);
  if (axis === null) return a.size;
  return a.shape[normalize_axis(axis, a.shape.length)];
}

export function astype(x: NDArray, dtype: DType, copy: boolean): NDArray {
  x = asarray(x);
  if (x.dtype === dtype && !copy && !x._simpleIndexes) return x;
  return new NDArray(convert_buffer(x.flat, dtype), x.shape, dtype);
}
export const copy = (a: NDArray) => astype(asarray(a), asarray(a).dtype, true);

/** Copies values from src into dst (in place) where the mask is true */
export function copyto(dst: NDArray, src: NDArray, casting: string, where: NDArray | boolean): NDArray {
  src = asarray(src);
  if (where === true) return op_assign["="](dst, src, []);
  if (where === false) return dst;
  const mask = broadcast_to(asarray(where), dst.shape);
  const [bshape] = broadcast_shapes(dst.shape, src.shape);
  const values = broadcast_to(src, dst.shape);
  const current = dst.flat as any, m = mask.flat as any, v = values.flat as any;
  const out = new_buffer(current.length, dst.dtype);
  for (let i = 0; i < out.length; i++) out[i] = truthy(m[i]) ? v[i] : current[i];
  dst.flat = out;
  return dst;
}

// ==============================
//   Axes
// ==============================

export function moveaxis(a: NDArray, source: number | number[], destination: number | number[]): NDArray {
  a = asarray(a);
  const nd = a.shape.length;
  const src = as_list(source).map(s => normalize_axis(s, nd));
  const dst = as_list(destination).map(d => normalize_axis(d, nd));
  if (src.length != dst.length) throw new Error(`\`source\` and \`destination\` arguments must have the same number of elements`);
  if (new Set(src).size != src.length) throw new Error(`repeated axis in \`source\` argument`);
  if (new Set(dst).size != dst.length) throw new Error(`repeated axis in \`destination\` argument`);
  const order = range(nd).filter(n => !src.includes(n));
  src.map((s, i) => [dst[i], s]).sort((p, q) => p[0] - q[0]).forEach(([d, s]) => order.splice(d, 0, s));
  return transpose(a, order);
}

export function rollaxis(a: NDArray, axis: number, start: number): NDArray {
  a = asarray(a);
  const n = a.shape.length;
  axis = normalize_axis(axis, n);
  if (start < 0) start += n;
  if (start < 0 || start > n) throw new Error(`'start' arg requires ${-n} <= start < ${n + 1}, but ${start} was passed in`);
  if (axis < start) start -= 1;
  if (axis == start) return a;
  const axes = range(n).filter(i => i != axis);
  axes.splice(start, 0, axis);
  return transpose(a, axes);
}

export function matrix_transpose(x: NDArray): NDArray {
  x = asarray(x);
  if (x.shape.length < 2) throw new Error(`Input array must be at least 2-dimensional, but it is ${x.shape.length}`);
  return swapAxes(x, -2, -1);
}

export function atleast_nd(arys: NDArray[], n: 1 | 2 | 3): NDArray | NDArray[] {
  const res = arys.map(a => {
    a = asarray(a);
    const s = a.shape;
    if (s.length >= n) return a;
    if (n == 1) return reshape(a, [1]);
    if (n == 2) return reshape(a, s.length == 0 ? [1, 1] : [1, s[0]]);
    return reshape(a, s.length == 0 ? [1, 1, 1] : s.length == 1 ? [1, s[0], 1] : [s[0], s[1], 1]);
  });
  return res.length == 1 ? res[0] : res;
}

export function broadcast_to(array: NDArray, shape: number[] | number): NDArray {
  array = asarray(array);
  const shp = typeof shape === "number" ? [shape] : Array.from(shape as any) as number[];
  if (shp.some(d => d < 0)) throw new Error(`all elements of broadcast shape must be non-negative`);
  if (array.shape.length > shp.length) throw new Error(`input operand has more dimensions than allowed by the axis remapping`);
  const off = shp.length - array.shape.length;
  array.shape.forEach((d, i) => { if (d !== shp[i + off] && d !== 1) throw new Error(`operands could not be broadcast together with remapped shapes [original->remapped]: (${array.shape}) and requested shape (${shp})`); });
  const buffer = new_buffer(prod(shp), array.dtype);
  map_broadcast(shp, [broadcast_strides(shp, array.shape)], [array.flat as any], (x) => x, buffer);
  return new NDArray(buffer, shp, array.dtype);
}

export function broadcast_arrays(args: NDArray[]): NDArray[] {
  const arrays = args.map(a => asarray(a));
  if (!arrays.length) return [];
  const [, shape] = broadcast_n_shapes(...arrays.map(a => a.shape));
  return arrays.map(a => broadcast_to(a, shape as number[]));
}

export function broadcast(args: NDArray[]) {
  const arrays = args.map(a => asarray(a));
  const [, shape] = broadcast_n_shapes(...arrays.map(a => a.shape));
  return { shape: shape as number[], ndim: (shape as number[]).length, size: prod(shape as number[]), numiter: arrays.length };
}

export function expand_dims(a: NDArray, axis: number | number[]): NDArray {
  a = asarray(a);
  const axes = as_list(axis);
  const out_nd = a.shape.length + axes.length;
  const norm = axes.map(ax => normalize_axis(ax, out_nd));
  if (new Set(norm).size != norm.length) throw new Error(`repeated axis`);
  const shape: number[] = [];
  let j = 0;
  for (let i = 0; i < out_nd; i++) shape.push(norm.includes(i) ? 1 : a.shape[j++]);
  return reshape(a, shape);
}

export function squeeze(a: NDArray, axis: number | number[] | null): NDArray {
  a = asarray(a);
  if (axis === null) return reshape(a, a.shape.filter(d => d != 1));
  const axes = as_list(axis).map(ax => normalize_axis(ax, a.shape.length));
  for (const ax of axes) if (a.shape[ax] != 1) throw new Error(`cannot select an axis to squeeze out which has size not equal to one`);
  return reshape(a, a.shape.filter((_, i) => !axes.includes(i)));
}

// ==============================
//   Joining and splitting
// ==============================

export function vstack(tup: NDArray[]): NDArray {
  const arrs = tup.map(a => atleast_nd([a], 2) as NDArray);
  return concatenate(arrs, 0);
}
export function hstack(tup: NDArray[]): NDArray {
  const arrs = tup.map(a => atleast_nd([a], 1) as NDArray);
  return concatenate(arrs, arrs[0].shape.length == 1 ? 0 : 1);
}
export function dstack(tup: NDArray[]): NDArray {
  return concatenate(tup.map(a => atleast_nd([a], 3) as NDArray), 2);
}
export function column_stack(tup: NDArray[]): NDArray {
  const arrs = tup.map(a => { a = asarray(a); return a.shape.length < 2 ? reshape(a, [a.size, 1]) : a; });
  return concatenate(arrs, 1);
}

/** Concatenates nested lists of arrays: the innermost list along the last axis, the next one along the second to last, ... */
export function block(arrays: any): NDArray {
  const depth_of = (x: any): number => Array.isArray(x) ? (x.length ? 1 + Math.max(...x.map(depth_of)) : 1) : 0;
  const depth = depth_of(arrays);
  if (depth == 0) return asarray(arrays);
  let ndmax = depth;
  const scan = (x: any) => { if (Array.isArray(x)) x.forEach(scan); else ndmax = Math.max(ndmax, asarray(x).shape.length); };
  scan(arrays);
  const rec = (x: any, level: number): NDArray => {
    if (!Array.isArray(x)) {
      const a = asarray(x);
      return reshape(a, [...Array(ndmax - a.shape.length).fill(1), ...a.shape]);
    }
    if (!x.length) throw new Error(`List at arrays with index ${level} is empty`);
    return concatenate(x.map(e => rec(e, level + 1)), ndmax - (depth - level));
  };
  return rec(arrays, 0);
}

function slice_along(ary: NDArray, axis: number, start: number, stop: number): NDArray {
  const where: any[] = ary.shape.map(() => ':');
  where[axis] = `${start}:${stop}`;
  return ary.index(...where) as NDArray;
}

export function array_split(ary: NDArray, indices_or_sections: number | number[], axis: number, equal: boolean = false): NDArray[] {
  ary = asarray(ary);
  axis = normalize_axis(axis, ary.shape.length);
  const n = ary.shape[axis];
  let bounds: number[];
  if (typeof indices_or_sections === "number") {
    const k = indices_or_sections;
    if (k <= 0) throw new Error(`number sections must be larger than 0.`);
    if (equal && n % k != 0) throw new Error(`array split does not result in an equal division`);
    const each = Math.floor(n / k), extras = n % k;
    bounds = [0];
    for (let i = 0; i < k; i++) bounds.push(bounds[i] + each + (i < extras ? 1 : 0));
  } else {
    bounds = [0, ...as_list(indices_or_sections), n];
  }
  const out: NDArray[] = [];
  for (let i = 0; i + 1 < bounds.length; i++) {
    let lo = bounds[i], hi = bounds[i + 1];
    // python slice semantics for out of range / reversed indices
    const clampi = (v: number) => v < 0 ? Math.max(0, v + n) : Math.min(v, n);
    lo = clampi(lo); hi = clampi(hi);
    out.push(hi > lo ? slice_along(ary, axis, lo, hi) : reindex(ary, ary.shape.map((_, ax) => ax == axis ? [] : null)));
  }
  return out;
}
export const split = (ary: NDArray, ios: number | number[], axis: number) => array_split(ary, ios, axis, true);
export function hsplit(ary: NDArray, ios: number | number[]) {
  ary = asarray(ary);
  if (ary.shape.length == 0) throw new Error(`hsplit only works on arrays of 1 or more dimensions`);
  return split(ary, ios, ary.shape.length > 1 ? 1 : 0);
}
export function vsplit(ary: NDArray, ios: number | number[]) {
  ary = asarray(ary);
  if (ary.shape.length < 2) throw new Error(`vsplit only works on arrays of 2 or more dimensions`);
  return split(ary, ios, 0);
}
export function dsplit(ary: NDArray, ios: number | number[]) {
  ary = asarray(ary);
  if (ary.shape.length < 3) throw new Error(`dsplit only works on arrays of 3 or more dimensions`);
  return split(ary, ios, 2);
}
export function unstack(x: NDArray, axis: number): NDArray[] {
  x = asarray(x);
  if (x.shape.length == 0) throw new Error(`Input array must be at least 1-d.`);
  axis = normalize_axis(axis, x.shape.length);
  return range(x.shape[axis]).map(i => {
    const where: any[] = x.shape.map(() => ':');
    where[axis] = i;
    return asarray(x.index(...where));
  });
}

// ==============================
//   Repeating, inserting, deleting
// ==============================

export function tile(A: NDArray, reps: number | number[]): NDArray {
  A = asarray(A);
  let r = as_list(reps);
  if (r.some(v => v < 0 || !Number.isInteger(v))) throw new Error(`reps must be non-negative integers`);
  const d = Math.max(A.shape.length, r.length);
  r = [...Array(d - r.length).fill(1), ...r];
  const shape = [...Array(d - A.shape.length).fill(1), ...A.shape];
  const src = reshape(A, shape);
  return reindex(src, shape.map((n, ax) => range(n * r[ax]).map(i => i % n)));
}

export function repeat(a: NDArray, repeats: number | number[] | NDArray, axis: number | null): NDArray {
  a = asarray(a);
  if (axis === null) { a = ravel(a); axis = 0; }
  if (a.shape.length == 0) a = reshape(a, [1]);
  axis = normalize_axis(axis, a.shape.length);
  const n = a.shape[axis];
  let reps = as_list(repeats);
  if (reps.length == 1) reps = Array(n).fill(reps[0]);
  if (reps.length != n) throw new Error(`operands could not be broadcast together with shape (${n},) (${reps.length},)`);
  if (reps.some(v => v < 0 || !Number.isInteger(v))) throw new Error(`repeats may not contain negative values.`);
  const map: number[] = [];
  reps.forEach((c, i) => { for (let k = 0; k < c; k++) map.push(i); });
  return reindex(a, a.shape.map((_, ax) => ax == axis ? map : null));
}

function normalize_indices(obj: any, n: number, what: string): number[] {
  if (typeof obj === "string") {
    const parts = obj.split(':').map(s => s.trim() === '' ? null : parseInt(s));
    const [a, b, c] = [...parts, null, null, null];
    const step = c ?? 1;
    const clamp = (v: number | null, dflt: number, lo: number, hi: number) => v === null ? dflt : Math.min(hi, Math.max(lo, v < 0 ? v + n : v));
    const out: number[] = [];
    if (step > 0) for (let i = clamp(a, 0, 0, n); i < clamp(b, n, 0, n); i += step) out.push(i);
    else for (let i = clamp(a, n - 1, -1, n - 1); i > clamp(b, -1, -1, n - 1); i += step) out.push(i);
    return out;
  }
  const arr = asarray(obj);
  if (arr.dtype === bool) {
    if (arr.size != n) throw new Error(`boolean array argument obj to ${what} must be one dimensional and match the axis length of ${n}`);
    return range(n).filter(i => (arr.flat as any)[i]);
  }
  return as_list(arr).map(i => {
    if (!Number.isInteger(i)) throw new Error(`index ${i} is not an integer`);
    const j = i < 0 ? i + n : i;
    if (j < 0 || j >= n) throw new Error(`index ${i} is out of bounds for axis with size ${n}`);
    return j;
  });
}

export function del(arr: NDArray, obj: any, axis: number | null): NDArray {
  arr = asarray(arr);
  if (axis === null) { arr = ravel(arr); axis = 0; }
  axis = normalize_axis(axis, arr.shape.length);
  const drop = new Set(normalize_indices(obj, arr.shape[axis], 'delete'));
  const keep = range(arr.shape[axis]).filter(i => !drop.has(i));
  return reindex(arr, arr.shape.map((_, ax) => ax == axis ? keep : null));
}

export function insert(arr: NDArray, obj: any, values: any, axis: number | null): NDArray {
  arr = asarray(arr);
  if (axis === null) { arr = ravel(arr); axis = 0; }
  else axis = normalize_axis(axis, arr.shape.length);
  const n = arr.shape[axis];
  const scalar_obj = typeof obj === "number" || (isarray(obj) && obj.shape.length == 0);
  let vals = asarray(values);
  let indices: number[];
  if (typeof obj === "string") indices = normalize_indices(obj, n, 'insert').map(i => i);
  else if (scalar_obj) {
    let i = +(isarray(obj) ? obj.item() : obj);
    if (i < -n || i > n) throw new Error(`index ${i} is out of bounds for axis ${axis} with size ${n}`);
    if (i < 0) i += n;
    if (vals.shape.length < arr.shape.length) vals = reshape(vals, [...Array(arr.shape.length - vals.shape.length).fill(1), ...vals.shape]);
    vals = moveaxis(vals, 0, axis);
    const numnew = vals.shape[axis];
    indices = range(numnew).map(k => i + k);
    return insert_at(arr, axis, indices, vals, n + numnew, true);
  } else {
    indices = as_list(obj).map(i => { if (i < -n || i > n) throw new Error(`index ${i} is out of bounds for axis ${axis} with size ${n}`); return i < 0 ? i + n : i; });
  }
  const numnew = indices.length;
  // insertion points refer to positions of the original array; shift by the amount of earlier insertions (stable)
  const order = range(numnew).sort((p, q) => indices[p] - indices[q] || p - q);
  const final: number[] = new Array(numnew);
  order.forEach((p, rank) => { final[p] = indices[p] + rank; });
  if (vals.shape.length < arr.shape.length) vals = reshape(vals, [...Array(arr.shape.length - vals.shape.length).fill(1), ...vals.shape]);
  const target = arr.shape.map((d, ax) => ax == axis ? numnew : d);
  return insert_at(arr, axis, final, broadcast_to(vals, target), n + numnew, false);
}

function insert_at(arr: NDArray, axis: number, positions: number[], vals: NDArray, newLen: number, contiguous: boolean): NDArray {
  const dtype = arr.dtype; // as numpy, values are cast to the dtype of the array
  const shape = arr.shape.map((d, ax) => ax == axis ? newLen : d);
  const out = empty(shape, dtype);
  const posSet = new Set(positions);
  const oldPos = range(newLen).filter(i => !posSet.has(i));
  const where = (idx: number[]) => shape.map((_, ax) => ax == axis ? idx : ':');
  const a = arr.dtype === dtype ? arr : new NDArray(convert_buffer(arr.flat, dtype), arr.shape, dtype);
  const v = vals.dtype === dtype ? vals : new NDArray(convert_buffer(vals.flat, dtype), vals.shape, dtype);
  op_assign["="](out, a, where(oldPos) as any);
  const [vshape] = broadcast_shapes(vals.shape, shape.map((d, ax) => ax == axis ? positions.length : d));
  op_assign["="](out, broadcast_to(v, shape.map((d, ax) => ax == axis ? positions.length : d)), where(positions) as any);
  return out;
}

export function append(arr: NDArray, values: NDArray, axis: number | null): NDArray {
  arr = asarray(arr); values = asarray(values);
  if (axis === null) {
    return concatenate([ravel(arr), ravel(values)], 0);
  }
  return concatenate([arr, values], axis);
}

export function resize(a: NDArray, new_shape: number | number[]): NDArray {
  a = asarray(a);
  const shape = as_list(new_shape);
  if (shape.some(d => d < 0)) throw new Error(`all elements of \`new_shape\` must be non-negative`);
  const n = prod(shape);
  const flat = a.flat as any;
  const buffer = new_buffer(n, a.dtype);
  if (flat.length > 0) for (let i = 0; i < n; i++) buffer[i] = flat[i % flat.length];
  return new NDArray(buffer, shape, a.dtype);
}

export function trim_zeros(filt: NDArray, trim: string, axis: number | number[] | null): NDArray {
  filt = asarray(filt);
  const t = trim.toLowerCase();
  if (!/^[fb]*$/.test(t)) throw new Error(`unexpected character(s) in \`trim\`: ${JSON.stringify(trim)}`);
  const nd = filt.shape.length;
  const axes = axis === null ? range(nd) : as_list(axis).map(ax => normalize_axis(ax, nd));
  const maps: (number[] | null)[] = filt.shape.map(() => null);
  for (const ax of axes) {
    const others = range(nd).filter(i => i != ax);
    const nz = reduce_lanes(filt, others, false, bool, (lane) => { for (let i = 0; i < lane.length; i++) if (truthy(lane[i]) && lane[i] != 0) return true; return false; });
    const flags = nz.flat as any;
    let lo = 0, hi = flags.length;
    if (t.includes('f')) while (lo < hi && !flags[lo]) lo++;
    if (t.includes('b')) while (hi > lo && !flags[hi - 1]) hi--;
    maps[ax] = range(hi - lo).map(i => lo + i);
  }
  return reindex(filt, maps);
}

// ==============================
//   Reordering
// ==============================

export function flip(m: NDArray, axis: number | number[] | null): NDArray {
  m = asarray(m);
  const axes = axis === null ? range(m.shape.length) : as_list(axis).map(ax => normalize_axis(ax, m.shape.length));
  if (new Set(axes).size !== axes.length) throw new Error('repeated axis');
  return reindex(m, m.shape.map((n, ax) => axes.includes(ax) ? range(n).reverse() : null));
}
export function fliplr(m: NDArray): NDArray {
  m = asarray(m);
  if (m.shape.length < 2) throw new Error(`Input must be >= 2-d.`);
  return flip(m, 1);
}
export function flipud(m: NDArray): NDArray {
  m = asarray(m);
  if (m.shape.length < 1) throw new Error(`Input must be >= 1-d.`);
  return flip(m, 0);
}

export function roll(a: NDArray, shift: number | number[], axis: number | number[] | null): NDArray {
  a = asarray(a);
  if (axis === null) {
    const flat = ravel(a);
    const s = as_list(shift).reduce((x, y) => x + y, 0);
    const n = flat.size;
    const rolled = reindex(flat, [range(n).map(i => ((i - s) % n + n) % n)]);
    return reshape(rolled, a.shape);
  }
  const axes = as_list(axis).map(ax => normalize_axis(ax, a.shape.length));
  const shifts = as_list(shift);
  const [, bshape] = broadcast_n_shapes([axes.length], [shifts.length]);
  const count = (bshape as number[])[0];
  if (axes.length != count && axes.length != 1 || shifts.length != count && shifts.length != 1) throw new Error(`'shift' and 'axis' should be scalars or 1D sequences`);
  const total: { [ax: number]: number } = {};
  for (let i = 0; i < count; i++) {
    const ax = axes[axes.length == 1 ? 0 : i], sh = shifts[shifts.length == 1 ? 0 : i];
    total[ax] = (total[ax] || 0) + sh;
  }
  return reindex(a, a.shape.map((n, ax) => total[ax] === undefined || n == 0 ? null : range(n).map(i => (((i - total[ax]) % n) + n) % n)));
}

export function rot90(m: NDArray, k: number, axes: number[]): NDArray {
  m = asarray(m);
  if (axes.length != 2) throw new Error(`len(axes) must be 2.`);
  const nd = m.shape.length;
  const [a0, a1] = axes.map(ax => normalize_axis(ax, nd));
  if (a0 == a1) throw new Error(`Axes must be different.`);
  k = ((k % 4) + 4) % 4;
  if (k == 0) return copy(m);
  if (k == 2) return flip(flip(m, a0), a1);
  const perm = range(nd);
  [perm[a0], perm[a1]] = [perm[a1], perm[a0]];
  if (k == 1) return transpose(flip(m, a1), perm);
  return flip(transpose(m, perm), a1);
}

// ==============================
//   Padding
// ==============================

export type PadMode = 'constant' | 'edge' | 'linear_ramp' | 'maximum' | 'mean' | 'median' | 'minimum' | 'reflect' | 'symmetric' | 'wrap' | 'empty';

function pad_widths(pad_width: any, nd: number): [number, number][] {
  const w = asarray(pad_width);
  const f = Array.from(w.flat as any).map(x => +(x as any));
  let pairs: [number, number][];
  if (w.shape.length == 0) pairs = range(nd).map(() => [f[0], f[0]]);
  else if (w.shape.length == 1 && w.size == 1) pairs = range(nd).map(() => [f[0], f[0]]);
  else if (w.shape.length == 1 && w.size == 2) pairs = range(nd).map(() => [f[0], f[1]]);
  else if (w.shape.length == 2 && w.shape[1] == 2 && w.shape[0] == nd) pairs = range(nd).map(i => [f[2 * i], f[2 * i + 1]]);
  else if (w.shape.length == 2 && w.shape[0] == 1 && w.shape[1] == 2) pairs = range(nd).map(() => [f[0], f[1]]);
  else throw new Error(`Unable to create correctly shaped tuple from pad_width ${JSON.stringify(pad_width)}`);
  for (const [b, a] of pairs) if (b < 0 || a < 0 || !Number.isInteger(b) || !Number.isInteger(a)) throw new Error(`index can't contain negative values`);
  return pairs;
}

/** Per-axis (before, after) pair of arbitrary numbers: scalar, (before, after) or one pair per axis */
function per_axis(values: any, nd: number, dflt: number): [number, number][] {
  if (values === null || values === undefined) return range(nd).map(() => [dflt, dflt]);
  const w = asarray(values);
  const f = Array.from(w.flat as any).map(x => +(x as any));
  if (w.size == 1) return range(nd).map(() => [f[0], f[0]]);
  if (w.shape.length == 1 && w.size == 2) return range(nd).map(() => [f[0], f[1]]);
  if (w.shape.length == 2 && w.shape[1] == 2 && w.shape[0] == nd) return range(nd).map(i => [f[2 * i], f[2 * i + 1]]);
  if (w.shape.length == 2 && w.shape[0] == 1 && w.shape[1] == 2) return range(nd).map(() => [f[0], f[1]]);
  throw new Error(`Unable to create correctly shaped tuple from ${JSON.stringify(values)}`);
}

export function pad(array: NDArray, pad_width: any, mode: PadMode, options: { constant_values?: any, end_values?: any, stat_length?: any, reflect_type?: string }): NDArray {
  array = asarray(array);
  if (is_cx(array.dtype)) {
    // real and imaginary parts are padded separately (exact for every mode that does not order the values)
    if (mode === 'maximum' || mode === 'minimum' || mode === 'median') throw new Error(`mode '${mode}' is not supported for complex arrays`);
    const split = (v: any, k: 're' | 'im') => v === undefined || v === null ? v : Array.isArray(v) ? v.map(x => split(x, k)) : (v instanceof Complex ? v[k] : (k === 're' ? v : 0));
    const part = (k: 're' | 'im') => pad(new NDArray(Array.from(array.flat as any, (z: any) => z[k]) as any, array.shape, 'float64'), pad_width, mode, { ...options, constant_values: split(options.constant_values, k), end_values: split(options.end_values, k) });
    const re = part('re'), im = part('im');
    const rf = re.flat as any, imf = im.flat as any;
    return new NDArray(Array.from({ length: rf.length }, (_, i) => new Complex(rf[i], imf[i])) as any, re.shape, array.dtype);
  }
  const nd = array.shape.length;
  const widths = pad_widths(pad_width, nd);
  const modes = ['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap', 'empty'];
  if (!modes.includes(mode)) throw new Error(`mode '${mode}' is not supported`);
  const consts = per_axis(options.constant_values, nd, 0);
  const ends = per_axis(options.end_values, nd, 0);
  const odd = options.reflect_type === 'odd';
  let stats: [number | null, number | null][] = range(nd).map(() => [null, null]);
  if (options.stat_length !== undefined && options.stat_length !== null) stats = per_axis(options.stat_length, nd, 0) as any;
  let out = array;
  for (let ax = 0; ax < nd; ax++) {
    const [before, after] = widths[ax];
    if (before == 0 && after == 0) continue;
    const n = out.shape[ax];
    if (n == 0 && mode != 'constant' && mode != 'empty' && mode != 'linear_ramp') throw new Error(`can't extend empty axis ${ax} using modes other than 'constant' or 'empty'`);
    const dtype = out.dtype;
    const lane_fn = (lane: any): ArrayLike<number> => {
      const L: number[] = Array.from(lane as any);
      const res: number[] = new Array(before + L.length + after);
      for (let i = 0; i < L.length; i++) res[before + i] = L[i];
      const stat = (side: 0 | 1, f: (v: number[]) => number) => {
        const len = stats[ax][side] === null ? L.length : Math.min(stats[ax][side]!, L.length);
        return f(side == 0 ? L.slice(0, len) : L.slice(L.length - len));
      };
      const edge_val = (side: 0 | 1) => side == 0 ? L[0] : L[L.length - 1];
      for (let side = 0; side < 2; side++) {
        const cnt = side == 0 ? before : after;
        const sign = side == 0 ? -1 : 1;
        for (let k = 1; k <= cnt; k++) {
          const dst = side == 0 ? before - k : before + L.length + k - 1;
          let v: number;
          switch (mode) {
            case 'constant': v = consts[ax][side]; break;
            case 'empty': v = 0; break;
            case 'edge': v = edge_val(side as 0 | 1); break;
            case 'maximum': v = stat(side as 0 | 1, a => Math.max(...a)); break;
            case 'minimum': v = stat(side as 0 | 1, a => Math.min(...a)); break;
            case 'mean': v = round_if(stat(side as 0 | 1, a => a.reduce((s, x) => s + x, 0) / a.length), dtype); break;
            case 'median': v = round_if(stat(side as 0 | 1, a => { const s = a.slice().sort((p, q) => p - q); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }), dtype); break;
            case 'linear_ramp': {
              // from the edge value (next to the array) to end_values (outermost element)
              const e = ends[ax][side], start = edge_val(side as 0 | 1);
              v = start + (e - start) * (k / cnt);
              if (dtype_is_integer(dtype)) v = Math.trunc(v); // numpy casts the float ramp to the integer dtype
              break;
            }
            case 'reflect': case 'symmetric': {
              if (L.length == 1) { v = L[0]; break; }
              v = L[reflect_index(side == 0 ? -k : L.length - 1 + k, L.length, mode == 'symmetric')];
              if (odd) v = 2 * edge_val(side as 0 | 1) - v;
              break;
            }
            case 'wrap': {
              const total = L.length;
              const pos = side == 0 ? -k : total - 1 + k;
              v = L[((pos % total) + total) % total];
              break;
            }
            default: v = 0;
          }
          res[dst] = v;
        }
      }
      return res;
    };
    out = apply_along_axis(() => dtype, lane_fn, out, ax);
  }
  return out;
}

function reflect_index(pos: number, n: number, symmetric: boolean): number {
  if (n == 1) return 0;
  // reflect: ... 2 1 [0 1 2 3] 2 1 ... (period 2(n-1)); symmetric: ... 1 0 [0 1 2 3] 3 2 ... (period 2n)
  if (symmetric) {
    const period = 2 * n;
    let p = ((pos % period) + period) % period;
    return p < n ? p : period - 1 - p;
  }
  const period = 2 * (n - 1);
  let p = ((pos % period) + period) % period;
  return p < n ? p : period - p;
}
function round_if(v: number, dtype: DType) { return dtype_is_integer(dtype) ? rint_scalar(v) : v; }

// ==============================
//   Bit counting
// ==============================

export function bitwise_count(x: NDArray): NDArray {
  x = asarray(x);
  if (!dtype_is_integer(x.dtype) && x.dtype !== bool) throw new Error(`ufunc 'bitwise_count' not supported for the input types`);
  const f = x.flat as any;
  const buffer = new_buffer(f.length, int32);
  const pop32 = (v: number) => { v = v - ((v >>> 1) & 0x55555555); v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24; };
  for (let i = 0; i < f.length; i++) {
    const v = Math.abs(+f[i]);
    buffer[i] = pop32(v % 4294967296) + pop32(Math.floor(v / 4294967296));
  }
  return new NDArray(buffer, x.shape, int32);
}

/** Vandermonde matrix: columns are powers of x (decreasing unless `increasing`) */
export function vander(x: NDArray, N: number | null, increasing: boolean): NDArray {
  x = asarray(x);
  if (x.shape.length != 1) throw new Error(`x must be a one-dimensional array or sequence.`);
  const n = N === null ? x.size : N;
  const dtype = x.dtype === bool ? int32 : promote_types_(x.dtype, 'int64'); // numpy: promote_types(x.dtype, int)
  const f = x.flat as any;
  const buffer = new_buffer(x.size * n, dtype);
  if (is_cx(dtype)) {
    for (let i = 0; i < x.size; i++) for (let j = 0; j < n; j++) { const e = increasing ? j : n - 1 - j; let r = new Complex(1, 0); for (let t = 0; t < e; t++) r = r.mul(f[i]); buffer[i * n + j] = r; }
    return new NDArray(fix_buffer(buffer, dtype), [x.size, n], dtype);
  }
  for (let i = 0; i < x.size; i++) for (let j = 0; j < n; j++) buffer[i * n + j] = Math.pow(+f[i], increasing ? j : n - 1 - j);
  return new NDArray(fix_buffer(buffer, dtype), [x.size, n], dtype);
}
