//@ts-check

import { NDArray, isarray, asarray, shape_shifts, empty } from '../NDArray';
import { ravel, reshape } from "./shape_operations";
import { allEq } from '../utils-js';
import { ArrayOrConstant } from './operators';
import { argmax_out, bool, DType, dtype_max, DtypeResolver, new_buffer, same_out, int32, fix_buffer } from '../dtypes';

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Normalizes a (possibly negative) axis and checks that it is valid for an array with `ndim` dimensions */
export function normalize_axis(axis: number, ndim: number): number {
  if (typeof axis !== "number" || !Number.isInteger(axis)) throw new Error(`Axis must be an integer. Found ${axis}`);
  if (axis < -ndim || axis >= ndim) throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  return axis < 0 ? axis + ndim : axis;
}

/**
 * Applies `transform` to every 1D lane of `arr` along `axis`. The transform receives the lane
 * (as a JS array / typed array) and returns an array-like whose length may differ from the lane length
 * (but must be the same for all lanes). The new axis length replaces the old one.
 */
export function apply_along_axis(
  dtype_resolver: DtypeResolver,
  transform: (arr: any) => ArrayLike<any>,
  arr: NDArray, axis: number
): NDArray {
  arr = asarray(arr);
  const nDims = arr.shape.length;
  if (nDims == 0) throw new Error(`Expected an array with at least one dimension`);
  axis = normalize_axis(axis, nDims);
  const moved = axis == nDims - 1 ? arr : swapAxes(arr, axis, nDims - 1);
  const n = moved.shape[nDims - 1];
  const nLanes = n == 0 ? prod(moved.shape.slice(0, -1)) : moved.size / n;
  const flat = moved.flat;
  const lanes: ArrayLike<any>[] = new Array(nLanes);
  let m = -1;
  for (let i = 0; i < nLanes; i++) {
    const lane = transform(flat.slice(i * n, (i + 1) * n));
    if (m == -1) m = lane.length;
    else if (m != lane.length) throw new Error(`Inconsistent lane lengths: ${m} and ${lane.length}`);
    lanes[i] = lane;
  }
  if (m == -1) m = transform(flat.slice(0, 0)).length;
  const dtype = dtype_resolver([arr.dtype], null);
  const buffer = new_buffer(nLanes * m, dtype);
  for (let i = 0; i < nLanes; i++) {
    const lane = lanes[i];
    if (dtype === bool) for (let j = 0; j < m; j++) buffer[i * m + j] = lane[j] ? 1 : 0;
    else for (let j = 0; j < m; j++) buffer[i * m + j] = lane[j];
  }
  fix_buffer(buffer, dtype);
  const out = new NDArray(buffer, [...moved.shape.slice(0, -1), m], dtype);
  return axis == nDims - 1 ? out : swapAxes(out, axis, nDims - 1);
}

/** Total order for sorting numbers: nans at the end */
export const cmp_nan_at_the_end = (a: number, b: number) => {
  if (a < b) return -1;
  if (a > b) return 1;
  if (a === b) return 0;
  const na = a !== a, nb = b !== b;
  return na ? (nb ? 0 : 1) : -1;
}

export function sort(a: NDArray, axis: number) {
  const transform = (arr) => {
    const cpy = arr.slice();
    // Typed arrays sort numerically with nans at the end (and much faster than with a comparator)
    if (Array.isArray(cpy)) cpy.sort(cmp_nan_at_the_end); else cpy.sort();
    return cpy;
  }
  return apply_along_axis(same_out, transform, a, axis) as NDArray;
}

export function argsort(a: NDArray, axis: number) {
  const func = (arr) => {
    const idx = Array.from(arr).map((_: any, i: number) => i);
    idx.sort((i: number, j: number) => cmp_nan_at_the_end(arr[i], arr[j]));
    return idx;
  }
  return apply_along_axis(argmax_out, func, a, axis) as NDArray;
}

/** Copy of the array with its axes permuted (`axes[i]` is the source axis of output axis i) */
export function transpose(arr: NDArray, axes: null | number[] = null) {
  arr = asarray(arr);
  const nDims = arr.shape.length;
  if (axes == null) return transpose(arr, Array.from({ length: nDims }, (_, i) => i).reverse());
  if (axes.length !== nDims) throw new Error(`Axes must have length ${nDims}. Found ${axes.length}`);
  axes = axes.map(ax => normalize_axis(ax, nDims));
  const seen = new Array(nDims).fill(false);
  for (const ax of axes) { if (seen[ax]) throw new Error(`repeated axis in transpose`); seen[ax] = true; }
  const srcShifts = shape_shifts(arr.shape);
  const shape = axes.map((j) => arr.shape[j]);
  const shifts = axes.map((j) => srcShifts[j]);
  const src = arr.flat;
  const n = src.length;
  const buffer = new_buffer(n, arr.dtype);
  if (n > 0) {
    const tuple = new Array(nDims).fill(0);
    let cursor = 0;
    for (let i = 0; i < n; i++) {
      buffer[i] = src[cursor];
      let axis = nDims - 1;
      while (axis >= 0) {
        tuple[axis]++;
        cursor += shifts[axis];
        if (tuple[axis] < shape[axis]) break;
        cursor -= shifts[axis] * shape[axis];
        tuple[axis] = 0;
        axis--;
      }
    }
  }
  return new NDArray(buffer, shape, arr.dtype);
}


export function swapAxes(arr: NDArray, axisA: number, axisB: number) {
  arr = asarray(arr)
  const nDims = arr.shape.length;
  axisA = normalize_axis(axisA, nDims);
  axisB = normalize_axis(axisB, nDims);
  const perm = Array.from({ length: nDims }, (_, i) => i);
  perm[axisA] = axisB;
  perm[axisB] = axisA;
  return transpose(arr, perm);
}


export function concatenate(arrays: NDArray[], axis: number | null = 0) {
  if ((axis as any) instanceof Object) ({ axis } = axis as any);
  if (isarray(arrays)) arrays = [...arrays];
  arrays = arrays.map((a) => asarray(a));
  if (!arrays.length) throw new Error(`Need at least one array to concatenate`);
  if (axis == null) {
    arrays = arrays.map(arr => ravel(arr));
    axis = 0;
  }
  const nDims = arrays[0].shape.length;
  if (nDims == 0) throw new Error(`zero-dimensional arrays cannot be concatenated`);
  axis = normalize_axis(axis, nDims);
  const shapeIn = [...arrays[0].shape];
  const dtype = dtype_max(...arrays.map(arr => arr.dtype));
  for (const arr of arrays) {
    if (arr.shape.length != nDims || !allEq(arr.shape.filter((_, i) => i != axis), shapeIn.filter((_, i) => i != axis)))
      throw new Error(`all the input array dimensions except for the concatenation axis must match exactly. Found shapes ${JSON.stringify(shapeIn)} and ${JSON.stringify(arr.shape)}`);
  }
  const shape = shapeIn.slice();
  shape[axis] = arrays.reduce((n, arr) => n + arr.shape[axis], 0);
  const outer = prod(shape.slice(0, axis));
  const inner = prod(shape.slice(axis + 1));
  const buffer = new_buffer(prod(shape), dtype);
  const rowLen = shape[axis] * inner;
  let offset = 0;
  for (const arr of arrays) {
    const flat = arr.flat;
    const chunk = arr.shape[axis] * inner;
    for (let o = 0; o < outer; o++) {
      for (let k = 0; k < chunk; k++) {
        const v = flat[o * chunk + k];
        buffer[o * rowLen + offset + k] = dtype === bool ? (v ? 1 : 0) : v;
      }
    }
    offset += chunk;
  }
  return new NDArray(buffer, shape, dtype);
}


export function stack(arrays: NDArray[], axis: number = 0) {
  if ((axis as any) instanceof Object) ({ axis } = axis as any);
  if (isarray(arrays)) arrays = [...arrays];
  if (!Array.isArray(arrays)) throw new Error(`Expected list of arrays. Found ${typeof arrays}`);
  arrays = arrays.map((a) => asarray(a));
  if (!arrays.length) throw new Error(`Need at least one array to stack`);
  const shapeIn = [...arrays[0].shape];
  axis = normalize_axis(axis, shapeIn.length + 1);
  const shapeBroadcast = [...shapeIn.slice(0, axis), 1, ...shapeIn.slice(axis)];
  const bArrays = [];
  for (let arr of arrays) {
    if (!allEq(arr.shape, shapeIn)) {
      throw new Error(`all input arrays must have the same shape. Found ${JSON.stringify(shapeIn)} and ${JSON.stringify(arr.shape)}`);
    }
    bArrays.push(reshape(arr, shapeBroadcast));
  }
  return concatenate(bArrays, axis);
}


/** Indirect stable sort with multiple keys. The last key is the primary sort key. keys: array of equally-shaped arrays */
export function lexsort(keys: NDArray[], axis: number): NDArray {
  let ks = (isarray(keys) && (keys as any).shape.length > 1) ? Array.from({ length: (keys as any).shape[0] }, (_, i) => (keys as any).index(i) as NDArray) : (keys as NDArray[]).map(k => asarray(k));
  if (!ks.length) throw new Error(`need sequence of keys with len > 0 in lexsort`);
  const shape = ks[0].shape;
  for (const k of ks) if (k.shape.length != shape.length || k.shape.some((d, i) => d != shape[i])) throw new Error(`all keys need to be the same shape`);
  if (shape.length == 0) return new NDArray(new_buffer([0], int32), [], int32);
  axis = normalize_axis(axis, shape.length);
  // Sort along the last axis of each key
  const moved = ks.map(k => axis == shape.length - 1 ? k : swapAxes(k, axis, -1));
  const n = shape[axis];
  const flats = moved.map(k => k.flat as any);
  const nLanes = n == 0 ? 0 : flats[0].length / n;
  const buffer = new_buffer(nLanes * n, int32);
  for (let l = 0; l < nLanes; l++) {
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((i, j) => {
      for (let k = flats.length - 1; k >= 0; k--) {
        const c = cmp_nan_at_the_end(flats[k][l * n + i], flats[k][l * n + j]);
        if (c) return c;
      }
      return 0;
    });
    for (let i = 0; i < n; i++) buffer[l * n + i] = idx[i];
  }
  const out = new NDArray(buffer, moved[0].shape, int32);
  return axis == shape.length - 1 ? out : swapAxes(out, axis, -1);
}

/** Partial sort: element at each kth position is in its sorted position, smaller before, larger after */
export function partition(a: NDArray, kth: number | number[], axis: number): NDArray {
  a = asarray(a);
  const nd = a.shape.length;
  if (nd == 0) throw new Error(`Cannot partition a 0-d array`);
  axis = normalize_axis(axis, nd);
  const n = a.shape[axis];
  const ks = (Array.isArray(kth) ? kth : [kth]).map(k => { if (k < -n || k >= n) throw new Error(`kth(=${k}) out of bounds (${n})`); return k < 0 ? k + n : k; });
  // A full sort is a valid partition
  return sort(a, axis);
}

export function argpartition(a: NDArray, kth: number | number[], axis: number): NDArray {
  a = asarray(a);
  const nd = a.shape.length;
  if (nd == 0) throw new Error(`Cannot partition a 0-d array`);
  axis = normalize_axis(axis, nd);
  const n = a.shape[axis];
  (Array.isArray(kth) ? kth : [kth]).forEach(k => { if (k < -n || k >= n) throw new Error(`kth(=${k}) out of bounds (${n})`); });
  return argsort(a, axis);
}


/**
 * numpy.apply_along_axis: applies func1d(lane, ...args) to every 1D lane of `arr` along `axis`.
 * func1d may return a scalar or an array; the axis is replaced by the shape of the result.
 */
export function apply_along_axis_func(func1d: (lane: NDArray, ...args: any[]) => any, axis: number, arr: NDArray, args: any[] = []): NDArray {
  arr = asarray(arr);
  const nd = arr.shape.length;
  if (nd == 0) throw new Error(`apply_along_axis requires an array with at least one dimension`);
  axis = normalize_axis(axis, nd);
  const moved = axis == nd - 1 ? arr : swapAxes(arr, axis, nd - 1);
  const n = moved.shape[nd - 1];
  const flat = moved.flat;
  const nLanes = n == 0 ? prod(moved.shape.slice(0, -1)) : flat.length / n;
  const results: NDArray[] = [];
  for (let l = 0; l < nLanes; l++) {
    const lane = new NDArray(flat.slice(l * n, (l + 1) * n), [n], arr.dtype);
    results.push(asarray(func1d(lane, ...args)));
  }
  if (nLanes == 0) throw new Error(`Cannot apply_along_axis when any iteration dimensions are 0`);
  const inner = results[0].shape;
  const size = prod(inner);
  const dtype = results.map(r => r.dtype).reduce((a, b) => dtype_max(a, b));
  const buffer = new_buffer(nLanes * size, dtype);
  results.forEach((r, l) => {
    if (r.shape.length != inner.length || r.shape.some((d, i) => d != inner[i])) throw new Error(`func1d must return arrays of the same shape for every lane`);
    const f = r.flat;
    for (let i = 0; i < size; i++) buffer[l * size + i] = dtype === bool ? (f[i] ? 1 : 0) : f[i];
  });
  fix_buffer(buffer, dtype);
  const lead = moved.shape.slice(0, -1);
  // result: lead dims + inner dims; then move the inner dims to the position of `axis`
  const out = new NDArray(buffer, [...lead, ...inner], dtype);
  if (axis == nd - 1) return out;
  const k = inner.length;
  const perm: number[] = [];
  for (let i = 0; i < axis; i++) perm.push(i);
  for (let j = 0; j < k; j++) perm.push(lead.length + j);
  for (let i = axis; i < lead.length; i++) perm.push(i);
  return transpose(out, perm);
}
