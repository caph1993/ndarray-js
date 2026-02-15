//@ts-check

import { isarray, asarray, new_NDArray, _NDArray, new_from, number_collapse, ravel, shape_shifts, reshape } from './basic';
import { fromlist } from './js-interface';
import { allEq, extend } from '../utils-js';
import type NDArray from "../NDArray";
import { ArrayOrConstant } from './operators';
import { AxisArg, Func_a_lastAxis } from './kwargs';
import { TypedArrayConstructor, dtype_least_ancestor, new_buffer } from '../dtypes';


/**
 * This function can reduce, sort, operate pointwise, or increase the dimensionality.
 */
export function apply_along_axis(arr: NDArray<any>, axis: number, transform, dtype: TypedArrayConstructor = Float64Array): ArrayOrConstant {
  arr = asarray(arr);
  if (axis == null) return transform(arr.flat);
  const nDims = arr.shape.length;
  if (axis < 0) axis = nDims + axis;
  if (axis !== nDims - 1) {
    // Transpose to end, apply, and transpose back:
    const tmp = swapAxes(arr, axis, -1);
    const out = apply_along_axis(tmp, -1, transform, dtype);
    //@ts-ignore
    return swapAxes(out, axis, -1);
  }

  let m = arr.shape[axis];
  let shift = shape_shifts(arr.shape)[axis];
  const groups = Array.from({ length: m }, (_) => ([] as number[]));
  arr.flat.forEach((value, i) => groups[(Math.floor(i / shift)) % m].push(value));
  // Transpose it:
  let nCols = arr.size / m;
  const groupsT = [];
  for (let j = 0; j < nCols; j++) {
    const newRow = [];
    for (let i = 0; i < m; i++) newRow.push(groups[i][j]);
    groupsT.push(newRow);
  }
  const data = groupsT.map(transform);
  const tmp = fromlist(data);
  const shape = [...arr.shape.slice(0, axis), ...tmp.shape.slice(1), ...arr.shape.slice(axis + 1),];

  const out = new_NDArray(new_buffer(tmp.flat, dtype), shape);
  return number_collapse(out);
}

export const cmp_nan_at_the_end = (a: number, b: number) => {
  // https://stackoverflow.com/a/56265258/3671939
  //@ts-ignore
  return a - b || Number.isNaN(a) - Number.isNaN(b) || Object.is(b, 0) - Object.is(a, 0);
}

export function sort(a: NDArray<any>, axis: number) {
  return apply_along_axis(a, axis, (arr) => {
    const cpy = [...arr];
    cpy.sort(cmp_nan_at_the_end)
    return cpy;
  }, a.dtype) as NDArray;
}

export function argsort(a: NDArray<any>, axis: number) {
  return apply_along_axis(a, axis, (arr) => {
    const idx = Array.from(arr).map((_: any, i: number) => i);
    idx.sort((i: number, j: number) => cmp_nan_at_the_end(arr[i], arr[j]));
    return idx;
  }, Int32Array) as NDArray;
}

export function transpose(arr: NDArray<any>, axes: null | number[] = null) {
  ({ axes } = Object.assign({ axes }, this));
  //@ts-ignore
  if (axes !== null && axes["axes"]) ({ axes } = axes);

  let nDims = arr.shape.length;
  if (axes == null) return transpose(arr, Array.from({ length: nDims }, (_, i) => i).reverse());
  if (axes.length !== nDims) throw new Error(`Axes must have length ${nDims}. Found ${axes.length}`);
  let inv = Array.from({ length: nDims }, () => -1);

  for (let i = 0; i < nDims; i++) {
    if (axes[i] < 0 || axes[i] >= nDims) throw new Error(`Expected axis in [0..${nDims}). Found ${axes[i]}`)
    inv[axes[i]] = i;
  }
  for (let i = 0; i < nDims; i++) if (inv[i] == -1) throw new Error(`Axes must contain all dimensions. [${axes.join(", ")}] is missing ${i}.`);
  const srcShifts = shape_shifts(arr.shape);
  let shape = axes.map((j) => arr.shape[j]);
  let shifts = axes.map((j) => srcShifts[j]);
  // Copied from slice:
  const indices = [];
  const tuple = new Array(shape.length).fill(0);
  let cursor = 0;
  while (true) {
    if (!isFinite(cursor)) throw new Error(`Programming error`);
    indices.push(cursor);
    let axis = nDims - 1;
    while (axis >= 0) {
      tuple[axis]++;
      cursor += shifts[axis];
      if (tuple[axis] < shape[axis]) break;
      // Overflow
      cursor -= shifts[axis] * shape[axis];
      tuple[axis] = 0;
      axis--;
    };
    if (axis < 0) break;
  }
  // Now, just copy the data:
  const src = arr.flat;
  return new_NDArray(arr.dtype.from(indices.map((i) => src[i])), shape);
}


export function swapAxes(arr: NDArray<any>, axisA: number, axisB: number) {
  arr = asarray(arr)
  const nDims = arr.shape.length;
  if (axisA < 0) axisA = nDims + axisA;
  if (axisB < 0) axisB = nDims + axisB;
  const perm = Array.from({ length: nDims }, (_, i) => i);
  perm[axisA] = axisB;
  perm[axisB] = axisA;
  return transpose(arr, perm);
}


export function concatenate(arrays: NDArray<any>[], axis: number | null = null) {
  // ({ axis } = Object.assign({ axis }, this));
  //@ts-ignore
  if (axis instanceof Object) ({ axis } = axis);
  if (isarray(arrays)) arrays = [...arrays];
  arrays = arrays.map(asarray);
  if (axis == null) {
    arrays = arrays.map(arr => ravel(arr));
    axis = 0;
  }
  if (!arrays.length) throw new Error(`Expected at least two arrays`);
  const nDims = arrays[0].shape.length;
  if (axis < 0) axis = nDims + axis;
  const shapeIn = [...arrays[0].shape];
  const flat = [];
  const shape = shapeIn.map((_, i) => i == 0 ? 0 : shapeIn[i == axis ? 0 : i]);
  for (let arr of arrays) {
    if (!allEq(arr.shape.filter((_, i) => i != axis), shapeIn.filter((_, i) => i != axis))) throw new Error(`Inconsistent input shape ${shapeIn} with respect to ${arr.shape.map((v, i) => i == axis ? '?' : v)}`);
    shape[0] += arr.shape[axis];
    arr = axis == 0 ? arr : swapAxes(arr, axis, 0);
    extend(flat, arr.flat);
  }
  // TO DO: infer or expect dtype here:
  const dtype = dtype_least_ancestor(...arrays.map(arr => arr.dtype));
  const out = new_NDArray(dtype.from(flat), shape);
  if (axis == 0) return out;
  else return swapAxes(out, axis, 0);
}


export function stack(arrays: NDArray[], axis: number = 0) {
  //@ts-ignore
  if (axis instanceof Object) ({ axis } = axis);
  if (isarray(arrays)) arrays = [...arrays];
  if (!Array.isArray(arrays)) throw new Error(`Expected list of arrays. Found ${typeof arrays}`);
  arrays = arrays.map(asarray);
  if (!arrays.length) throw new Error(`Expected at least two arrays`);
  const shapeIn = [...arrays[0].shape];
  if (axis < 0) axis = shapeIn.length + 1 + axis;
  const shapeBroadcast = [...shapeIn.slice(0, axis), 1, ...shapeIn.slice(axis)];
  const bArrays = [];
  for (let arr of arrays) {
    if (!allEq(arr.shape, shapeIn)) {
      throw new Error(`Inconsistent input shape ${JSON.stringify(shapeIn)} with respect to ${JSON.stringify(arr.shape)}`);
    }
    bArrays.push(reshape(arr, shapeBroadcast));
  }
  return concatenate(bArrays, axis);
}







export const kw_exported = {
  sort: Func_a_lastAxis.defaultDecorator(sort),
  argsort: Func_a_lastAxis.defaultDecorator(argsort),
}