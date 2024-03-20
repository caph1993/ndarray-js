//@ts-check

import { isarray, asarray, new_NDArray, _NDArray, new_from, number_collapse, ravel, shape_shifts, reshape } from './basic';
import { fromlist } from './js-interface';
import { allEq, extend } from '../utils-js';
import { DType } from '../NDArray';
import type NDArray from "../NDArray";
import { ArrayOrConstant } from './operators';

export function apply_along_axis(arr: NDArray, axis: number, transform, dtype: DType = Number): ArrayOrConstant {
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
  const groups = Array.from({ length: m }, (_) =>/**@type {number[]}*/([]));
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
  const out = new_NDArray(tmp.flat, shape, dtype);
  return number_collapse(out);
}


export function sort(A: NDArray, axis = -1) {
  ({ axis } = Object.assign({ axis }, this));
  //@ts-ignore
  if (axis instanceof Object) ({ axis } = axis);
  return apply_along_axis(A, axis, (arr) => {
    const cpy = [...arr];
    cpy.sort((a, b) => a - b);
    return cpy;
  }, A.dtype);
}

export function transpose(arr: NDArray, axes: null | number[] = null) {
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
  return new_NDArray(indices.map((i) => src[i]), shape, arr.dtype);
}


export function swapAxes(arr: NDArray, axisA: number, axisB: number) {
  arr = asarray(arr)
  const nDims = arr.shape.length;
  if (axisA < 0) axisA = nDims + axisA;
  if (axisB < 0) axisB = nDims + axisB;
  const perm = Array.from({ length: nDims }, (_, i) => i);
  perm[axisA] = axisB;
  perm[axisB] = axisA;
  return transpose(arr, perm);
}


export function concatenate(arrays: NDArray[], axis: number | null = null) {
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
  const out = new_NDArray(flat, shape, arrays[0].dtype);
  if (axis == 0) return out;
  else return swapAxes(out, axis, 0);
}


export function stack(arrays: NDArray[], axis: number = 0) {
  // ({ axis } = Object.assign({ axis }, this));
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
    if (!allEq(arr.shape, shapeIn)) throw new Error(`Inconsistent input shape ${arr.shape} with respect to ${arr.shape}`);
    bArrays.push(reshape(arr, shapeBroadcast));
  }
  return concatenate(bArrays, axis);
}
