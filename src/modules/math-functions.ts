//@ts-check
import { Arr } from "../array/kwargs";
import { Func_clip } from "./math-functions.kwargs";
import { op_binary } from "../array/operators";
import { apply_along_axis, concatenate } from "../array/transform";


const { max, min } = op_binary;

export function clip(a: Arr, a_min: Arr, a_max: Arr, out: Arr): Arr {
  min(a, a_max, out);
  max(a, a_min, out);
  return out;
}

export function prepend_append(a: Arr, axis: number, prepend: Arr | null, append: Arr | null) {
  if (prepend !== null || append !== null) {
    const arrs: Arr[] = [];
    if (prepend !== null) arrs.push(prepend);
    arrs.push(a);
    if (append !== null) arrs.push(append);
    a = concatenate(arrs, axis);
  }
  return a;
}

export function diff(a: Arr, axis: number, prepend: Arr | null, append: Arr | null) {
  a = prepend_append(a, axis, prepend, append);
  return apply_along_axis(a, axis, (arr) => {
    return arr.map((v, i) => i + 1 == arr.length ? 0 : arr[i + 1] - v).splice(0, arr.length - 1);
  }, a.dtype);
}

export function cumsum(a: Arr, axis: number, prepend: Arr | null, append: Arr | null) {
  a = prepend_append(a, axis, prepend, append);
  return apply_along_axis(a, axis, (arr) => {
    let sum = 0;
    return arr.map((v) => sum += v);
  }, a.dtype);
}

export function nancumsum(a: Arr, axis: number, prepend: Arr | null, append: Arr | null) {
  a = prepend_append(a, axis, prepend, append);
  return apply_along_axis(a, axis, (arr) => {
    let sum = 0;
    return arr.map((v) => isNaN(v) ? sum : (sum += v));
  }, a.dtype);
}


export function cumprod(a: Arr, axis: number, prepend: Arr | null, append: Arr | null) {
  a = prepend_append(a, axis, prepend, append);
  return apply_along_axis(a, axis, (arr) => {
    let prod = 1;
    return arr.map((v) => prod *= v);
  }, a.dtype);
}

export function nancumprod(a: Arr, axis: number, prepend: Arr | null, append: Arr | null) {
  a = prepend_append(a, axis, prepend, append);
  return apply_along_axis(a, axis, (arr) => {
    let prod = 1;
    return arr.map((v) => isNaN(v) ? prod : (prod *= v));
  }, a.dtype);
}


export function cumtrapz(y: Arr, x: Arr | null, dx: number, axis: number, initial: 0 | null) {
  return apply_along_axis(y, axis, (arr) => {
    let sum = 0;
    return arr.map((v, i) => {
      if (i == 0) return 0;
      const dxi = x ? x[i] - x[i - 1] : dx;
      return sum += (v + arr[i - 1]) / 2 * dxi;
    }).splice(initial ? 0 : 1);
  }, y.dtype);
}




export const kw_exported = {
  clip: Func_clip.defaultDecorator(clip),
};