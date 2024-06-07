//@ts-check
import { Arr } from "../array/_globals";
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

export function trapz(y: Arr, x: Arr | null, dx: number, axis: number) {
  return apply_along_axis(y, axis, (arr: any[]) => {
    let sum = 0;
    for (let i = 1; i < arr.length; i++) {
      const dxi = x ? x[i] - x[i - 1] : dx;
      sum += (arr[i] + arr[i - 1]) / 2 * dxi;
    }
    return sum;
  }, y.dtype);
}

export function interp(x: Arr, xp: Arr, fp: Arr, left: number | null, right: number | null) {
  const x_flat = x.flat;
  const xp_flat: number[] = xp.flat;
  const fp_flat = fp.flat;
  // Use binary search instead of findIndex:
  return x_flat.map((v) => {
    let lo = 0, hi = xp_flat.length;
    while (lo < hi) {
      const mid = lo + hi >> 1;
      if (xp_flat[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    if (lo == 0) return left;
    if (lo == xp_flat.length) return right;
    const x0 = xp_flat[lo - 1];
    const x1 = xp_flat[lo];
    const f0 = fp_flat[lo - 1];
    const f1 = fp_flat[lo];
    return f0 + (f1 - f0) * (v - x0) / (x1 - x0);
  });
}


