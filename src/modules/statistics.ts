//@ts-check
import assert = require("assert");
import NDArray from "../NDArray";
import { ArrOrAny, asarray, new_NDArray } from "../array/_globals";
import { as_number } from "../array/basic";
import { Arr, AxisArg } from "../array/kwargs";

import { cmp_nan_at_the_end, swapAxes } from "../array/transform";
import { np } from "./_globals";




export const sum_list = (arr: number[] | boolean[]) => arr.reduce((a, b) => (a as number) + (b as number), 0) as number;
export const product_list = (arr: number[] | boolean[]) => arr.reduce((a, b) => (a as number) + (b as number), 1) as number;
export function argmax_list(arr: number[] | boolean[]) {
  if (arr.length == 0) throw new Error("argmax of an empty array");
  let i = 0, v = arr[0];
  for (let j = 1; j < arr.length; j++) if (arr[j] > v) v = arr[i = j];
  return i;
}
export function argmin_list(arr: number[] | boolean[]) {
  if (arr.length == 0) throw new Error("argmin of an empty array");
  let i = 0, v = arr[0];
  for (let j = 1; j < arr.length; j++) if (arr[j] < v) v = arr[i = j];
  return i;
}
export function max_list<T>(arr: T[]): T {
  if (arr.length == 0) throw new Error("max of an empty array");
  if (typeof arr[0] == "boolean") return any_list(arr as boolean[]) as T;
  return Math.max(...arr as number[]) as T;
}
export function min_list<T>(arr: T[]): T {
  if (arr.length == 0) throw new Error("min of an empty array");
  if (typeof arr[0] == "boolean") return all_list(arr as boolean[]) as T;
  return Math.min(...arr as number[]) as T;
}
export const any_list = (arr: number[] | boolean[]) => {
  for (let x of arr) if (x) return true;
  return false;
};
export const all_list = (arr: number[] | boolean[]) => {
  for (let x of arr) if (!x) return false;
  return true;
}
export const mean_list = (arr: number[] | boolean[]) => sum_list(arr) / arr.length;
export const var_list = (arr: number[] | boolean[], ddof: number = 0) => {
  const mean = mean_list(arr);
  const n = arr.length;
  return sum_list(arr.map(x => (x - mean) ** 2)) / (n - ddof);
}
export const std_list = (arr: number[] | boolean[], ddof: number = 0) => {
  return Math.pow(var_list(arr, ddof), 0.5);
}


/**
 * a must be a list.
 * q can be a number in [0,1] or a list.
 * the output is shaped as q.
 * @param a
 * @param q
 * @param _nan_handling 
 * @returns 
 */
function quantile_list<T extends number | number[] = number>(
  a: number[] | boolean[],
  q: T,
  _nan_handling = false,
): T {
  const x = a.map(v => +v).sort(cmp_nan_at_the_end);
  let n = x.length;
  if (_nan_handling) n -= x.reduce((cum, x) => cum + (isNaN(x) ? 1 : 0), 0);
  const f = (q: number) => {
    const nq = q * (n - 1);
    const lo = Math.floor(nq);
    if (nq == lo) return x[lo];
    const hi = lo + 1;
    if (nq == hi) return x[hi];
    return x[lo] * (hi - nq) + x[hi] * (nq - lo);
  }
  return (Array.isArray(q) ? q.map(f) : f(q)) as T;
}

// const norm_list = (arr: number[], ord: number = 2) => {
//   if (ord % 2 != 0) arr = abs_list(arr);
//   if (ord == Infinity) return max_list(arr);
//   if (ord == 1) return sum_list(arr);
//   return Math.pow(sum_list(pow_list(arr, ord)), 1 / ord);
// }


// MOVE THIS!!!
export function linspace_list(start: number, stop: number, steps = 50, endpoint = true) {
  let n = (steps - (endpoint ? 1 : 0));
  const shift = (stop - start) / n;
  let arr = new Array(steps);
  for (let i = 0; i < steps; i++) arr[i] = start + i * shift;
  return arr;
}
export const median_list = (a: number[] | boolean[]) => quantile_list(a, 0.5);





/**
 * Compute the q-th percentile of the data along the specified axis.
 */
export function percentile(a: Arr, q: Arr | number, axis: AxisArg | null) {
  q = np.divide(q, 100);
  return quantile(a, q, axis);
}

/**
 * Compute the q-th quantile of the data along the specified axis.
 */
export function quantile(a: Arr, q: Arr | number, axis: number, _nan_handling = false): Arr {
  q = asarray(q);
  if (axis != a.shape.length - 1) { a = swapAxes(a, axis, -1); }
  const outer_shape = q.shape;
  const inner_shape = a.shape.slice(0, -1);
  a = a.reshape(-1, a.shape.at(-1));
  q = q.reshape(-1);
  const [nrows, ncols] = a.shape;
  const a_flat = a.flat;
  const q_flat = q.flat;
  const out = new Float64Array(q.size * nrows);
  for (let i = 0; i < nrows; i++) {
    const row = a_flat.slice(i * ncols, (i + 1) * ncols);
    const values = quantile_list(row, q_flat, _nan_handling);
    let j = i;
    for (let k in values) {
      out[j] = values[k];
      j += nrows;
    }
  }
  return new_NDArray(out, [...outer_shape, ...inner_shape]);

  // a = np.sort(a, { axis: null });
  // const n = _nan_handling ? np.isnan(a).logical_not().sum(0) : a.shape[0];
  // // TODO: throw if out of bounds 
  // const indices_float = op_binary["*"](q, op_binary["-"](n, 1));
  // return n_ary_operation([indices_float], a.shape.slice(1), (iq) => {
  //   const lo = Math.floor(iq);
  //   if (iq == lo) return np.asarray(a.index(iq));
  //   const hi = lo + 1;
  //   if (iq == hi) return np.asarray(a.index(iq));
  //   return average(a.index(`${lo}:${hi + 1}`), 0, asarray([hi - iq, iq - lo]), false);
  // });
}

/**
 * Compute the median along the specified axis.
 */
export function median(a: Arr, axis: AxisArg | null, keepdims: boolean) {
  let out = quantile(a, 0.5, axis);
  if (keepdims) {
    const shape = a.shape.slice();
    shape[axis] = 1;
    out = out.reshape(shape);
  }
  return out;
}

/**
 * Compute the weighted average along the specified axis.
 */
export function average(a: Arr, axis: AxisArg | null, weights: NDArray | null, keepdims: boolean) {
  if (weights === null) return np.mean(a, axis);
  // MISSING: assert weights is 1D
  const denominator = np.sum(weights);
  const weights_shape = a.shape.map(() => 1);
  weights_shape[axis] = weights.size;
  weights = weights.reshape(weights_shape);
  const numerator = np.sum(np.multiply(a, weights), axis, keepdims);
  return np.asarray(np.divide(numerator, denominator));
}


/**
 * Compute the q-th percentile of the data along the specified axis, while ignoring nan values.
*/
export function nanpercentile(a: Arr, q: Arr | number, axis: AxisArg | null) {
  q = np.divide(q, 100);
  return nanquantile(a, q, axis);
}


/**
 * Compute the q-th quantile of the data along the specified axis, while ignoring nan values.
 */
export function nanquantile(a: Arr, q: Arr | number, axis: AxisArg | null) {
  return quantile(a, q, axis, true)
}


/**
 * Compute the median along the specified axis, while ignoring NaNs.
 */
export function nanmedian(a: Arr, axis: number, keepdims: boolean) {
  let out = nanquantile(a, 0.5, axis);
  if (keepdims) {
    const shape = a.shape.slice();
    shape[axis] = 1;
    out = out.reshape(shape);
  }
  return out;
}





// /**
//  * Return Pearson product-moment correlation coefficients.
//  */
// export function corrcoef(x: Arr, y: Arr, rowvar = true, bias = false, ddof = 0, dtype = Number) {
//   const x_shape = x.shape;
//   const y_shape = y.shape;
//   if (x_shape.length !== 1 || y_shape.length !== 1) throw new Error("Expected 1D arrays");
//   if (x_shape[0] !== y_shape[0]) throw new Error("Expected arrays of the same length");
//   const x_mean = np.mean(x);
//   const y_mean = np.mean(y);
//   const x_std = np.std(x);
//   const y_std = np.std(y);
//   const n = x_shape[0];
//   const cov = np.mean(np.multiply(np.subtract(x, x_mean), np.subtract(y, y_mean)));
//   const corr = np.divide(cov, np.multiply(x_std, y_std));
//   return corr;
// }

// /**
//  * Cross-correlation of two 1-dimensional sequences.
//  */
// export function correlate(a: Arr, v: Arr, mode) {
//   throw new Error("Not implemented");
// }

// /**
//  * Estimate a covariance matrix, given data and weights.
//  */
// export function cov(m: Arr, y: Arr, rowvar = true, bias = false, ddof = 0, fweights = null, aweights = null) {
//   const m_shape = m.shape;
//   const y_shape = y.shape;
//   if (m_shape.length !== 2 || y_shape.length !== 2) throw new Error("Expected 2D arrays");
//   if (rowvar) {
//     m = np.transpose(m);
//     y = np.transpose(y);
//   }
//   const m_mean = np.mean(m, 1);
//   const y_mean = np.mean(y, 1);
//   const m_centered = np.subtract(m, m_mean.reshape(m_shape[0], 1));
//   const y_centered = np.subtract(y, y_mean.reshape(y_shape[0], 1));
//   const m_weights = fweights || np.ones(m_shape[1]);
//   const y_weights = aweights || np.ones(y_shape[1]);
//   const m_weighted = np.multiply(m_centered, m_weights);
//   const y_weighted = np.multiply(y_centered, y_weights);
//   const cov = np.divide(np.sum(np.multiply(m_weighted, y_weighted), 1), m_shape[1] - ddof);
//   return cov;
// }

// /**
//  * Compute the histogram of a dataset.
//  */
// export function histogram(a: Arr, bins, range, density, weights) {
//   throw new Error("Not implemented");
// }

// /**
//  * Compute the bi-dimensional histogram of two data samples.
//  */
// export function histogram2d(x: Arr, y: Arr, bins, range, density, weights) {
//   throw new Error("Not implemented");
// }

// /**
//  * Compute the multidimensional histogram of some data.
//  */
// export function histogramdd(sample: Arr, bins, range, density, weights) {
//   throw new Error("Not implemented");
// }

// /**
//  * Count number of occurrences of each value in array of non-negative ints.
//  */
// export function bincount(x: Arr, weights, minlength) {
//   throw new Error("Not implemented");
// }

/**
 * Function to calculate only the edges of the bins used by the histogram function.
 */

export function _n_bins(a: number[], bins: number | number[] | 'auto' | 'rice' | 'fd' | 'scott' | 'sqrt' | 'sturges', range: [number, number]) {
  const n = a.length;
  if (bins === 'auto') {
    return Math.max(_n_bins(a, 'fd', range), _n_bins(a, 'sturges', range));
  }
  if (bins == 'sturges') {
    return Math.ceil(Math.log2(n) + 1);
  }
  if (bins == 'fd') {
    const IQR = quantile_list(a, 0.75) - quantile_list(a, 0.25);
    return Math.ceil(2 * IQR / Math.pow(n, 1 / 3));
  }
  if (bins == 'rice') {
    const nh = 2 * Math.pow(n, 1 / 3);
    return Math.ceil(n / nh);
  }
  if (bins == 'scott') {
    const std = std_list(a);
    const h = std * Math.pow(24 * Math.sqrt(Math.PI) / n, 1 / 3);
    return Math.ceil((max_list(a) - min_list(a)) / h);
  }
  if (bins == 'sqrt') {
    return Math.ceil(Math.sqrt(n));
  }
  throw new Error("Unsupported bins value");
}

export function histogram_bin_edges_list(a: number[], bins: number | number[] | 'auto' | 'rice' | 'fd' | 'scott' | 'sqrt' = 'auto', range: [number, number] | null = null): number[] {
  if (range === null) range = [min_list(a), max_list(a)];
  else {
    // Exclude values outside the range
    a = a.filter(x => x >= range[0] && x <= range[1]);
  }
  if (typeof bins === 'string') bins = _n_bins(a, bins, range);
  if (typeof bins === 'number') bins = linspace_list(min_list(a), max_list(a), bins + 1);
  assert(Array.isArray(bins), "bins must have at least 2 elements");
  return bins;
}


export function histogram_list(a: number[], bins: number | number[] | 'auto' | 'rice' | 'fd' | 'scott' | 'sqrt', range: [number, number] | null = null, weights: number[] | null = null): [number[], number[]] {
  const edges = histogram_bin_edges_list(a, bins, range);
  const counts = new Array(edges.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    let j = 0;
    while (j < edges.length - 1 && x >= edges[j + 1]) j++;
    counts[j] += weights ? weights[i] : 1;
  }
  return [edges, counts];
}


// /**
//  * Return the indices of the bins to which each value in input array belongs.
//  */
export function digitize_list(x: number[], bins: number[], right = false): number[] {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    let j = 0;
    while (j < bins.length && (right ? xi > bins[j] : xi >= bins[j])) j++;
    out[i] = j;
  }
  return out;
}



/*

Correlating
  - corrcoef(x[, y, rowvar, bias, ddof, dtype]) Return Pearson product-moment correlation coefficients.
  - correlate(a, v[, mode]) Cross-correlation of two 1-dimensional sequences.
  - cov(m[, y, rowvar, bias, ddof, fweights, ...]) Estimate a covariance matrix, given data and weights.

Histograms
 - histogram(a[, bins, range, density, weights]) Compute the histogram of a dataset.
 - histogram2d(x, y[, bins, range, density, ...]) Compute the bi-dimensional histogram of two data samples.
 - histogramdd(sample[, bins, range, density, ...]) Compute the multidimensional histogram of some data.
 - bincount(x, /[, weights, minlength]) Count number of occurrences of each value in array of non-negative ints.
 - histogram_bin_edges(a[, bins, range, weights]) Function to calculate only the edges of the bins used by the histogram function.
 - digitize(x, bins[, right]) Return the indices of the bins to which each value in input array belongs.

Averages and variances

  - median(a[, axis, out, overwrite_input, keepdims]) Compute the median along the specified axis.
  - average(a[, axis, weight  s, returned, keepdims]) Compute the weighted average along the specified axis.
  - mean(a[, axis, dtype, out, keepdims, where]) Compute the arithmetic mean along the specified axis.
  - std(a[, axis, dtype, out, ddof, keepdims, where]) Compute the standard deviation along the specified axis.
  - var(a[, axis, dtype, out, ddof, keepdims, where]) Compute the variance along the specified axis.
  - nanmedian(a[, axis, out, overwrite_input, ...]) Compute the median along the specified axis, while ignoring NaNs.
  - nanmean(a[, axis, dtype, out, keepdims, where]) Compute the arithmetic mean along the specified axis, ignoring NaNs.
  - nanstd(a[, axis, dtype, out, ddof, ...]) Compute the standard deviation along the specified axis, while ignoring NaNs.
  - nanvar(a[, axis, dtype, out, ddof, ...]) Compute the variance along the specified axis, while ignoring NaNs.
 */

