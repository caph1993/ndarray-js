//@ts-check
import NDArray from "../NDArray";
import { asarray, new_NDArray } from "../array/_globals";
import { AxisArg, Func_a_q_axis, Func_a_axis_keepdims } from "../array/kwargs";
import { n_ary_operation, op_binary } from "../array/operators";
import { cmp_nan_at_the_end, swapAxes } from "../array/transform";
import { np } from "./_globals";


/**
 * Compute the q-th percentile of the data along the specified axis.
 */
export function percentile(a: NDArray<any>, q: NDArray<any> | number, axis: AxisArg | null) {
  q = np.divide(q, 100);
  return quantile(a, q, axis);
}


function js_quantile_a_1D_q_01D(
  a_flat: number[],
  q_flat: number[],
  _nan_handling = false,
): number[] {
  a_flat = [...a_flat].sort(cmp_nan_at_the_end);
  let n = a_flat.length;
  if (_nan_handling) n -= a_flat.reduce((cum, x) => cum + (isNaN(x) ? 1 : 0), 0);
  return q_flat.map(q => {
    const nq = q * (n - 1);
    const lo = Math.floor(nq);
    if (nq == lo) return a_flat[lo];
    const hi = lo + 1;
    if (nq == hi) return a_flat[hi];
    return a_flat[lo] * (hi - nq) + a_flat[hi] * (nq - lo);
  });
}

/**
 * Compute the q-th quantile of the data along the specified axis.
 */
export function quantile(a: NDArray<any>, q: NDArray<any> | number, axis: number, _nan_handling = false) {
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
    const values = js_quantile_a_1D_q_01D(row, q_flat, _nan_handling);
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
export function median(a: NDArray, axis: AxisArg | null, keepdims: boolean) {
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
export function average(a: NDArray, axis: AxisArg | null, weights: NDArray | null, keepdims: boolean) {
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
export function nanpercentile(a: NDArray, q: NDArray | number, axis: AxisArg | null) {
  q = np.divide(q, 100);
  return nanquantile(a, q, axis);
}


/**
 * Compute the q-th quantile of the data along the specified axis, while ignoring nan values.
 */
export function nanquantile(a: NDArray, q: NDArray | number, axis: AxisArg | null) {
  return quantile(a, q, axis, true)
}


/**
 * Compute the median along the specified axis, while ignoring NaNs.
 */
export function nanmedian(a: NDArray, axis: number, keepdims: boolean) {
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
// export function corrcoef(x: NDArray, y: NDArray, rowvar = true, bias = false, ddof = 0, dtype = Number) {
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
// export function correlate(a: NDArray, v: NDArray, mode) {
//   throw new Error("Not implemented");
// }

// /**
//  * Estimate a covariance matrix, given data and weights.
//  */
// export function cov(m: NDArray, y: NDArray, rowvar = true, bias = false, ddof = 0, fweights = null, aweights = null) {
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
// export function histogram(a: NDArray, bins, range, density, weights) {
//   throw new Error("Not implemented");
// }

// /**
//  * Compute the bi-dimensional histogram of two data samples.
//  */
// export function histogram2d(x: NDArray, y: NDArray, bins, range, density, weights) {
//   throw new Error("Not implemented");
// }

// /**
//  * Compute the multidimensional histogram of some data.
//  */
// export function histogramdd(sample: NDArray, bins, range, density, weights) {
//   throw new Error("Not implemented");
// }

// /**
//  * Count number of occurrences of each value in array of non-negative ints.
//  */
// export function bincount(x: NDArray, weights, minlength) {
//   throw new Error("Not implemented");
// }

// /**
//  * Function to calculate only the edges of the bins used by the histogram function.
//  */
// export function histogram_bin_edges(a: NDArray, bins, range, weights) {
// }

// /**
//  * Return the indices of the bins to which each value in input array belongs.
//  */
// export function digitize(x: NDArray, bins, right) {
//   throw new Error("Not implemented");
// }




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


export const kw_exported = {
  quantile: Func_a_q_axis.defaultDecorator(quantile),
  nanquantile: Func_a_q_axis.defaultDecorator(nanquantile),
  percentile: Func_a_q_axis.defaultDecorator(percentile),
  nanpercentile: Func_a_q_axis.defaultDecorator(nanpercentile),
  median: Func_a_axis_keepdims.defaultDecorator(median),
  nanmedian: Func_a_axis_keepdims.defaultDecorator(nanmedian),
};