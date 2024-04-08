//@ts-check
import NDArray from "../NDArray";
import { AxisArg, kwDecorators, Parsed_q_axis, Signature_q_axis } from "../array/kwargs";
import { apply_along_axis } from "../array/transform";
import { np } from "./_globals";


/**
 * Compute the q-th percentile of the data along the specified axis.
 */
export function percentile(a: NDArray, q: NDArray | number, axis: AxisArg) {
  return quantile(a, np.divide(q, 100), axis);
}

/**
 * Compute the q-th quantile of the data along the specified axis.
 */
export function quantile(a: NDArray, q: NDArray | number, axis: AxisArg) {
  if (axis == null) {
    a = a.reshape(-1);
    axis = 0;
  }
  const sorted = np.sort(a, axis);
  const n = sorted.shape[axis];
  const indices_float = np.multiply(q, n - 1);
  const lo = np.floor(indices_float);
  const hi = np.ceil(indices_float);
  const weights_hi = np.subtract(indices_float, lo);
  const weights_lo = np.subtract(1, weights_hi);
  let out_lo = np.take(sorted, lo, axis);
  let out_hi = np.take(sorted, hi, axis);
  // Fix the problem of neighbor with infinity with weight of 0
  out_lo = np.multiply(weights_lo, out_lo);
  out_hi = np.multiply(weights_hi, out_hi);
  out_lo = np.where(np.equal(weights_lo, 0), np.asarray(0), out_lo);
  out_hi = np.where(np.equal(weights_hi, 0), np.asarray(0), out_hi);
  return np.add(out_lo, out_hi);
}
/**
 * Compute the median along the specified axis.
 */
export function median(a: NDArray, axis, keepdims) {
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
export function average(a: NDArray, axis, weights, keepdims) {
  if (weights === undefined) return np.mean(a, axis);
  const denominator = np.sum(weights);
  const weights_shape = a.shape.map(() => 1);
  weights_shape[axis] = a.shape[axis];
  weights = weights.reshape(weights_shape);
  const numerator = np.sum(np.multiply(a, weights), axis, keepdims);
  return np.divide(numerator, denominator);
}


// /**
//  * Compute the q-th percentile of the data along the specified axis, while ignoring nan values.
// */
// export function nanpercentile(a: NDArray, q: NDArray | number, axis) {
//   return nanquantile(a, np.divide(q, 100), axis);
// }


// /**
//  * Compute the q-th quantile of the data along the specified axis, while ignoring nan values.
//  */
// export function nanquantile(a: NDArray, q: NDArray | number, axis) {
//   throw new Error("Not implemented");
//   return a;
// }


// /**
//  * Compute the median along the specified axis, while ignoring NaNs.
//  */
// export function nanmedian(a: NDArray, axis, keepdims) {
//   let out = nanquantile(a, 0.5, axis);
//   if (keepdims) {
//     const shape = a.shape.slice();
//     shape[axis] = 1;
//     out = out.reshape(shape);
//   }
//   return out;
// }





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
  quantile: kwDecorators<Signature_q_axis, Parsed_q_axis>({
    defaults: [["q", undefined], ["axis", null]],
    func: quantile,
  }).as_function,
};