//@ts-check

/**
 * Parser and main namespace for the ndarray-js package.
 * <script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>
 */
// const np = function (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) {
//   const usage = 'Usage example: np`np.arange(10)+${5}` or np([0,1,2]).';
//   if (typeof template == "number" || typeof template == "boolean") return template;
//   if (template instanceof np.NDArray) return template;
//   if (!Array.isArray(template)) throw new Error(`Expected template or array. ${usage}`);
//   if (!template.length) throw new Error(`Expected argument. ${usage}`);
//   if (typeof template[0] == "string") {
//     if (variables.length + 1 != template.length) throw new Error(`Wrong input. ${usage}`);
//     //@ts-ignore
//     return np.modules.grammar.parse(/**@type {*}*/(template), ...variables);
//   }
//   else {
//     if (variables.length) throw new Error(`Wrong input. ${usage}`);
//     //@ts-ignore
//     return np.asarray(template);
//   }
// }


function np(template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) {
  const usage = 'Usage example: np`np.arange(10)+${5}` or np([0,1,2]).';
  if (typeof template == "number" || typeof template == "boolean") return template;
  if (template instanceof np.NDArray) return template;
  if (!Array.isArray(template)) throw new Error(`Expected template or array. ${usage}`);
  if (!template.length) throw new Error(`Expected argument. ${usage}`);
  if (typeof template[0] == "string") {
    if (variables.length + 1 != template.length) throw new Error(`Wrong input. ${usage}`);
    //@ts-ignore
    return np.modules.grammar.parse(/**@type {*}*/(template), ...variables);
  }
  else {
    if (variables.length) throw new Error(`Wrong input. ${usage}`);
    //@ts-ignore
    return np.asarray(template);
  }
}

import { set_np } from './np_circular_import';
set_np(np);

// ==============================
//    Define casting and core before importing any other module
// ==============================
import { NDArray, isarray, asarray } from './NDArray';
import * as array from './array';

import { Func_a_a_min_a_max_out, Func_a_axis_ddof_keepdims, Func_a_axis_keepdims, Func_a_axis_keepdims_dtype_out, Func_a_decimals_out, Func_a_lastAxis, Func_a_ord_axis_keepdims, Func_a_other_out, Func_a_out, Func_x1_x2_out, Func_y_x_out, Method_a_axis_ddof_keepdims, Method_a_axis_keepdims, Method_a_axis_keepdims_dtype_out, Method_a_min_a_max_out, Method_a_decimals_out, Method_a_ord_axis_keepdims, Method_other_out, Method_out, Method_values_where } from './kwargs/kwargs';
import * as sig from './kwargs/signatures';

/** @category Main */
np.NDArray = NDArray;

// ==============================
//  Define core-related functions
// ==============================

const { tolist } = array;
/** @category Casting and reshaping */
np.tolist = (template: TemplateStringsArray | any[] | number | boolean | NDArray, ...variables: any[]) => {
  if (template instanceof NDArray) return tolist(template);
  return tolist(np(template, ...variables));
}
/** @category Casting and reshaping */
np.fromlist = array.fromlist;

/** @category Casting and reshaping */
np.ravel = sig.ravel.func(array.ravel);
/** @category Casting and reshaping */
np.reshape = sig.reshape.func((a, shape) => (array.reshape as any)(a, ...shape));
/** @category Casting and reshaping */
np.array = sig.array.func(array.array);
/** @category Casting and reshaping */
np.asarray = sig.asarray_sig.func(asarray);


import * as reducers from './array/reduce';
/** @category Reducers */
np.sum = Func_a_axis_keepdims_dtype_out.decorate(reducers.sum);
/** @category Reducers */
np.prod = Func_a_axis_keepdims_dtype_out.decorate(reducers.product);
/** @category Reducers */
np.product = np.prod;
/** @category Reducers */
np.nansum = Func_a_axis_keepdims_dtype_out.decorate(reducers.nansum);
/** @category Reducers */
np.nanprod = Func_a_axis_keepdims_dtype_out.decorate(reducers.nanprod);
/** @category Reducers */
np.mean = Func_a_axis_keepdims_dtype_out.decorate(reducers.mean);
/** @category Reducers */
np.nanmean = Func_a_axis_keepdims_dtype_out.decorate(reducers.nanmean);
/** @category Reducers */
np.max = Func_a_axis_keepdims.decorate(reducers.max);
/** @category Reducers */
np.min = Func_a_axis_keepdims.decorate(reducers.min);
/** @category Reducers */
np.nanmax = Func_a_axis_keepdims.decorate(reducers.nanmax);
/** @category Reducers */
np.nanmin = Func_a_axis_keepdims.decorate(reducers.nanmin);
/** @category Reducers */
np.argmax = Func_a_axis_keepdims.decorate(reducers.argmax);
/** @category Reducers */
np.argmin = Func_a_axis_keepdims.decorate(reducers.argmin);
/** @category Reducers */
np.nanargmax = Func_a_axis_keepdims.decorate(reducers.nanargmax);
/** @category Reducers */
np.nanargmin = Func_a_axis_keepdims.decorate(reducers.nanargmin);
/** @category Reducers */
np.len = Func_a_axis_keepdims.decorate(reducers.len);
/** @category Reducers */
np.any = Func_a_axis_keepdims.decorate(reducers.any);
/** @category Reducers */
np.all = Func_a_axis_keepdims.decorate(reducers.all);
/** @category Reducers */
np.count_nonzero = Func_a_axis_keepdims.decorate(reducers.count_nonzero);
/** @category Reducers */
np.ptp = Func_a_axis_keepdims.decorate(reducers.ptp);
/** @category Reducers */
np.norm = Func_a_ord_axis_keepdims.decorate(reducers.norm);
/** @category Reducers */
np.var = Func_a_axis_ddof_keepdims.decorate(reducers.variance);
/** @category Reducers */
np.std = Func_a_axis_ddof_keepdims.decorate(reducers.std);
/** @category Reducers */
np.nanvar = Func_a_axis_ddof_keepdims.decorate(reducers.nanvar);
/** @category Reducers */
np.nanstd = Func_a_axis_ddof_keepdims.decorate(reducers.nanstd);

/** @category Transformations */
np.transpose = sig.transpose.func(array.transpose);
/** @category Transformations */
np.apply_along_axis = sig.apply_along_axis.func((func1d, axis, arr, args) => array.apply_along_axis_func(func1d, axis, arr, args));

/** @category Sorting */
np.sort = Func_a_lastAxis.decorate(array.sort);
/** @category Sorting */
np.argsort = Func_a_lastAxis.decorate(array.argsort);
/** Sorts along the last axis. Complex numbers are not supported: the values stay real. @category Sorting */
np.sort_complex = (a: any) => np.sort(a, -1);
/** @category Sorting */
np.lexsort = sig.lexsort.func(array.lexsort);
/** @category Sorting */
np.partition = sig.partition.func(array.partition);
/** @category Sorting */
np.argpartition = sig.partition.func(array.argpartition);
/** @category Transformations */
np.concatenate = sig.concatenate.func(array.concatenate);
/** @category Transformations */
np.stack = sig.stack.func(array.stack);


import * as operators from './array/operators';

/** @category Binary operators */
np.add = Func_a_other_out.decorate(operators.add);
/** @category Binary operators */
np.subtract = Func_a_other_out.decorate(operators.subtract);
/** @category Binary operators */
np.multiply = Func_a_other_out.decorate(operators.multiply);
/** @category Binary operators */
np.divide = Func_a_other_out.decorate(operators.divide);
/** @category Binary operators */
np.true_divide = np.divide;
/** @category Binary operators */
np.mod = Func_a_other_out.decorate(operators.mod);
/** @category Binary operators */
np.fmod = Func_a_other_out.decorate(operators.fmod);
/** @category Binary operators */
np.remainder = np.mod;
/** @category Binary operators */
np.divide_int = Func_a_other_out.decorate(operators.divide_int);
/** @category Binary operators */
np.floor_divide = np.divide_int;
/** @category Binary operators */
np.pow = Func_a_other_out.decorate(operators.pow);
/** @category Binary operators */
np.power = np.pow;
/** @category Binary operators */
np.float_power = Func_a_other_out.decorate(operators.float_power);
/** @category Binary operators */
np.bitwise_or = Func_a_other_out.decorate(operators.bitwise_or);
/** @category Binary operators */
np.bitwise_and = Func_a_other_out.decorate(operators.bitwise_and);
/** @category Binary operators */
np.bitwise_xor = Func_a_other_out.decorate(operators.bitwise_xor);
/** @category Binary operators */
np.bitwise_shift_left = Func_a_other_out.decorate(operators.bitwise_shift_left);
/** @category Binary operators */
np.bitwise_shift_right = Func_a_other_out.decorate(operators.bitwise_shift_right);
/** @category Binary operators */
np.left_shift = np.bitwise_shift_left;
/** @category Binary operators */
np.right_shift = np.bitwise_shift_right;
/** @category Binary operators */
np.bitwise_left_shift = np.bitwise_shift_left;
/** @category Binary operators */
np.bitwise_right_shift = np.bitwise_shift_right;
/** @category Binary operators */
np.greater = Func_a_other_out.decorate(operators.greater);
/** @category Binary operators */
np.less = Func_a_other_out.decorate(operators.less);
/** @category Binary operators */
np.greater_equal = Func_a_other_out.decorate(operators.greater_equal);
/** @category Binary operators */
np.less_equal = Func_a_other_out.decorate(operators.less_equal);
/** @category Binary operators */
np.equal = Func_a_other_out.decorate(operators.equal);
/** @category Binary operators */
np.not_equal = Func_a_other_out.decorate(operators.not_equal);
/** @category Binary operators */
np.maximum = Func_a_other_out.decorate(operators.maximum);
/** @category Binary operators */
np.minimum = Func_a_other_out.decorate(operators.minimum);
/** @category Binary operators */
np.fmax = Func_a_other_out.decorate(operators.fmax);
/** @category Binary operators */
np.fmin = Func_a_other_out.decorate(operators.fmin);
/** @category Binary operators */
np.logical_or = Func_a_other_out.decorate(operators.logical_or);
/** @category Binary operators */
np.logical_and = Func_a_other_out.decorate(operators.logical_and);
/** @category Binary operators */
np.logical_xor = Func_a_other_out.decorate(operators.logical_xor);
/** @category Binary operators */
np.hypot = Func_a_other_out.decorate(operators.hypot);
/** @category Binary operators */
np.copysign = Func_a_other_out.decorate(operators.copysign);
/** @category Binary operators */
np.logaddexp = Func_a_other_out.decorate(operators.logaddexp);
/** @category Binary operators */
np.logaddexp2 = Func_a_other_out.decorate(operators.logaddexp2);
/** @category Binary operators */
np.ldexp = Func_a_other_out.decorate(operators.ldexp);
/** @category Binary operators */
np.gcd = Func_a_other_out.decorate(operators.gcd);
/** @category Binary operators */
np.lcm = Func_a_other_out.decorate(operators.lcm);

np.assign = operators.op_assign['='];

/** @category Comparison */
np.allclose = sig.allclose.func(operators.allclose);
/** @category Comparison */
np.isclose = sig.isclose.func(operators.isclose);
/** @category Comparison */
np.array_equal = sig.array_equal.func(operators.array_equal);
/** @category Comparison */
np.array_equiv = sig.array_equiv.func(operators.array_equiv);


import * as elementwise from './array/elementwise';

/** @category Elementwise operators */
np.sign = Func_a_out.decorate(elementwise.sign);
/** @category Elementwise operators */
np.sqrt = Func_a_out.decorate(elementwise.sqrt);
/** @category Elementwise operators */
np.square = Func_a_out.decorate(elementwise.square);
/** @category Elementwise operators */
np.exp = Func_a_out.decorate(elementwise.exp);
/** @category Elementwise operators */
np.exp2 = Func_a_out.decorate(elementwise.exp2);
/** @category Elementwise operators */
np.expm1 = Func_a_out.decorate(elementwise.expm1);
/** @category Elementwise operators */
np.log = Func_a_out.decorate(elementwise.log);
/** @category Elementwise operators */
np.log2 = Func_a_out.decorate(elementwise.log2);
/** @category Elementwise operators */
np.log10 = Func_a_out.decorate(elementwise.log10);
/** @category Elementwise operators */
np.log1p = Func_a_out.decorate(elementwise.log1p);
/** @category Elementwise operators */
np.sin = Func_a_out.decorate(elementwise.sin);
/** @category Elementwise operators */
np.cos = Func_a_out.decorate(elementwise.cos);
/** @category Elementwise operators */
np.tan = Func_a_out.decorate(elementwise.tan);
/** @category Elementwise operators */
np.asin = Func_a_out.decorate(elementwise.asin);
/** @category Elementwise operators */
np.acos = Func_a_out.decorate(elementwise.acos);
/** @category Elementwise operators */
np.atan = Func_a_out.decorate(elementwise.atan);
/** @category Elementwise operators */
np.cosh = Func_a_out.decorate(elementwise.cosh);
/** @category Elementwise operators */
np.sinh = Func_a_out.decorate(elementwise.sinh);
/** @category Elementwise operators */
np.tanh = Func_a_out.decorate(elementwise.tanh);
/** @category Elementwise operators */
np.acosh = Func_a_out.decorate(elementwise.acosh);
/** @category Elementwise operators */
np.asinh = Func_a_out.decorate(elementwise.asinh);
/** @category Elementwise operators */
np.atanh = Func_a_out.decorate(elementwise.atanh);
/** @category Elementwise operators */
np.degrees = Func_a_out.decorate(elementwise.degrees);
/** @category Elementwise operators */
np.radians = Func_a_out.decorate(elementwise.radians);
/** @category Elementwise operators */
np.deg2rad = np.radians;
/** @category Elementwise operators */
np.rad2deg = np.degrees;
/** @category Elementwise operators */
np.floor = Func_a_out.decorate(elementwise.floor);
/** @category Elementwise operators */
np.ceil = Func_a_out.decorate(elementwise.ceil);
/** @category Elementwise operators */
np.trunc = Func_a_out.decorate(elementwise.trunc);
/** @category Elementwise operators */
np.fix = Func_a_out.decorate(elementwise.fix);
/** @category Elementwise operators */
np.rint = Func_a_out.decorate(elementwise.rint);
/** @category Elementwise operators */
np.negative = Func_a_out.decorate(elementwise.negative);
/** @category Elementwise operators */
np.bitwise_not = Func_a_out.decorate(elementwise.bitwise_not);
/** @category Elementwise operators */
np.invert = np.bitwise_not;
/** @category Elementwise operators */
np.logical_not = Func_a_out.decorate(elementwise.logical_not);
/** @category Elementwise operators */
np.valueOf = Func_a_out.decorate(elementwise.valueOf);
/** @category Elementwise operators */
np.abs = Func_a_out.decorate(elementwise.abs);

/** @category Elementwise operators */
np.isfinite = Func_a_out.decorate(elementwise.isfinite);
/** @category Elementwise operators */
np.isinf = Func_a_out.decorate(elementwise.isinf);
/** @category Elementwise operators */
np.isposinf = Func_a_out.decorate(elementwise.isposinf);
/** @category Elementwise operators */
np.isneginf = Func_a_out.decorate(elementwise.isneginf);
/** @category Elementwise operators */
np.isnan = Func_a_out.decorate(elementwise.isnan);
/** @category Elementwise operators */
np.signbit = Func_a_out.decorate(elementwise.signbit);
/** @category Elementwise operators */
np.iscomplex = Func_a_out.decorate(elementwise.iscomplex);
/** @category Elementwise operators */
np.isreal = Func_a_out.decorate(elementwise.isreal);
/** @category Elementwise operators */
np.reciprocal = Func_a_out.decorate(elementwise.reciprocal);
/** @category Elementwise operators */
np.positive = Func_a_out.decorate(elementwise.positive);
/** @category Elementwise operators */
np.angle = Func_a_out.decorate(elementwise.angle);
/** @category Elementwise operators */
np.real = Func_a_out.decorate(elementwise.real);
/** @category Elementwise operators */
np.imag = Func_a_out.decorate(elementwise.imag);
/** @category Elementwise operators */
np.conj = Func_a_out.decorate(elementwise.conj);
/** @category Elementwise operators */
np.conjugate = np.conj;
/** @category Elementwise operators */
np.cbrt = Func_a_out.decorate(elementwise.cbrt);
/** @category Elementwise operators */
np.sinc = Func_a_out.decorate(elementwise.sinc);
/** @category Elementwise operators */
np.i0 = Func_a_out.decorate(elementwise.i0);
/** @category Elementwise operators */
np.spacing = Func_a_out.decorate(elementwise.spacing);
/** @category Elementwise operators */
np.nan_to_num = sig.nan_to_num.func(elementwise.nan_to_num);
/** @category Elementwise operators */
np.real_if_close = Func_a_out.decorate(elementwise.real_if_close);
/** @category Elementwise operators */
np.round = Func_a_decimals_out.decorate(elementwise.round);
/** @category Elementwise operators */
np.around = np.round;
/** @category Elementwise operators */
np.modf = sig.a_to_pair.func(elementwise.modf);
/** @category Elementwise operators */
np.frexp = sig.a_to_pair.func(elementwise.frexp);
/** @category Binary operators */
np.nextafter = Func_a_other_out.decorate(elementwise.nextafter as any);

// Aliases and special functions
np.arcsin = np.asin;
np.arccos = np.acos;
np.arctan = np.atan;
np.arcsinh = np.asinh;
np.arccosh = np.acosh;
np.arctanh = np.atanh;
np.absolute = np.abs;
np.fabs = Func_a_out.decorate(elementwise.abs);

np.amax = np.max;
np.amin = np.min;

/** @category Binary operators */
np.divmod = operators.divmod;

/** @category Binary operators */
np.heaviside = Func_x1_x2_out.decorate(operators.heaviside);
/** @category Elementwise operators */
np.clip = Func_a_a_min_a_max_out.decorate(operators.clip);
/** @category Binary operators */
np.atan2 = Func_y_x_out.decorate(operators.atan2);
/** @category Binary operators */
np.arctan2 = np.atan2;


/** @category Types */
np.isscalar = (obj: any) => {
  if (isarray(obj)) return false;
  if (Array.isArray(obj)) return false;
  if (obj === null || obj === undefined) return false;
  const t = typeof obj;
  return t === 'number' || t === 'boolean' || t === 'string' || t === 'bigint' || t === 'symbol';
}

import * as math from './modules/math-functions';
/** @category Cumulative */
np.cumsum = sig.cumulative.func(math.cumsum);
/** @category Cumulative */
np.cumprod = sig.cumulative.func(math.cumprod);
/** @category Cumulative */
np.nancumsum = sig.cumulative.func(math.nancumsum);
/** @category Cumulative */
np.nancumprod = sig.cumulative.func(math.nancumprod);
/** @category Cumulative */
np.cumulative_sum = sig.cumulative_xxx.func(math.cumulative_sum);
/** @category Cumulative */
np.cumulative_prod = sig.cumulative_xxx.func(math.cumulative_prod);
/** @category Differences */
np.diff = sig.diff.func(math.diff);
/** @category Differences */
np.ediff1d = sig.ediff1d.func(math.ediff1d);
/** @category Differences */
np.gradient = sig.gradient.func(math.gradient);
/** @category Math */
np.cross = sig.cross.func(math.cross);
/** @category Math */
np.trapezoid = sig.trapezoid.func(math.trapezoid);
/** @category Math */
np.convolve = sig.convolve.func(math.convolve);
/** @category Statistics */
np.correlate = sig.correlate.func(math.correlate);
/** @category Math */
np.interp = sig.interp.func(math.interp);
/** @category Math */
np.unwrap = sig.unwrap.func(math.unwrap);

import * as sets from './modules/sets';
/** @category Sets */
np.unique = sig.unique.func(sets.unique);
/** @category Sets */
np.unique_values = sig.x_only.func(sets.unique_values as any);
/** @category Sets */
np.unique_counts = sig.x_only.func(sets.unique_counts as any);
/** @category Sets */
np.unique_inverse = sig.x_only.func(sets.unique_inverse as any);
/** @category Sets */
np.unique_all = sig.x_only.func(sets.unique_all as any);
/** @category Sets */
np.intersect1d = sig.intersect1d.func(sets.intersect1d);
/** @category Sets */
np.isin = sig.isin.func(sets.isin);
/** @category Sets */
np.setdiff1d = sig.setdiff1d.func(sets.setdiff1d);
/** @category Sets */
np.setxor1d = sig.setdiff1d.func(sets.setxor1d);
/** @category Sets */
np.union1d = sig.union1d.func(sets.union1d);

import * as stats from './modules/statistics';
/** @category Statistics */
np.quantile = sig.quantile.func(stats.quantile);
/** @category Statistics */
np.nanquantile = sig.quantile.func(stats.nanquantile);
/** @category Statistics */
np.percentile = sig.quantile.func(stats.percentile);
/** @category Statistics */
np.nanpercentile = sig.quantile.func(stats.nanpercentile);
/** @category Statistics */
np.median = Func_a_axis_keepdims.decorate(stats.median);
/** @category Statistics */
np.nanmedian = Func_a_axis_keepdims.decorate(stats.nanmedian);
/** @category Statistics */
np.average = sig.average.func(stats.average);
/** @category Statistics */
np.cov = sig.cov.func(stats.cov);
/** @category Statistics */
np.corrcoef = sig.corrcoef.func(stats.corrcoef);
/** @category Histograms */
np.histogram = sig.histogram.func(stats.histogram);
/** @category Histograms */
np.histogram_bin_edges = sig.histogram_bin_edges.func(stats.histogram_bin_edges);
/** @category Histograms */
np.histogram2d = sig.histogram2d.func(stats.histogram2d);
/** @category Histograms */
np.histogramdd = sig.histogramdd.func(stats.histogramdd);
/** @category Histograms */
np.bincount = sig.bincount.func(stats.bincount);
/** @category Histograms */
np.digitize = sig.digitize.func(stats.digitize);
/** @category Searching */
np.searchsorted = sig.searchsorted.func(stats.searchsorted);

import * as misc from './modules/misc';
/** @category Memory */
np.shares_memory = misc.shares_memory;
/** @category Memory */
np.may_share_memory = misc.may_share_memory;
/** @category Broadcasting */
np.broadcast_shapes = misc.broadcast_shapes;
/** @category Settings */
np.getbufsize = misc.getbufsize;
/** @category Settings */
np.setbufsize = misc.setbufsize;
/** @category Settings */
np.get_include = misc.get_include;
/** @category Settings */
np.show_config = misc.show_config;
/** @category Settings */
np.show_runtime = misc.show_runtime;
/** @category Settings */
np.info = misc.info;


// ==============================
//   import np modules
// ============================== 

import { modules } from './modules';
import { op_assign } from './array/operators';
import { fromlist } from './array';

/** @category Main @namespace*/
np.modules = modules;
/** @category Modules */
np.random = np.modules.random;

import * as constructors from './modules/constructors';
import * as indexing from './modules/indexing';
import * as ctor from './kwargs/constructors.kwargs';
import { Func_shape_dtype } from './kwargs/constructors.kwargs';

/** @category Constructors */
np.empty = Func_shape_dtype.decorate(constructors.empty);
/** @category Constructors */
np.zeros = Func_shape_dtype.decorate(constructors.zeros);
/** @category Constructors */
np.ones = Func_shape_dtype.decorate(constructors.ones);
/** @category Constructors */
np.arange = ctor.arange.func(constructors.arange);
/** @category Constructors */
np.linspace = ctor.linspace.func(constructors.linspace);
/** @category Constructors */
np.geomspace = ctor.geomspace.func(constructors.geomspace);

/** @category Indexing */
np.take = sig.take.func(indexing.take);
/** @category Searching */
np.where = sig.where.func((condition, x, y, out) => {
  if (x === null) return indexing.nonzero(asarray(condition));
  return indexing.where(condition, x, y, out);
});
/** @category Searching */
np.nonzero = sig.a_only.func(indexing.nonzero as any);
/** @category Searching */
np.argwhere = sig.a_only.func(indexing.argwhere);
/** @category Searching */
np.flatnonzero = sig.a_only.func(indexing.flatnonzero);
/** @category Searching */
np.extract = sig.extract.func(indexing.extract);


NDArray.prototype.add = Method_other_out.decorate(array.add);
NDArray.prototype.subtract = Method_other_out.decorate(array.subtract);
NDArray.prototype.multiply = Method_other_out.decorate(array.multiply);
NDArray.prototype.divide = Method_other_out.decorate(array.divide);
NDArray.prototype.mod = Method_other_out.decorate(array.mod);
NDArray.prototype.divide_int = Method_other_out.decorate(array.divide_int);
NDArray.prototype.pow = Method_other_out.decorate(array.pow);
NDArray.prototype.maximum = Method_other_out.decorate(array.maximum);
NDArray.prototype.minimum = Method_other_out.decorate(array.minimum);
NDArray.prototype.bitwise_or = Method_other_out.decorate(array.bitwise_or);
NDArray.prototype.bitwise_and = Method_other_out.decorate(array.bitwise_and);
NDArray.prototype.bitwise_xor = Method_other_out.decorate(array.bitwise_xor);
NDArray.prototype.bitwise_shift_left = Method_other_out.decorate(array.bitwise_shift_left);
NDArray.prototype.bitwise_shift_right = Method_other_out.decorate(array.bitwise_shift_right);
NDArray.prototype.logical_or = Method_other_out.decorate(array.logical_or);
NDArray.prototype.logical_and = Method_other_out.decorate(array.logical_and);
NDArray.prototype.logical_xor = Method_other_out.decorate(array.logical_xor);
NDArray.prototype.greater = Method_other_out.decorate(array.greater);
NDArray.prototype.less = Method_other_out.decorate(array.less);
NDArray.prototype.greater_equal = Method_other_out.decorate(array.greater_equal);
NDArray.prototype.less_equal = Method_other_out.decorate(array.less_equal);
NDArray.prototype.equal = Method_other_out.decorate(array.equal);
NDArray.prototype.not_equal = Method_other_out.decorate(array.not_equal);

NDArray.prototype.isclose = sig.isclose.method(operators.isclose);
NDArray.prototype.allclose = sig.allclose.method(operators.allclose);
NDArray.prototype.assign = Method_values_where.decorate(op_assign["="]);
NDArray.prototype.add_assign = Method_values_where.decorate(op_assign["+="]);
NDArray.prototype.subtract_assign = Method_values_where.decorate(op_assign["-="]);
NDArray.prototype.multiply_assign = Method_values_where.decorate(op_assign["*="]);
NDArray.prototype.divide_assign = Method_values_where.decorate(op_assign["/="]);
NDArray.prototype.mod_assign = Method_values_where.decorate(op_assign["%="]);
NDArray.prototype.divide_int_assign = Method_values_where.decorate(op_assign["//="]);
NDArray.prototype.pow_assign = Method_values_where.decorate(op_assign["**="]);

NDArray.prototype.maximum_assign = Method_values_where.decorate(op_assign["max="]);
NDArray.prototype.minimum_assign = Method_values_where.decorate(op_assign["min="]);

NDArray.prototype.bitwise_or_assign = Method_values_where.decorate(op_assign["|="]);
NDArray.prototype.bitwise_and_assign = Method_values_where.decorate(op_assign["&="]);
NDArray.prototype.bitwise_shift_left_assign = Method_values_where.decorate(op_assign["<<="]);
NDArray.prototype.bitwise_shift_right_assign = Method_values_where.decorate(op_assign[">>="]);

NDArray.prototype.logical_or_assign = Method_values_where.decorate(op_assign["or="]);
NDArray.prototype.logical_and_assign = Method_values_where.decorate(op_assign["and="]);

/** @param {null|number[]} axes */
NDArray.prototype.transpose = function (axes: null | number[] = null) {
  if (axes !== null && !Array.isArray(axes) && (axes as any).axes !== undefined) axes = (axes as any).axes;
  return array.transpose(this, axes);
};

/** Sorts the array in place along the given axis (default: last). Returns the array itself. */
NDArray.prototype.sort = function (axis: any = -1) {
  if (axis !== null && typeof axis === "object") axis = (axis as any).axis === undefined ? -1 : (axis as any).axis;
  const sorted = Func_a_lastAxis.decorate(array.sort)(this, axis) as NDArray;
  this.flat = sorted.flat;
  return this;
};
NDArray.prototype.argsort = function (axis: any = -1) {
  return np.argsort(this, axis);
};

// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = Method_out.decorate(elementwise.bitwise_not);
NDArray.prototype.logical_not = Method_out.decorate(elementwise.logical_not);
NDArray.prototype.negative = Method_out.decorate(elementwise.negative);
NDArray.prototype.abs = Method_out.decorate(elementwise.abs);
NDArray.prototype.round = Method_a_decimals_out.decorate(elementwise.round);
NDArray.prototype.conj = Method_out.decorate(elementwise.conj);


NDArray.prototype.any = Method_a_axis_keepdims.decorate(reducers.any);
NDArray.prototype.all = Method_a_axis_keepdims.decorate(reducers.all);
NDArray.prototype.sum = Method_a_axis_keepdims_dtype_out.decorate(reducers.sum);
NDArray.prototype.product = Method_a_axis_keepdims_dtype_out.decorate(reducers.product);
NDArray.prototype.prod = NDArray.prototype.product;
NDArray.prototype.max = Method_a_axis_keepdims.decorate(reducers.max);
NDArray.prototype.min = Method_a_axis_keepdims.decorate(reducers.min);
NDArray.prototype.ptp = Method_a_axis_keepdims.decorate(reducers.ptp);
NDArray.prototype.argmax = Method_a_axis_keepdims.decorate(reducers.argmax);
NDArray.prototype.argmin = Method_a_axis_keepdims.decorate(reducers.argmin);
NDArray.prototype.mean = Method_a_axis_keepdims_dtype_out.decorate(reducers.mean);
NDArray.prototype.var = Method_a_axis_ddof_keepdims.decorate(reducers.variance);
NDArray.prototype.std = Method_a_axis_ddof_keepdims.decorate(reducers.std);
NDArray.prototype.norm = Method_a_ord_axis_keepdims.decorate(reducers.norm);
NDArray.prototype.cumsum = sig.cumulative.method(math.cumsum);
NDArray.prototype.cumprod = sig.cumulative.method(math.cumprod);
NDArray.prototype.nonzero = sig.a_only.method(indexing.nonzero as any);
NDArray.prototype.searchsorted = sig.searchsorted.method(stats.searchsorted);
NDArray.prototype.clip = Method_a_min_a_max_out.decorate(operators.clip);

NDArray.prototype.fromlist = fromlist;
NDArray.prototype.tolist = function () {
  return tolist(this);
}

/** @category Math constants */
np.pi = Math.PI;
/** @category Math constants */
np.e = Math.E;
/** @category Math constants */
np.nan = NaN;
/** @category Math constants */
np.inf = Infinity;
/** @category Math constants */
np.euler_gamma = 0.5772156649015328606065120900824024310421;
/** @category Math constants */
np.newaxis = null;

export { np };
