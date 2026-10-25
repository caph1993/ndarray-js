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

import { kwargs_decorator, Func_a_a_min_a_max_out, Func_a_axis_ddof_keepdims, Func_a_axis_keepdims, Func_a_axis_keepdims_dtype_out, Func_a_decimals_out, Func_a_lastAxis, Func_a_ord_axis_keepdims, Func_a_other_out, Func_a_out, Func_x1_x2_out, Func_y_x_out, Method_a_axis_ddof_keepdims, Method_a_axis_keepdims, Method_a_axis_keepdims_dtype_out, Method_a_min_a_max_out, Method_a_decimals_out, Method_a_ord_axis_keepdims, Method_other_out, Method_out, Method_values_where } from './kwargs/kwargs';
import * as sig from './kwargs/signatures';
import { Complex } from './complex';

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
np.sort = sig.sort_.func((a: any, axis: any) => array.sort(a, axis));
/** @category Sorting */
np.argsort = sig.sort_.func((a: any, axis: any) => array.argsort(a, axis));
/** Sorts along the last axis, with complex values: sorted by real part, then imaginary part. @category Sorting */
np.sort_complex = (a: any) => np.sort(asarray(asarray(a), 'complex128'), -1);
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
np.angle = sig.angle.func(elementwise.angle);
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
np.real_if_close = sig.real_if_close.func(elementwise.real_if_close);
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
np.fabs = Func_a_out.decorate(elementwise.fabs);

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
np.group_by = sig.group_by.func(sets.group_by);
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

import * as manip from './modules/manipulation';
/** @category Constructors */
np.eye = sig.eye.func(manip.eye);
/** @category Constructors */
np.identity = sig.identity.func(manip.identity);
/** @category Constructors */
np.tri = sig.tri.func(manip.tri);
/** @category Constructors */
np.tril = sig.tril.func(manip.tril);
/** @category Constructors */
np.triu = sig.tril.func(manip.triu);
/** @category Constructors */
np.diag = sig.diag.func(manip.diag);
/** @category Constructors */
np.diagflat = sig.diag.func(manip.diagflat);
/** @category Constructors */
np.full = sig.full.func(manip.full);
/** @category Constructors */
np.full_like = sig.full_like.func(manip.full_like);
/** @category Constructors */
np.zeros_like = sig.like.func(manip.zeros_like);
/** @category Constructors */
np.ones_like = sig.like.func(manip.ones_like);
/** @category Constructors */
np.empty_like = sig.like.func(manip.empty_like);
/** @category Constructors */
np.logspace = sig.logspace.func(constructors.logspace);
/** @category Constructors */
np.vander = sig.vander.func(manip.vander);
/** @category Constructors */
np.meshgrid = sig.meshgrid.func(manip.meshgrid);
/** Dense multi-dimensional grid from slice strings: `np.mgrid('0:3', '0:2')`. @category Constructors */
np.mgrid = sig.grid.func((specs: any[]) => manip.mgrid(...specs));
/** Open multi-dimensional grid from slice strings: `np.ogrid('0:3', '0:2')`. @category Constructors */
np.ogrid = sig.grid.func((specs: any[]) => manip.ogrid(...specs));
/** @category Attributes */
np.ndim = sig.ndim_.func(manip.ndim);
/** @category Attributes */
np.shape = sig.ndim_.func(manip.shape as any);
/** @category Attributes */
np.size = sig.size_.func(manip.size);
/** @category Casting and reshaping */
np.astype = sig.astype.func(manip.astype);
/** @category Casting and reshaping */
np.copy = sig.copy_.func(manip.copy);
/** @category Casting and reshaping */
np.asanyarray = np.asarray;
/** @category Casting and reshaping */
np.require = sig.require_.func((a, dtype) => asarray(a, dtype));
/** @category Casting and reshaping */
np.copyto = sig.copyto.func(manip.copyto);
/** @category Axes */
np.moveaxis = sig.moveaxis.func(manip.moveaxis);
/** @category Axes */
np.rollaxis = sig.rollaxis.func(manip.rollaxis);
/** @category Axes */
np.swapaxes = sig.swapaxes.func(array.swapAxes);
/** @category Axes */
np.permute_dims = np.transpose;
/** @category Axes */
np.matrix_transpose = sig.a_only_arr.func(manip.matrix_transpose);
/** @category Axes */
np.atleast_1d = sig.atleast.func((arys: any) => manip.atleast_nd(arys, 1));
/** @category Axes */
np.atleast_2d = sig.atleast.func((arys: any) => manip.atleast_nd(arys, 2));
/** @category Axes */
np.atleast_3d = sig.atleast.func((arys: any) => manip.atleast_nd(arys, 3));
/** @category Broadcasting */
np.broadcast_to = sig.broadcast_to.func(manip.broadcast_to);
/** @category Broadcasting */
np.broadcast_arrays = sig.broadcast_arrays.func(manip.broadcast_arrays);
/** @category Broadcasting */
np.broadcast = sig.broadcast_arrays.func(manip.broadcast as any);
/** @category Axes */
np.expand_dims = sig.expand_dims.func(manip.expand_dims);
/** @category Axes */
np.squeeze = sig.squeeze.func(manip.squeeze);
/** @category Joining */
np.concat = np.concatenate;
/** @category Joining */
np.block = sig.block.func(manip.block);
/** @category Joining */
np.vstack = sig.stack_like.func(manip.vstack);
/** @category Joining */
np.hstack = sig.stack_like.func(manip.hstack);
/** @category Joining */
np.dstack = sig.stack_like.func(manip.dstack);
/** @category Joining */
np.column_stack = sig.stack_like.func(manip.column_stack);
/** @category Splitting */
np.split = sig.split.func(manip.split);
/** @category Splitting */
np.array_split = sig.split.func((ary, ios, axis) => manip.array_split(ary, ios, axis));
/** @category Splitting */
np.hsplit = sig.hsplit.func(manip.hsplit);
/** @category Splitting */
np.vsplit = sig.hsplit.func(manip.vsplit);
/** @category Splitting */
np.dsplit = sig.hsplit.func(manip.dsplit);
/** @category Splitting */
np.unstack = sig.unstack.func(manip.unstack);
/** @category Repeating */
np.tile = sig.tile.func(manip.tile);
/** @category Repeating */
np.repeat = sig.repeat.func(manip.repeat);
/** @category Inserting and deleting */
np.delete = sig.del.func(manip.del);
/** @category Inserting and deleting */
np.insert = sig.insert.func(manip.insert);
/** @category Inserting and deleting */
np.append = sig.append.func(manip.append);
/** @category Inserting and deleting */
np.resize = sig.resize.func(manip.resize);
/** @category Inserting and deleting */
np.trim_zeros = sig.trim_zeros.func(manip.trim_zeros);
/** @category Padding */
np.pad = sig.pad.func((array, pad_width, mode, constant_values, end_values, stat_length, reflect_type) => manip.pad(array, pad_width, mode, { constant_values, end_values, stat_length, reflect_type }));
/** @category Reordering */
np.flip = sig.flip.func(manip.flip);
/** @category Reordering */
np.fliplr = sig.flip1.func(manip.fliplr);
/** @category Reordering */
np.flipud = sig.flip1.func(manip.flipud);
/** @category Reordering */
np.roll = sig.roll.func(manip.roll);
/** @category Reordering */
np.rot90 = sig.rot90.func(manip.rot90);
/** @category Binary operators */
np.bitwise_invert = np.bitwise_not;
/** @category Binary operators */
np.bitwise_count = sig.bitwise_count.func(manip.bitwise_count);

/** @category Indexing */
np.take_along_axis = sig.take_along_axis.func(ix.take_along_axis);
/** @category Indexing */
np.put_along_axis = sig.put_along_axis.func(ix.put_along_axis as any);
/** @category Indexing */
np.choose = sig.choose.func(ix.choose);
/** @category Indexing */
np.compress = sig.compress.func(ix.compress);
/** @category Indexing */
np.diagonal = sig.diagonal.func(ix.diagonal);
/** @category Indexing */
np.select = sig.select.func(ix.select);
/** @category Indexing */
np.place = sig.place.func(ix.place as any);
/** @category Indexing */
np.put = sig.put.func(ix.put as any);
/** @category Indexing */
np.putmask = sig.putmask.func(ix.putmask as any);
/** @category Indexing */
np.fill_diagonal = sig.fill_diagonal.func(ix.fill_diagonal as any);
/** @category Index generators */
np.indices = sig.indices.func(ix.indices);
/** @category Index generators */
np.ix_ = sig.ix_.func(ix.ix_);
/** @category Index generators */
np.unravel_index = sig.unravel_index.func(ix.unravel_index);
/** @category Index generators */
np.ravel_multi_index = sig.ravel_multi_index.func(ix.ravel_multi_index);
/** @category Index generators */
np.diag_indices = sig.diag_indices.func(ix.diag_indices);
/** @category Index generators */
np.diag_indices_from = sig.diag_indices_from.func(ix.diag_indices_from);
/** @category Index generators */
np.mask_indices = sig.mask_indices.func(ix.mask_indices);
/** @category Index generators */
np.tril_indices = sig.tri_indices.func(ix.tril_indices);
/** @category Index generators */
np.triu_indices = sig.tri_indices.func(ix.triu_indices);
/** @category Index generators */
np.tril_indices_from = sig.tri_indices_from.func(ix.tril_indices_from);
/** @category Index generators */
np.triu_indices_from = sig.tri_indices_from.func(ix.triu_indices_from);
/** @category Functional */
np.apply_over_axes = sig.apply_over_axes.func(fn.apply_over_axes);
/** @category Functional */
np.vectorize = (pyfunc: (...args: any[]) => any, options: fn.VectorizeOptions = {}) => fn.vectorize(pyfunc, options);
/** @category Functional */
np.frompyfunc = fn.frompyfunc;
/** @category Functional */
np.piecewise = (x: any, condlist: any, funclist: any, ...args: any[]) => fn.piecewise(x, condlist, funclist, args, {});
/** @category Iteration */
np.ndindex = (...shape: any[]) => fn.ndindex(shape.length == 1 && Array.isArray(shape[0]) ? shape[0] : shape);
/** @category Iteration */
np.ndenumerate = fn.ndenumerate;
/** @category Iteration */
np.nditer = (op: any, flags?: string[]) => new fn.nditer(op, flags);
/** @category Iteration */
np.nested_iters = fn.nested_iters;
/** @category Iteration */
np.flatiter = fn.flatiter;
/** @category Iteration */
np.iterable = fn.iterable;
/** Mathematical functions with automatic domain: results must be real (complex numbers are not supported). @category Math */
np.emath = { sqrt: fn.emath_sqrt, log: fn.emath_log, log2: fn.emath_log2, log10: fn.emath_log10, logn: fn.emath_logn, power: fn.emath_power, arccos: fn.emath_arccos, arcsin: fn.emath_arcsin, arctanh: fn.emath_arctanh };

/** @category Linear algebra */
np.dot = sig.dot.func(la.dot);
/** @category Linear algebra */
np.matmul = sig.matmul.func(la.matmul);
/** @category Linear algebra */
np.vdot = sig.arr2.func(la.vdot);
/** @category Linear algebra */
np.inner = sig.arr2.func(la.inner);
/** @category Linear algebra */
np.outer = sig.outer.func(la.outer);
/** @category Linear algebra */
np.matvec = sig.x1x2.func(la.matvec);
/** @category Linear algebra */
np.vecmat = sig.x1x2.func(la.vecmat);
/** @category Linear algebra */
np.vecdot = sig.vecdot.func(la.vecdot);
/** @category Linear algebra */
np.tensordot = sig.tensordot.func(la.tensordot);
/** @category Linear algebra */
np.kron = sig.kron.func(la.kron);
/** @category Linear algebra */
np.trace = sig.trace.func(la.trace);
/** @category Linear algebra */
np.einsum = sig.einsum.func((subscripts, operands, dtype, out) => la.einsum(subscripts, operands, dtype, out));
/** @category Linear algebra */
np.einsum_path = (subscripts: string, ...operands: any[]) => la.einsum_path(subscripts, operands.filter(o => !(typeof o === 'string') && !(o && typeof o === 'object' && !isarray(o) && !Array.isArray(o))));

operators.op_binary['@'] = la.matmul as any;
const cross_func = np.cross;
/** numpy.linalg: linear algebra on stacks of matrices (real numbers only) @category Linear algebra */
np.linalg = {
  LinAlgError: la.LinAlgError,
  matmul: np.matmul, vecdot: np.vecdot, outer: np.outer, tensordot: np.tensordot, trace: np.trace, diagonal: np.diagonal,
  matrix_transpose: np.matrix_transpose,
  cross: sig.linalg_cross.func((a: any, b: any, axis: number) => cross_func(a, b, { axis })),
  matrix_power: sig.matrix_power.func(la.matrix_power),
  multi_dot: sig.multi_dot.func(la.multi_dot),
  det: sig.x_only_la.func(la.det),
  slogdet: sig.x_only_la.func(la.slogdet),
  inv: sig.x_only_la.func(la.inv),
  cholesky: sig.cholesky.func(la.cholesky),
  qr: sig.qr.func(la.qr),
  svd: sig.svd.func(la.svd),
  svdvals: sig.x_only_la.func(la.svdvals),
  eig: sig.x_only_la.func(la.eig),
  eigvals: sig.x_only_la.func(la.eigvals),
  eigh: sig.eigh.func(la.eigh),
  eigvalsh: sig.eigvalsh.func(la.eigvalsh),
  solve: sig.solve.func(la.solve),
  lstsq: sig.lstsq.func(la.lstsq),
  pinv: sig.pinv.func(la.pinv),
  matrix_rank: sig.matrix_rank.func(la.matrix_rank),
  cond: sig.cond.func(la.cond),
  tensorsolve: sig.tensorsolve.func(la.tensorsolve),
  tensorinv: sig.tensorinv.func(la.tensorinv),
  norm: sig.linalg_norm.func(la.norm),
  matrix_norm: sig.matrix_norm.func(la.matrix_norm),
  vector_norm: sig.vector_norm.func((x: any, axis: any, keepdims: any, ord: any) => la.vector_norm(x, axis, keepdims, ord)),
};
/** @category Linear algebra */
np.linalg.vecdot = np.vecdot;

import * as testing from './modules/testing';
/** Assertion helpers (numpy.testing). @category Testing */
np.testing = testing;

import * as fftm from './modules/fft';
/** Discrete Fourier transforms (numpy.fft) @category FFT */
np.fft = {
  fft: sig.fft1.func(fftm.fft), ifft: sig.fft1.func(fftm.ifft), rfft: sig.fft1.func(fftm.rfft), irfft: sig.fft1.func(fftm.irfft),
  hfft: sig.fft1.func(fftm.hfft), ihfft: sig.fft1.func(fftm.ihfft),
  fftn: sig.fftn.func(fftm.fftn), ifftn: sig.fftn.func(fftm.ifftn), rfftn: sig.fftn.func(fftm.rfftn), irfftn: sig.fftn.func(fftm.irfftn),
  fft2: sig.fft2.func(fftm.fft2), ifft2: sig.fft2.func(fftm.ifft2), rfft2: sig.fft2.func(fftm.rfft2), irfft2: sig.fft2.func(fftm.irfft2),
  fftfreq: sig.fftfreq.func(fftm.fftfreq), rfftfreq: sig.fftfreq.func(fftm.rfftfreq),
  fftshift: sig.fftshift.func(fftm.fftshift), ifftshift: sig.fftshift.func(fftm.ifftshift),
};

import * as polyapi from './modules/polynomial-api';
/** numpy.polynomial: power series and Chebyshev, Hermite, HermiteE, Laguerre and Legendre series @category Polynomials */
np.polynomial = polyapi.build_polynomial_package();
/** Polynomial with the coefficients in decreasing powers (numpy.poly1d) @category Polynomials */
np.poly1d = polyapi.poly1d;
const legacy = (defaults: [string, any][], impl: (...a: any[]) => any) => kwargs_decorator<any, any, any>({ defaults, implementation: impl, parsers: [] }) as (...args: any[]) => any;
/** @category Polynomials */
np.polyval = legacy([["p", undefined], ["x", undefined]], polyapi.legacy_polyval);
/** @category Polynomials */
np.poly = legacy([["seq_of_zeros", undefined]], polyapi.legacy_poly);
/** @category Polynomials */
np.roots = legacy([["p", undefined]], polyapi.legacy_roots);
/** @category Polynomials */
np.polyfit = legacy([["x", undefined], ["y", undefined], ["deg", undefined], ["rcond", null], ["full", false], ["w", null], ["cov", false]], polyapi.legacy_polyfit);
/** @category Polynomials */
np.polyder = legacy([["p", undefined], ["m", 1]], polyapi.legacy_polyder);
/** @category Polynomials */
np.polyint = legacy([["p", undefined], ["m", 1], ["k", null]], polyapi.legacy_polyint);
/** @category Polynomials */
np.polyadd = legacy([["a1", undefined], ["a2", undefined]], polyapi.legacy_polyadd);
/** @category Polynomials */
np.polysub = legacy([["a1", undefined], ["a2", undefined]], polyapi.legacy_polysub);
/** @category Polynomials */
np.polymul = legacy([["a1", undefined], ["a2", undefined]], polyapi.legacy_polymul);
/** @category Polynomials */
np.polydiv = legacy([["u", undefined], ["v", undefined]], polyapi.legacy_polydiv);

import * as dtu from './modules/dtype-utils';
import { to_bytes, from_bytes } from './modules/binary';
import { TYPE_BY_NAME, canonical_dtype, DType } from './dtypes';
import * as fmt from './modules/format';
const unwrap0d = (a: any) => (isarray(a) && a.shape.length == 0) ? a.item() : a;
const kw = (defaults: [string, any][], impl: (...a: any[]) => any) => kwargs_decorator<any, any, any>({ defaults, implementation: (...args: any[]) => impl(...args.map(unwrap0d)), parsers: [] }) as (...args: any[]) => any;
/** @category Types */
np.dtype = (spec: any) => new dtu.DType_(spec);
// Scalar type names: np.float32 === 'float32', np.number === 'number', ...
for (const name of ['bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64', 'complex64', 'complex128', 'object']) (np as any)[name] = name;
Object.assign(np, {
  bool_: 'bool', intp: 'int64', uintp: 'uint64', int_: 'int64', uint: 'uint64', float_: 'float64', double: 'float64', single: 'float32', half: 'float16', csingle: 'complex64', cdouble: 'complex128', byte: 'int8', ubyte: 'uint8', short: 'int16', ushort: 'uint16', intc: 'int32', uintc: 'uint32', longlong: 'int64', ulonglong: 'uint64',
  number: 'number', integer: 'integer', signedinteger: 'signedinteger', unsignedinteger: 'unsignedinteger', inexact: 'inexact', floating: 'floating', complexfloating: 'complexfloating', generic: 'generic',
});
/** @category Types */
np.can_cast = kw([["from_", undefined], ["to", undefined], ["casting", 'safe']], dtu.can_cast);
/** @category Types */
np.promote_types = kw([["type1", undefined], ["type2", undefined]], dtu.promote_types);
/** @category Types */
np.result_type = (...args: any[]) => dtu.result_type(...args);
/** @category Types */
np.min_scalar_type = kw([["a", undefined]], dtu.min_scalar_type);
/** @category Types */
np.common_type = (...arrays: any[]) => dtu.common_type(...arrays);
/** @category Types */
np.isdtype = kw([["dtype", undefined], ["kind", undefined]], dtu.isdtype);
/** @category Types */
np.issubdtype = kw([["arg1", undefined], ["arg2", undefined]], dtu.issubdtype);
/** @category Types */
np.typename = kw([["char", undefined]], dtu.typename);
/** @category Types */
np.mintypecode = kw([["typechars", undefined], ["typeset", 'GDFgdf'], ["default", 'd']], dtu.mintypecode);
/** @category Types */
np.finfo = (spec?: any) => new dtu.finfo(spec);
/** @category Types */
np.iinfo = (spec?: any) => new dtu.iinfo(spec);
/** @category Error handling */
np.seterr = kw([["all", null], ["divide", null], ["over", null], ["under", null], ["invalid", null]], dtu.seterr);
/** @category Error handling */
np.geterr = dtu.geterr;
/** @category Error handling */
np.seterrcall = dtu.seterrcall;
/** @category Error handling */
np.geterrcall = dtu.geterrcall;
/** @category Error handling */
np.errstate = (options?: any, fn?: () => any) => dtu.errstate(options, fn);
/** @category Error handling */
np.exceptions = dtu.exceptions;
/** @category Formatting */
np.binary_repr = kw([["num", undefined], ["width", null]], fmt.binary_repr);
/** @category Formatting */
np.base_repr = kw([["number", undefined], ["base", 2], ["padding", 0]], fmt.base_repr);
/** @category Formatting */
np.packbits = kw([["a", undefined], ["axis", null], ["bitorder", 'big']], (a, axis, bitorder) => fmt.packbits(asarray(a), axis, bitorder));
/** @category Formatting */
np.unpackbits = kw([["a", undefined], ["axis", null], ["count", null], ["bitorder", 'big']], (a, axis, count, bitorder) => fmt.unpackbits(asarray(a), axis, count, bitorder));
/** @category Formatting */
np.array2string = kw([["a", undefined], ["max_line_width", null], ["precision", null], ["suppress_small", null], ["separator", ' '], ["prefix", ''], ["style", null], ["formatter", null], ["threshold", null], ["edgeitems", null], ["sign", null], ["floatmode", null], ["suffix", ''], ["legacy", null]],
  (a, max_line_width, precision, suppress_small, separator, prefix, style, formatter, threshold, edgeitems, sign, floatmode, suffix, legacy) =>
    fmt.array2string(asarray(a), { max_line_width, precision, suppress_small, separator, prefix, formatter, threshold, edgeitems, sign, floatmode, suffix }));
/** @category Formatting */
np.array_repr = kw([["arr", undefined], ["max_line_width", null], ["precision", null], ["suppress_small", null]], (a, w, p, s) => fmt.array_repr(asarray(a), w, p, s));
/** @category Formatting */
np.array_str = kw([["a", undefined], ["max_line_width", null], ["precision", null], ["suppress_small", null]], (a, w, p, s) => fmt.array_str(asarray(a), w, p, s));
/** @category Formatting */
np.format_float_positional = kw([["x", undefined], ["precision", null], ["unique", true], ["fractional", true], ["trim", 'k'], ["sign", false], ["pad_left", null], ["pad_right", null], ["min_digits", null]], fmt.format_float_positional);
/** @category Formatting */
np.format_float_scientific = kw([["x", undefined], ["precision", null], ["unique", true], ["trim", 'k'], ["sign", false], ["pad_left", null], ["exp_digits", null], ["min_digits", null]], fmt.format_float_scientific);
/** @category Formatting */
np.set_printoptions = kw([["precision", null], ["threshold", null], ["edgeitems", null], ["linewidth", null], ["suppress", null], ["nanstr", null], ["infstr", null], ["formatter", null], ["sign", null], ["floatmode", null], ["legacy", null]], (precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, formatter, sign, floatmode, legacy) =>
  fmt.set_printoptions({ precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, formatter, sign, floatmode, legacy } as any));
/** @category Formatting */
np.get_printoptions = fmt.get_printoptions;
/** @category Formatting */
np.printoptions = (opts: any, fn?: () => any) => fmt.printoptions(opts, fn);

import * as strs from './modules/strings';
const S1 = (impl: any, names: string[] = ["a"]) => kw(names.map(n => [n, undefined] as [string, any]), impl);
const strings_api: { [name: string]: any } = {
  add: kw([["x1", undefined], ["x2", undefined]], strs.add), multiply: kw([["a", undefined], ["i", undefined]], strs.multiply), mod: kw([["a", undefined], ["values", undefined]], strs.mod),
  capitalize: S1(strs.capitalize), lower: S1(strs.lower), upper: S1(strs.upper), swapcase: S1(strs.swapcase), title: S1(strs.title),
  center: kw([["a", undefined], ["width", undefined], ["fillchar", ' ']], strs.center), ljust: kw([["a", undefined], ["width", undefined], ["fillchar", ' ']], strs.ljust), rjust: kw([["a", undefined], ["width", undefined], ["fillchar", ' ']], strs.rjust),
  zfill: kw([["a", undefined], ["width", undefined]], strs.zfill), expandtabs: kw([["a", undefined], ["tabsize", 8]], strs.expandtabs),
  strip: kw([["a", undefined], ["chars", null]], strs.strip), lstrip: kw([["a", undefined], ["chars", null]], strs.lstrip), rstrip: kw([["a", undefined], ["chars", null]], strs.rstrip),
  replace: kw([["a", undefined], ["old", undefined], ["new", undefined], ["count", -1]], strs.replace), slice: kw([["a", undefined], ["start", null], ["stop", strs.NO_VALUE], ["step", null]], strs.slice),
  translate: kw([["a", undefined], ["table", undefined], ["deletechars", null]], strs.translate), decode: kw([["a", undefined], ["encoding", 'utf-8'], ["errors", 'strict']], strs.decode), encode: kw([["a", undefined], ["encoding", 'utf-8'], ["errors", 'strict']], strs.encode),
  partition: kw([["a", undefined], ["sep", undefined]], strs.partition), rpartition: kw([["a", undefined], ["sep", undefined]], strs.rpartition),
  equal: kw([["x1", undefined], ["x2", undefined]], strs.equal), not_equal: kw([["x1", undefined], ["x2", undefined]], strs.not_equal), greater: kw([["x1", undefined], ["x2", undefined]], strs.greater),
  greater_equal: kw([["x1", undefined], ["x2", undefined]], strs.greater_equal), less: kw([["x1", undefined], ["x2", undefined]], strs.less), less_equal: kw([["x1", undefined], ["x2", undefined]], strs.less_equal),
  count: kw([["a", undefined], ["sub", undefined], ["start", 0], ["end", null]], strs.count), find: kw([["a", undefined], ["sub", undefined], ["start", 0], ["end", null]], strs.find), rfind: kw([["a", undefined], ["sub", undefined], ["start", 0], ["end", null]], strs.rfind),
  index: kw([["a", undefined], ["sub", undefined], ["start", 0], ["end", null]], strs.index), rindex: kw([["a", undefined], ["sub", undefined], ["start", 0], ["end", null]], strs.rindex),
  startswith: kw([["a", undefined], ["prefix", undefined], ["start", 0], ["end", null]], strs.startswith), endswith: kw([["a", undefined], ["suffix", undefined], ["start", 0], ["end", null]], strs.endswith),
  isalpha: S1(strs.isalpha, ["x"]), isalnum: S1(strs.isalnum, ["x"]), isdecimal: S1(strs.isdecimal, ["x"]), isdigit: S1(strs.isdigit, ["x"]), isnumeric: S1(strs.isnumeric, ["x"]),
  isspace: S1(strs.isspace, ["x"]), islower: S1(strs.islower, ["x"]), isupper: S1(strs.isupper, ["x"]), istitle: S1(strs.istitle, ["x"]), str_len: S1(strs.str_len, ["x"]),
};
/** Elementwise string functions (numpy.strings) @category Strings */
np.strings = strings_api;
/** Legacy string functions (numpy.char) @category Strings */
np.char = {
  ...strings_api,
  join: kw([["sep", undefined], ["seq", undefined]], strs.join), split: kw([["a", undefined], ["sep", null], ["maxsplit", null]], strs.split), rsplit: kw([["a", undefined], ["sep", null], ["maxsplit", null]], strs.rsplit),
  splitlines: kw([["a", undefined], ["keepends", false]], strs.splitlines),
  array: kw([["obj", undefined], ["itemsize", null], ["copy", true], ["unicode", null]], strs.char_array), asarray: kw([["obj", undefined], ["itemsize", null], ["unicode", null]], (obj: any, i: any, u: any) => strs.char_array(obj, i, false, u)),
};
(np.char as any).chararray = np.char.array;

import * as ioMod from './modules/io';
import * as cre from './modules/creation';
/** Saves an array in .npy format (path, or null to get the bytes). @category Input and output */
np.save = kw([["file", undefined], ["arr", undefined], ["allow_pickle", true], ["fix_imports", true]], ioMod.save);
/** @category Input and output */
np.savez = (file: any, ...args: any[]) => { const kw_ = args.length && args[args.length - 1] && Object.getPrototypeOf(args[args.length - 1]) === Object.prototype ? args.pop() : {}; return ioMod.savez(file, args, kw_); };
/** @category Input and output */
np.savez_compressed = (file: any, ...args: any[]) => { const kw_ = args.length && args[args.length - 1] && Object.getPrototypeOf(args[args.length - 1]) === Object.prototype ? args.pop() : {}; return ioMod.savez_compressed(file, args, kw_); };
/** Loads .npy / .npz files (path or bytes). @category Input and output */
np.load = kw([["file", undefined], ["mmap_mode", null], ["allow_pickle", false], ["fix_imports", true], ["encoding", 'ASCII'], ["max_header_size", 10000]], ioMod.load);
/** @category Input and output */
np.loadtxt = kw([["fname", undefined], ["dtype", 'float64'], ["comments", '#'], ["delimiter", null], ["converters", null], ["skiprows", 0], ["usecols", null], ["unpack", false], ["ndmin", 0], ["encoding", null], ["max_rows", null], ["quotechar", null]],
  (fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, max_rows, quotechar) => ioMod.loadtxt(fname, { dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, max_rows, quotechar }));
/** @category Input and output */
np.genfromtxt = kw([["fname", undefined], ["dtype", 'float64'], ["comments", '#'], ["delimiter", null], ["skip_header", 0], ["skip_footer", 0], ["converters", null], ["missing_values", null], ["filling_values", null], ["usecols", null], ["unpack", false], ["invalid_raise", true], ["max_rows", null]],
  (fname, dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values, filling_values, usecols, unpack, invalid_raise, max_rows) => ioMod.genfromtxt(fname, { dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values: missing_values ?? undefined, filling_values: filling_values ?? undefined, usecols, unpack, invalid_raise, max_rows }));
/** @category Input and output */
np.savetxt = kw([["fname", undefined], ["X", undefined], ["fmt", '%.18e'], ["delimiter", ' '], ["newline", '\n'], ["header", ''], ["footer", ''], ["comments", '# '], ["encoding", null]], ioMod.savetxt);
/** @category Input and output */
np.fromregex = kw([["file", undefined], ["regexp", undefined], ["dtype", undefined]], ioMod.fromregex);
/** @category Input and output */
np.fromstring = kw([["string", undefined], ["dtype", 'float64'], ["count", -1], ["sep", '']], ioMod.fromstring);
/** @category Input and output */
np.fromfile = kw([["file", undefined], ["dtype", 'float64'], ["count", -1], ["sep", ''], ["offset", 0]], ioMod.fromfile);
/** @category Input and output */
np.memmap = kw([["filename", undefined], ["dtype", 'uint8'], ["mode", 'r+'], ["offset", 0], ["shape", null], ["order", 'C']], ioMod.memmap);
/** @category Input and output */
np.lib = { format: { ...ioMod.lib_format }, npyio: { NpzFile: ioMod.NpzFile, DataSource: ioMod.DataSource } };
np.DataSource = ioMod.DataSource;
NDArray.prototype.tofile = function (fid: any, sep: string = '', format: string = '%s') { ioMod.tofile(this, fid, sep, format); };
NDArray.prototype.tobytes = function () { return to_bytes(this); };
/** @category Index tricks */
np.r_ = cre.r_;
/** @category Index tricks */
np.c_ = cre.c_;
/** @category Index tricks */
np.s_ = cre.s_;
/** @category Index tricks */
np.index_exp = cre.index_exp;
(np.mgrid as any).index = np.mgrid; (np.ogrid as any).index = np.ogrid;
/** @category Constructors */
np.asmatrix = kw([["data", undefined], ["dtype", null]], cre.asmatrix);
/** @category Constructors */
np.mat = np.asmatrix;
/** @category Constructors */
np.bmat = kw([["obj", undefined], ["ldict", null], ["gdict", null]], cre.bmat);
/** @category Constructors */
np.ascontiguousarray = kw([["a", undefined], ["dtype", null]], cre.ascontiguousarray);
/** @category Constructors */
np.asfortranarray = kw([["a", undefined], ["dtype", null]], cre.asfortranarray);
/** @category Constructors */
np.asarray_chkfinite = kw([["a", undefined], ["dtype", null], ["order", null]], cre.asarray_chkfinite);
/** @category Constructors */
np.frombuffer = kw([["buffer", undefined], ["dtype", 'float64'], ["count", -1], ["offset", 0]], cre.frombuffer);
/** @category Constructors */
np.fromiter = kw([["iter", undefined], ["dtype", undefined], ["count", -1]], cre.fromiter);
/** @category Constructors */
np.fromfunction = kw([["function", undefined], ["shape", undefined], ["dtype", 'float64'], ["kwargs", {}]], cre.fromfunction);
/** @category Constructors */
np.from_dlpack = cre.from_dlpack;
/** Record arrays are not supported (only the format parser is available). @category Constructors */
np.rec = cre.rec;

import * as misc from './modules/misc';
/** @category Memory */
np.shares_memory = misc.shares_memory;
/** @category Memory */
np.may_share_memory = misc.may_share_memory;
/** @category Broadcasting */
np.broadcast_shapes = misc.broadcast_shapes;

import * as datetime from './modules/datetime';
np.busdaycalendar = sig.busdaycalendar.func((weekmask, holidays) => new datetime.busdaycalendar(weekmask, holidays));
np.is_busday = sig.is_busday.func(datetime.is_busday);
np.busday_offset = sig.busday_offset.func(datetime.busday_offset);
np.busday_count = sig.busday_count.func(datetime.busday_count);
np.datetime_as_string = sig.datetime_as_string.func(datetime.datetime_as_string);
np.datetime_data = datetime.datetime_data;
np.isnat = datetime.isnat;
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
import * as ix from './modules/indexing-extra';
import * as fn from './modules/functional';
import * as la from './modules/linalg';
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
np.take = sig.take_.func(ix.take);
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
NDArray.prototype.transpose = function (...args: any[]) {
  // x.transpose(), x.transpose([1, 0]), x.transpose(1, 0), x.transpose({axes})
  let axes: null | number[] = args.length === 0 ? null : (args.length > 1 || typeof args[0] === 'number') ? args : args[0];
  if (axes !== null && !Array.isArray(axes) && (axes as any).axes !== undefined) axes = (axes as any).axes;
  return array.transpose(this, axes);
};

/** Fills the array (also through views) with a scalar value. @category Transformations */
(NDArray.prototype as any).fill = function (value: any) {
  const n = this.size, buf = np.broadcast_to(np.array(value, this.dtype), [n]).flat;
  this.flat = buf;
  return undefined;
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

NDArray.prototype.astype = sig.astype.method(manip.astype);
NDArray.prototype.flatten = function () { return manip.copy(array.ravel(this)); };
NDArray.prototype.squeeze = sig.squeeze.method(manip.squeeze);
NDArray.prototype.swapaxes = sig.swapaxes.method(array.swapAxes);
NDArray.prototype.repeat = sig.repeat.method(manip.repeat);
NDArray.prototype.take = sig.take_.method(ix.take);
NDArray.prototype.compress = function (...args: any[]) {
  // ndarray.compress(condition, axis=None, out=None): the array is the second parameter of np.compress
  const names = ['condition', 'axis', 'out'];
  const kw: any = {};
  const last = args[args.length - 1];
  if (last !== null && typeof last === 'object' && !Array.isArray(last) && !isarray(last) && Object.getPrototypeOf(last) === Object.prototype) { Object.assign(kw, last); args = args.slice(0, -1); }
  args.forEach((v, i) => { kw[names[i]] = v; });
  return np.compress({ ...kw, a: this });
} as any;
NDArray.prototype.choose = sig.choose.method(ix.choose);
NDArray.prototype.put = sig.put.method(ix.put as any);
NDArray.prototype.trace = sig.trace.method(la.trace);
NDArray.prototype.dot = sig.dot.method(la.dot);
NDArray.prototype.diagonal = sig.diagonal.method(ix.diagonal);
NDArray.prototype.argpartition = sig.partition.method(array.argpartition);
NDArray.prototype.conjugate = NDArray.prototype.conj as any;
NDArray.prototype.partition = function (kth: any, axis: any = -1) {
  if (kth !== null && typeof kth === 'object' && !Array.isArray(kth)) ({ kth, axis = -1 } = kth);
  const r = np.partition(this, kth, axis) as NDArray;
  this.flat = r.flat;
};
NDArray.prototype.resize = function (...shape: any[]) {
  const r = np.resize(this, shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape) as NDArray;
  this._flat = r._flat; this.shape = r.shape; this._simpleIndexes = null;
};
NDArray.prototype.view = function (dtype?: DType) {
  if (dtype === undefined || dtype === this.dtype) return new NDArray(this._flat, this.shape.slice(), this.dtype);
  const bytes = to_bytes(this);
  const sz = TYPE_BY_NAME[canonical_dtype(dtype)].bits / 8;
  if (this.shape.length === 0 || bytes.length % sz !== 0) throw new Error(`When changing to a dtype of different size, the total size of the array must be divisible by the new itemsize`);
  const flat = from_bytes(bytes, canonical_dtype(dtype) as DType);
  const shape = this.shape.slice(); shape[shape.length - 1] = shape[shape.length - 1] * (this.itemsize / sz);
  return new NDArray(flat, shape, canonical_dtype(dtype) as DType);
};
NDArray.prototype.byteswap = function (inplace: boolean = false) {
  const bytes = to_bytes(this), sz = this.itemsize;
  for (let i = 0; i < bytes.length; i += sz) bytes.subarray(i, i + sz).reverse();
  const out = new NDArray(from_bytes(bytes, this.dtype), this.shape.slice(), this.dtype);
  if (inplace) { this.flat = out.flat; return this; }
  return out;
};
NDArray.prototype.to_device = function (_device: string) { return this; };
NDArray.prototype.setflags = function () { };
Object.defineProperties(NDArray.prototype, {
  ndim: { get() { return this.shape.length; } },
  itemsize: { get() { return TYPE_BY_NAME[this.dtype].kind === 'O' ? 8 : TYPE_BY_NAME[this.dtype].bits / 8; } },
  nbytes: { get() { return this.size * this.itemsize; } },
  strides: { get() { const out: number[] = []; let acc = this.itemsize; for (let i = this.shape.length - 1; i >= 0; i--) { out[i] = acc; acc *= this.shape[i]; } return out; } },
  mT: { get() { if (this.shape.length < 2) throw new Error('matrix transpose with ndim < 2 is undefined'); return np.swapaxes(this, -1, -2); } },
  flags: { get() { const c = this._simpleIndexes === null; return { c_contiguous: c, f_contiguous: c && this.shape.length <= 1, owndata: c, writeable: true, aligned: true }; } },
  device: { get() { return 'cpu'; } },
  base: { get() { return null; } },
});
NDArray.prototype.fromlist = fromlist;
NDArray.prototype.tolist = function () {
  return tolist(this);
}

/** Complex number class: `new np.Complex(re, im)` @category Types */
np.Complex = Complex;
/** @category Types */
np.iscomplexobj = (x: any) => asarray(x).dtype === 'complex128';
/** @category Types */
np.isrealobj = (x: any) => asarray(x).dtype !== 'complex128';
/** Complex number from real and imaginary parts. @category Types */
np.complex = (re: number, im: number = 0) => new Complex(re, im);
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
