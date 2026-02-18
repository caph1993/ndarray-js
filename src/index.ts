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


import { Func_a_a_min_a_max_out, Func_a_axis_ddof_keepdims, Func_a_axis_keepdims, Func_a_decimals_out, Func_a_lastAxis, Func_a_ord_axis_keepdims, Func_a_other_out, Func_a_out, Func_x1_x2_out, Func_y_x_out, Method_a_axis_ddof_keepdims, Method_a_axis_keepdims, Method_a_decimals_out, Method_a_ord_axis_keepdims, Method_other_out, Method_out, Method_values_where } from './kwargs/kwargs';

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
np.ravel = array.ravel;
/** @category Casting and reshaping */
np.reshape = array.reshape;
/** @category Casting and reshaping */
np.array = array.array;
/** @category Casting and reshaping */
np.asarray = asarray;


import * as reducers from './array/reduce';
/** @category Reducers */
np.sum = Func_a_axis_keepdims.decorate(reducers.sum);
/** @category Reducers */
np.product = Func_a_axis_keepdims.decorate(reducers.product);
/** @category Reducers */
np.mean = Func_a_axis_keepdims.decorate(reducers.mean);
/** @category Reducers */
np.max = Func_a_axis_keepdims.decorate(reducers.max);
/** @category Reducers */
np.min = Func_a_axis_keepdims.decorate(reducers.min);
/** @category Reducers */
np.argmax = Func_a_axis_keepdims.decorate(reducers.argmax);
/** @category Reducers */
np.argmin = Func_a_axis_keepdims.decorate(reducers.argmin);
/** @category Reducers */
np.len = Func_a_axis_keepdims.decorate(reducers.len);
/** @category Reducers */
np.any = Func_a_axis_keepdims.decorate(reducers.any);
/** @category Reducers */
np.all = Func_a_axis_keepdims.decorate(reducers.all);
/** @category Reducers */
np.norm = Func_a_ord_axis_keepdims.decorate(reducers.norm);
/** @category Reducers */
np.var = Func_a_axis_ddof_keepdims.decorate(reducers.variance);
/** @category Reducers */
np.std = Func_a_axis_ddof_keepdims.decorate(reducers.std);

/** @category Transformations */
np.transpose = array.transpose;
/** @category Transformations */
np.apply_along_axis = array.apply_along_axis;

/** @category Transformations */
np.sort = Func_a_lastAxis.decorate(array.sort);
/** @category Transformations */
np.argsort = Func_a_lastAxis.decorate(array.argsort);
/** @category Transformations */
np.concatenate = array.concatenate;
/** @category Transformations */
np.stack = array.stack;


import * as operators from './array/operators';

/** @category Binary operators */
np.add = Func_a_other_out.decorate(operators.op_binary["+"]);
/** @category Binary operators */
np.subtract = Func_a_other_out.decorate(operators.op_binary["-"]);
/** @category Binary operators */
np.multiply = Func_a_other_out.decorate(operators.op_binary["*"]);
/** @category Binary operators */
np.divide = Func_a_other_out.decorate(operators.op_binary["divide"]);
/** @category Binary operators */
np.true_divide = Func_a_other_out.decorate(operators.op_binary["true_divide"]);
/** @category Binary operators */
np.mod = Func_a_other_out.decorate(operators.op_binary["%"]);
/** @category Binary operators */
np.fmod = Func_a_other_out.decorate(operators.op_binary["fmod"]);
/** @category Binary operators */
np.remainder = Func_a_other_out.decorate(operators.op_binary["remainder"]);
/** @category Binary operators */
np.divide_int = Func_a_other_out.decorate(operators.op_binary["//"]);
/** @category Binary operators */
np.floor_divide = Func_a_other_out.decorate(operators.op_binary["//"]);
/** @category Binary operators */
np.pow = Func_a_other_out.decorate(operators.op_binary["**"]);
/** @category Binary operators */
np.power = np.pow;
/** @category Binary operators */
np.float_power = Func_a_other_out.decorate(operators.float_power);
/** @category Binary operators */
np.bitwise_or = Func_a_other_out.decorate(operators.op_binary["|"]);
/** @category Binary operators */
np.bitwise_and = Func_a_other_out.decorate(operators.op_binary["&"]);
/** @category Binary operators */
np.bitwise_xor = Func_a_other_out.decorate(operators.op_binary["^"]);
/** @category Binary operators */
np.bitwise_shift_left = Func_a_other_out.decorate(operators.op_binary["<<"]);
/** @category Binary operators */
np.bitwise_shift_right = Func_a_other_out.decorate(operators.op_binary[">>"]);
/** @category Binary operators */
np.greater = Func_a_other_out.decorate(operators.op_binary[">"]);
/** @category Binary operators */
np.less = Func_a_other_out.decorate(operators.op_binary["<"]);
/** @category Binary operators */
np.greater_equal = Func_a_other_out.decorate(operators.op_binary[">="]);
/** @category Binary operators */
np.less_equal = Func_a_other_out.decorate(operators.op_binary["<="]);
/** @category Binary operators */
np.equal = Func_a_other_out.decorate(operators.op_binary["=="]);
/** @category Binary operators */
np.not_equal = Func_a_other_out.decorate(operators.op_binary["!="]);
/** @category Binary operators */
np.maximum = Func_a_other_out.decorate(operators.op_binary["max"]);
/** @category Binary operators */
np.minimum = Func_a_other_out.decorate(operators.op_binary["min"]);
/** @category Binary operators */
np.fmax = Func_a_other_out.decorate(operators.op_binary["fmax"]);
/** @category Binary operators */
np.fmin = Func_a_other_out.decorate(operators.op_binary["fmin"]);
/** @category Binary operators */
np.logical_or = Func_a_other_out.decorate(operators.op_binary["or"]);
/** @category Binary operators */
np.logical_and = Func_a_other_out.decorate(operators.op_binary["and"]);
/** @category Binary operators */
np.logical_xor = Func_a_other_out.decorate(operators.op_binary["xor"]);
np.atan2 = operators.atan2;
np.hypot = Func_a_other_out.decorate(operators.op_binary["hypot"]);






np.assign = operators.op_assign['='];

np.allclose = operators.allclose;
np.isclose = operators.isclose;
np.array_equal = operators.array_equal;
np.array_equiv = operators.array_equiv;


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
np.floor = Func_a_out.decorate(elementwise.floor);
/** @category Elementwise operators */
np.ceil = Func_a_out.decorate(elementwise.ceil);
/** @category Elementwise operators */
np.negative = Func_a_out.decorate(elementwise.negative);
/** @category Elementwise operators */
np.bitwise_not = Func_a_out.decorate(elementwise.bitwise_not);
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
np.conjugate = Func_a_out.decorate(elementwise.conjugate);
/** @category Elementwise operators */
np.cbrt = Func_a_out.decorate(elementwise.cbrt);
/** @category Elementwise operators */
np.nan_to_num = Func_a_out.decorate(elementwise.nan_to_num);
/** @category Elementwise operators */
np.real_if_close = Func_a_out.decorate(elementwise.real_if_close);
/** @category Elementwise operators */
np.round = Func_a_decimals_out.decorate(elementwise.round);




np.arcsin = np.asin;
np.arccos = np.acos;
np.arctan = np.atan;
np.arctan2 = np.atan2;

// Aliases and special functions
np.absolute = np.abs;
np.fabs = np.abs;

np.amax = np.max;
np.amin = np.min;

np.divmod = operators.divmod;

np.heaviside = Func_x1_x2_out.decorate(operators.heaviside);
np.clip = Func_a_a_min_a_max_out.decorate(operators.clip);
np.atan2 = Func_y_x_out.decorate(operators.atan2);



np.isscalar = (obj: any) => {
  if (isarray(obj)) return false;
  if (Array.isArray(obj)) return false;
  if (obj === null || obj === undefined) return false;
  const t = typeof obj;
  return t === 'number' || t === 'boolean' || t === 'string' || t === 'bigint' || t === 'symbol';
}


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

/** @category Constructors */
np.empty = np.modules.constructors.empty;
/** @category Constructors */
np.zeros = np.modules.constructors.zeros;
/** @category Constructors */
np.ones = np.modules.constructors.ones;
/** @category Constructors */
np.arange = np.modules.constructors.arange;
/** @category Constructors */
np.linspace = np.modules.constructors.linspace;
/** @category Constructors */
np.geomspace = np.modules.constructors.geomspace;


np.take = np.modules.indexing.take;
np.where = np.modules.indexing.where;
np.nonzero = np.modules.indexing.nonzero;
np.quantile = np.modules.statistics.quantile;
np.nanquantile = np.modules.statistics.nanquantile;
// np.percentile = np.modules.statistics.kw_exported.percentile;
// np.median = np.modules.statistics.kw_exported.median;
// np.average = np.modules.statistics.kw_exported.average;



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
// NDArray.prototype.bitwise_xor = Method_other_out.decorate(array.prototype);
// NDArray.prototype.bitwise_shift_left = Method_other_out.decorate(array.prototype);
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

NDArray.prototype.isclose = array.isclose;
NDArray.prototype.allclose = array.allclose;
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
  return array.transpose(this, axes);
};

NDArray.prototype.sort = function (axis = -1) {
  array.sort(this, axis);
  return null;
};

// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = Method_out.decorate(elementwise.bitwise_not);
NDArray.prototype.logical_not = Method_out.decorate(elementwise.logical_not);
NDArray.prototype.negative = Method_out.decorate(elementwise.negative);
NDArray.prototype.abs = Method_out.decorate(elementwise.abs);
NDArray.prototype.round = Method_a_decimals_out.decorate(elementwise.round);


NDArray.prototype.any = Method_a_axis_keepdims.decorate(reducers.any);
NDArray.prototype.all = Method_a_axis_keepdims.decorate(reducers.all);
NDArray.prototype.sum = Method_a_axis_keepdims.decorate(reducers.sum);
NDArray.prototype.product = Method_a_axis_keepdims.decorate(reducers.product);
NDArray.prototype.max = Method_a_axis_keepdims.decorate(reducers.max);
NDArray.prototype.min = Method_a_axis_keepdims.decorate(reducers.min);
NDArray.prototype.argmax = Method_a_axis_keepdims.decorate(reducers.argmax);
NDArray.prototype.argmin = Method_a_axis_keepdims.decorate(reducers.argmin);
NDArray.prototype.mean = Method_a_axis_keepdims.decorate(reducers.mean);
NDArray.prototype.var = Method_a_axis_ddof_keepdims.decorate(reducers.variance);
NDArray.prototype.std = Method_a_axis_ddof_keepdims.decorate(reducers.std);
NDArray.prototype.norm = Method_a_ord_axis_keepdims.decorate(reducers.norm);

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

export { np };
