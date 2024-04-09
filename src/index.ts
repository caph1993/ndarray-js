//@ts-check

/**
 * Parser and main namespace for the ndarray-js package.
 * <script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>
 */
const np = function (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) {
  const usage = 'Usage example: np`np.arange(10)+${5}` or np([0,1,2]).';
  if (typeof template == "number") return template;
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


// ==============================
//    Define Global before importing any module
// ==============================
import { GLOBALS } from './_globals';
GLOBALS.np = np;

// ==============================
//    Define casting and core before importing any other module
// ==============================
import NDArray from './NDArray';
/** @category Main */
np.NDArray = NDArray;

// ==============================
//  Define core-related functions
// ==============================

const { tolist } = NDArray.prototype.modules.jsInterface;
/** @category Casting and reshaping */
np.tolist = (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) => {
  return tolist(np(template, ...variables));
}
/** @category Casting and reshaping */
np.fromlist = NDArray.prototype.modules.jsInterface.fromlist;

/** @category Casting and reshaping */
np.ravel = NDArray.prototype.modules.basic.ravel;
/** @category Casting and reshaping */
np.reshape = NDArray.prototype.modules.basic.reshape;
/** @category Casting and reshaping */
np.array = NDArray.prototype.modules.basic.array;
/** @category Casting and reshaping */
np.asarray = NDArray.prototype.modules.basic.asarray;

const reduce = NDArray.prototype.modules.reduce;
/** @category Reducers */
np.sum = reduce.kw_reducers.sum.as_function;
/** @category Reducers */
np.product = reduce.kw_reducers.product.as_function;
/** @category Reducers */
np.prod = np.product;
/** @category Reducers */
np.any = reduce.kw_reducers.any.as_function;
/** @category Reducers */
np.all = reduce.kw_reducers.all.as_function;
/** @category Reducers */
np.max = reduce.kw_reducers.max.as_function;
/** @category Reducers */
np.min = reduce.kw_reducers.min.as_function;
/** @category Reducers */
np.argmax = reduce.kw_reducers.argmax.as_function;
/** @category Reducers */
np.argmin = reduce.kw_reducers.argmin.as_function;
/** @category Reducers */
np.mean = reduce.kw_reducers.mean.as_function;
/** @category Reducers */
np.norm = reduce.kw_reducers.norm.as_function;
/** @category Reducers */
np.var = reduce.kw_reducers.var.as_function;
/** @category Reducers */
np.std = reduce.kw_reducers.std.as_function;



const transform = NDArray.prototype.modules.transform;

/** @category Transformations */
np.transpose = transform.transpose;
/** @category Transformations */
np.apply_along_axis = transform.apply_along_axis;
/** @category Transformations */
np.sort = transform.kw_exported.sort;
/** @category Transformations */
np.concatenate = transform.concatenate;
/** @category Transformations */
np.stack = transform.stack;

const operators = NDArray.prototype.modules.operators;

/** @category Binary operators */
np.add = operators.kw_op_binary["+"].as_function;
/** @category Binary operators */
np.subtract = operators.kw_op_binary["-"].as_function;
/** @category Binary operators */
np.multiply = operators.kw_op_binary["*"].as_function;
/** @category Binary operators */
np.divide = operators.kw_op_binary["/"].as_function;
/** @category Binary operators */
np.mod = operators.kw_op_binary["%"].as_function;
/** @category Binary operators */
np.divide_int = operators.kw_op_binary["//"].as_function;
/** @category Binary operators */
np.pow = operators.kw_op_binary["**"].as_function;
/** @category Binary operators */
np.bitwise_or = operators.kw_op_binary["|"].as_function;
/** @category Binary operators */
np.bitwise_and = operators.kw_op_binary["&"].as_function;
/** @category Binary operators */
np.bitwise_xor = operators.kw_op_binary["^"].as_function;
/** @category Binary operators */
np.bitwise_shift_left = operators.kw_op_binary["<<"].as_function;
/** @category Binary operators */
np.bitwise_shift_right = operators.kw_op_binary[">>"].as_function;
/** @category Binary operators */
np.greater = operators.kw_op_binary[">"].as_function;
/** @category Binary operators */
np.less = operators.kw_op_binary["<"].as_function;
/** @category Binary operators */
np.greater_equal = operators.kw_op_binary[">="].as_function;
/** @category Binary operators */
np.less_equal = operators.kw_op_binary["<="].as_function;
/** @category Binary operators */
np.equal = operators.kw_op_binary["=="].as_function;
/** @category Binary operators */
np.not_equal = operators.kw_op_binary["!="].as_function;
/** @category Binary operators */
np.maximum = operators.kw_op_binary["max"].as_function;
/** @category Binary operators */
np.minimum = operators.kw_op_binary["min"].as_function;
/** @category Binary operators */
np.logical_or = operators.kw_op_binary["or"].as_function;
/** @category Binary operators */
np.logical_and = operators.kw_op_binary["and"].as_function;
np.atan2 = operators.atan2;


np.assign = operators.op_assign['='];

np.allclose = operators.allclose;
np.isclose = operators.isclose;


const ew = NDArray.prototype.modules.elementwise;

/** @category Elementwise operators */
np.sign = ew.kw_funcs.sign;
/** @category Elementwise operators */
np.sqrt = ew.kw_funcs.sqrt;
/** @category Elementwise operators */
np.square = ew.kw_funcs.square;
/** @category Elementwise operators */
np.exp = ew.kw_funcs.exp;
/** @category Elementwise operators */
np.log = ew.kw_funcs.log;
/** @category Elementwise operators */
np.log2 = ew.kw_funcs.log2;
/** @category Elementwise operators */
np.log10 = ew.kw_funcs.log10;
/** @category Elementwise operators */
np.log1p = ew.kw_funcs.log1p;
/** @category Elementwise operators */
np.sin = ew.kw_funcs.sin;
/** @category Elementwise operators */
np.cos = ew.kw_funcs.cos;
/** @category Elementwise operators */
np.tan = ew.kw_funcs.tan;
/** @category Elementwise operators */
np.asin = ew.kw_funcs.asin;
/** @category Elementwise operators */
np.acos = ew.kw_funcs.acos;
/** @category Elementwise operators */
np.atan = ew.kw_funcs.atan;
/** @category Elementwise operators */
np.cosh = ew.kw_funcs.cosh;
/** @category Elementwise operators */
np.sinh = ew.kw_funcs.sinh;
/** @category Elementwise operators */
np.tanh = ew.kw_funcs.tanh;
/** @category Elementwise operators */
np.acosh = ew.kw_funcs.acosh;
/** @category Elementwise operators */
np.asinh = ew.kw_funcs.asinh;
/** @category Elementwise operators */
np.atanh = ew.kw_funcs.atanh;
/** @category Elementwise operators */
np.floor = ew.kw_funcs.floor;
/** @category Elementwise operators */
np.ceil = ew.kw_funcs.ceil;

/** @category Elementwise operators */
np.isnan = ew.kw_funcs.isnan;
/** @category Elementwise operators */
np.isfinite = ew.kw_funcs.isfinite;

/** @category Elementwise operators */
np.abs = ew.kw_ops.abs.as_function;
/** @category Elementwise operators */
np.bitwise_not = ew.kw_ops.bitwise_not.as_function;
/** @category Elementwise operators */
np.logical_not = ew.kw_ops.logical_not.as_function;
/** @category Elementwise operators */
np.negative = ew.kw_ops.negative.as_function;
/** @category Elementwise operators */
np.round = ew.kw_ops.round.as_function;


// ==============================
//   import np modules
// ============================== 

import { modules } from './modules';

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
np.quantile = np.modules.statistics.kw_exported.quantile;
np.nanquantile = np.modules.statistics.kw_exported.nanquantile;
// np.percentile = np.modules.statistics.kw_exported.percentile;
// np.median = np.modules.statistics.kw_exported.median;
// np.average = np.modules.statistics.kw_exported.average;


/** @category Math constants */
np.pi = Math.PI;
/** @category Math constants */
np.e = Math.E;

export { np };
