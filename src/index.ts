//@ts-check

/**
 * Namespace for the ndarray-js package.
 * 
 * np is both the main namespace and a numpy parser: np`...` is equivalent to np.numpy`...`.
 * @param template 
 * @param variables 
 * @returns 
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
import NDArray from './NDArray-class';
np.NDArray = NDArray;

// ==============================
//  Define core-related functions
// ==============================

const { tolist } = NDArray.prototype.modules.jsInterface;
np.tolist = (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) => {
  return tolist(np(template, ...variables));
}
np.fromlist = NDArray.prototype.modules.jsInterface.fromlist;


np.ravel = NDArray.prototype.modules.basic.ravel;
np.reshape = NDArray.prototype.modules.basic.reshape;
np.array = NDArray.prototype.modules.basic.array;
np.asarray = NDArray.prototype.modules.basic.asarray;

const reduce = NDArray.prototype.modules.reduce;
np.sum = reduce.kw_reducers.sum.as_function;
np.product = reduce.kw_reducers.product.as_function;
np.prod = np.product;
np.any = reduce.kw_reducers.any.as_function;
np.all = reduce.kw_reducers.all.as_function;
np.max = reduce.kw_reducers.max.as_function;
np.min = reduce.kw_reducers.min.as_function;
np.argmax = reduce.kw_reducers.argmax.as_function;
np.argmin = reduce.kw_reducers.argmin.as_function;
np.mean = reduce.kw_reducers.mean.as_function;
np.norm = reduce.kw_reducers.norm.as_function;
np.var = reduce.kw_reducers.var.as_function;
np.std = reduce.kw_reducers.std.as_function;



const transform = NDArray.prototype.modules.transform;

np.transpose = transform.transpose;
np.apply_along_axis = transform.apply_along_axis;
np.sort = transform.sort;
np.concatenate = transform.concatenate;
np.stack = transform.stack;

const operators = NDArray.prototype.modules.operators;
np.add = operators.kw_op_binary["+"].as_function;
np.subtract = operators.kw_op_binary["-"].as_function;
np.multiply = operators.kw_op_binary["*"].as_function;
np.divide = operators.kw_op_binary["/"].as_function;
np.mod = operators.kw_op_binary["%"].as_function;
np.divide_int = operators.kw_op_binary["//"].as_function;
np.pow = operators.kw_op_binary["**"].as_function;
np.bitwise_or = operators.kw_op_binary["|"].as_function;
np.bitwise_and = operators.kw_op_binary["&"].as_function;
np.bitwise_xor = operators.kw_op_binary["^"].as_function;
np.bitwise_shift_left = operators.kw_op_binary["<<"].as_function;
np.bitwise_shift_right = operators.kw_op_binary[">>"].as_function;
np.greater = operators.kw_op_binary[">"].as_function;
np.less = operators.kw_op_binary["<"].as_function;
np.greater_equal = operators.kw_op_binary[">="].as_function;
np.less_equal = operators.kw_op_binary["<="].as_function;
np.equal = operators.kw_op_binary["=="].as_function;
np.not_equal = operators.kw_op_binary["!="].as_function;
np.maximum = operators.kw_op_binary["max"].as_function;
np.minimum = operators.kw_op_binary["min"].as_function;
np.logical_or = operators.kw_op_binary["or"].as_function;
np.logical_and = operators.kw_op_binary["and"].as_function;
np.atan2 = operators.atan2;


np.allclose = operators.allclose;
np.isclose = operators.isclose;


const ew = NDArray.prototype.modules.elementwise;

np.sign = ew.funcs.sign;
np.sqrt = ew.funcs.sqrt;
np.square = ew.funcs.square;
np.exp = ew.funcs.exp;
np.log = ew.funcs.log;
np.log2 = ew.funcs.log2;
np.log10 = ew.funcs.log10;
np.log1p = ew.funcs.log1p;
np.sin = ew.funcs.sin;
np.cos = ew.funcs.cos;
np.tan = ew.funcs.tan;
np.asin = ew.funcs.asin;
np.acos = ew.funcs.acos;
np.atan = ew.funcs.atan;
np.cosh = ew.funcs.cosh;
np.sinh = ew.funcs.sinh;
np.tanh = ew.funcs.tanh;
np.acosh = ew.funcs.acosh;
np.asinh = ew.funcs.asinh;
np.atanh = ew.funcs.atanh;

np.abs = ew.kw_ops.abs.as_function;
np.bitwise_not = ew.kw_ops.bitwise_not.as_function;
np.logical_not = ew.kw_ops.logical_not.as_function;
np.abs = ew.kw_ops.abs.as_function;
np.negative = ew.kw_ops.negative.as_function;
np.round = ew.kw_ops.round.as_function;


// ==============================
//   import np modules
// ============================== 

import { modules } from './modules';

np.modules = modules;

np.empty = np.modules.constructors.empty;
np.zeros = np.modules.constructors.zeros;
np.ones = np.modules.constructors.ones;
np.arange = np.modules.constructors.arange;
np.linspace = np.modules.constructors.linspace;
np.geomspace = np.modules.constructors.geomspace;

np.random = np.modules.random;

np.pi = Math.PI;
np.e = Math.E;

export { np };
