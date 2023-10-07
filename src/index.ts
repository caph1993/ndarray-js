//@ts-check

/**
 * @param {TemplateStringsArray|any[]|number|boolean} template
 * @param {any[]} variables
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
import { GLOBALS } from './globals';
GLOBALS.np = np;

// ==============================
//    Define casting and core before importing any other module
// ==============================
import NDArray from './core';
np.NDArray = NDArray;



np.transpose = function () { };

// np.modules = {
//   NDArray: NDArray,
//   grammar: require('./grammar'),
//   constructors: require('./constructors'),
//   jsUtils: require('./js-utils'),
//   random: require('./random'),
// }

// np.random = np.modules.random;


// // ==============================
// //    Grammar parser (must be declared after GLOBALS.np assignment)
// // ==============================

import * as jsInterface from './core-js-interface';

np.fromJS = jsInterface.fromJS;

np.JS = (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) => {
  return jsInterface.toJS(np(template, ...variables));
}


// np.ravel = NDArray.prototype.modules.basic.ravel;
// np.reshape = NDArray.prototype.modules.basic.reshape;
// np.array = NDArray.prototype.modules.basic.array;
// np.asarray = NDArray.prototype.modules.basic.asarray;



// const reduce = NDArray.prototype.modules.reduce;
// np.sum = reduce.reducers.sum;
// np.product = reduce.reducers.product;
// np.prod = np.product;
// np.any = reduce.reducers.any;
// np.all = reduce.reducers.all;
// np.max = reduce.reducers.max;
// np.min = reduce.reducers.min;
// np.argmax = reduce.reducers.argmax;
// np.argmin = reduce.reducers.argmin;
// np.mean = reduce.reducers.mean;
// np.var = reduce.reducers.var;
// np.std = reduce.reducers.std;


// const transform = NDArray.prototype.modules.transform;
// np.transpose = transform.transpose;
// np.apply_along_axis = transform.apply_along_axis;
// np.sort = transform.sort;
// np.concatenate = transform.concatenate;
// np.stack = transform.stack;



// const operators = NDArray.prototype.modules.operators;
// np.add = operators.op_binary["+"];
// np.subtract = operators.op_binary["-"];
// np.multiply = operators.op_binary["*"];
// np.divide = operators.op_binary["/"];
// np.mod = operators.op_binary["%"];
// np.divide_int = operators.op_binary["//"];
// np.pow = operators.op_binary["**"];
// np.bitwise_or = operators.op_binary["|"];
// np.bitwise_and = operators.op_binary["&"];
// np.bitwise_xor = operators.op_binary["^"];
// np.bitwise_shift_left = operators.op_binary["<<"];
// np.bitwise_shift_right = operators.op_binary[">>"];
// np.greater = operators.op_binary[">"];
// np.less = operators.op_binary["<"];
// np.greater_equal = operators.op_binary[">="];
// np.less_equal = operators.op_binary["<="];
// np.equal = operators.op_binary["=="];
// np.not_equal = operators.op_binary["!="];
// np.maximum = operators.op_binary["↑"];
// np.minimum = operators.op_binary["↓"];
// np.logical_or = operators.op_binary["or"];
// np.logical_and = operators.op_binary["and"];

// np.bitwise_not = operators.op_unary["~"];
// np.logical_not = operators.op_unary["not"];



// const ew = NDArray.prototype.modules.elementwise;
// np.sign = ew.__make_elementwise(Math.sign);
// np.sqrt = ew.__make_elementwise(Math.sqrt);
// np.square = ew.__make_elementwise((a) => a * a);
// np.abs = ew.__make_elementwise(Math.abs);
// np.exp = ew.__make_elementwise(Math.exp);
// np.log = ew.__make_elementwise(Math.log);
// np.log2 = ew.__make_elementwise(Math.log2);
// np.log10 = ew.__make_elementwise(Math.log10);
// np.log1p = ew.__make_elementwise(Math.log1p);
// np.sin = ew.__make_elementwise(Math.sin);
// np.cos = ew.__make_elementwise(Math.cos);
// np.tan = ew.__make_elementwise(Math.tan);
// np.asin = ew.__make_elementwise(Math.asin);
// np.acos = ew.__make_elementwise(Math.acos);
// np.atan = ew.__make_elementwise(Math.atan);
// np.atan2 = ew.__make_elementwise(Math.atan2);
// np.cosh = ew.__make_elementwise(Math.cosh);
// np.sinh = ew.__make_elementwise(Math.sinh);
// np.tanh = ew.__make_elementwise(Math.tanh);
// np.acosh = ew.__make_elementwise(Math.acosh);
// np.asinh = ew.__make_elementwise(Math.asinh);
// np.atanh = ew.__make_elementwise(Math.atanh);
// np.round = ew.round;


// const constructors = np.modules.constructors;

// np.empty = constructors.empty;
// np.zeros = constructors.zeros;
// np.ones = constructors.ones;
// np.arange = constructors.arange;
// np.linspace = constructors.linspace;
// np.geomspace = constructors.geomspace;




export default np;