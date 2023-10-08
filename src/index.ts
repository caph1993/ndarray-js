//@ts-check

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

const { toJS } = NDArray.prototype.modules.jsInterface;
np.JS = (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]) => {
  return toJS(np(template, ...variables));
}
np.fromJS = NDArray.prototype.modules.jsInterface.fromJS;


np.ravel = NDArray.prototype.modules.basic.ravel;
np.reshape = NDArray.prototype.modules.basic.reshape;
np.array = NDArray.prototype.modules.basic.array;
np.asarray = NDArray.prototype.modules.basic.asarray;

const reduce = NDArray.prototype.modules.reduce;
np.sum = reduce.reducers.sum;
np.product = reduce.reducers.product;
np.prod = np.product;
np.any = reduce.reducers.any;
np.all = reduce.reducers.all;
np.max = reduce.reducers.max;
np.min = reduce.reducers.min;
np.argmax = reduce.reducers.argmax;
np.argmin = reduce.reducers.argmin;
np.mean = reduce.reducers.mean;
np.var = reduce.reducers.var;
np.std = reduce.reducers.std;


const transform = NDArray.prototype.modules.transform;
np.transpose = transform.transpose;
np.apply_along_axis = transform.apply_along_axis;
np.sort = transform.sort;
np.concatenate = transform.concatenate;
np.stack = transform.stack;


const operators = NDArray.prototype.modules.operators;
np.add = operators.op_binary["+"];
np.subtract = operators.op_binary["-"];
np.multiply = operators.op_binary["*"];
np.divide = operators.op_binary["/"];
np.mod = operators.op_binary["%"];
np.divide_int = operators.op_binary["//"];
np.pow = operators.op_binary["**"];
np.bitwise_or = operators.op_binary["|"];
np.bitwise_and = operators.op_binary["&"];
np.bitwise_xor = operators.op_binary["^"];
np.bitwise_shift_left = operators.op_binary["<<"];
np.bitwise_shift_right = operators.op_binary[">>"];
np.greater = operators.op_binary[">"];
np.less = operators.op_binary["<"];
np.greater_equal = operators.op_binary[">="];
np.less_equal = operators.op_binary["<="];
np.equal = operators.op_binary["=="];
np.not_equal = operators.op_binary["!="];
np.maximum = operators.op_binary["↑"];
np.minimum = operators.op_binary["↓"];
np.logical_or = operators.op_binary["or"];
np.logical_and = operators.op_binary["and"];

np.bitwise_not = operators.op_unary["~"];
np.logical_not = operators.op_unary["not"];

np.allclose = operators.allclose;
np.isclose = operators.isclose;


const ew = NDArray.prototype.modules.elementwise;

np.sign = ew.ops.sign;
np.sqrt = ew.ops.sqrt;
np.square = ew.ops.square;
np.abs = ew.ops.abs;
np.exp = ew.ops.exp;
np.log = ew.ops.log;
np.log2 = ew.ops.log2;
np.log10 = ew.ops.log10;
np.log1p = ew.ops.log1p;
np.sin = ew.ops.sin;
np.cos = ew.ops.cos;
np.tan = ew.ops.tan;
np.asin = ew.ops.asin;
np.acos = ew.ops.acos;
np.atan = ew.ops.atan;
np.atan2 = ew.ops.atan2;
np.cosh = ew.ops.cosh;
np.sinh = ew.ops.sinh;
np.tanh = ew.ops.tanh;
np.acosh = ew.ops.acosh;
np.asinh = ew.ops.asinh;
np.atanh = ew.ops.atanh;
np.round = ew.ops.round;


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

export { np };
