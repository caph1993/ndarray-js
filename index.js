//@ts-check
var NDArray = require('./core');

/**
 * @param {TemplateStringsArray|any[]|number|boolean} template
 * @param {any[]} variables
 */
var np = function (template, ...variables) {
  const usage = 'Usage example: np`np.arange(10)+${5}` or np([0,1,2]).';
  if (typeof template == "number") return template;
  if (!Array.isArray(template)) throw new Error(`Expected template or array .${usage}`);
  if (!template.length) throw new Error(`Expected argument. ${usage}`);
  if (typeof template[0] == "string") {
    if (variables.length + 1 != template.length) throw new Error(`Wrong input. ${usage}`);
    return np.grammar.parse(/**@type {*}*/(template), ...variables);
  }
  else {
    if (variables.length) throw new Error(`Wrong input. ${usage}`);
    return np.NDArray.prototype.asarray(template)
  }
}

require('./globals').GLOBALS.np = np;

// ==============================
//    Grammar parser (must be declared after GLOBALS.np assignment)
// ==============================

var grammar = require('./core-grammar');
np.grammar = grammar;

/**
 * @param {TemplateStringsArray|any[]|number|boolean} template
 * @param {any[]} variables
 */
np.JS = function (template, ...variables) {
  return np.toJS(np(template, ...variables));
}


var jsInterface = require('./core-js-interface');
np.nested = jsInterface.nested;
np.fromJS = jsInterface.fromJS;
np.toJS = jsInterface.toJS;



np.NDArray = NDArray;

np.zeros = function (shape, /**@type {import('./core').DType} */dtype = Number) {
  const c = dtype == Boolean ? false : 0;
  return NDArray.prototype._new(shape, (_) => c, dtype)
};
np.ones = function (shape, /**@type {import('./core').DType} */dtype = Number) {
  const c = dtype == Boolean ? true : 1;
  return NDArray.prototype._new(shape, (_) => c, dtype)
};
np.arange = function (arg0, arg1 = null) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return NDArray.prototype._new(end - start, (_, i) => start + i, Number)
};


np.ravel = NDArray.prototype.ravel;
np.reshape = NDArray.prototype.reshape;
np.array = NDArray.prototype.array;


var reduce = require('./core-reduce');
np.all = reduce.reducers.all;
np.any = reduce.reducers.any;
np.sum = reduce.reducers.sum;
np.product = reduce.reducers.product;


var op = require('./core-op');
np.add = op.op["+"];
np.subtract = op.op["-"];
np.multiply = op.op["*"];
np.divide = op.op["/"];
np.mod = op.op["%"];
np.divide_int = op.op["//"];
np.pow = op.op["**"];
np.boolean_or = op.op["|"];
np.boolean_and = op.op["&"];
np.boolean_xor = op.op["^"];
np.boolean_shift_left = op.op["<<"];
np.boolean_shift_right = op.op[">>"];
np.gt = op.op[">"];
np.lt = op.op["<"];
np.geq = op.op[">="];
np.leq = op.op["<="];
np.eq = op.op["=="];
np.neq = op.op["!="];
np.maximum = op.op["↑"];
np.minimum = op.op["↓"];




np.apply_pointwise = NDArray.prototype.apply_pointwise;

function __make_pointwise(func, dtype = Number) {
  return function (A) {
    return np.apply_pointwise(A, func, dtype);
  }
}

np.sign = __make_pointwise(Math.sign);
np.sqrt = __make_pointwise(Math.sqrt);
np.abs = __make_pointwise(Math.abs);
np.exp = __make_pointwise(Math.exp);
np.log = __make_pointwise(Math.log);
np.log2 = __make_pointwise(Math.log2);
np.log10 = __make_pointwise(Math.log10);
np.log1p = __make_pointwise(Math.log1p);
np.sin = __make_pointwise(Math.sin);
np.cos = __make_pointwise(Math.cos);
np.tan = __make_pointwise(Math.tan);
np.asin = __make_pointwise(Math.asin);
np.acos = __make_pointwise(Math.acos);
np.atan = __make_pointwise(Math.atan);
np.atan2 = __make_pointwise(Math.atan2);
np.cosh = __make_pointwise(Math.cosh);
np.sinh = __make_pointwise(Math.sinh);
np.tanh = __make_pointwise(Math.tanh);
np.acosh = __make_pointwise(Math.acosh);
np.asinh = __make_pointwise(Math.asinh);
np.atanh = __make_pointwise(Math.atanh);
np._round = __make_pointwise(Math.round);
np.round = function (A, decimals = 0) {
  if (decimals == 0) np._round(A);
  return np.apply_pointwise(A, x => parseFloat(x.toFixed(decimals)), Number);
}






np.random = {
  random(shape) {
    return np.NDArray.prototype.__random(shape);
  },
  uniform(a, b, shape) {
    return np.add(a, np.multiply(np.random.random(shape), (b - a)));
  },
  exponential(mean, shape) {
    return np.multiply(mean, np.subtract(0, np.log(np.random.random(shape))));
  },
  // normal,
  // shuffle,
  // permutation,
};





np.linspace = function (start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  const { __as_number } = NDArray.prototype;
  start = __as_number(start);
  stop = __as_number(stop);
  let n = (num - (endpoint ? 1 : 0))
  let arr = np.add(np.multiply(np.arange(num), (stop - start) / n), start);
  return arr;
}

np.geomspace = function (start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  start = np.log(start);
  stop = np.log(stop);
  return np.exp(np.linspace(start, stop, num, endpoint));
}


module.exports = np;