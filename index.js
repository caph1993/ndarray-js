//@ts-check

/**
 * @param {TemplateStringsArray|any[]|number|boolean} template
 * @param {any[]} variables
 */
const np = function (template, ...variables) {
  const usage = 'Usage example: np`np.arange(10)+${5}` or np([0,1,2]).';
  if (typeof template == "number") return template;
  if (!Array.isArray(template)) throw new Error(`Expected template or array .${usage}`);
  if (!template.length) throw new Error(`Expected argument. ${usage}`);
  if (typeof template[0] == "string") {
    if (variables.length + 1 != template.length) throw new Error(`Wrong input. ${usage}`);
    return np.modules.grammar.parse(/**@type {*}*/(template), ...variables);
  }
  else {
    if (variables.length) throw new Error(`Wrong input. ${usage}`);
    return np.asarray(template)
  }
}


// ==============================
//    Define Global before importing any module
// ==============================
require('./globals').GLOBALS.np = np;

const NDArray = require('./core');
np.NDArray = NDArray;


np.modules = {
  NDArray: NDArray,
  grammar: require('./grammar'),
  constructors: require('./constructors'),
  jsUtils: require('./js-utils'),
}


// ==============================
//    Grammar parser (must be declared after GLOBALS.np assignment)
// ==============================


const jsInterface = NDArray.prototype.modules.jsInterface;
np.fromJS = jsInterface.fromJS;
/**
 * @param {TemplateStringsArray|any[]|number|boolean} template
 * @param {any[]} variables
 */
np.toJS = function (template, ...variables) {
  return jsInterface.toJS(np(template, ...variables));
}


np.ravel = NDArray.prototype.modules.basic.ravel;
np.reshape = NDArray.prototype.modules.basic.reshape;
np.array = NDArray.prototype.modules.basic.array;
np.asarray = NDArray.prototype.modules.basic.asarray;

np.slice = NDArray.prototype.modules.indexes.slice;


const reduce = NDArray.prototype.modules.reduce;
np.all = reduce.reducers.all;
np.any = reduce.reducers.any;
np.sum = reduce.reducers.sum;
np.product = reduce.reducers.product;


const op = NDArray.prototype.modules.op;
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


const ew = NDArray.prototype.modules.elementwise;
np.sign = ew.__make_elementwise(Math.sign);
np.sqrt = ew.__make_elementwise(Math.sqrt);
np.abs = ew.__make_elementwise(Math.abs);
np.exp = ew.__make_elementwise(Math.exp);
np.log = ew.__make_elementwise(Math.log);
np.log2 = ew.__make_elementwise(Math.log2);
np.log10 = ew.__make_elementwise(Math.log10);
np.log1p = ew.__make_elementwise(Math.log1p);
np.sin = ew.__make_elementwise(Math.sin);
np.cos = ew.__make_elementwise(Math.cos);
np.tan = ew.__make_elementwise(Math.tan);
np.asin = ew.__make_elementwise(Math.asin);
np.acos = ew.__make_elementwise(Math.acos);
np.atan = ew.__make_elementwise(Math.atan);
np.atan2 = ew.__make_elementwise(Math.atan2);
np.cosh = ew.__make_elementwise(Math.cosh);
np.sinh = ew.__make_elementwise(Math.sinh);
np.tanh = ew.__make_elementwise(Math.tanh);
np.acosh = ew.__make_elementwise(Math.acosh);
np.asinh = ew.__make_elementwise(Math.asinh);
np.atanh = ew.__make_elementwise(Math.atanh);
np.round = ew.round;


const constructors = np.modules.constructors;

np.empty = constructors.empty;
np.zeros = constructors.zeros;
np.ones = constructors.ones;
np.arange = constructors.arange;
np.linspace = constructors.linspace;
np.geomspace = constructors.geomspace;

module.exports = np;