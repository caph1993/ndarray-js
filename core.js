//@ts-check
/** @typedef {NumberConstructor|BooleanConstructor} DType */
/** @typedef {NDArray|number|boolean} ArrayOrConstant */

class NDArray {
  /**
   * @param {number[]} flat actually number|boolean
   * @param {number[]} shape
   * @param {*} dtype
   */
  constructor(flat, shape, dtype = Number) {
    this.shape = shape; // invariant: immutable
    this._flat = flat;
    this.dtype = dtype;
    this._simpleIndexes = null;
  }
  /** @type {import("./core-indexes").AxesIndex|null} */ _simpleIndexes;

  get size() {
    return this._simpleIndexes == null ? this._flat.length : this._simpleIndexes.size;
  }
  get flat() {
    if (this._simpleIndexes == null) return this._flat;
    const indices = this._simpleIndexes.indices;
    return indices.map(i => this._flat[i]);
  }
  set flat(list) {
    if (list.length != this.size)
      throw new Error(`Length mismatch. Can't write ${list.length} values into ${this.size} available positions.`);
    const n = this.size;
    if (this._simpleIndexes == null) {
      for (let i = 0; i < n; i++) this._flat[i] = list[i];
    } else {
      const indices = this._simpleIndexes.indices;
      for (let i = 0; i < n; i++) this._flat[indices[i]] = list[i];
    }
  }
  get T() {
    return this.transpose();
  }
  __popKwArgs() {
    let out = this['__kwArgs'];
    if (out === undefined) return {};
    delete this['__kwArgs'];
    return out;
  }
  /** @param {Object<string, any>} kwArgs */
  withKwArgs(kwArgs) {
    this['__kwArgs'] = kwArgs;
    return this;
  }
  *[Symbol.iterator]() {
    for (let i = 0; i < this.shape[0]; i++) yield this.index(i);
  }
  get length() {
    return this.shape[0] || 0;
  }
}

require('./globals').GLOBALS.NDArray = NDArray;


const modules = {};
NDArray.prototype.modules = modules; // NEEDED before loading the modules!

modules.basic = require('./core-basic');
modules.indexes = require('./core-indexes');
modules.elementwise = require('./core-elementwise');
modules.print = require('./core-print');
modules.reduce = require('./core-reduce');
modules.operators = require('./core-operators');
modules.transform = require('./core-transform');
modules.jsInterface = require('./core-js-interface');



// ==============================
//    Basic methods
// ==============================

const basic = modules.basic;

NDArray.prototype.reshape = function (shape, ...more_shape) {
  return basic.reshape(this, shape, ...more_shape);
};

NDArray.prototype.ravel = function () {
  return basic.ravel(this);
};

NDArray.prototype.copy = function () {
  return basic.copy(this);
};

/** @returns {NDArray} */
NDArray.prototype.asarray = function () {
  return basic.asarray(this);
}

/** @returns {NDArray} */
NDArray.prototype.array = function () {
  return basic.array(this);
}

NDArray.prototype.__shape_shifts = function (shape) {
  // increasing one by one on a given axis is increasing by shifts[axis] in flat representation
  const shifts = Array.from({ length: shape.length }, (_) => 0);
  shifts[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) shifts[i] = shifts[i + 1] * shape[i + 1];
  return shifts;
}

/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 * @param {NDArray} arr 
 * @returns {NDArray|number}
 */
NDArray.prototype.__number_collapse = function (arr, expect = false) {
  if (!arr.shape.length) return arr.flat[0];
  if (expect) throw new Error(`Expected constant. Got array with shape ${arr.shape}`);
  return arr;
}

NDArray.prototype.__parse_shape = function (list) {
  if (Array.isArray(list)) return list;
  if (list instanceof NDArray) {
    if (list.shape.length > 1) {
      throw new Error(`Expected flat list. Got array with shape ${list.shape}`);
    }
    return list.flat;
  }
  if (typeof list == "number") return [list];
  throw new Error(`Expected list. Got ${list}`);
}

NDArray.prototype.__as_boolean = function (obj) {
  if (obj instanceof NDArray) obj = NDArray.prototype.__number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return !!(0 + obj);
}
NDArray.prototype.__as_number = function (obj) {
  if (obj instanceof NDArray) obj = NDArray.prototype.__number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return parseFloat(obj);
}


NDArray.prototype._new = function (shape, f, dtype) {
  shape = NDArray.prototype.__parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  const flat = Array.from({ length: size }, f);
  return new NDArray(flat, shape, dtype);
};


// ==============================
//    Indexing
// ==============================

/**
 * @param {import("./core-indexes").GeneralIndexSpec[]} indexesSpec
 */
NDArray.prototype.index = function (...indexesSpec) {
  return modules.indexes.index(this, ...indexesSpec);
}

// ==============================
//    Printing
// ==============================

NDArray.prototype.toString = function () {
  return modules.print.humanReadable(this);
}


// ==============================
//    Reduce
// ==============================






function reduceDecorator(func) {
  /** @param {import("./core-reduce").AxisArg} axis  @param {boolean} keepdims */
  return function (axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this.__popKwArgs()));
    return func(this, axis, keepdims);
  }
}

NDArray.prototype.sum = reduceDecorator(modules.reduce.reducers.sum);
NDArray.prototype.product = reduceDecorator(modules.reduce.reducers.product);
NDArray.prototype.any = reduceDecorator(modules.reduce.reducers.any);
NDArray.prototype.all = reduceDecorator(modules.reduce.reducers.all);
NDArray.prototype.max = reduceDecorator(modules.reduce.reducers.max);
NDArray.prototype.min = reduceDecorator(modules.reduce.reducers.min);
NDArray.prototype.argmax = reduceDecorator(modules.reduce.reducers.argmax);
NDArray.prototype.argmin = reduceDecorator(modules.reduce.reducers.argmin);
NDArray.prototype.mean = reduceDecorator(modules.reduce.reducers.mean);
NDArray.prototype.var = reduceDecorator(modules.reduce.reducers.var);
NDArray.prototype.std = reduceDecorator(modules.reduce.reducers.std);



// ==============================
//       Operators: Binary operations, assignment operations and unary boolean_not
// ==============================

NDArray.prototype._binary_operation = modules.operators.binary_operation;


function opDecorator(func) {
  /**
   * @param {import("./core-operators").ArrayOrConstant} other
   * @param {NDArray?} out
   * @returns {NDArray}
   */
  return function (other, out = null) {
    ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, other, out);
  }
}

NDArray.prototype.add = opDecorator(modules.operators.op["+"]);
NDArray.prototype.subtract = opDecorator(modules.operators.op["-"]);
NDArray.prototype.multiply = opDecorator(modules.operators.op["*"]);
NDArray.prototype.divide = opDecorator(modules.operators.op["/"]);
NDArray.prototype.mod = opDecorator(modules.operators.op["%"]);
NDArray.prototype.divide_int = opDecorator(modules.operators.op["//"]);
NDArray.prototype.pow = opDecorator(modules.operators.op["**"]);

NDArray.prototype.maximum = opDecorator(modules.operators.op["↑"]);
NDArray.prototype.minimum = opDecorator(modules.operators.op["↓"]);

NDArray.prototype.bitwise_or = opDecorator(modules.operators.op["|"]);
NDArray.prototype.bitwise_and = opDecorator(modules.operators.op["&"]);
NDArray.prototype.bitwise_shift_left = opDecorator(modules.operators.op["<<"]);
NDArray.prototype.bitwise_shift_right = opDecorator(modules.operators.op[">>"]);

NDArray.prototype.logical_or = opDecorator(modules.operators.op["or"]);
NDArray.prototype.logical_and = opDecorator(modules.operators.op["and"]);

NDArray.prototype.greater = opDecorator(modules.operators.op[">"]);
NDArray.prototype.less = opDecorator(modules.operators.op["<"]);
NDArray.prototype.greater_equal = opDecorator(modules.operators.op[">="]);
NDArray.prototype.less_equal = opDecorator(modules.operators.op["<="]);
NDArray.prototype.equal = opDecorator(modules.operators.op["=="]);
NDArray.prototype.not_equal = opDecorator(modules.operators.op["!="]);


function unaryOpDecorator(func) {
  /** @param {NDArray?} out  @returns {NDArray} */
  return function (out = null) {
    ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, out);
  }
}
// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = unaryOpDecorator(modules.operators.op["~"]);
NDArray.prototype.logical_not = unaryOpDecorator(modules.operators.op["not"]);
NDArray.prototype.bitwise_xor = unaryOpDecorator(modules.operators.op["^"]);
NDArray.prototype.logical_xor = unaryOpDecorator(modules.operators.op["xor"]);


NDArray.prototype.isclose = modules.operators.isclose;
NDArray.prototype.allclose = modules.operators.allclose;

// ==============================
//    array instantiation and reshaping
// ==============================

NDArray.prototype.JS = function () {
  return modules.jsInterface.toJS(this);
}
NDArray.prototype.fromJS = function (A) {
  return modules.jsInterface.fromJS(A);
}

// ==============================
//    elementwise methods
// ==============================

NDArray.prototype.round = function (decimals = 0) {
  ({ decimals } = Object.assign({ decimals }, this.__popKwArgs()));
  return modules.elementwise.round(this, decimals);
};

// ==============================
//    transform methods
// ==============================

/** @param {null|number[]} axes */
NDArray.prototype.transpose = function (axes = null) {
  ({ axes } = Object.assign({ axes }, this.__popKwArgs()));
  return modules.transform.transpose(this, axes);
};

NDArray.prototype.sort = function (axis = -1) {
  ({ axis } = Object.assign({ axis }, this.__popKwArgs()));
  modules.transform.sort(this, axis);
  return null;
};


//=============================


module.exports = NDArray;