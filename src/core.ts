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
 * @param {import("./core-indexes").GeneralIndexSpec[]} where
 */
NDArray.prototype.index = function (...where) {
  return modules.indexes.index(this, where);
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


/**
 * @param {import("./core-operators").BinaryOperator} func
 * @returns {import("./core-operators").SelfBinaryOperator}
 */
function binaryOpDecorator(func) {
  return function (other, out = null) {
    ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, other, out);
  }
}

NDArray.prototype.add = binaryOpDecorator(modules.operators.op_binary["+"]);
NDArray.prototype.subtract = binaryOpDecorator(modules.operators.op_binary["-"]);
NDArray.prototype.multiply = binaryOpDecorator(modules.operators.op_binary["*"]);
NDArray.prototype.divide = binaryOpDecorator(modules.operators.op_binary["/"]);
NDArray.prototype.mod = binaryOpDecorator(modules.operators.op_binary["%"]);
NDArray.prototype.divide_int = binaryOpDecorator(modules.operators.op_binary["//"]);
NDArray.prototype.pow = binaryOpDecorator(modules.operators.op_binary["**"]);

NDArray.prototype.maximum = binaryOpDecorator(modules.operators.op_binary["↑"]);
NDArray.prototype.minimum = binaryOpDecorator(modules.operators.op_binary["↓"]);

NDArray.prototype.bitwise_or = binaryOpDecorator(modules.operators.op_binary["|"]);
NDArray.prototype.bitwise_and = binaryOpDecorator(modules.operators.op_binary["&"]);
NDArray.prototype.bitwise_xor = binaryOpDecorator(modules.operators.op_binary["^"]);
NDArray.prototype.bitwise_shift_left = binaryOpDecorator(modules.operators.op_binary["<<"]);
NDArray.prototype.bitwise_shift_right = binaryOpDecorator(modules.operators.op_binary[">>"]);

NDArray.prototype.logical_or = binaryOpDecorator(modules.operators.op_binary["or"]);
NDArray.prototype.logical_and = binaryOpDecorator(modules.operators.op_binary["and"]);
NDArray.prototype.logical_xor = binaryOpDecorator(modules.operators.op_binary["xor"]);

NDArray.prototype.greater = binaryOpDecorator(modules.operators.op_binary[">"]);
NDArray.prototype.less = binaryOpDecorator(modules.operators.op_binary["<"]);
NDArray.prototype.greater_equal = binaryOpDecorator(modules.operators.op_binary[">="]);
NDArray.prototype.less_equal = binaryOpDecorator(modules.operators.op_binary["<="]);
NDArray.prototype.equal = binaryOpDecorator(modules.operators.op_binary["=="]);
NDArray.prototype.not_equal = binaryOpDecorator(modules.operators.op_binary["!="]);



/**
 * @param {import("./core-operators").UnaryOperator} func
 * @returns {import("./core-operators").SelfUnaryOperator}
 */
function unaryOpDecorator(func) {
  return function (out = null) {
    ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, out);
  }
}
// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = unaryOpDecorator(modules.operators.op_unary["~"]);
NDArray.prototype.logical_not = unaryOpDecorator(modules.operators.op_unary["not"]);


NDArray.prototype.isclose = modules.operators.isclose;
NDArray.prototype.allclose = modules.operators.allclose;


/**
 * @param {import("./core-operators").AssignmentOperator} func
 * @returns {import("./core-operators").SelfAssignmentOperator}
 */
function assignOpDecorator(func) {
  //@ts-ignore
  return function (...args) { return func(this, ...args); }
}
NDArray.prototype.assign = assignOpDecorator(modules.operators.op_assign["="]);
NDArray.prototype.add_assign = assignOpDecorator(modules.operators.op_assign["+="]);
NDArray.prototype.subtract_assign = assignOpDecorator(modules.operators.op_assign["-="]);
NDArray.prototype.multiply_assign = assignOpDecorator(modules.operators.op_assign["*="]);
NDArray.prototype.divide_assign = assignOpDecorator(modules.operators.op_assign["/="]);
NDArray.prototype.mod_assign = assignOpDecorator(modules.operators.op_assign["%="]);
NDArray.prototype.divide_int_assign = assignOpDecorator(modules.operators.op_assign["//="]);
NDArray.prototype.pow_assign = assignOpDecorator(modules.operators.op_assign["**="]);

NDArray.prototype.maximum_assign = assignOpDecorator(modules.operators.op_assign["↑="]);
NDArray.prototype.minimum_assign = assignOpDecorator(modules.operators.op_assign["↓="]);

NDArray.prototype.bitwise_or_assign = assignOpDecorator(modules.operators.op_assign["|="]);
NDArray.prototype.bitwise_and_assign = assignOpDecorator(modules.operators.op_assign["&="]);
NDArray.prototype.bitwise_shift_left_assign = assignOpDecorator(modules.operators.op_assign["<<="]);
NDArray.prototype.bitwise_shift_right_assign = assignOpDecorator(modules.operators.op_assign[">>="]);

NDArray.prototype.logical_or_assign = assignOpDecorator(modules.operators.op_assign["or="]);
NDArray.prototype.logical_and_assign = assignOpDecorator(modules.operators.op_assign["and="]);



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





// /** @typedef {'+'|'-'|'*'} BinaryOperatorSymbol */
// /** @typedef {'~'|'!'} UnaryOperatorSymbol */
// /** @typedef {'='|'+='|'*='} AssignmentOperatorSymbol */

// /**
//  * @callback BinaryOp
//  * @param  {BinaryOperatorSymbol} symbol
//  * @param  {NDArray} other
//  * @param  {NDArray?} out
//  * @returns {NDArray}
//  */
// /**
//  * @callback AssignmentOp
//  * @param  {AssignmentOperatorSymbol} symbol
//  * @param  {NDArray} other
//  * @param  {...import("./core-indexes").GeneralIndexSpec} indexSpec
//  * @returns {NDArray}
//  */
// /**
//  * @callback UnaryOp
//  * @param  {UnaryOperatorSymbol} symbol
//  * @returns {NDArray}
//  */

// // /**@type {BinaryOp|AssignmentOp}*/

// /**
//  * @overload
//  * @param  {BinaryOperatorSymbol} symbol
//  * @param  {NDArray} other
//  * @param  {NDArray?} out
//  * @returns {NDArray}
// */
// /**
//  * @overload
//  * @param  {AssignmentOperatorSymbol} symbol
//  * @param  {NDArray} other
//  * @param  {...import("./core-indexes").GeneralIndexSpec} indexSpec
//  * @returns {NDArray}
// */
// /**
//  * @param  {UnaryOperatorSymbol} symbol
//  * @returns {NDArray}
// */
// NDArray.prototype.op = function (...arguments) {

//   return this;
// }




/** @typedef {"+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min"} BinaryOpSymbol */
/** @typedef {"=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and="} AssignmentOpSymbol */
/** @typedef {"~" | "not" | "-"} UnaryOpSymbol */
/** @typedef {import("./core-indexes").Where} Where */

/**
 * @type {{():NDArray; (where:Where):ArrayOrConstant; (where:Where, op:AssignmentOpSymbol, B:ArrayOrConstant):NDArray; (op:AssignmentOpSymbol, B:ArrayOrConstant):NDArray; ( op:BinaryOpSymbol, B:ArrayOrConstant):NDArray; (UnaryOpSymbol):NDArray; }}  */
NDArray.prototype.op = function (...arguments) {
  if (!arguments.length) return this;
  if (typeof arguments[0] == "string") {
    const symbol = arguments[0];
    if (arguments.length == 1) {
      let func = modules.operators.op_unary[symbol];
      if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.operators.op_unary)]}`);
      return func(this, symbol);
    }
    if (arguments.length > 2) throw new Error(`Too many arguments provided: ${[...arguments]}`);
    const other = arguments[1];
    let func = modules.operators.op_binary[symbol];
    if (func) return func(this, other);
    func = modules.operators.op_assign[symbol];
    if (func) return func(this, other);
    if (symbol.includes(':')) throw new Error(`Expected index or operator symbol. Found "${symbol}". Did you mean ${[symbol]}?`);
    throw new Error(`Expected index or operator symbol. Found "${symbol}"`);
  }
  const where = arguments[0];
  if (where instanceof NDArray) throw new Error(`Expected operator or index. Found numpy array`);
  //@ts-ignore
  if (arguments.length == 1) return this.index(where);
  const symbol = arguments[1];
  let func = modules.operators.op_assign[symbol];
  if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.operators.op_unary)]}`);
  if (arguments.length > 2) throw new Error(`Too many arguments provided: ${[...arguments]}`);
  const other = arguments[2];
  return func(this, where, other);
}





module.exports = NDArray;