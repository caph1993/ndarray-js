//@ts-check
export type DType = NumberConstructor | BooleanConstructor;
export type ArrayOrConstant = NDArray | number | boolean;


class NDArray {
  _flat: number[];
  shape: number[];
  dtype: DType;
  modules: typeof import("./core-modules").modules;
  reshape: (shape: any, ...more_shape: any[]) => any;
  index: (...where: any[]) => any;
  sum: (axis?:
    any, keepdims?: boolean) => any;
  any: (axis?: any, keepdims?: boolean) => any;
  all: (axis?:
    any, keepdims?: boolean) => any;
  max: (axis?: any, keepdims?: boolean) => any;
  product: (axis?:
    any, keepdims?: boolean) => any;
  min: (axis?: any, keepdims?: boolean) => any;
  argmax: (axis?: any, keepdims?: boolean) => any;
  argmin: (axis?: any, keepdims?: boolean) => any;
  mean: (axis?: any, keepdims?: boolean) => any;
  var: (axis?: any, keepdims?: boolean) => any;
  std: (axis?: any, keepdims?: boolean) => any;
  add: any;
  subtract: any;
  multiply: any;
  divide: any;
  mod: any;
  divide_int: any;
  pow: any;
  maximum: any;
  _binary_operation: (A: number | boolean | NDArray, B: number | boolean |
    NDArray, func: any, dtype: any, out?: NDArray) => number | boolean | NDArray;
  bitwise_or: any;
  bitwise_and: any;
  bitwise_shift_right: any;
  logical_xor: any;
  logical_or: any;
  minimum: any;
  logical_and: any;
  greater: any;
  less: any;
  greater_equal: any;
  less_equal: any;
  equal: any;
  not_equal: any;
  isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray;
  allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;
  assign: any;
  add_assign: any;
  subtract_assign: any;
  multiply_assign: any;
  divide_assign: any;
  mod_assign: any;
  pow_assign: any;
  divide_int_assign: any;
  maximum_assign: any;
  minimum_assign: any;
  bitwise_and_assign: any;
  bitwise_or_assign: any;
  logical_or_assign: any;
  bitwise_shift_right_assign: any;
  bitwise_shift_left_assign: any;
  logical_and_assign: any;
  JS: () => any;
  fromJS: (A: any) => any;
  round: (decimals?: number) => any;
  sort: (axis?: number) => any;
  transpose: (axes?: number[]) => any;
  op: (...args: any[]) => NDArray;
  ravel: () => any;

  constructor(flat: number[], shape: number[], dtype: any = Number) {
    this.shape = shape; // invariant: immutable
    this._flat = flat;
    this.dtype = dtype;
    this._simpleIndexes = null;
  }

  _simpleIndexes: import("./core-modules/core-indexes").AxesIndex | null;

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
  withKwArgs(kwArgs: { [s: string]: any; }) {
    this['__kwArgs'] = kwArgs;
    return this;
  }
  *[Symbol.iterator]() {
    for (let i = 0; i < this.shape[0]; i++) yield this.index(i);
  }
  get length() {
    return this.shape[0] || 0;
  }
  copy: () => NDArray;
}

import { GLOBALS } from './globals';
GLOBALS.NDArray = NDArray;

import { modules } from "./core-modules";
NDArray.prototype.modules = modules;





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


// ==============================
//    Indexing
// ==============================


NDArray.prototype.index = function (...where: import("./core-modules/core-indexes").GeneralIndexSpec[]) {
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
  return function (axis: import("./core-modules/core-reduce").AxisArg = null, keepdims: boolean = false) {
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
function binaryOpDecorator(func: import("./core-modules/core-operators").BinaryOperator): import("./core-modules/core-operators").SelfBinaryOperator {
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
NDArray.prototype.bitwise_or = binaryOpDecorator(modules.operators.op_binary["^"]);
NDArray.prototype.bitwise_shift_right = binaryOpDecorator(modules.operators.op_binary["<<"]);
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



function unaryOpDecorator(func: import("./core-modules/core-operators").UnaryOperator): import("./core-modules/core-operators").SelfUnaryOperator {
  return function (out = null) {
    ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, out);
  }
}
// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_or = unaryOpDecorator(modules.operators.op_unary["~"]);
NDArray.prototype.logical_or = unaryOpDecorator(modules.operators.op_unary["not"]);


NDArray.prototype.isclose = modules.operators.isclose;
NDArray.prototype.allclose = modules.operators.allclose;


function assignOpDecorator(func: import("./core-modules/core-operators").AssignmentOperator): import("./core-modules/core-operators").SelfAssignmentOperator {
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
NDArray.prototype.transpose = function (axes: null | number[] = null) {
  ({ axes } = Object.assign({ axes }, this.__popKwArgs()));
  return modules.transform.transpose(this, axes);
};

NDArray.prototype.sort = function (axis = -1) {
  ({ axis } = Object.assign({ axis }, this.__popKwArgs()));
  modules.transform.sort(this, axis);
  return null;
};


//=============================





/** @typedef {"+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min"} BinaryOpSymbol */
/** @typedef {"=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and="} AssignmentOpSymbol */
/** @typedef {"~" | "not" | "-"} UnaryOpSymbol */
/** @typedef {import("./core-modules/core-indexes").Where} Where */

/**
 * @type {{():NDArray; (where:Where):ArrayOrConstant; (where:Where, op:AssignmentOpSymbol, B:ArrayOrConstant):NDArray; (op:AssignmentOpSymbol, B:ArrayOrConstant):NDArray; ( op:BinaryOpSymbol, B:ArrayOrConstant):NDArray; (UnaryOpSymbol):NDArray; }}  */
NDArray.prototype.op = function (...args): NDArray {
  if (!args.length) return this;
  if (typeof args[0] == "string") {
    const symbol = args[0];
    if (args.length == 1) {
      let func = modules.operators.op_unary[symbol];
      if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.operators.op_unary)]}`);
      return func(this, symbol);
    }
    if (args.length > 2) throw new Error(`Too many arguments provided: ${[...args]}`);
    const other = args[1];
    let func = modules.operators.op_binary[symbol];
    if (func) return func(this, other);
    func = modules.operators.op_assign[symbol];
    if (func) return func(this, other);
    if (symbol.includes(':')) throw new Error(`Expected index or operator symbol. Found "${symbol}". Did you mean ${[symbol]}?`);
    throw new Error(`Expected index or operator symbol. Found "${symbol}"`);
  }
  const where = args[0];
  if (where instanceof NDArray) throw new Error(`Expected operator or index. Found numpy array`);
  if (args.length == 1) return this.index(where);
  const symbol = args[1];
  let func = modules.operators.op_assign[symbol];
  if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.operators.op_unary)]}`);
  if (args.length > 3) throw new Error(`Too many arguments provided: ${[...args]}`);
  const other = args[2];
  return func(this, where, other);
}





export default NDArray;
