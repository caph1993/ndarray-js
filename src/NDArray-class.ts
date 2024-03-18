//@ts-check
export type DType = NumberConstructor | BooleanConstructor;
export type ArrayOrConstant = NDArray | number | boolean;


class NDArray {
  _flat: number[];
  shape: number[];
  dtype: DType;

  index: (...where: any[]) => any;
  reshape: (shape: any, ...more_shape: any[]) => any;

  modules: typeof import("./NDArray").modules;

  any: ReduceSignatureBool;
  all: ReduceSignatureBool;
  sum: ReduceSignature;
  product: ReduceSignature;
  max: ReduceSignature;
  min: ReduceSignature;
  argmax: ReduceSignature;
  argmin: ReduceSignature;
  mean: ReduceSignature;

  var: ReduceSignature;
  std: ReduceStdSignature;
  norm: ReduceNormSignature;

  add: SelfBinaryOperator;
  subtract: SelfBinaryOperator;
  multiply: SelfBinaryOperator;
  divide: SelfBinaryOperator;
  mod: SelfBinaryOperator;
  divide_int: SelfBinaryOperator;
  pow: SelfBinaryOperator;
  maximum: SelfBinaryOperator;
  minimum: SelfBinaryOperator;
  bitwise_or: SelfBinaryOperator;
  bitwise_and: SelfBinaryOperator;
  bitwise_shift_right: SelfBinaryOperator;
  logical_xor: SelfBinaryOperator;
  logical_or: SelfBinaryOperator;

  logical_and: SelfBinaryOperator;
  greater: SelfBinaryOperator;
  less: SelfBinaryOperator;
  greater_equal: SelfBinaryOperator;
  less_equal: SelfBinaryOperator;
  equal: SelfBinaryOperator;
  not_equal: SelfBinaryOperator;
  isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray;
  allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;

  abs: SelfUnaryOperator;

  assign: SelfAssignmentOperator;
  add_assign: SelfAssignmentOperator;
  subtract_assign: SelfAssignmentOperator;
  multiply_assign: SelfAssignmentOperator;
  divide_assign: SelfAssignmentOperator;
  mod_assign: SelfAssignmentOperator;
  pow_assign: SelfAssignmentOperator;
  divide_int_assign: SelfAssignmentOperator;
  maximum_assign: SelfAssignmentOperator;
  minimum_assign: SelfAssignmentOperator;
  bitwise_and_assign: SelfAssignmentOperator;
  bitwise_or_assign: SelfAssignmentOperator;
  logical_or_assign: SelfAssignmentOperator;
  bitwise_shift_right_assign: SelfAssignmentOperator;
  bitwise_shift_left_assign: SelfAssignmentOperator;
  logical_and_assign: SelfAssignmentOperator;

  tolist: () => any;
  // fromJS: (A: any) => NDArray;
  round: RoundSignature;
  sort: (axis?: number) => NDArray;
  transpose: (axes?: number[]) => NDArray;
  op: (...args: any[]) => NDArray;
  ravel: () => NDArray;

  constructor(flat: number[], shape: number[], dtype: any = Number) {
    this.shape = shape; // invariant: immutable
    this._flat = flat;
    this.dtype = dtype;
    this._simpleIndexes = null;
  }

  _simpleIndexes: import("./NDArray/indexes").AxesIndex | null;

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
  // __popKwArgs() {
  //   let out = this['__kwArgs'];
  //   if (out === undefined) return {};
  //   delete this['__kwArgs'];
  //   return out;
  // }
  // /** @param {Object<string, any>} kwArgs */
  // withKwArgs(kwArgs: { [s: string]: any; }) {
  //   this['__kwArgs'] = kwArgs;
  //   return this;
  // }
  *[Symbol.iterator]() {
    for (let i = 0; i < this.shape[0]; i++) yield this.index(i);
  }
  get length() {
    return this.shape[0] || 0;
  }
  copy: () => NDArray;
  item() {
    if (this.size != 1) throw new Error(`Can't convert array of size ${this.size} to scalar`);
    return this._flat[0];
  }
}

import { GLOBALS } from './_globals';
GLOBALS.NDArray = NDArray;

import { modules } from "./NDArray";
import { SelfAssignmentOperator, SelfBinaryOperator, SelfUnaryOperator } from './NDArray/operators';
// import { AxisArg, ReduceKwArgs } from './NDArray/reduce';
import { AxisArg, KwParser, ReduceKwargs, ReduceNormSignature, ReduceSignature, ReduceSignatureBool, ReduceStdSignature, RoundKwargs, RoundParsedKwargs, RoundSignature } from './NDArray/kwargs';
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


NDArray.prototype.index = function (...where: import("./NDArray/indexes").GeneralIndexSpec[]) {
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


NDArray.prototype.any = modules.reduce.kw_reducers.any.as_method;
NDArray.prototype.all = modules.reduce.kw_reducers.all.as_method;

NDArray.prototype.sum = modules.reduce.kw_reducers.sum.as_method;
NDArray.prototype.product = modules.reduce.kw_reducers.product.as_method;
NDArray.prototype.max = modules.reduce.kw_reducers.max.as_method;
NDArray.prototype.min = modules.reduce.kw_reducers.min.as_method;
NDArray.prototype.argmax = modules.reduce.kw_reducers.argmax.as_method;
NDArray.prototype.argmin = modules.reduce.kw_reducers.argmin.as_method;
NDArray.prototype.mean = modules.reduce.kw_reducers.mean.as_method;

NDArray.prototype.var = modules.reduce.kw_reducers.var.as_method;
NDArray.prototype.std = modules.reduce.kw_reducers.std.as_method;
NDArray.prototype.norm = modules.reduce.kw_reducers.norm.as_method;


// ==============================
//       Operators: Binary operations, assignment operations and unary boolean_not
// ==============================


function binaryOpDecorator(func: import("./NDArray/operators").BinaryOperator): import("./NDArray/operators").SelfBinaryOperator {
  return function (other, out = null) {
    // ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, other, out);
  }
}

NDArray.prototype.add = modules.operators.kw_op_binary["+"].as_method;
NDArray.prototype.subtract = modules.operators.kw_op_binary["-"].as_method;
NDArray.prototype.multiply = modules.operators.kw_op_binary["*"].as_method;
NDArray.prototype.divide = modules.operators.kw_op_binary["/"].as_method;
NDArray.prototype.mod = modules.operators.kw_op_binary["%"].as_method;
NDArray.prototype.divide_int = modules.operators.kw_op_binary["//"].as_method;
NDArray.prototype.pow = modules.operators.kw_op_binary["**"].as_method;

NDArray.prototype.maximum = modules.operators.kw_op_binary["max"].as_method;
NDArray.prototype.minimum = modules.operators.kw_op_binary["min"].as_method;

NDArray.prototype.bitwise_or = modules.operators.kw_op_binary["|"].as_method;
NDArray.prototype.bitwise_and = modules.operators.kw_op_binary["&"].as_method;
NDArray.prototype.bitwise_or = modules.operators.kw_op_binary["^"].as_method;
NDArray.prototype.bitwise_shift_right = modules.operators.kw_op_binary["<<"].as_method;
NDArray.prototype.bitwise_shift_right = modules.operators.kw_op_binary[">>"].as_method;

NDArray.prototype.logical_or = modules.operators.kw_op_binary["or"].as_method;
NDArray.prototype.logical_and = modules.operators.kw_op_binary["and"].as_method;
NDArray.prototype.logical_xor = modules.operators.kw_op_binary["xor"].as_method;

NDArray.prototype.greater = modules.operators.kw_op_binary[">"].as_method;
NDArray.prototype.less = modules.operators.kw_op_binary["<"].as_method;
NDArray.prototype.greater_equal = modules.operators.kw_op_binary[">="].as_method;
NDArray.prototype.less_equal = modules.operators.kw_op_binary["<="].as_method;
NDArray.prototype.equal = modules.operators.kw_op_binary["=="].as_method;
NDArray.prototype.not_equal = modules.operators.kw_op_binary["!="].as_method;



function unaryOpDecorator(func: import("./NDArray/operators").UnaryOperator): import("./NDArray/operators").SelfUnaryOperator {
  return function (out = null) {
    ({ out } = Object.assign({ out }, this.__popKwArgs()));
    return func(this, out);
  }
}
// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_or = unaryOpDecorator(modules.operators.op_unary["~"]);
NDArray.prototype.logical_or = unaryOpDecorator(modules.operators.op_unary["not"]);
NDArray.prototype.abs = unaryOpDecorator(modules.operators.op_unary["abs"]);



NDArray.prototype.isclose = modules.operators.isclose;
NDArray.prototype.allclose = modules.operators.allclose;


function assignOpDecorator(func: import("./NDArray/operators").AssignmentOperator): import("./NDArray/operators").SelfAssignmentOperator {
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

NDArray.prototype.tolist = function () {
  return modules.jsInterface.tolist(this);
}
// NDArray.prototype.fromJS = function (A) {
//   return modules.jsInterface.fromJS(A);
// }

// ==============================
//    elementwise methods
// ==============================

NDArray.prototype.round = modules.elementwise.round_kw.as_method;

// ==============================
//    transform methods
// ==============================

/** @param {null|number[]} axes */
NDArray.prototype.transpose = function (axes: null | number[] = null) {
  // ({ axes } = Object.assign({ axes }, this.__popKwArgs()));
  return modules.transform.transpose(this, axes);
};

NDArray.prototype.sort = function (axis = -1) {
  // ({ axis } = Object.assign({ axis }, this.__popKwArgs()));
  modules.transform.sort(this, axis);
  return null;
};


//=============================



type BinaryOpSymbol = "+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min";
type AssignmentOpSymbol = "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and=";
type UnaryOpSymbol = "~" | "not" | "-";
type Where = import("./NDArray/indexes").Where;

type Op = { (): NDArray; (where: Where): ArrayOrConstant; (where: Where, op: AssignmentOpSymbol, B: ArrayOrConstant): NDArray; (op: AssignmentOpSymbol, B: ArrayOrConstant): NDArray; (op: BinaryOpSymbol, B: ArrayOrConstant): NDArray; (UnaryOpSymbol): NDArray; };

const op: Op = function (...args): NDArray {
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
NDArray.prototype.op = op;


export { NDArray };

export default NDArray;
