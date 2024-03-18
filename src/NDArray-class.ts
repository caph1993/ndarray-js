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



NDArray.prototype.any = modules.reduce.self_reducers.any;
NDArray.prototype.all = modules.reduce.self_reducers.all;

NDArray.prototype.sum = modules.reduce.self_reducers.sum;
NDArray.prototype.product = modules.reduce.self_reducers.product;
NDArray.prototype.max = modules.reduce.self_reducers.max;
NDArray.prototype.min = modules.reduce.self_reducers.min;
NDArray.prototype.argmax = modules.reduce.self_reducers.argmax;
NDArray.prototype.argmin = modules.reduce.self_reducers.argmin;
NDArray.prototype.mean = modules.reduce.self_reducers.mean;

NDArray.prototype.var = modules.reduce.self_reducers.var;
NDArray.prototype.std = modules.reduce.self_reducers.std;


// function reduceDecorator(func) {
//   return function (axis: import("./NDArray/reduce").AxisArg = null, keepdims: boolean = false) {
//     ({ axis, keepdims } = Object.assign({ axis, keepdims }, this.__popKwArgs()));
//     return func(this, axis, keepdims);
//   }
// }
// reducersExtra
// NDArray.prototype.var = reduceDecorator(modules.reduce.self_reducers.var);

// NDArray.prototype.norm = function (axis: import("./NDArray/reduce").AxisArg = null, keepdims: boolean = false, ord = 2) {
//   ({ axis, keepdims, ord } = Object.assign({ axis, keepdims, ord }, this.__popKwArgs()));
//   return modules.reduce.reducers.norm(this, axis, keepdims, ord);
// } as any;
// NDArray.prototype.std = function (axis: import("./NDArray/reduce").AxisArg = null, keepdims: boolean = false, ddof = 0) {
//   ({ axis, keepdims, ddof } = Object.assign({ axis, keepdims, ddof }, this.__popKwArgs()));
//   return modules.reduce.reducers.std(this, axis, keepdims, ddof);
// }



// ==============================
//       Operators: Binary operations, assignment operations and unary boolean_not
// ==============================


function binaryOpDecorator(func: import("./NDArray/operators").BinaryOperator): import("./NDArray/operators").SelfBinaryOperator {
  return function (other, out = null) {
    // ({ out } = Object.assign({ out }, this.__popKwArgs()));
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
