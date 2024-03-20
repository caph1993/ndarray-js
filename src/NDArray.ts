//@ts-check

/** @ignore */
export type DType = NumberConstructor | BooleanConstructor;
/** @ignore */
export type ArrayOrConstant = NDArray | number | boolean;


/**
 * Multi dimensional array.
 */
class NDArray {

  /** @ignore */
  _flat: number[];

  /** @category Attributes @readonly */
  shape: number[];
  /** @category Attributes @readonly */
  dtype: DType;

  /** @category Indexing / slicing */
  index: (...where: Where) => number | NDArray;

  /** @ignore */
  modules: typeof import("./array").modules;

  /** @category Reducers */
  any: ReduceSignatureBool;
  /** @category Reducers */
  all: ReduceSignatureBool;
  /** @category Reducers */
  sum: ReduceSignature;
  /** @category Reducers */
  product: ReduceSignature;
  /** @category Reducers */
  max: ReduceSignature;
  /** @category Reducers */
  min: ReduceSignature;
  /** @category Reducers */
  argmax: ReduceSignature;
  /** @category Reducers */
  argmin: ReduceSignature;
  /** @category Reducers */
  mean: ReduceSignature;

  /** @category Reducers */
  var: ReduceSignature;
  /** @category Reducers */
  std: ReduceStdSignature;
  /** @category Reducers */
  norm: ReduceNormSignature;

  /** @category Binary operators */
  add: BinaryOperatorMethod;
  /** @category Binary operators */
  subtract: BinaryOperatorMethod;
  /** @category Binary operators */
  multiply: BinaryOperatorMethod;
  /** @category Binary operators */
  divide: BinaryOperatorMethod;
  /** @category Binary operators */
  mod: BinaryOperatorMethod;
  /** @category Binary operators */
  divide_int: BinaryOperatorMethod;
  /** @category Binary operators */
  pow: BinaryOperatorMethod;
  /** @category Binary operators */
  maximum: BinaryOperatorMethod;
  /** @category Binary operators */
  minimum: BinaryOperatorMethod;
  /** @category Binary operators */
  bitwise_or: BinaryOperatorMethod;
  /** @category Binary operators */
  bitwise_and: BinaryOperatorMethod;
  /** @category Binary operators */
  bitwise_shift_right: BinaryOperatorMethod;

  /** @category Binary logical operators */
  logical_xor: BinaryOperatorMethod<boolean>;
  /** @category Binary logical operators */
  logical_or: BinaryOperatorMethod<boolean>;
  /** @category Binary logical operators */
  logical_and: BinaryOperatorMethod<boolean>;

  /** @category Comparison operators */
  greater: BinaryOperatorMethod;
  /** @category Comparison operators */
  less: BinaryOperatorMethod;
  /** @category Comparison operators */
  greater_equal: BinaryOperatorMethod;
  /** @category Comparison operators */
  less_equal: BinaryOperatorMethod;
  /** @category Comparison operators */
  equal: BinaryOperatorMethod;
  /** @category Comparison operators */
  not_equal: BinaryOperatorMethod;
  /** @category Comparison operators */
  isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray;
  /** @category Comparison operators */
  allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;

  /** @category Unary operators */
  round: RoundSignature;
  /** @category Unary operators */
  abs: UnaryOperatorMethod;
  /** @category Unary operators */
  negative: UnaryOperatorMethod;
  /** @category Unary operators */
  bitwise_not: UnaryOperatorMethod;
  /** @category Unary logical operators */
  logical_not: UnaryOperatorMethod;

  /** @category Assignment operators */
  assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  add_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  subtract_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  multiply_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  divide_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  mod_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  pow_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  divide_int_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  maximum_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  minimum_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  bitwise_and_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  bitwise_or_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  logical_or_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  bitwise_shift_right_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  bitwise_shift_left_assign: SelfAssignmentOperator;
  /** @category Assignment operators */
  logical_and_assign: SelfAssignmentOperator;

  /** @category Transformations */
  ravel: () => NDArray;
  /** @category Transformations */
  reshape: (shape: any, ...more_shape: any[]) => any;
  /** @category Transformations */
  sort: (axis?: number) => NDArray;
  /** @category Transformations */
  transpose: (axes?: number[]) => NDArray;


  /** @category Casting */
  tolist: () => any;
  // fromJS: (A: any) => NDArray;

  /**
   * Generic operator function. See {@link GenericOperatorFunction} for details.
   */
  op: GenericOperatorFunction;

  constructor(flat: number[], shape: number[], dtype: any = Number) {
    this.shape = shape; // invariant: immutable
    this._flat = flat;
    this.dtype = dtype;
    this._simpleIndexes = null;
  }

  /** @ignore */
  _simpleIndexes: import("./array/indexes").AxesIndex | null;

  /** @category Attributes @readonly */
  get size() {
    return this._simpleIndexes == null ? this._flat.length : this._simpleIndexes.size;
  }
  get flat() {
    if (this._simpleIndexes == null) return this._flat;
    const indices = this._simpleIndexes.indices;
    return indices.map(i => this._flat[i]);
  }
  /** @category Attributes @readonly */
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

  /**
   * @category Transformations
   * Transpose.
  */
  get T() {
    return this.transpose();
  }
  /**
   * Iterator over the first axis.
   * For 1D arrays, yields numbers.
   * For multidimensional arrays, yields array views.
   */
  *[Symbol.iterator]() {
    for (let i = 0; i < this.shape[0]; i++) yield this.index(i);
  }

  /** @category Attributes @readonly */
  get length() {
    return this.shape[0] || 0;
  }

  /** @category Transformations */
  copy: () => NDArray;
  /** @category Casting */
  item() {
    if (this.size != 1) throw new Error(`Can't convert array of size ${this.size} to scalar`);
    return this._flat[0];
  }
}

import { GLOBALS } from './_globals';
GLOBALS.NDArray = NDArray;

import { modules } from "./array";
import { SelfAssignmentOperator } from './array/operators';
// import { AxisArg, ReduceKwArgs } from './NDArray/reduce';
import { AxisArg, BinaryOperatorMethod, KwParser, ReduceKwargs, ReduceNormSignature, ReduceSignature, ReduceSignatureBool, ReduceStdSignature, RoundKwargs, RoundParsedKwargs, RoundSignature, UnaryOperatorMethod } from './array/kwargs';
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


NDArray.prototype.index = function (...where: import("./array/indexes").GeneralIndexSpec[]) {
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


function binaryOpDecorator(func: import("./array/operators").BinaryOperator): import("./array/operators").SelfBinaryOperator {
  return function (other, out = null) {
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


// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = modules.elementwise.kw_ops.bitwise_not.as_method;
NDArray.prototype.logical_not = modules.elementwise.kw_ops.logical_not.as_method;
NDArray.prototype.negative = modules.elementwise.kw_ops.negative.as_method;
NDArray.prototype.abs = modules.elementwise.kw_ops.abs.as_method;

NDArray.prototype.isclose = modules.operators.isclose;
NDArray.prototype.allclose = modules.operators.allclose;


function assignOpDecorator(func: import("./array/operators").AssignmentOperator): import("./array/operators").SelfAssignmentOperator {
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

NDArray.prototype.round = modules.elementwise.kw_ops.round.as_method;

// ==============================
//    transform methods
// ==============================

/** @param {null|number[]} axes */
NDArray.prototype.transpose = function (axes: null | number[] = null) {
  return modules.transform.transpose(this, axes);
};

NDArray.prototype.sort = function (axis = -1) {
  modules.transform.sort(this, axis);
  return null;
};


//=============================



type BinaryOpSymbol = "+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min";
type AssignmentOpSymbol = "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and=";
type UnaryOpSymbol = "~" | "not" | "-";
type Where = import("./array/indexes").Where;



/**
 * Generic function to apply an operation to an array.
 * It can be used to apply unary, binary or assignment operations.
 * 
 * @Example
 * ```javascript
 * const A = np.arange(10).op("+", 1);
 * A.index('::2').op("=", 0);
 * console.log(A.tolist());
 * ```
 */
export type GenericOperatorFunction = { (): NDArray; (where: Where): ArrayOrConstant; (where: Where, op: AssignmentOpSymbol, B: ArrayOrConstant): NDArray; (op: AssignmentOpSymbol, B: ArrayOrConstant): NDArray; (op: BinaryOpSymbol, B: ArrayOrConstant): NDArray; (UnaryOpSymbol): NDArray; };

const op: GenericOperatorFunction = function (...args): NDArray {
  if (!args.length) return this;
  if (typeof args[0] == "string") {
    const symbol = args[0];
    if (args.length == 1) {
      let func = modules.elementwise.ops[symbol];
      if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.elementwise.ops)]}`);
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
  if (!func) throw new Error(`Unknown assign operator "${symbol}". Options:${[...Object.keys(modules.operators.op_assign)]}`);
  if (args.length > 3) throw new Error(`Too many arguments provided: ${[...args]}`);
  const other = args[2];
  return func(this, where, other);
}

NDArray.prototype.op = op;


export { NDArray };

export default NDArray;
