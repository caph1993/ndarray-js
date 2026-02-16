//@ts-check

import { DType, HasDType, bool, infer_dtype, new_buffer, object } from './dtypes';
/** @ignore */
export type ArrayOrConstant = NDArray | number | boolean;





/**
 * Multi dimensional array.
 */
class NDArray implements HasDType {


  /** @ignore */
  _flat: InstanceType<DType["BufferType"]>;

  /** @category Attributes @readonly */
  shape: number[];

  /** @ignore */
  _dtype?: DType;

  /** @category Attributes @readonly */
  get dtype(): DType {
    if (this._dtype) return this._dtype;
    return infer_dtype[this._flat.constructor.name] || object;
  }

  /** @category Indexing / slicing */
  index: (...where: Where) => NDArray;

  /** @ignore */
  modules: typeof import("./array").modules;

  /** @category Reducers */
  any: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  all: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  sum: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  product: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  max: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  min: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  argmax: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  argmin: Method_a_axis_keepdims.Wrapper;

  /** @category Reducers */
  mean: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  var: Method_a_axis_ddof_keepdims.Wrapper;
  /** @category Reducers */
  std: Method_a_axis_ddof_keepdims.Wrapper;
  /** @category Reducers */
  norm: Method_a_ord_axis_keepdims.Wrapper;

  /** @category Binary operators */
  add: Method_other_out.Wrapper;
  /** @category Binary operators */
  subtract: Method_other_out.Wrapper;
  /** @category Binary operators */
  multiply: Method_other_out.Wrapper;
  /** @category Binary operators */
  divide: Method_other_out.Wrapper;
  /** @category Binary operators */
  mod: Method_other_out.Wrapper;
  /** @category Binary operators */
  divide_int: Method_other_out.Wrapper;
  /** @category Binary operators */
  pow: Method_other_out.Wrapper;
  /** @category Binary operators */
  maximum: Method_other_out.Wrapper;
  /** @category Binary operators */
  minimum: Method_other_out.Wrapper;
  /** @category Binary operators */
  bitwise_or: Method_other_out.Wrapper;
  /** @category Binary operators */
  bitwise_and: Method_other_out.Wrapper;
  /** @category Binary operators */
  bitwise_shift_right: Method_other_out.Wrapper;

  /** @category Binary logical operators */
  logical_xor: Method_other_out.Wrapper;
  /** @category Binary logical operators */
  logical_or: Method_other_out.Wrapper;
  /** @category Binary logical operators */
  logical_and: Method_other_out.Wrapper;

  /** @category Comparison operators */
  greater: Method_other_out.Wrapper;
  /** @category Comparison operators */
  less: Method_other_out.Wrapper;
  /** @category Comparison operators */
  greater_equal: Method_other_out.Wrapper;
  /** @category Comparison operators */
  less_equal: Method_other_out.Wrapper;
  /** @category Comparison operators */
  equal: Method_other_out.Wrapper;
  /** @category Comparison operators */
  not_equal: Method_other_out.Wrapper;
  /** @category Comparison operators */
  isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray;
  /** @category Comparison operators */
  allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;

  /** @category Unary operators */
  round: Method_a_decimals_out.Wrapper;
  /** @category Unary operators */
  abs: Method_out.Wrapper;
  /** @category Unary operators */
  negative: Method_out.Wrapper;
  /** @category Unary operators */
  bitwise_not: Method_out.Wrapper;
  /** @category Unary logical operators */
  logical_not: Method_out.Wrapper;

  /** @category Operators with assignment */
  assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  add_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  subtract_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  multiply_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  divide_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  mod_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  pow_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  divide_int_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  maximum_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  minimum_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  bitwise_and_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  bitwise_or_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  logical_or_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  bitwise_shift_right_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  bitwise_shift_left_assign: Method_values_where.Wrapper;
  /** @category Operators with assignment */
  logical_and_assign: Method_values_where.Wrapper;

  /** @category Transformations */
  ravel: () => NDArray;
  /** @category Transformations */
  reshape: (shape: any, ...more_shape: any[]) => NDArray;
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

  constructor(flat: InstanceType<DType["BufferType"]>, shape?: number[], dtype?: DType) {
    this.shape = shape || [flat.length]; // invariant: immutable
    this._flat = flat;
    this._simpleIndexes = null;
    if (dtype) this._dtype = dtype;
  }

  /** @ignore */
  _simpleIndexes: import("./array/indexes").AxesIndex | null;

  /** @category Attributes @readonly */
  get size() {
    return this._simpleIndexes == null ? this._flat.length : this._simpleIndexes.size;
  }
  /** @category Attributes @readonly */
  get flat(): InstanceType<DType["BufferType"]> {
    if (this._simpleIndexes == null) return this._flat;
    const indices = this._simpleIndexes.indices;
    //@ts-ignore
    return new_buffer(indices.length, this.dtype).map((_: any, i: number) => this._flat[indices[i]]);
  }


  /** @internal */
  set flat(list) {
    console.log(list, this.size)
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
    return this.dtype === bool ? !!this._flat[0] : this._flat[0];
  }
}

import { GLOBALS } from './_globals';
GLOBALS.NDArray = NDArray;

import { modules } from "./array";
// import { AxisArg, ReduceKwArgs } from './NDArray/reduce';
import { Method_other_out, Method_a_axis_keepdims, Method_values_where, Method_out, Method_a_decimals_out, Method_a_ord_axis_keepdims, Method_a_axis_ddof_keepdims } from './array/kwargs';
NDArray.prototype.modules = modules;



// const a = new NDArray(new Int32Array(20), [], Number)

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


NDArray.prototype.any = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.any);
NDArray.prototype.all = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.all);

NDArray.prototype.sum = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.sum);
NDArray.prototype.product = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.product);
NDArray.prototype.max = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.max);
NDArray.prototype.min = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.min);
NDArray.prototype.argmax = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.argmax);
NDArray.prototype.argmin = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.argmin);
NDArray.prototype.mean = Method_a_axis_keepdims.defaultDecorator(modules.reduce.reducers.mean);

NDArray.prototype.var = Method_a_axis_ddof_keepdims.defaultDecorator(modules.reduce.reducers.var);

NDArray.prototype.std = Method_a_axis_ddof_keepdims.defaultDecorator(modules.reduce.reducers.std);
NDArray.prototype.norm = Method_a_ord_axis_keepdims.defaultDecorator(modules.reduce.reducers.norm);


// ==============================
//       Operators: Binary operations, assignment operations and unary boolean_not
// ==============================
import { Func_a_other_out } from "./array/kwargs";

NDArray.prototype.add = Method_other_out.defaultDecorator(modules.operators.op_binary["+"]);
NDArray.prototype.subtract = Method_other_out.defaultDecorator(modules.operators.op_binary["-"]);
NDArray.prototype.multiply = Method_other_out.defaultDecorator(modules.operators.op_binary["*"]);
NDArray.prototype.divide = Method_other_out.defaultDecorator(modules.operators.op_binary["/"]);
NDArray.prototype.mod = Method_other_out.defaultDecorator(modules.operators.op_binary["%"]);
NDArray.prototype.divide_int = Method_other_out.defaultDecorator(modules.operators.op_binary["//"]);
NDArray.prototype.pow = Method_other_out.defaultDecorator(modules.operators.op_binary["**"]);
NDArray.prototype.maximum = Method_other_out.defaultDecorator(modules.operators.op_binary["max"]);
NDArray.prototype.minimum = Method_other_out.defaultDecorator(modules.operators.op_binary["min"]);

NDArray.prototype.bitwise_or = Method_other_out.defaultDecorator(modules.operators.op_binary["|"]);
NDArray.prototype.bitwise_and = Method_other_out.defaultDecorator(modules.operators.op_binary["&"]);
NDArray.prototype.bitwise_or = Method_other_out.defaultDecorator(modules.operators.op_binary["^"]);
NDArray.prototype.bitwise_shift_right = Method_other_out.defaultDecorator(modules.operators.op_binary["<<"]);
NDArray.prototype.bitwise_shift_right = Method_other_out.defaultDecorator(modules.operators.op_binary[">>"]);

NDArray.prototype.logical_or = Method_other_out.defaultDecorator(modules.operators.op_binary["or"]);
NDArray.prototype.logical_and = Method_other_out.defaultDecorator(modules.operators.op_binary["and"]);
NDArray.prototype.logical_xor = Method_other_out.defaultDecorator(modules.operators.op_binary["xor"]);

NDArray.prototype.greater = Method_other_out.defaultDecorator(modules.operators.op_binary[">"]);
NDArray.prototype.less = Method_other_out.defaultDecorator(modules.operators.op_binary["<"]);
NDArray.prototype.greater_equal = Method_other_out.defaultDecorator(modules.operators.op_binary[">="]);
NDArray.prototype.less_equal = Method_other_out.defaultDecorator(modules.operators.op_binary["<="]);
NDArray.prototype.equal = Method_other_out.defaultDecorator(modules.operators.op_binary["=="]);
NDArray.prototype.not_equal = Method_other_out.defaultDecorator(modules.operators.op_binary["!="]);


// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = Method_out.defaultDecorator(modules.elementwise.funcs.bitwise_not);
NDArray.prototype.logical_not = Method_out.defaultDecorator(modules.elementwise.funcs.logical_not);
NDArray.prototype.negative = Method_out.defaultDecorator(modules.elementwise.funcs.negative);
NDArray.prototype.abs = Method_out.defaultDecorator(modules.elementwise.funcs.abs);


NDArray.prototype.isclose = modules.operators.isclose;
NDArray.prototype.allclose = modules.operators.allclose;


NDArray.prototype.assign = Method_values_where.defaultDecorator(modules.operators.op_assign["="]);
NDArray.prototype.add_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["+="]);
NDArray.prototype.subtract_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["-="]);
NDArray.prototype.multiply_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["*="]);
NDArray.prototype.divide_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["/="]);
NDArray.prototype.mod_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["%="]);
NDArray.prototype.divide_int_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["//="]);
NDArray.prototype.pow_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["**="]);

NDArray.prototype.maximum_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["max="]);
NDArray.prototype.minimum_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["min="]);

NDArray.prototype.bitwise_or_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["|="]);
NDArray.prototype.bitwise_and_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["&="]);
NDArray.prototype.bitwise_shift_left_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["<<="]);
NDArray.prototype.bitwise_shift_right_assign = Method_values_where.defaultDecorator(modules.operators.op_assign[">>="]);

NDArray.prototype.logical_or_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["or="]);
NDArray.prototype.logical_and_assign = Method_values_where.defaultDecorator(modules.operators.op_assign["and="]);



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

NDArray.prototype.round = Method_a_decimals_out.defaultDecorator(modules.elementwise.funcs.round);

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
