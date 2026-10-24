//@ts-check

import { DType, HasDType, bool, infer_dtype, new_buffer, object, Buffer } from './dtypes';
/** @ignore */
export type ArrayOrConstant = NDArray | number | boolean;


import type * as sigs from './kwargs/signatures';
import type { Method_other_out, Method_a_axis_keepdims, Method_a_axis_keepdims_dtype_out, Method_a_min_a_max_out, Method_values_where, Method_out, Method_a_decimals_out, Method_a_ord_axis_keepdims, Method_a_axis_ddof_keepdims } from './kwargs/kwargs';


/**
 * Multi dimensional array.
 */
class NDArray implements HasDType {


  /** @ignore */
  _flat: Buffer;

  /** @category Attributes @readonly */
  shape: number[];

  /** @ignore */
  _dtype?: DType;

  /**
   * True for 0-d arrays that come from a plain JS number/boolean (like python scalars,
   * they do not upcast the dtype of the other operand in binary operations).
   * @ignore
   */
  _weak?: boolean;

  /** @category Attributes @readonly */
  get dtype(): DType {
    if (this._dtype) return this._dtype;
    return infer_dtype[this._flat.constructor.name] || object;
  }

  /** @category Indexing / slicing */
  index: (...where: Where) => NDArray;

  /** @category Reducers */
  any: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  all: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  sum: Method_a_axis_keepdims_dtype_out.Wrapper;
  /** @category Reducers */
  product: Method_a_axis_keepdims_dtype_out.Wrapper;
  /** @category Reducers */
  prod: Method_a_axis_keepdims_dtype_out.Wrapper;
  /** @category Reducers */
  ptp: Method_a_axis_keepdims.Wrapper;
  /** @category Cumulative */
  cumsum: ReturnType<typeof sigs.cumulative.method>;
  /** @category Cumulative */
  cumprod: ReturnType<typeof sigs.cumulative.method>;
  /** @category Searching */
  nonzero: ReturnType<typeof sigs.a_only.method>;
  /** @category Searching */
  searchsorted: ReturnType<typeof sigs.searchsorted.method>;
  /** @category Elementwise operators */
  clip: Method_a_min_a_max_out.Wrapper;
  /** @category Elementwise operators */
  conj: Method_out.Wrapper;
  /** @category Reducers */
  max: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  min: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  argmax: Method_a_axis_keepdims.Wrapper;
  /** @category Reducers */
  argmin: Method_a_axis_keepdims.Wrapper;

  /** @category Reducers */
  mean: Method_a_axis_keepdims_dtype_out.Wrapper;
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
  bitwise_xor: Method_other_out.Wrapper;
  /** @category Binary operators */
  bitwise_shift_left: Method_other_out.Wrapper;
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
  isclose: ReturnType<typeof sigs.isclose.method>;
  /** @category Comparison operators */
  allclose: ReturnType<typeof sigs.allclose.method>;

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
  sort: (axis?: number | { axis?: number }) => NDArray;
  /** @category Sorting */
  argsort: (axis?: number | { axis?: number }) => NDArray;
  /** @category Transformations */
  transpose: (axes?: number[]) => NDArray;


  /** @category Casting */
  astype: ReturnType<typeof sigs.astype.method>;
  /** @category Transformations */
  flatten: () => NDArray;
  /** @category Transformations */
  squeeze: ReturnType<typeof sigs.squeeze.method>;
  /** @category Transformations */
  swapaxes: ReturnType<typeof sigs.swapaxes.method>;
  /** @category Transformations */
  repeat: ReturnType<typeof sigs.repeat.method>;
  /** @category Casting */
  tolist: () => any;
  fromlist: (A: any, dtype?: DType) => NDArray;

  /**
   * Generic operator function. See {@link GenericOperatorFunction} for details.
   */
  op: GenericOperatorFunction;

  constructor(flat: Buffer, shape?: number[], dtype?: DType) {
    this.shape = shape || [flat.length]; // invariant: immutable
    this._flat = flat;
    this._simpleIndexes = null;
    if (dtype && this.dtype !== dtype) this._dtype = dtype;
  }

  /** @ignore */
  _simpleIndexes: import("./array/indexes").AxesIndex | null;

  /** @category Attributes @readonly */
  get size() {
    return this._simpleIndexes == null ? this._flat.length : this._simpleIndexes.size;
  }
  /** @category Attributes @readonly */
  get flat(): Buffer {
    if (this._simpleIndexes == null) return this._flat;
    const indices = this._simpleIndexes.indices;
    //@ts-ignore
    return new_buffer(indices.length, this.dtype).map((_: any, i: number) => this._flat[indices[i]]);
  }


  /** @internal */
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
    return this.dtype === bool ? !!this._flat[0] : this._flat[0];
  }
  /** Returns a plain JS value for 0-d arrays, and the array itself otherwise */
  item_if_scalar() {
    if (this.shape.length == 0) return this.item();
    else return this;
  }
}

export type Shape = number | number[] | NDArray;


export function shape_shifts(shape) {
  // increasing one by one on a given axis is increasing by shifts[axis] in flat representation
  const shifts = Array.from({ length: shape.length }, (_) => 0);
  shifts[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) shifts[i] = shifts[i + 1] * shape[i + 1];
  return shifts;
}

export function parse_shape(shape: Shape): number[] {
  if (typeof shape == "number") return [shape];
  if (isarray(shape)) {
    if (shape.shape.length > 1) {
      throw new Error(`Expected flat shape. Got array with shape ${shape.shape}`);
    }
    return [...shape.flat] as number[];
  }
  if (Array.isArray(shape)) return shape;
  throw new Error(`Expected shape. Got ${shape}`);
}

export function broadcast_shapes(shapeA: number[], shapeB: number[]) {
  const shape = [];
  const maxDim = Math.max(shapeA.length, shapeB.length);
  shapeA = [...Array.from({ length: maxDim - shapeA.length }, () => 1), ...shapeA];
  shapeB = [...Array.from({ length: maxDim - shapeB.length }, () => 1), ...shapeB];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dim1 = shapeA[axis];
    const dim2 = shapeB[axis];
    if (dim1 !== 1 && dim2 !== 1 && dim1 !== dim2)
      throw new Error(`Can not broadcast axis ${axis} with sizes ${dim1} and ${dim2}`);
    shape.unshift(dim1 === 1 ? dim2 : dim1);
  }
  return [shape, shapeA, shapeB];
}

/**
 * Strides (in the flat buffer of an array with shape `srcShape`) to iterate it as if it had
 * been broadcast to `shape`. Broadcast axes have stride 0.
 */
export function broadcast_strides(shape: number[], srcShape: number[]): number[] {
  const offset = shape.length - srcShape.length;
  const strides = shape_shifts(srcShape);
  return shape.map((_, i) => (i < offset || srcShape[i - offset] === 1) ? 0 : strides[i - offset]);
}

export function broadcast_n_shapes(...shapes: number[][]) {
  const maxDim = Math.max(...shapes.map(shape => shape.length));
  const broadcastShapes = shapes.map(shape => {
    return [...Array.from({ length: maxDim - shape.length }, () => 1), ...shape];
  });
  const outputShape = [];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dims = broadcastShapes.map(shape => shape[axis]);
    const dim = dims.find(d => d !== 1) ?? 1;
    if (dims.some(d => d !== 1 && d !== dim)) throw new Error(`Can not broadcast axis ${axis} with sizes ${dims}`);
    outputShape.unshift(dim);
  }
  return [broadcastShapes, outputShape];
}



export { NDArray };

export default NDArray;

// //=============================

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
export function copy(A: NDArray) {
  return new NDArray(new_buffer(A.flat, A.dtype), A.shape);
}
NDArray.prototype.copy = function () {
  return copy(this);
};
// ====================
// Constructors
// ====================
export function empty(shape: Shape, dtype?: DType) {
  const size = parse_shape(shape).reduce((a, b) => a * b, 1);
  const buffer = new_buffer(size, dtype);
  return new NDArray(buffer, parse_shape(shape), dtype);
}
export function asarray(A: NDArray | any, dtype: DType = null): NDArray {
  if (isarray(A) && (!dtype || A.dtype === dtype)) return A;
  else return NDArray.prototype.fromlist(A, dtype);
}
export function isarray(A: any): A is NDArray {
  return A instanceof NDArray;
}

// const op: GenericOperatorFunction = function (...args): NDArray {
//   if (!args.length) return this;
//   if (typeof args[0] == "string") {
//     const symbol = args[0];
//     if (args.length == 1) {
//       let func = modules.elementwise.ops[symbol];
//       if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.elementwise.ops)]}`);
//       return func(this, symbol);
//     }
//     if (args.length > 2) throw new Error(`Too many arguments provided: ${[...args]}`);
//     const other = args[1];
//     let func = modules.operators.op_binary[symbol];
//     if (func) return func(this, other);
//     func = modules.operators.op_assign[symbol];
//     if (func) return func(this, other);
//     if (symbol.includes(':')) throw new Error(`Expected index or operator symbol. Found "${symbol}". Did you mean ${[symbol]}?`);
//     throw new Error(`Expected index or operator symbol. Found "${symbol}"`);
//   }
//   const where = args[0];
//   if (where instanceof NDArray) throw new Error(`Expected operator or index. Found numpy array`);
//   if (args.length == 1) return this.index(where);
//   const symbol = args[1];
//   let func = modules.operators.op_assign[symbol];
//   if (!func) throw new Error(`Unknown assign operator "${symbol}". Options:${[...Object.keys(modules.operators.op_assign)]}`);
//   if (args.length > 3) throw new Error(`Too many arguments provided: ${[...args]}`);
//   const other = args[2];
//   return func(this, where, other);
// }

// NDArray.prototype.op = op;

