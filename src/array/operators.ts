//@ts-check
import * as indexes from './indexes';
import { isarray, asarray, _NDArray, new_from, number_collapse, ravel, shape_shifts } from './basic';
import { tolist } from './js-interface';

import type NDArray from "../NDArray";
import { BinaryOperatorParsedKwargs, BinaryOperatorMethod, UnaryOperatorParsedKwargs, UnaryOperatorMethod, kwDecorator, kwDecorators, AssignmentOperatorParsedKwargs, AssignmentOperatorMethod } from './kwargs';
import { extend } from '../utils-js';
import { Where } from './indexes';

export type ArrayOrConstant = NDArray | number | boolean;
type Index = indexes.Where;


export function binary_operation(A: ArrayOrConstant, B: ArrayOrConstant, func: any, dtype: any, out: NDArray | null = null): ArrayOrConstant {

  if (isarray(this)) return func.bind(_NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);
  const [shape, shapeA, shapeB] = _broadcast_shapes(A.shape, B.shape);
  if (out == null) out = new_from(shape, (_) => undefined, dtype);
  else if (!(isarray(out))) throw new Error(`Out must be of type ${_NDArray}. Got ${typeof out}`);
  // Iterate with broadcasted indices
  const flatOut = [];
  const shiftsA = shape_shifts(shapeA);
  const shiftsB = shape_shifts(shapeB);
  const flatA = A.flat;
  const flatB = B.flat;
  for (let i = 0; i < out.size; i++) {
    let idxA = 0, idxB = 0, idx = i;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      idxA += shiftsA[axis] * (idx % shapeA[axis]);
      idxB += shiftsB[axis] * (idx % shapeB[axis]);
      idx = Math.floor(idx / shape[axis]);
    }
    flatOut.push(func(flatA[idxA], flatB[idxB]));
  };
  out.flat = flatOut;
  return number_collapse(out);
}

export function _broadcast_shapes(shapeA, shapeB) {
  const shape = [];
  const maxDim = Math.max(shapeA.length, shapeB.length);
  shapeA = [...Array.from({ length: maxDim - shapeA.length }, () => 1), ...shapeA];
  shapeB = [...Array.from({ length: maxDim - shapeB.length }, () => 1), ...shapeB];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dim1 = shapeA[axis];
    const dim2 = shapeB[axis];
    if (dim1 !== 1 && dim2 !== 1 && dim1 !== dim2)
      throw new Error(`Can not broadcast axis ${axis} with sizes ${dim1} and ${dim2}`);
    shape.unshift(Math.max(dim1, dim2));
  }
  return [shape, shapeA, shapeB];
}

export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray

export function __make_operator(dtype, func): BinaryOperator {
  function operator(A: ArrayOrConstant, B: ArrayOrConstant, out = null) {
    return binary_operation(A, B, func, dtype, out);
  };
  //@ts-ignore
  return operator;
}

export function __make_operator_special(funcNum, funcBool): BinaryOperator {
  function operator(arr, other, out: NDArray | null = null) {
    arr = asarray(arr);
    other = asarray(other);
    let dtype = arr.dtype, func;
    if (arr.dtype != other.dtype) console.warn(`Warning: operating arrays of different dtypes. Using ${dtype}`);
    if (dtype == Boolean) func = funcBool;
    else func = funcNum;
    return binary_operation(arr, other, func, dtype, out);
  };
  //@ts-ignore
  return operator;
}

export const op_binary = {
  "+": __make_operator(Number, (a, b) => a + b),
  "-": __make_operator(Number, (a, b) => a - b),
  "*": __make_operator(Number, (a, b) => a * b),
  "/": __make_operator(Number, (a, b) => a / b),
  "%": __make_operator(Number, (a, b) => (a % b)),
  "//": __make_operator(Number, (a, b) => Math.floor(a / b)),
  "**": __make_operator(Number, (a, b) => Math.pow(a, b)),
  "<": __make_operator(Boolean, (a, b) => a < b),
  ">": __make_operator(Boolean, (a, b) => a > b),
  ">=": __make_operator(Boolean, (a, b) => a >= b),
  "<=": __make_operator(Boolean, (a, b) => a <= b),
  "==": __make_operator(Boolean, (a, b) => a == b),
  "!=": __make_operator(Boolean, (a, b) => a != b),
  "|": __make_operator_special((a, b) => a | b, (a, b) => a || b),
  "&": __make_operator_special((a, b) => a & b, (a, b) => a && b),
  "^": __make_operator(Number, (a, b) => a ^ b),
  "<<": __make_operator(Number, (a, b) => a << b),
  ">>": __make_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "or": __make_operator(Boolean, (a, b) => a || b),
  "and": __make_operator(Boolean, (a, b) => a && b),
  "xor": __make_operator(Boolean, (a, b) => (!a) != (!b)),
  "max": __make_operator(Number, (a, b) => Math.max(a, b)),
  "min": __make_operator(Number, (a, b) => Math.min(a, b)),
  // "isclose": ,
}


op_binary["↑"] = op_binary["max"];
op_binary["↓"] = op_binary["min"];
op_binary["≤"] = op_binary["leq"];
op_binary["≥"] = op_binary["geq"];
op_binary["≠"] = op_binary["neq"];


export function assign_operation(tgt: NDArray, src: ArrayOrConstant, where: Index, func, dtype) {

  if (tgt['__warnAssignment']) {
    console.warn(`Warning: You are assigning on a copy that resulted from an advanced index on a source array.\nIf this is intentional, use yourArray = source.withKwArgs({copy:true}).index(...yourIndex) to make explicit your awareness of the copy operation.\nInstead, if you want to assign to the source array, use source.op('=', other) or source.op(['::3', -1, '...', [5,4]], '*=', other).\n`);
    delete tgt['__warnAssignment'];
  }
  if (!(isarray(tgt))) return _assign_operation_toJS(/**@type {*}*/(tgt), src, where, func, dtype)
  if (!where) {
    binary_operation(tgt, src, func, dtype, tgt);
  } else {
    src = asarray(src);
    let { indices } = indexes.AxesIndex.prototype.parse(tgt.shape, where);
    let tmpTgt;
    if (func == null) {
      // Small optimization: unlike "+=", "*=", etc., for "=", we don't need to reed the target
      func = (a, b) => b;
      tmpTgt = new_from(indices.length, () => undefined, tgt.dtype);
    } else {
      tmpTgt = asarray(indices.map(i => tgt._flat[i]));
    }
    binary_operation(tmpTgt, ravel(src), func, dtype, tmpTgt);
    for (let i of indices) tgt._flat[i] = tmpTgt._flat[i];
  }
};



export function _assign_operation_toJS(tgtJS: any[], src: any, where: Index, func: any, dtype: any) {
  if (!Array.isArray(tgtJS)) throw new Error(`Can not assign to a non-array. Found ${typeof tgtJS}: ${tgtJS}`);
  console.warn('Assignment to JS array is experimental and slow.')
  // Parse the whole array
  const cpy = asarray(tgtJS);
  assign_operation(cpy, src, where, func, dtype);
  // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
  const outJS = tolist(cpy);
  while (tgtJS.length) tgtJS.pop();
  // @ts-ignore
  extend(tgtJS, outJS);
}


// export type AssignmentOperator = { (tgt: NDArray, src: ArrayOrConstant): NDArray; (tgt: NDArray, where: Index, src: ArrayOrConstant): NDArray; }
// export type SelfAssignmentOperator = { (other: ArrayOrConstant): NDArray; (where: Index, other: ArrayOrConstant): NDArray; }


export function __make_assignment_operator(dtype, func) {
  return function operator(a: NDArray, values: NDArray, ...where: Where) {
    return assign_operation(a, values, where, func, dtype);
  }
}

export const op_assign = {
  "=": __make_assignment_operator(Number, (a, b) => b),
  "+=": __make_assignment_operator(Number, (a, b) => a + b),
  "-=": __make_assignment_operator(Number, (a, b) => a - b),
  "*=": __make_assignment_operator(Number, (a, b) => a * b),
  "/=": __make_assignment_operator(Number, (a, b) => a / b),
  "%=": __make_assignment_operator(Number, (a, b) => (a % b)),
  "//=": __make_assignment_operator(Number, (a, b) => Math.floor(a / b)),
  "**=": __make_assignment_operator(Number, (a, b) => Math.pow(a, b)),
  "|=": __make_assignment_operator(Number, (a, b) => a | b),
  "&=": __make_assignment_operator(Number, (a, b) => a & b),
  "^=": __make_assignment_operator(Number, (a, b) => a ^ b),
  "<<=": __make_assignment_operator(Number, (a, b) => a << b),
  ">>=": __make_assignment_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": __make_assignment_operator(Number, (a, b) => Math.max(a, b)),
  "min=": __make_assignment_operator(Number, (a, b) => Math.min(a, b)),
  "or=": __make_assignment_operator(Boolean, (a, b) => a || b),
  "and=": __make_assignment_operator(Boolean, (a, b) => a && b),
};


// op_assign["↑="] = op_assign["max="];
// op_assign["↓="] = op_assign["min="];


// ====================================

export function isclose(A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  const func = (a, b) => {
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  return binary_operation(A, B, func, Boolean)
}

export function allclose(A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  // Equivalent to all(isclose(A, B, rtol, atol, equal_nan)), but shortcutting if false 
  const func = (a, b) => { //copied from isclose
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  const different = new Error('');
  const wrapper = (a, b) => {
    if (!func(a, b)) throw different;
    return 0;
  }
  try { binary_operation(A, B, wrapper, Number) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
}


//op_binary["≈≈"] = op[MyArray.prototype.isclose,

function mk_kw_operator<T = number>(op: BinaryOperator) {
  return kwDecorators<BinaryOperatorMethod<T>, BinaryOperatorParsedKwargs>({
    defaults: [["other", undefined, asarray], ["out", null]],
    func: op,
  })
}

export const kw_op_binary = {
  "+": mk_kw_operator(op_binary["+"]),
  "-": mk_kw_operator(op_binary["-"]),
  "*": mk_kw_operator(op_binary["*"]),
  "/": mk_kw_operator(op_binary["/"]),
  "%": mk_kw_operator(op_binary["%"]),
  "//": mk_kw_operator(op_binary["//"]),
  "**": mk_kw_operator(op_binary["**"]),
  "<": mk_kw_operator(op_binary["<"]),
  ">": mk_kw_operator(op_binary[">"]),
  ">=": mk_kw_operator(op_binary[">="]),
  "<=": mk_kw_operator(op_binary["<="]),
  "==": mk_kw_operator(op_binary["=="]),
  "!=": mk_kw_operator(op_binary["!="]),
  "|": mk_kw_operator(op_binary["|"]),
  "&": mk_kw_operator(op_binary["&"]),
  "^": mk_kw_operator(op_binary["^"]),
  "<<": mk_kw_operator(op_binary["<<"]),
  ">>": mk_kw_operator(op_binary[">>"]),
  "max": mk_kw_operator(op_binary["max"]),
  "min": mk_kw_operator(op_binary["min"]),
  "or": mk_kw_operator<boolean>(op_binary["or"]),
  "and": mk_kw_operator<boolean>(op_binary["and"]),
  "xor": mk_kw_operator<boolean>(op_binary["xor"]),
}

export const atan2 = kwDecorator<(y: NDArray, x: NDArray, out?: NDArray | null) => NDArray, any>({
  defaults: [["y", undefined, asarray], ["x", undefined, asarray], ["out", null]],
  func: __make_operator(Number, (y, x) => Math.atan2(y, x)),
});




function mk_kw_assign_operator(op) {
  return kwDecorators<AssignmentOperatorMethod, AssignmentOperatorParsedKwargs>({
    defaults: [["values", undefined, asarray], ['...where', []]],
    func: op,
  })
}

export const kw_op_assign = {
  "=": mk_kw_assign_operator(op_assign["="]),
  "+=": mk_kw_assign_operator(op_assign["+="]),
  "-=": mk_kw_assign_operator(op_assign["-="]),
  "*=": mk_kw_assign_operator(op_assign["*="]),
  "/=": mk_kw_assign_operator(op_assign["/="]),
  "%=": mk_kw_assign_operator(op_assign["%="]),
  "//=": mk_kw_assign_operator(op_assign["//="]),
  "**=": mk_kw_assign_operator(op_assign["**="]),
  "|=": mk_kw_assign_operator(op_assign["|="]),
  "&=": mk_kw_assign_operator(op_assign["&="]),
  "^=": mk_kw_assign_operator(op_assign["^="]),
  "<<=": mk_kw_assign_operator(op_assign["<<="]),
  ">>=": mk_kw_assign_operator(op_assign[">>="]),
  // Operators with custom ascii identifiers:
  "max=": mk_kw_assign_operator(op_assign["max="]),
  "min=": mk_kw_assign_operator(op_assign["min="]),
  "or=": mk_kw_assign_operator(op_assign["or="]),
  "and=": mk_kw_assign_operator(op_assign["and="]),
}

