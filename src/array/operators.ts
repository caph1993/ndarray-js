//@ts-check
import * as indexes from './indexes';
import { isarray, asarray, _NDArray, new_from, number_collapse, ravel, shape_shifts, new_NDArray } from './basic';
import { tolist } from './js-interface';

import NDArray from "../NDArray";
import { Func_y_x_out } from './kwargs';
import { extend } from '../utils-js';
import { Where } from './indexes';
import { concatenate } from './transform';
import { broadcast_n_shapes, broadcast_shapes } from './_globals';
import { TypedArrayConstructor, new_buffer } from '../dtypes';

export type ArrayOrConstant = NDArray<any> | number | boolean;





export function binary_operation<
  T extends TypedArrayConstructor = Float64ArrayConstructor,
>(A: ArrayOrConstant, B: ArrayOrConstant, func: any, dtype: T, out: NDArray<T> | null = null): NDArray<T> {

  if (isarray(this)) return func.bind(_NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);
  const [shape, shapeA, shapeB] = broadcast_shapes(A.shape, B.shape);

  //@ts-ignore
  if (out) dtype = out.dtype;
  //@ts-ignore
  if (out == null) out = new_from(shape, undefined, dtype);

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
  out.flat = new_buffer(flatOut, dtype);
  return out;
}


export function __make_operator<
  T extends TypedArrayConstructor = Float64ArrayConstructor,
>(dtype: T, func): BinaryOperator {
  function operator(A: ArrayOrConstant, B: ArrayOrConstant, out = null) {
    return binary_operation<T>(A, B, func, dtype, out);
  };
  //@ts-ignore
  return operator;
}


export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray



export const op_binary = {

  "+": __make_operator(Float64Array, (a, b) => a + b),
  "-": __make_operator(Float64Array, (a, b) => a - b),
  "*": __make_operator(Float64Array, (a, b) => a * b),
  "/": __make_operator(Float64Array, (a, b) => a / b),
  "%": __make_operator(Float64Array, (a, b) => (a % b)),
  "|": __make_operator(Float64Array, (a, b) => a | b),
  "&": __make_operator(Float64Array, (a, b) => a & b),
  "^": __make_operator(Float64Array, (a, b) => a ^ b),
  "<<": __make_operator(Float64Array, (a, b) => a << b),
  ">>": __make_operator(Float64Array, (a, b) => a >> b),
  "**": __make_operator(Float64Array, (a, b) => Math.pow(a, b)),
  "//": __make_operator(Float64Array, (a, b) => Math.floor(a / b)),

  "<": __make_operator(Uint8Array, (a, b) => a < b),
  ">": __make_operator(Uint8Array, (a, b) => a > b),
  ">=": __make_operator(Uint8Array, (a, b) => a >= b),
  "<=": __make_operator(Uint8Array, (a, b) => a <= b),
  "==": __make_operator(Uint8Array, (a, b) => a == b),
  "!=": __make_operator(Uint8Array, (a, b) => a != b),
  // Operators with custom ascii identifiers:
  "or": __make_operator(Uint8Array, (a, b) => a || b),
  "and": __make_operator(Uint8Array, (a, b) => a && b),
  "xor": __make_operator(Uint8Array, (a, b) => (!a) != (!b)),
  "max": __make_operator(Float64Array, (a, b) => Math.max(a, b)),
  "min": __make_operator(Float64Array, (a, b) => Math.min(a, b)),
  // "approx": ,
}


op_binary["↑"] = op_binary["max"];
op_binary["↓"] = op_binary["min"];
op_binary["≤"] = op_binary["leq"];
op_binary["≥"] = op_binary["geq"];
op_binary["≠"] = op_binary["neq"];


export function assign_operation<
  T extends TypedArrayConstructor = Float64ArrayConstructor,
>(tgt: NDArray<T>, src: ArrayOrConstant, where: Where, func): NDArray<T> {

  if (tgt['__warnAssignment']) {
    console.warn(`Warning: You are assigning on a copy that resulted from an advanced index on a source array.\nIf this is intentional, use yourArray = source.withKwArgs({copy:true}).index(...yourIndex) to make explicit your awareness of the copy operation.\nInstead, if you want to assign to the source array, use source.op('=', other) or source.op(['::3', -1, '...', [5,4]], '*=', other).\n`);
    delete tgt['__warnAssignment'];
  }
  if (!(isarray(tgt))) {
    return _assign_operation_toJS(tgt as any, src, where, func) as any;
  }
  if (!where) {
    binary_operation(tgt, src, func, tgt.dtype, tgt);
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
    binary_operation(tmpTgt, ravel(src), func, tgt.dtype, tmpTgt);
    for (let i in indices) tgt._flat[indices[i]] = tmpTgt._flat[i];
  }
  return tgt;
};



export function _assign_operation_toJS(tgtJS: any[], src: any, where: Where, func: any) {
  if (!Array.isArray(tgtJS)) throw new Error(`Can not assign to a non-array. Found ${typeof tgtJS}: ${tgtJS}`);
  console.warn('Assignment to JS array is experimental and slow.')
  // Parse the whole array
  const cpy = asarray(tgtJS);
  assign_operation(cpy, src, where, func);
  // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
  const outJS = tolist(cpy);
  while (tgtJS.length) tgtJS.pop();
  // @ts-ignore
  extend(tgtJS, outJS);
  return tgtJS;
}


// export type AssignmentOperator = { (tgt: NDArray, src: ArrayOrConstant): NDArray; (tgt: NDArray, where: Where, src: ArrayOrConstant): NDArray; }
// export type SelfAssignmentOperator = { (other: ArrayOrConstant): NDArray; (where: Where, other: ArrayOrConstant): NDArray; }


export function __make_assignment_operator(func) {
  return function operator<
    T extends TypedArrayConstructor,
  >(a: NDArray<T>, values: NDArray, where: Where) {
    // console.log(`where=${JSON.stringify(where)}`)
    if (where.length === 1 && where[0] === false) return a;
    if (where.length === 1 && where[0] === true) where = [];
    return assign_operation<T>(a, values, where, func);
  }
}

export const op_assign = {
  "=": __make_assignment_operator((a, b) => b),
  "+=": __make_assignment_operator((a, b) => a + b),
  "-=": __make_assignment_operator((a, b) => a - b),
  "*=": __make_assignment_operator((a, b) => a * b),
  "/=": __make_assignment_operator((a, b) => a / b),
  "%=": __make_assignment_operator((a, b) => (a % b)),
  "//=": __make_assignment_operator((a, b) => Math.floor(a / b)),
  "**=": __make_assignment_operator((a, b) => Math.pow(a, b)),
  "|=": __make_assignment_operator((a, b) => a | b),
  "&=": __make_assignment_operator((a, b) => a & b),
  "^=": __make_assignment_operator((a, b) => a ^ b),
  "<<=": __make_assignment_operator((a, b) => a << b),
  ">>=": __make_assignment_operator((a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": __make_assignment_operator((a, b) => Math.max(a, b)),
  "min=": __make_assignment_operator((a, b) => Math.min(a, b)),
  "or=": __make_assignment_operator((a, b) => a || b),
  "and=": __make_assignment_operator((a, b) => a && b),
};


// op_assign["↑="] = op_assign["max="];
// op_assign["↓="] = op_assign["min="];


// ====================================

export function isclose(A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  const func = (a, b) => {
    if (a == b) return true; // shortcut for equality
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    console.log('NOT FINITE', a, b)
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  return binary_operation(A, B, func, Uint8Array)
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
  try { binary_operation(A, B, wrapper, Float64Array) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
}


//op_binary["≈≈"] = op[MyArray.prototype.isclose,

export const atan2 = Func_y_x_out.defaultDecorator(__make_operator(Float64Array, Math.atan2));




export function n_ary_operation<
  F extends (...args: any) => NDArray | number | number[] | boolean | boolean[]
>(
  arrs: NDArray[],
  elem_shape: number[],
  func: F,
): NDArray {
  // Find output shape and input broadcast shapes
  arrs = arrs.map(asarray);

  // // Some optimization for common low-complexity cases
  // if (arrs.length == 1 && elem_shape.length == 0) {
  //   const a = arrs[0];
  //   //@ts-ignore
  //   return new_NDArray(a.flat.map(x => number_collapse(func(x))), a.shape, Number); // FIX
  // } else if (arrs.length == 1) {
  //   const a = arrs[0];
  //   const a_size = a.size;
  //   const out_elem_size = elem_shape.reduce((a, b) => a * b, 1);
  //   const outs = [];
  //   a.flat.forEach((x, i) => {
  //     //@ts-ignore
  //     const values = number_collapse(func(x));
  //     const end = (i + 1) * out_elem_size;
  //     for (let j = i * out_elem_size; j < end; j++) outs[j] = values[i];
  //   });
  //   //@ts-ignore
  //   return new_NDArray(outs, a.shape, Number); // FIX
  // }

  const rawShapes = arrs.map(a => a.shape);
  // Broadcast shapes
  const [shapes, shape] = broadcast_n_shapes(...rawShapes);
  const size = shape.reduce((a, b) => a * b, 1);
  if (shape.length == 0) {
    return asarray(func(...arrs.map((a) => a.flat[0])));
  }
  const shifts = shapes.map(shape => shape_shifts(shape));

  function* iter_n_ary() {
    for (let i = 0; i < size; i++) {
      let idx = i;
      // compute index per array
      let idxs = arrs.map(_ => 0);
      for (let axis = shape.length - 1; axis >= 0; axis--) {
        for (let j = 0; j < arrs.length; j++) {
          idxs[j] += shifts[j][axis] * (idx % shapes[j][axis]);
        }
        idx = Math.floor(idx / shape[axis]);
      }
      yield [i, idxs] as [number, number[]];
    }
  }

  // Iterate with broadcasted indices
  const flat = [];
  let funcOutIsNDArray = false;
  for (let [i, idxs] of iter_n_ary()) {
    // apply function
    const value = func(...arrs.map((a, j) => a.flat[idxs[j]]));
    if (i == 0) funcOutIsNDArray = isarray(value);
    flat.push(value);
  };
  let out = funcOutIsNDArray ? concatenate(flat) : asarray(flat);
  out = out.reshape([shape, ...out.shape.slice(1)]);
  return out;
}
