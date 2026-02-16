//@ts-check
import * as indexes from './indexes';
import { isarray, asarray, _NDArray, empty, ravel, shape_shifts, new_NDArray } from './basic';
import { DType, DtypeResolver } from '../dtypes';
import { tolist } from './js-interface';

import NDArray from "../NDArray";
import { Func_y_x_out, Func_a_a_min_a_max_out, Func_x1_x2_out } from './kwargs';
import { extend } from '../utils-js';
import { Where } from './indexes';
import { concatenate } from './transform';
import { broadcast_n_shapes, broadcast_shapes } from './_globals';
import { new_buffer } from '../dtypes';

export type ArrayOrConstant = NDArray | number | boolean;


export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | DType | null) => NDArray
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray




function freeze_args_bin_op(get_dtype: DtypeResolver, func): BinaryOperator {
  return function (A: NDArray, B: NDArray, out: NDArray | DType = null): NDArray {
    return apply_binary_operation(get_dtype, func, A, B, out);
  };
}

function apply_binary_operation(
  get_dtype: DtypeResolver,
  func,
  A: NDArray,
  B: NDArray,
  out: NDArray | DType = null
) {
  if (isarray(this)) return func.bind(_NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);
  const dtype = get_dtype([A.dtype, B.dtype], out);
  out = isarray(out) ? out : null;
  return _apply_binary_operation(A, B, func, dtype, out);
}

function _apply_binary_operation(
  A: NDArray,
  B: NDArray,
  func,
  dtype: DType = null,
  out: NDArray = null,
) {
  const [shape, shapeA, shapeB] = broadcast_shapes(A.shape, B.shape);

  //@ts-ignore
  if (out) dtype = out.dtype;
  //@ts-ignore
  if (out == null) out = empty(shape, dtype);

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



export const op_binary = {

  "+": freeze_args_bin_op(Float64Array, (a, b) => a + b),
  "-": freeze_args_bin_op(Float64Array, (a, b) => a - b),
  "*": freeze_args_bin_op(Float64Array, (a, b) => a * b),
  "/": freeze_args_bin_op(Float64Array, (a, b) => a / b),
  "%": freeze_args_bin_op(Float64Array, (a, b) => (a % b)),
  "|": freeze_args_bin_op(Float64Array, (a, b) => a | b),
  "&": freeze_args_bin_op(Float64Array, (a, b) => a & b),
  "^": freeze_args_bin_op(Float64Array, (a, b) => a ^ b),
  "<<": freeze_args_bin_op(Float64Array, (a, b) => a << b),
  ">>": freeze_args_bin_op(Float64Array, (a, b) => a >> b),
  "**": freeze_args_bin_op(Float64Array, (a, b) => Math.pow(a, b)),
  "//": freeze_args_bin_op(Float64Array, (a, b) => Math.floor(a / b)),

  "<": freeze_args_bin_op(Uint8Array, (a, b) => a < b),
  ">": freeze_args_bin_op(Uint8Array, (a, b) => a > b),
  ">=": freeze_args_bin_op(Uint8Array, (a, b) => a >= b),
  "<=": freeze_args_bin_op(Uint8Array, (a, b) => a <= b),
  "==": freeze_args_bin_op(Uint8Array, (a, b) => a == b),
  "!=": freeze_args_bin_op(Uint8Array, (a, b) => a != b),
  // Operators with custom ascii identifiers:
  "or": freeze_args_bin_op(Uint8Array, (a, b) => a || b),
  "and": freeze_args_bin_op(Uint8Array, (a, b) => a && b),
  "xor": freeze_args_bin_op(Uint8Array, (a, b) => (!a) != (!b)),
  "max": freeze_args_bin_op(Float64Array, (a, b) => Math.max(a, b)),
  "min": freeze_args_bin_op(Float64Array, (a, b) => Math.min(a, b)),
  "hypot": freeze_args_bin_op(Float64Array, Math.hypot),
  "fmod": freeze_args_bin_op(Float64Array, (a, b) => a % b),
  "remainder": freeze_args_bin_op(Float64Array, (a, b) => a % b),
  "divide": freeze_args_bin_op(Float64Array, (a, b) => a / b),
  "true_divide": freeze_args_bin_op(Float64Array, (a, b) => a / b),
  "fmax": freeze_args_bin_op(Float64Array, (a, b) => {
    if (Number.isNaN(a)) return b;
    if (Number.isNaN(b)) return a;
    return Math.max(a, b);
  }),
  "fmin": freeze_args_bin_op(Float64Array, (a, b) => {
    if (Number.isNaN(a)) return b;
    if (Number.isNaN(b)) return a;
    return Math.min(a, b);
  }),
  // "approx": ,
}


export const heaviside = Func_x1_x2_out.defaultDecorator(
  freeze_args_bin_op(Float64Array, (x1, x2) => x1 > 0 ? 1 : (x1 === 0 ? x2 : 0))
);

op_binary["↑"] = op_binary["max"];
op_binary["↓"] = op_binary["min"];
op_binary["≤"] = op_binary["leq"];
op_binary["≥"] = op_binary["geq"];
op_binary["≠"] = op_binary["neq"];


export function assign_operation(tgt: NDArray, src: ArrayOrConstant, where: Where, func): NDArray {

  if (tgt['__warnAssignment']) {
    console.warn(`Warning: You are assigning on a copy that resulted from an advanced index on a source array.\nIf this is intentional, use yourArray = source.withKwArgs({copy:true}).index(...yourIndex) to make explicit your awareness of the copy operation.\nInstead, if you want to assign to the source array, use source.op('=', other) or source.op(['::3', -1, '...', [5,4]], '*=', other).\n`);
    delete tgt['__warnAssignment'];
  }
  if (!(isarray(tgt))) {
    return _assign_operation_toJS(tgt as any, src, where, func) as any;
  }
  if (!where) {
    src = asarray(src);
    _apply_binary_operation(tgt, src, func, tgt.dtype, tgt);
  } else {
    src = asarray(src);
    let { indices } = indexes.AxesIndex.prototype.parse(tgt.shape, where);
    let tmpTgt;
    if (func == null) {
      // Small optimization: unlike "+=", "*=", etc., for "=", we don't need to reed the target
      func = (a, b) => b;
      tmpTgt = empty([indices.length], tgt.dtype);
    } else {
      tmpTgt = asarray(indices.map(i => tgt._flat[i]));
    }
    _apply_binary_operation(tmpTgt, ravel(src), func, tgt.dtype, tmpTgt);
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
  >(a: NDArray, values: NDArray, where: Where) {
    // console.log(`where=${JSON.stringify(where)}`)
    if (where.length === 1 && where[0] === false) return a;
    if (where.length === 1 && where[0] === true) where = [];
    return assign_operation(a, values, where, func);
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

export function array_equal(A, B, equal_nan = false) {
  ({ equal_nan } = Object.assign({ equal_nan }, this));
  A = asarray(A);
  B = asarray(B);
  if (A.shape.length !== B.shape.length) return false;
  for (let i = 0; i < A.shape.length; i++) if (A.shape[i] !== B.shape[i]) return false;
  const flatA = A.flat;
  const flatB = B.flat;
  if (flatA.length !== flatB.length) return false;
  for (let i = 0; i < flatA.length; i++) {
    const a = flatA[i];
    const b = flatB[i];
    if (a === b) continue;
    if (equal_nan && Number.isNaN(a) && Number.isNaN(b)) continue;
    return false;
  }
  return true;
}

export function array_equiv(A, B, equal_nan = false) {
  ({ equal_nan } = Object.assign({ equal_nan }, this));
  A = asarray(A);
  B = asarray(B);
  let shape: number[];
  let shapeA: number[];
  let shapeB: number[];
  try {
    [shape, shapeA, shapeB] = broadcast_shapes(A.shape, B.shape);
  } catch (_err) {
    return false;
  }
  const shiftsA = shape_shifts(shapeA);
  const shiftsB = shape_shifts(shapeB);
  const flatA = A.flat;
  const flatB = B.flat;
  const size = shape.reduce((a, b) => a * b, 1);
  for (let i = 0; i < size; i++) {
    let idxA = 0, idxB = 0, idx = i;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      idxA += shiftsA[axis] * (idx % shapeA[axis]);
      idxB += shiftsB[axis] * (idx % shapeB[axis]);
      idx = Math.floor(idx / shape[axis]);
    }
    const a = flatA[idxA];
    const b = flatB[idxB];
    if (a === b) continue;
    if (equal_nan && Number.isNaN(a) && Number.isNaN(b)) continue;
    return false;
  }
  return true;
}


//op_binary["≈≈"] = op[MyArray.prototype.isclose,

export const atan2 = Func_y_x_out.defaultDecorator(freeze_args_bin_op(Float64Array, Math.atan2));

export function modf(x: any) {
  x = asarray(x);
  const frac = elementwise(x, (v) => v - Math.floor(v), Float64Array);
  const integ = elementwise(x, Math.floor, Float64Array);
  return [frac, integ];
}

export function divmod(x: any, y: any) {
  const { elementwise } = require('./elementwise');
  x = asarray(x);
  y = asarray(y);
  const quotient = binary_operation(x, y, (a, b) => Math.floor(a / b), Float64Array);
  const remainder = binary_operation(x, y, (a, b) => a % b, Float64Array);
  return [quotient, remainder];
}

export function float_power(x: any, y: any, out?: any) {
  return binary_operation(x, y, Math.pow, Float64Array, out);
}

const { elementwise } = require('./elementwise');


export function ternary_operation(A: ArrayOrConstant, B: ArrayOrConstant, C: ArrayOrConstant, func: any, dtype: DType, out: NDArray | null = null): NDArray {

  if (isarray(this)) return func.bind(_NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);
  C = asarray(C);
  const [[shapeA, shapeB, shapeC], shape] = broadcast_n_shapes(A.shape, B.shape, C.shape);

  //@ts-ignore
  if (out) dtype = out.dtype;
  //@ts-ignore
  if (out == null) out = empty(shape, dtype);

  else if (!(isarray(out))) throw new Error(`Out must be of type ${_NDArray}. Got ${typeof out}`);
  // Iterate with broadcasted indices
  const flatOut = [];
  const shiftsA = shape_shifts(shapeA);
  const shiftsB = shape_shifts(shapeB);
  const shiftsC = shape_shifts(shapeC);
  const flatA = A.flat;
  const flatB = B.flat;
  const flatC = C.flat;
  for (let i = 0; i < out.size; i++) {
    let idxA = 0, idxB = 0, idxC = 0, idx = i;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      idxA += shiftsA[axis] * (idx % shapeA[axis]);
      idxB += shiftsB[axis] * (idx % shapeB[axis]);
      idxC += shiftsC[axis] * (idx % shapeC[axis]);
      idx = Math.floor(idx / shape[axis]);
    }
    flatOut.push(func(flatA[idxA], flatB[idxB], flatC[idxC]));
  };
  out.flat = new_buffer(flatOut, dtype);
  return out;
}


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


export function clip(a: any, a_min: any = null, a_max: any = null, out: any = null) {
  if (a_min === null && a_max === null) throw new Error('At least one of a_min or a_max must be provided');
  if (a_min !== null && a_max !== null) {
    return ternary_operation(a, a_min, a_max, (v, minVal, maxVal) => Math.max(minVal, Math.min(maxVal, v)), Float64Array, out);
  } else if (a_min !== null) {
    return binary_operation(a, a_min, (v, minVal) => Math.max(minVal, v), Float64Array, out);
  } else {
    return binary_operation(a, a_max, (v, maxVal) => Math.min(maxVal, v), Float64Array, out);
  }
}

export function convolve(a: any, v: any, mode: string = 'full') {
  a = asarray(a);
  v = asarray(v);
  if (a.shape.length !== 1 || v.shape.length !== 1) {
    throw new Error('convolve only supports 1D arrays');
  }
  const flatA = Array.from(a.flat as any) as number[];
  const flatV = Array.from(v.flat as any) as number[];
  const n = flatA.length + flatV.length - 1;
  const result = new Array(n).fill(0) as number[];
  for (let i = 0; i < flatA.length; i++) {
    for (let j = 0; j < flatV.length; j++) {
      result[i + j] += flatA[i] * flatV[j];
    }
  }
  if (mode === 'same') {
    const start = Math.floor(flatV.length / 2);
    return new_NDArray(new_buffer(result.slice(start, start + flatA.length), Float64Array), [flatA.length]);
  }
  if (mode === 'valid') {
    if (flatV.length > flatA.length) return new_NDArray(new_buffer([], Float64Array), [0]);
    return new_NDArray(new_buffer(result.slice(0, flatA.length - flatV.length + 1), Float64Array), [flatA.length - flatV.length + 1]);
  }
  return new_NDArray(new_buffer(result, Float64Array), [n]);
}

export function interp(x: any, xp: any, fp: any) {
  x = asarray(x);
  xp = asarray(xp);
  fp = asarray(fp);
  const flatX = Array.from(x.flat as any) as number[];
  const flatXp = Array.from(xp.flat as any) as number[];
  const flatFp = Array.from(fp.flat as any) as number[];
  const result = flatX.map((xi: number) => {
    if (xi <= flatXp[0]) return flatFp[0];
    if (xi >= flatXp[flatXp.length - 1]) return flatFp[flatFp.length - 1];
    for (let i = 0; i < flatXp.length - 1; i++) {
      if (flatXp[i] <= xi && xi <= flatXp[i + 1]) {
        const t = (xi - flatXp[i]) / (flatXp[i + 1] - flatXp[i]);
        return flatFp[i] + t * (flatFp[i + 1] - flatFp[i]);
      }
    }
    return flatFp[flatFp.length - 1];
  });
  return new_NDArray(new_buffer(result, Float64Array), x.shape);
}

export function unique(ar: any, return_index = false, return_inverse = false, return_counts = false) {
  ar = asarray(ar);
  const flatAr = Array.from(ar.flat as any).sort((a: any, b: any) => a - b);
  const unique_arr = [];
  const indices = [];
  const inverse = new Array(flatAr.length);
  const counts = [];
  let lastVal: any = Symbol('__init__');
  let count = 0;
  let idx = 0;
  for (let i = 0; i < flatAr.length; i++) {
    if (flatAr[i] !== lastVal) {
      if (count > 0) counts.push(count);
      unique_arr.push(flatAr[i]);
      indices.push(i);
      lastVal = flatAr[i];
      count = 1;
      inverse[i] = idx++;
    } else {
      inverse[i] = idx - 1;
      count++;
    }
  }
  if (count > 0) counts.push(count);

  const result: any[] = [new_NDArray(new_buffer(unique_arr, Float64Array), [unique_arr.length])];
  if (return_index) result.push(indices);
  if (return_inverse) result.push(inverse);
  if (return_counts) result.push(counts);
  return result.length === 1 ? result[0] : result;
}

export function intersect1d(...ar: any[]) {
  const arrays = ar.map(a => Array.from(asarray(a).flat as any).sort((x: any, y: any) => x - y));
  if (arrays.length === 0) return new_NDArray(new_buffer([], Float64Array), [0]);
  let result = arrays[0];
  for (let i = 1; i < arrays.length; i++) {
    result = result.filter(x => arrays[i].includes(x));
    result = Array.from(new Set(result)).sort((x: any, y: any) => x - y);
  }
  // return new_NDArray(new_buffer(result, Float64Array), [result.length]);
  throw new Error('intersect1d is not implemented yet');
}

export function union1d(...ar: any[]) {
  const all = ar.flatMap(a => Array.from(asarray(a).flat as any));
  const unique = Array.from(new Set(all)).sort((x: any, y: any) => x - y);
  // return new_NDArray(new_buffer(unique, Float64Array), [unique.length]);
  throw new Error('union1d is not implemented yet');
}

export function setdiff1d(ar1: any, ar2: any) {
  ar1 = Array.from(asarray(ar1).flat as any).sort((x: any, y: any) => x - y);
  ar2 = new Set(Array.from(asarray(ar2).flat as any));
  const result = ar1.filter((x: any) => !ar2.has(x));
  return new_NDArray(new_buffer(result, Float64Array), [result.length]);
}

export function setxor1d(ar1: any, ar2: any) {
  ar1 = Array.from(asarray(ar1).flat as any);
  ar2 = Array.from(asarray(ar2).flat as any);
  const set1 = new Set(ar1);
  const set2 = new Set(ar2);
  const result = Array.from(new Set([
    ...ar1.filter((x: any) => !set2.has(x)),
    ...ar2.filter((x: any) => !set1.has(x))
  ])).sort((x: any, y: any) => x - y);
  return new_NDArray(new_buffer(result, Float64Array), [result.length]);
}

export function isin(element: any, test_elements: any, invert = false) {
  element = asarray(element);
  test_elements = asarray(test_elements);
  const testSet = new Set(Array.from(test_elements.flat as any));
  const result = Array.from(element.flat as any).map((x: any) => {
    const isIn = testSet.has(x);
    return invert ? (isIn ? 0 : 1) : (isIn ? 1 : 0);
  });
  return new_NDArray(new_buffer(result, Uint8Array), element.shape);
}

export function lexsort(...keys: any[]) {
  const arrays = keys.map(k => asarray(k));
  const n = arrays[0].size;
  if (!arrays.every(a => a.size === n)) {
    throw new Error('lexsort: all input arrays must have the same size');
  }
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((i: number, j: number) => {
    for (let k = arrays.length - 1; k >= 0; k--) {
      const cmp = (arrays[k].flat as any)[i] - (arrays[k].flat as any)[j];
      if (cmp !== 0) return cmp;
    }
    return 0;
  });
  return new_NDArray(new_buffer(indices, Int32Array), [n]);
}

export const kw_export = {
  clip: Func_a_a_min_a_max_out.defaultDecorator(clip),
}