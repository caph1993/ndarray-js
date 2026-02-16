//@ts-check
import * as indexes from './indexes';
import { isarray, asarray, _NDArray, empty, ravel, shape_shifts, new_NDArray } from './basic';
import { addition_out, bitwise_out, bool, bool_out, DType, DtypeResolver, float_out } from '../dtypes';
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




function freeze_args_bin_op(dtype_resolver: DtypeResolver, func): BinaryOperator {
  return function (A: NDArray, B: NDArray, out: NDArray | DType = null): NDArray {
    return apply_binary_operation(dtype_resolver, func, A, B, out);
  };
}

function apply_binary_operation(
  dtype_resolver: DtypeResolver,
  func,
  A: NDArray,
  B: NDArray,
  out: NDArray | DType = null
) {
  if (isarray(this)) return func.bind(_NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);

  const [shape, shapeA, shapeB] = broadcast_shapes(A.shape, B.shape);


  const dtype = dtype_resolver([A.dtype, B.dtype], out);
  if (!isarray(out)) {
    out = empty(shape, dtype);
  }
  else if (out) throw new Error(`Out must be of type ${_NDArray}. Got ${typeof out}`);

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
  console.log(`flatOut=${flatOut}. len=${flatOut.length}, out.size=${out.size}, dtype=${dtype.name}`)

  console.log(new_buffer(flatOut, dtype));

  out.flat = new_buffer(flatOut, dtype);
  return out;
}



export const op_binary = {

  "+": freeze_args_bin_op(addition_out, (a, b) => a + b),
  "-": freeze_args_bin_op(addition_out, (a, b) => a - b),
  "*": freeze_args_bin_op(addition_out, (a, b) => a * b),
  "/": freeze_args_bin_op(addition_out, (a, b) => a / b),
  "%": freeze_args_bin_op(addition_out, (a, b) => (a % b)),
  "|": freeze_args_bin_op(addition_out, (a, b) => a | b),
  "&": freeze_args_bin_op(bitwise_out, (a, b) => a & b),
  "^": freeze_args_bin_op(bitwise_out, (a, b) => a ^ b),
  "<<": freeze_args_bin_op(bitwise_out, (a, b) => a << b),
  ">>": freeze_args_bin_op(bitwise_out, (a, b) => a >> b),
  "**": freeze_args_bin_op(addition_out, (a, b) => Math.pow(a, b)),
  "//": freeze_args_bin_op(addition_out, (a, b) => Math.floor(a / b)),

  "<": freeze_args_bin_op(bool_out, (a, b) => a < b),
  ">": freeze_args_bin_op(bool_out, (a, b) => a > b),
  ">=": freeze_args_bin_op(bool_out, (a, b) => a >= b),
  "<=": freeze_args_bin_op(bool_out, (a, b) => a <= b),
  "==": freeze_args_bin_op(bool_out, (a, b) => a == b),
  "!=": freeze_args_bin_op(bool_out, (a, b) => a != b),
  // Operators with custom ascii identifiers:
  "or": freeze_args_bin_op(bool_out, (a, b) => a || b),
  "and": freeze_args_bin_op(bool_out, (a, b) => a && b),
  "xor": freeze_args_bin_op(bool_out, (a, b) => (!a) != (!b)),
  "max": freeze_args_bin_op(bitwise_out, (a, b) => Math.max(a, b)),
  "min": freeze_args_bin_op(bitwise_out, (a, b) => Math.min(a, b)),
  "hypot": freeze_args_bin_op(float_out, Math.hypot),
  "fmax": freeze_args_bin_op(bitwise_out, (a, b) => {
    if (Number.isNaN(a)) return b;
    if (Number.isNaN(b)) return a;
    return Math.max(a, b);
  }),
  "fmin": freeze_args_bin_op(bitwise_out, (a, b) => {
    if (Number.isNaN(a)) return b;
    if (Number.isNaN(b)) return a;
    return Math.min(a, b);
  }),
  // "approx": ,
}


export const heaviside = Func_x1_x2_out.defaultDecorator(
  freeze_args_bin_op(float_out, (x1, x2) => x1 > 0 ? 1 : (x1 === 0 ? x2 : 0))
);

op_binary["↑"] = op_binary["max"];
op_binary["↓"] = op_binary["min"];
op_binary["≤"] = op_binary["leq"];
op_binary["≥"] = op_binary["geq"];
op_binary["≠"] = op_binary["neq"];
op_binary["divide"] = op_binary["/"];
op_binary["true_divide"] = op_binary["/"];


export function assign_operation(dtype_resolver: DtypeResolver, func, tgt: NDArray, src: ArrayOrConstant, where: Where): NDArray {

  if (tgt['__warnAssignment']) {
    console.warn(`Warning: You are assigning on a copy that resulted from an advanced index on a source array.\nIf this is intentional, use yourArray = source.withKwArgs({copy:true}).index(...yourIndex) to make explicit your awareness of the copy operation.\nInstead, if you want to assign to the source array, use source.op('=', other) or source.op(['::3', -1, '...', [5,4]], '*=', other).\n`);
    delete tgt['__warnAssignment'];
  }

  if (!(isarray(tgt))) {
    //Verify the dtype anyway
    return _assign_operation_toJS(dtype_resolver, func, tgt as any, src, where) as any;
  }
  if (!where) {
    src = asarray(src);
    apply_binary_operation(dtype_resolver, func, tgt, src, tgt);
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
    apply_binary_operation(dtype_resolver, func, tmpTgt, ravel(src), tmpTgt);
    for (let i in indices) tgt._flat[indices[i]] = tmpTgt._flat[i];
  }
  return tgt;
};



export function _assign_operation_toJS(dtype_resolver: DtypeResolver, func: any, tgtJS: any[], src: any, where: Where) {
  if (!Array.isArray(tgtJS)) throw new Error(`Can not assign to a non-array. Found ${typeof tgtJS}: ${tgtJS}`);
  console.warn('Assignment to JS array is experimental and slow.')
  // Parse the whole array
  const cpy = asarray(tgtJS);
  assign_operation(dtype_resolver, func, cpy, src, where);
  // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
  const outJS = tolist(cpy);
  while (tgtJS.length) tgtJS.pop();
  // @ts-ignore
  extend(tgtJS, outJS);
  return tgtJS;
}


// export type AssignmentOperator = { (tgt: NDArray, src: ArrayOrConstant): NDArray; (tgt: NDArray, where: Where, src: ArrayOrConstant): NDArray; }
// export type SelfAssignmentOperator = { (other: ArrayOrConstant): NDArray; (where: Where, other: ArrayOrConstant): NDArray; }


export function __make_assignment_operator(dtype_resolver: DtypeResolver, func) {
  return function operator(a: NDArray, values: NDArray, where: Where) {
    // console.log(`where=${JSON.stringify(where)}`)
    if (where.length === 1 && where[0] === false) return a;
    if (where.length === 1 && where[0] === true) where = [];
    return assign_operation(dtype_resolver, func, a, values, where);
  }
}

export const op_assign = {
  "=": __make_assignment_operator(bitwise_out, (a, b) => b),
  "+=": __make_assignment_operator(bitwise_out, (a, b) => a + b),
  "-=": __make_assignment_operator(bitwise_out, (a, b) => a - b),
  "*=": __make_assignment_operator(bitwise_out, (a, b) => a * b),
  "/=": __make_assignment_operator(float_out, (a, b) => a / b),
  "%=": __make_assignment_operator(bitwise_out, (a, b) => (a % b)),
  "//=": __make_assignment_operator(bitwise_out, (a, b) => Math.floor(a / b)),
  "**=": __make_assignment_operator(bitwise_out, (a, b) => Math.pow(a, b)),
  "|=": __make_assignment_operator(bitwise_out, (a, b) => a | b),
  "&=": __make_assignment_operator(bitwise_out, (a, b) => a & b),
  "^=": __make_assignment_operator(bitwise_out, (a, b) => a ^ b),
  "<<=": __make_assignment_operator(bitwise_out, (a, b) => a << b),
  ">>=": __make_assignment_operator(bitwise_out, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": __make_assignment_operator(bitwise_out, (a, b) => Math.max(a, b)),
  "min=": __make_assignment_operator(bitwise_out, (a, b) => Math.min(a, b)),
  "or=": __make_assignment_operator(bool_out, (a, b) => a || b),
  "and=": __make_assignment_operator(bool_out, (a, b) => a && b),
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
  return apply_binary_operation(bool_out, func, A, B)
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
  try { apply_binary_operation(bool_out, wrapper, A, B) }
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

export const atan2 = Func_y_x_out.defaultDecorator(freeze_args_bin_op(float_out, Math.atan2));

export function divmod(x: any, y: any) {
  x = asarray(x);
  y = asarray(y);
  const quotient = apply_binary_operation(float_out, ((a, b) => Math.floor(a / b)), x, y);
  const remainder = apply_binary_operation(float_out, ((a, b) => a % b), x, y);
  return [quotient, remainder];
}

export function float_power(x: any, y: any, out?: any) {
  return apply_binary_operation(float_out, Math.pow, x, y, out);
}


export function ternary_operation(dtype_resolver: DtypeResolver, func, A: ArrayOrConstant, B: ArrayOrConstant, C: ArrayOrConstant, out: NDArray | DType | null = null): NDArray {

  if (isarray(this)) return func.bind(_NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);
  C = asarray(C);
  const dtype = dtype_resolver([A.dtype, B.dtype, C.dtype], out);
  out = isarray(out) ? out : null;

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


export function clip(a: NDArray, a_min: NDArray = null, a_max: NDArray = null, out: any = null) {
  if (a_min === null && a_max === null) throw new Error('At least one of a_min or a_max must be provided');
  if (a_min !== null && a_max !== null) {
    const func = (v, minVal, maxVal) => Math.max(minVal, Math.min(maxVal, v))
    return ternary_operation(bitwise_out, func, a, a_min, a_max, out);
  } else if (a_min !== null) {
    return apply_binary_operation(bitwise_out, (v, minVal) => Math.max(minVal, v), a, a_min, out);
  } else {
    return apply_binary_operation(bitwise_out, (v, maxVal) => Math.min(maxVal, v), a, a_max, out);
  }
}

export const kw_export = {
  clip: Func_a_a_min_a_max_out.defaultDecorator(clip),
}