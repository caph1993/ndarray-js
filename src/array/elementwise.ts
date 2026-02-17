//@ts-check

import { asarray, isarray, new_NDArray } from './basic';
import NDArray from "../NDArray";
import { Func_a_out, Func_a_decimals_out, Method_out, Method_a_decimals_out } from './kwargs';
import { DType, dtype_cmp, new_buffer, float_out, bool, bool_out, DtypeResolver, HasDType } from "../dtypes";

// Here, we declare only the core functions (those that are methods)


export type ElementwiseOp = {
  (A: NDArray, out?: NDArray | DType): NDArray;
};


function applyFuncToArray_freeze(get_dtype: DtypeResolver, func): ElementwiseOp {
  return function (A: NDArray, out: NDArray | DType = null): NDArray {
    return applyFuncToArray(get_dtype, func, A, out);
  } as ElementwiseOp;
}

function applyFuncToArray(
  get_dtype: DtypeResolver,
  func,
  A: NDArray,
  out: NDArray | DType = null
) {
  A = asarray(A);
  const dtype = get_dtype([A.dtype], out);
  out = isarray(out) ? out : null;
  return _applyFuncToArray(A, func, dtype, out);
}

function _applyFuncToArray(
  A: NDArray,
  func,
  dtype: DType = null,
  out: NDArray = null,
) {
  A = asarray(A);
  //@ts-ignore
  let _out;
  if (!out) {
    let out_buffer = new_buffer(A.flat.length, dtype);
    for (let i = 0; i < A.flat.length; i++) {
      out_buffer[i] = func(A.flat[i]);
    }
    _out = new_NDArray(out_buffer, A.shape);
  } else {
    if (dtype_cmp(_out.dtype, out.dtype) > 0) {
      throw new Error(`Output array has dtype ${out.dtype}, which cannot hold all values of type ${dtype}.`);
    }
    //@ts-ignore
    out.flat = _out.flat;
    _out = out;
  }
  return _out;
}


export const funcs = {
  sign: applyFuncToArray_freeze(float_out, Math.sign),
  sqrt: applyFuncToArray_freeze(float_out, Math.sqrt),
  square: applyFuncToArray_freeze(float_out, (a) => a * a),
  exp: applyFuncToArray_freeze(float_out, Math.exp),
  log: applyFuncToArray_freeze(float_out, Math.log),
  log2: applyFuncToArray_freeze(float_out, Math.log2),
  log10: applyFuncToArray_freeze(float_out, Math.log10),
  log1p: applyFuncToArray_freeze(float_out, Math.log1p),
  sin: applyFuncToArray_freeze(float_out, Math.sin),
  cos: applyFuncToArray_freeze(float_out, Math.cos),
  tan: applyFuncToArray_freeze(float_out, Math.tan),
  asin: applyFuncToArray_freeze(float_out, Math.asin),
  acos: applyFuncToArray_freeze(float_out, Math.acos),
  atan: applyFuncToArray_freeze(float_out, Math.atan),
  cosh: applyFuncToArray_freeze(float_out, Math.cosh),
  sinh: applyFuncToArray_freeze(float_out, Math.sinh),
  tanh: applyFuncToArray_freeze(float_out, Math.tanh),
  acosh: applyFuncToArray_freeze(float_out, Math.acosh),
  asinh: applyFuncToArray_freeze(float_out, Math.asinh),
  atanh: applyFuncToArray_freeze(float_out, Math.atanh),
  floor: applyFuncToArray_freeze(float_out, Math.floor),
  ceil: applyFuncToArray_freeze(float_out, Math.ceil),
  isfinite: applyFuncToArray_freeze(bool_out, (x) => isFinite(x)),
  isinf: applyFuncToArray_freeze(bool_out, (x) => x === Infinity || x === -Infinity),
  isposinf: applyFuncToArray_freeze(bool_out, (x) => x === Infinity),
  isneginf: applyFuncToArray_freeze(bool_out, (x) => x === -Infinity),
  isnan: applyFuncToArray_freeze(bool_out, isNaN),
  iscomplex: applyFuncToArray_freeze(bool_out, (_x) => false),
  isreal: applyFuncToArray_freeze(bool_out, (_x) => true),
  reciprocal: applyFuncToArray_freeze(float_out, (x) => 1 / x),
  positive: applyFuncToArray_freeze(float_out, (x) => +x),
  angle: applyFuncToArray_freeze(float_out, (_x) => 0),
  real: applyFuncToArray_freeze(float_out, (x) => x),
  imag: applyFuncToArray_freeze(float_out, (_x) => 0),
  conj: applyFuncToArray_freeze(float_out, (x) => x),
  conjugate: applyFuncToArray_freeze(float_out, (x) => x),
  cbrt: applyFuncToArray_freeze(float_out, Math.cbrt),
  nan_to_num: applyFuncToArray_freeze(float_out, (x) => {
    if (Number.isNaN(x)) return 0;
    if (x === Infinity) return Number.MAX_VALUE;
    if (x === -Infinity) return -Number.MAX_VALUE;
    return x;
  }),
  real_if_close: applyFuncToArray_freeze(float_out, (x) => x),
  round: function round(arr: NDArray, decimals: number, out: NDArray | DType = null) {
    if (decimals == 0) applyFuncToArray(float_out, Math.round, arr, out);
    return applyFuncToArray(float_out, (x => parseFloat(x.toFixed(decimals))), arr, out);
  },
  negative: applyFuncToArray_freeze(float_out, x => -x),
  bitwise_not: applyFuncToArray_freeze(float_out, x => ~x),
  logical_not: applyFuncToArray_freeze(bool_out, x => !x),
  valueOf: applyFuncToArray_freeze(float_out, x => +x),
  abs: applyFuncToArray_freeze(float_out, Math.abs),
}

export const ops = {
  ...funcs,
  "~": funcs.bitwise_not,
  "not": funcs.logical_not,
  "+": funcs.valueOf,
  "-": funcs.negative,
}


export const kw_funcs = {
  sign: Func_a_out.defaultDecorator(funcs.sign),
  sqrt: Func_a_out.defaultDecorator(funcs.sqrt),
  square: Func_a_out.defaultDecorator(funcs.square),
  exp: Func_a_out.defaultDecorator(funcs.exp),
  log: Func_a_out.defaultDecorator(funcs.log),
  log2: Func_a_out.defaultDecorator(funcs.log2),
  log10: Func_a_out.defaultDecorator(funcs.log10),
  log1p: Func_a_out.defaultDecorator(funcs.log1p),
  sin: Func_a_out.defaultDecorator(funcs.sin),
  cos: Func_a_out.defaultDecorator(funcs.cos),
  tan: Func_a_out.defaultDecorator(funcs.tan),
  asin: Func_a_out.defaultDecorator(funcs.asin),
  acos: Func_a_out.defaultDecorator(funcs.acos),
  atan: Func_a_out.defaultDecorator(funcs.atan),
  cosh: Func_a_out.defaultDecorator(funcs.cosh),
  sinh: Func_a_out.defaultDecorator(funcs.sinh),
  tanh: Func_a_out.defaultDecorator(funcs.tanh),
  acosh: Func_a_out.defaultDecorator(funcs.acosh),
  asinh: Func_a_out.defaultDecorator(funcs.asinh),
  atanh: Func_a_out.defaultDecorator(funcs.atanh),
  floor: Func_a_out.defaultDecorator(funcs.floor),
  ceil: Func_a_out.defaultDecorator(funcs.ceil),
  negative: Func_a_out.defaultDecorator(funcs.negative),
  bitwise_not: Func_a_out.defaultDecorator(funcs.bitwise_not),
  logical_not: Func_a_out.defaultDecorator(funcs.logical_not),
  valueOf: Func_a_out.defaultDecorator(funcs.valueOf),
  abs: Func_a_out.defaultDecorator(funcs.abs),

  isfinite: Func_a_out.defaultDecorator(funcs.isfinite),
  isinf: Func_a_out.defaultDecorator(funcs.isinf),
  isposinf: Func_a_out.defaultDecorator(funcs.isposinf),
  isneginf: Func_a_out.defaultDecorator(funcs.isneginf),
  isnan: Func_a_out.defaultDecorator(funcs.isnan),
  iscomplex: Func_a_out.defaultDecorator(funcs.iscomplex),
  isreal: Func_a_out.defaultDecorator(funcs.isreal),
  reciprocal: Func_a_out.defaultDecorator(funcs.reciprocal),
  positive: Func_a_out.defaultDecorator(funcs.positive),
  angle: Func_a_out.defaultDecorator(funcs.angle),
  real: Func_a_out.defaultDecorator(funcs.real),
  imag: Func_a_out.defaultDecorator(funcs.imag),
  conj: Func_a_out.defaultDecorator(funcs.conj),
  conjugate: Func_a_out.defaultDecorator(funcs.conjugate),
  cbrt: Func_a_out.defaultDecorator(funcs.cbrt),
  nan_to_num: Func_a_out.defaultDecorator(funcs.nan_to_num),
  real_if_close: Func_a_out.defaultDecorator(funcs.real_if_close),

  round: Func_a_decimals_out.defaultDecorator(funcs.round),
}



// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.bitwise_not = Method_out.defaultDecorator(funcs.bitwise_not);
NDArray.prototype.logical_not = Method_out.defaultDecorator(funcs.logical_not);
NDArray.prototype.negative = Method_out.defaultDecorator(funcs.negative);
NDArray.prototype.abs = Method_out.defaultDecorator(funcs.abs);
NDArray.prototype.round = Method_a_decimals_out.defaultDecorator(funcs.round);