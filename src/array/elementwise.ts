//@ts-check

import { asarray, isarray, new_NDArray } from './basic';
import type NDArray from "../NDArray";
import { Func_a_out, Func_a_decimals_out } from './kwargs';
import { DType, dtype_cmp, new_buffer, float_out, bool, bool_out, DtypeResolver, HasDType } from "../dtypes";

// Here, we declare only the core functions (those that are methods)


export type ElementwiseOp = {
  (A: NDArray, out?: NDArray | DType): NDArray;
};


function applyFuncToArray_freeze(func, get_dtype: DtypeResolver): ElementwiseOp {
  return function (A: NDArray, out: NDArray | DType = null): NDArray {
    return applyFuncToArray(func, get_dtype, A, out);
  } as ElementwiseOp;
}

function applyFuncToArray(
  func,
  get_dtype: DtypeResolver,
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
  sign: applyFuncToArray_freeze(Math.sign, float_out),
  sqrt: applyFuncToArray_freeze(Math.sqrt, float_out),
  square: applyFuncToArray_freeze((a) => a * a, float_out),
  exp: applyFuncToArray_freeze(Math.exp, float_out),
  log: applyFuncToArray_freeze(Math.log, float_out),
  log2: applyFuncToArray_freeze(Math.log2, float_out),
  log10: applyFuncToArray_freeze(Math.log10, float_out),
  log1p: applyFuncToArray_freeze(Math.log1p, float_out),
  sin: applyFuncToArray_freeze(Math.sin, float_out),
  cos: applyFuncToArray_freeze(Math.cos, float_out),
  tan: applyFuncToArray_freeze(Math.tan, float_out),
  asin: applyFuncToArray_freeze(Math.asin, float_out),
  acos: applyFuncToArray_freeze(Math.acos, float_out),
  atan: applyFuncToArray_freeze(Math.atan, float_out),
  cosh: applyFuncToArray_freeze(Math.cosh, float_out),
  sinh: applyFuncToArray_freeze(Math.sinh, float_out),
  tanh: applyFuncToArray_freeze(Math.tanh, float_out),
  acosh: applyFuncToArray_freeze(Math.acosh, float_out),
  asinh: applyFuncToArray_freeze(Math.asinh, float_out),
  atanh: applyFuncToArray_freeze(Math.atanh, float_out),
  floor: applyFuncToArray_freeze(Math.floor, float_out),
  ceil: applyFuncToArray_freeze(Math.ceil, float_out),
  isfinite: applyFuncToArray_freeze((x) => isFinite(x), bool_out),
  isinf: applyFuncToArray_freeze((x) => x === Infinity || x === -Infinity, bool_out),
  isposinf: applyFuncToArray_freeze((x) => x === Infinity, bool_out),
  isneginf: applyFuncToArray_freeze((x) => x === -Infinity, bool_out),
  isnan: applyFuncToArray_freeze(isNaN, bool_out),
  iscomplex: applyFuncToArray_freeze((_x) => false, bool_out),
  isreal: applyFuncToArray_freeze((_x) => true, bool_out),
  reciprocal: applyFuncToArray_freeze((x) => 1 / x, float_out),
  positive: applyFuncToArray_freeze((x) => +x, float_out),
  angle: applyFuncToArray_freeze((_x) => 0, float_out),
  real: applyFuncToArray_freeze((x) => x, float_out),
  imag: applyFuncToArray_freeze((_x) => 0, float_out),
  conj: applyFuncToArray_freeze((x) => x, float_out),
  conjugate: applyFuncToArray_freeze((x) => x, float_out),
  cbrt: applyFuncToArray_freeze(Math.cbrt, float_out),
  nan_to_num: applyFuncToArray_freeze((x) => {
    if (Number.isNaN(x)) return 0;
    if (x === Infinity) return Number.MAX_VALUE;
    if (x === -Infinity) return -Number.MAX_VALUE;
    return x;
  }, float_out),
  real_if_close: applyFuncToArray_freeze((x) => x, float_out),
  round: function round(arr: NDArray, decimals: number, out: NDArray | DType = null) {
    if (decimals == 0) applyFuncToArray(Math.round, float_out, arr, out);
    return applyFuncToArray(x => parseFloat(x.toFixed(decimals)), float_out, arr, out);
  },
  negative: applyFuncToArray_freeze(x => -x, float_out),
  bitwise_not: applyFuncToArray_freeze(x => ~x, float_out),
  logical_not: applyFuncToArray_freeze(x => !x, bool_out),
  valueOf: applyFuncToArray_freeze(x => +x, float_out),
  abs: applyFuncToArray_freeze(Math.abs, float_out),
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
