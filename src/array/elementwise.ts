//@ts-check

import { asarray, new_NDArray } from './basic';
import type NDArray from "../NDArray";
import { Func_a_out, Func_a_decimals_out } from './kwargs';
import { TypedArrayConstructor, dtype_leq, new_buffer } from "../dtypes";

// Here, we declare only the core functions (those that are methods)

export function elementwise<
  T extends TypedArrayConstructor,
  T_a extends TypedArrayConstructor,
  T_out extends TypedArrayConstructor,
>(
  A: NDArray<T_a>,
  func,
  dtype: T,
  out: NDArray<T_out> = null,
) {
  A = asarray(A);
  //@ts-ignore
  if (out) dtype = out.dtype;
  //@ts-ignore
  const in_buffer = dtype_leq(dtype, A.dtype) ? A.flat : new_buffer(A.flat, dtype);
  const out_buffer = in_buffer.map(func) as InstanceType<T>;
  if (out) {
    //@ts-ignore
    out.flat = out_buffer;
    return out;
  }
  return new_NDArray(out_buffer, A.shape);
}

function mk_elementwise<
  T extends TypedArrayConstructor,
>(op, dtype: T) {
  return function <
    T_a extends TypedArrayConstructor,
    T_out extends TypedArrayConstructor,
  >(A: NDArray<T_a>, out: NDArray<T_out> = null) {
    return elementwise(A, op, dtype, out);
  }
}

export const funcs = {
  sign: mk_elementwise(Math.sign, Float64Array),
  sqrt: mk_elementwise(Math.sqrt, Float64Array),
  square: mk_elementwise((a) => a * a, Float64Array),
  exp: mk_elementwise(Math.exp, Float64Array),
  log: mk_elementwise(Math.log, Float64Array),
  log2: mk_elementwise(Math.log2, Float64Array),
  log10: mk_elementwise(Math.log10, Float64Array),
  log1p: mk_elementwise(Math.log1p, Float64Array),
  sin: mk_elementwise(Math.sin, Float64Array),
  cos: mk_elementwise(Math.cos, Float64Array),
  tan: mk_elementwise(Math.tan, Float64Array),
  asin: mk_elementwise(Math.asin, Float64Array),
  acos: mk_elementwise(Math.acos, Float64Array),
  atan: mk_elementwise(Math.atan, Float64Array),
  cosh: mk_elementwise(Math.cosh, Float64Array),
  sinh: mk_elementwise(Math.sinh, Float64Array),
  tanh: mk_elementwise(Math.tanh, Float64Array),
  acosh: mk_elementwise(Math.acosh, Float64Array),
  asinh: mk_elementwise(Math.asinh, Float64Array),
  atanh: mk_elementwise(Math.atanh, Float64Array),
  floor: mk_elementwise(Math.floor, Float64Array),
  ceil: mk_elementwise(Math.ceil, Float64Array),
  isfinite: mk_elementwise(isFinite, Uint8Array),
  isnan: mk_elementwise(isNaN, Uint8Array),
  round: function round(arr: NDArray, decimals: number, out: NDArray = null) {
    if (decimals == 0) elementwise(arr, Math.round, Float64Array, out);
    return elementwise(arr, x => parseFloat(x.toFixed(decimals)), Float64Array, out);
  },
  negative: mk_elementwise(x => -x, Float64Array),
  bitwise_not: mk_elementwise(x => ~x, Float64Array),
  logical_not: mk_elementwise(x => !x, Uint8Array),
  valueOf: mk_elementwise(x => +x, Float64Array),
  abs: mk_elementwise(Math.abs, Float64Array),
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
  isnan: Func_a_out.defaultDecorator(funcs.isnan),

  round: Func_a_decimals_out.defaultDecorator(funcs.round),
}
