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
  let _out;
  if (!out) {
    let out_buffer = new_buffer(A.flat.length, dtype) as InstanceType<T>;
    for (let i = 0; i < A.flat.length; i++) {
      out_buffer[i] = func(A.flat[i]);
    }
    _out = new_NDArray(out_buffer, A.shape);
  } else {
    if (!dtype_leq(_out.dtype, out.dtype)) {
      throw new Error(`Output array has dtype ${out.dtype}, which cannot hold all values of type ${dtype}.`);
    }
    //@ts-ignore
    out.flat = _out.flat;
    _out = out;
  }
  return _out;
}

function elementwise_factory<
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
  sign: elementwise_factory(Math.sign, Float64Array),
  sqrt: elementwise_factory(Math.sqrt, Float64Array),
  square: elementwise_factory((a) => a * a, Float64Array),
  exp: elementwise_factory(Math.exp, Float64Array),
  log: elementwise_factory(Math.log, Float64Array),
  log2: elementwise_factory(Math.log2, Float64Array),
  log10: elementwise_factory(Math.log10, Float64Array),
  log1p: elementwise_factory(Math.log1p, Float64Array),
  sin: elementwise_factory(Math.sin, Float64Array),
  cos: elementwise_factory(Math.cos, Float64Array),
  tan: elementwise_factory(Math.tan, Float64Array),
  asin: elementwise_factory(Math.asin, Float64Array),
  acos: elementwise_factory(Math.acos, Float64Array),
  atan: elementwise_factory(Math.atan, Float64Array),
  cosh: elementwise_factory(Math.cosh, Float64Array),
  sinh: elementwise_factory(Math.sinh, Float64Array),
  tanh: elementwise_factory(Math.tanh, Float64Array),
  acosh: elementwise_factory(Math.acosh, Float64Array),
  asinh: elementwise_factory(Math.asinh, Float64Array),
  atanh: elementwise_factory(Math.atanh, Float64Array),
  floor: elementwise_factory(Math.floor, Float64Array),
  ceil: elementwise_factory(Math.ceil, Float64Array),
  isfinite: elementwise_factory((x) => isFinite(x), Uint8Array),
  isinf: elementwise_factory((x) => x === Infinity || x === -Infinity, Uint8Array),
  isposinf: elementwise_factory((x) => x === Infinity, Uint8Array),
  isneginf: elementwise_factory((x) => x === -Infinity, Uint8Array),
  isnan: elementwise_factory(isNaN, Uint8Array),
  iscomplex: elementwise_factory((_x) => false, Uint8Array),
  isreal: elementwise_factory((_x) => true, Uint8Array),
  reciprocal: elementwise_factory((x) => 1 / x, Float64Array),
  positive: elementwise_factory((x) => +x, Float64Array),
  angle: elementwise_factory((_x) => 0, Float64Array),
  real: elementwise_factory((x) => x, Float64Array),
  imag: elementwise_factory((_x) => 0, Float64Array),
  conj: elementwise_factory((x) => x, Float64Array),
  conjugate: elementwise_factory((x) => x, Float64Array),
  cbrt: elementwise_factory(Math.cbrt, Float64Array),
  nan_to_num: elementwise_factory((x) => {
    if (Number.isNaN(x)) return 0;
    if (x === Infinity) return Number.MAX_VALUE;
    if (x === -Infinity) return -Number.MAX_VALUE;
    return x;
  }, Float64Array),
  real_if_close: elementwise_factory((x) => x, Float64Array),
  round: function round(arr: NDArray, decimals: number, out: NDArray = null) {
    if (decimals == 0) elementwise(arr, Math.round, Float64Array, out);
    return elementwise(arr, x => parseFloat(x.toFixed(decimals)), Float64Array, out);
  },
  negative: elementwise_factory(x => -x, Float64Array),
  bitwise_not: elementwise_factory(x => ~x, Float64Array),
  logical_not: elementwise_factory(x => !x, Uint8Array),
  valueOf: elementwise_factory(x => +x, Float64Array),
  abs: elementwise_factory(Math.abs, Float64Array),
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
