//@ts-check

import { asarray, new_NDArray } from './basic';
import type NDArray from "../NDArray";
import { KwParser, RoundParsedKwargs, RoundSignature, UnaryOperatorParsedKwargs, UnaryOperatorMethod, kwDecorators, Func_a_out } from './kwargs';
import { ArrayOrConstant } from '../NDArray';

// Here, we declare only the core functions (those that are methods)

export function elementwise(A: NDArray, func, dtype, out: NDArray = null) {
  A = asarray(A);
  if (out) {
    out.flat = A.flat.map(func);
    return out;
  }
  return new_NDArray(A.flat.map(func), A.shape, dtype);
}

function mk_elementwise(op, dtype) {
  return function (A: NDArray, out: NDArray = null) {
    return elementwise(A, op, dtype, out);
  }
}

export const funcs = {
  sign: mk_elementwise(Math.sign, Number),
  sqrt: mk_elementwise(Math.sqrt, Number),
  square: mk_elementwise((a) => a * a, Number),
  exp: mk_elementwise(Math.exp, Number),
  log: mk_elementwise(Math.log, Number),
  log2: mk_elementwise(Math.log2, Number),
  log10: mk_elementwise(Math.log10, Number),
  log1p: mk_elementwise(Math.log1p, Number),
  sin: mk_elementwise(Math.sin, Number),
  cos: mk_elementwise(Math.cos, Number),
  tan: mk_elementwise(Math.tan, Number),
  asin: mk_elementwise(Math.asin, Number),
  acos: mk_elementwise(Math.acos, Number),
  atan: mk_elementwise(Math.atan, Number),
  cosh: mk_elementwise(Math.cosh, Number),
  sinh: mk_elementwise(Math.sinh, Number),
  tanh: mk_elementwise(Math.tanh, Number),
  acosh: mk_elementwise(Math.acosh, Number),
  asinh: mk_elementwise(Math.asinh, Number),
  atanh: mk_elementwise(Math.atanh, Number),
  floor: mk_elementwise(Math.floor, Number),
  ceil: mk_elementwise(Math.ceil, Number),
  isfinite: mk_elementwise(isFinite, Boolean),
  isnan: mk_elementwise(isNaN, Boolean),
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
  isfinite: Func_a_out.defaultDecorator(funcs.isfinite),
  isnan: Func_a_out.defaultDecorator(funcs.isnan),
}


const _ops = {
  // Unary operators:
  round: function round(arr: NDArray, decimals: number, out: NDArray = null) {
    if (decimals == 0) elementwise(arr, Math.round, Number, out);
    return elementwise(arr, x => parseFloat(x.toFixed(decimals)), Number, out);
  },
  negative: mk_elementwise(x => -x, Number),
  bitwise_not: mk_elementwise(x => ~x, Number),
  logical_not: mk_elementwise(x => !x, Boolean),
  valueOf: mk_elementwise(x => +x, Number),
  abs: mk_elementwise(Math.abs, Number),
};
export const ops = {
  ..._ops,
  "~": _ops.bitwise_not,
  "not": _ops.logical_not,
  "+": _ops.valueOf,
  "-": _ops.negative,
}

export const kw_ops = {
  bitwise_not: kwDecorators<UnaryOperatorMethod, UnaryOperatorParsedKwargs>({
    defaults: [["out", null]],
    func: ops.bitwise_not,
  }),
  logical_not: kwDecorators<UnaryOperatorMethod, UnaryOperatorParsedKwargs>({
    defaults: [["out", null]],
    func: ops.logical_not,
  }),
  negative: kwDecorators<UnaryOperatorMethod, UnaryOperatorParsedKwargs>({
    defaults: [["out", null]],
    func: ops.negative,
  }),
  abs: kwDecorators<UnaryOperatorMethod, UnaryOperatorParsedKwargs>({
    defaults: [["out", null]],
    func: ops.abs,
  }),
  round: kwDecorators<RoundSignature, RoundParsedKwargs>({
    defaults: [["decimals", 0], ["out", null]],
    func: ops.round,
  }),
}