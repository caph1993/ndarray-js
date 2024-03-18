//@ts-check

import { asarray, new_NDArray } from './basic';
import type NDArray from "../NDArray-class";
import { KwParser, RoundParsedKwargs, RoundSignature } from './kwargs';

// Here, we declare only the core functions (those that are methods)

export function elementwise(A: NDArray, func, dtype) {
  A = asarray(A);
  return new_NDArray(A.flat.map(func), A.shape, dtype);
}

export function round(A: NDArray, decimals: number) {
  if (decimals == 0) elementwise(A, Math.round, Number);
  return elementwise(A, x => parseFloat(x.toFixed(decimals)), Number);
};
export const round_kw = new KwParser<RoundSignature, RoundParsedKwargs>([["decimals", 0]]).decorators(round)

export function bitwise_not(A: NDArray) {
  return elementwise(A, x => ~x, Number);
};
export function logical_not(A: NDArray) {
  return elementwise(A, x => !x, Boolean);
};



export function __make_elementwise(func, dtype = Number) {
  return function (A: NDArray) {
    return elementwise(A, func, dtype);
  }
}

export const ops = {
  sign: __make_elementwise(Math.sign),
  sqrt: __make_elementwise(Math.sqrt),
  square: __make_elementwise((a) => a * a),
  abs: __make_elementwise(Math.abs),
  exp: __make_elementwise(Math.exp),
  log: __make_elementwise(Math.log),
  log2: __make_elementwise(Math.log2),
  log10: __make_elementwise(Math.log10),
  log1p: __make_elementwise(Math.log1p),
  sin: __make_elementwise(Math.sin),
  cos: __make_elementwise(Math.cos),
  tan: __make_elementwise(Math.tan),
  asin: __make_elementwise(Math.asin),
  acos: __make_elementwise(Math.acos),
  atan: __make_elementwise(Math.atan),
  atan2: __make_elementwise(Math.atan2),
  cosh: __make_elementwise(Math.cosh),
  sinh: __make_elementwise(Math.sinh),
  tanh: __make_elementwise(Math.tanh),
  acosh: __make_elementwise(Math.acosh),
  asinh: __make_elementwise(Math.asinh),
  atanh: __make_elementwise(Math.atanh),
  round: round,
}