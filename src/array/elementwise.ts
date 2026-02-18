//@ts-check

import { isarray } from '../NDArray';
import { asarray } from '../NDArray';
import NDArray from "../NDArray";

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
    _out = new NDArray(out_buffer, A.shape);
  } else {
    if (dtype_cmp(_out.dtype, out.dtype) > 0) {
      throw new Error(`Output array has dtype ${out.dtype}, which cannot hold all values of type ${dtype}.`);
    }
    //@ts-ignore
    out.flat = _out.flat;
    _out = out;
  }
  return _out.item_if_scalar() as NDArray;
}


export const sign = applyFuncToArray_freeze(float_out, Math.sign);
export const sqrt = applyFuncToArray_freeze(float_out, Math.sqrt);
export const square = applyFuncToArray_freeze(float_out, (a) => a * a);
export const exp = applyFuncToArray_freeze(float_out, Math.exp);
export const log = applyFuncToArray_freeze(float_out, Math.log);
export const log2 = applyFuncToArray_freeze(float_out, Math.log2);
export const log10 = applyFuncToArray_freeze(float_out, Math.log10);
export const log1p = applyFuncToArray_freeze(float_out, Math.log1p);
export const sin = applyFuncToArray_freeze(float_out, Math.sin);
export const cos = applyFuncToArray_freeze(float_out, Math.cos);
export const tan = applyFuncToArray_freeze(float_out, Math.tan);
export const asin = applyFuncToArray_freeze(float_out, Math.asin);
export const acos = applyFuncToArray_freeze(float_out, Math.acos);
export const atan = applyFuncToArray_freeze(float_out, Math.atan);
export const cosh = applyFuncToArray_freeze(float_out, Math.cosh);
export const sinh = applyFuncToArray_freeze(float_out, Math.sinh);
export const tanh = applyFuncToArray_freeze(float_out, Math.tanh);
export const acosh = applyFuncToArray_freeze(float_out, Math.acosh);
export const asinh = applyFuncToArray_freeze(float_out, Math.asinh);
export const atanh = applyFuncToArray_freeze(float_out, Math.atanh);
export const floor = applyFuncToArray_freeze(float_out, Math.floor);
export const ceil = applyFuncToArray_freeze(float_out, Math.ceil);
export const isfinite = applyFuncToArray_freeze(bool_out, (x) => isFinite(x));
export const isinf = applyFuncToArray_freeze(bool_out, (x) => x === Infinity || x === -Infinity);
export const isposinf = applyFuncToArray_freeze(bool_out, (x) => x === Infinity);
export const isneginf = applyFuncToArray_freeze(bool_out, (x) => x === -Infinity);
export const isnan = applyFuncToArray_freeze(bool_out, isNaN);
export const iscomplex = applyFuncToArray_freeze(bool_out, (_x) => false);
export const isreal = applyFuncToArray_freeze(bool_out, (_x) => true);
export const reciprocal = applyFuncToArray_freeze(float_out, (x) => 1 / x);
export const positive = applyFuncToArray_freeze(float_out, (x) => +x);
export const angle = applyFuncToArray_freeze(float_out, (_x) => 0);
export const real = applyFuncToArray_freeze(float_out, (x) => x);
export const imag = applyFuncToArray_freeze(float_out, (_x) => 0);
export const conj = applyFuncToArray_freeze(float_out, (x) => x);
export const conjugate = applyFuncToArray_freeze(float_out, (x) => x);
export const cbrt = applyFuncToArray_freeze(float_out, Math.cbrt);
export const nan_to_num = applyFuncToArray_freeze(float_out, (x) => {
  if (Number.isNaN(x)) return 0;
  if (x === Infinity) return Number.MAX_VALUE;
  if (x === -Infinity) return -Number.MAX_VALUE;
  return x;
});
export const real_if_close = applyFuncToArray_freeze(float_out, (x) => x);
export const round = function round(arr: NDArray, decimals: number, out: NDArray | DType = null) {
  if (decimals == 0) applyFuncToArray(float_out, Math.round, arr, out);
  return applyFuncToArray(float_out, (x => parseFloat(x.toFixed(decimals))), arr, out);
};
export const negative = applyFuncToArray_freeze(float_out, x => -x);
export const bitwise_not = applyFuncToArray_freeze(float_out, x => ~x);
export const logical_not = applyFuncToArray_freeze(bool_out, x => !x);
export const valueOf = applyFuncToArray_freeze(float_out, x => +x);
export const abs = applyFuncToArray_freeze(float_out, Math.abs);

export const funcs = {
}

export const ops = {
  ...funcs,
  "~": bitwise_not,
  "not": logical_not,
  "+": valueOf,
  "-": negative,
}

