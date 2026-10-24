//@ts-check

import { isarray } from '../NDArray';
import { asarray } from '../NDArray';
import NDArray from "../NDArray";

import {
  DType, new_buffer, float_out, bool_out, bool, DtypeResolver, same_out, numeric_out, integer_out, sum_out,
  dtype_is_float, dtype_is_integer, float32, float64, int32, dtype_max, truthy
} from "../dtypes";
import { finish_output, apply_binary_operation } from './operators';


export type ElementwiseOp = {
  (A: NDArray, out?: NDArray | DType): NDArray;
};

/**
 * Applies `func` to every element of `A`.
 * The dtype of the result is given by `get_dtype`. If `out` is an NDArray the result is stored there.
 */
export function applyFuncToArray(
  get_dtype: DtypeResolver,
  func: (x: any) => any,
  A: NDArray,
  out: NDArray | DType = null
): NDArray {
  A = asarray(A);
  const dtype = get_dtype([A.dtype], out);
  const flat = A.flat;
  const n = flat.length;
  const buffer = new_buffer(n, dtype);
  if (dtype === bool) for (let i = 0; i < n; i++) buffer[i] = truthy(func(flat[i])) ? 1 : 0;
  else for (let i = 0; i < n; i++) buffer[i] = func(flat[i]);
  return finish_output(buffer, A.shape, dtype, out);
}

function applyFuncToArray_freeze(get_dtype: DtypeResolver, func): ElementwiseOp {
  return function (A: NDArray, out: NDArray | DType = null): NDArray {
    return applyFuncToArray(get_dtype, func, A, out);
  } as ElementwiseOp;
}

/** Round half to even (like numpy.rint) */
export function rint_scalar(x: number) {
  if (!Number.isFinite(x)) return x;
  if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2);
  return Math.round(x);
}

export const sign = applyFuncToArray_freeze(numeric_out, Math.sign);
export const sqrt = applyFuncToArray_freeze(float_out, Math.sqrt);
export const square = applyFuncToArray_freeze(sum_out, (a) => a * a);
export const exp = applyFuncToArray_freeze(float_out, Math.exp);
export const expm1 = applyFuncToArray_freeze(float_out, Math.expm1);
export const exp2 = applyFuncToArray_freeze(float_out, (x) => Math.pow(2, x));
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
export const degrees = applyFuncToArray_freeze(float_out, (x) => x * 180 / Math.PI);
export const radians = applyFuncToArray_freeze(float_out, (x) => x * Math.PI / 180);
export const deg2rad = radians;
export const rad2deg = degrees;
export const floor = applyFuncToArray_freeze(same_out, Math.floor);
export const ceil = applyFuncToArray_freeze(same_out, Math.ceil);
export const trunc = applyFuncToArray_freeze(same_out, Math.trunc);
export const fix = trunc;
export const rint = applyFuncToArray_freeze(float_out, rint_scalar);
export const isfinite = applyFuncToArray_freeze(bool_out, (x) => Number.isFinite(+x));
export const isinf = applyFuncToArray_freeze(bool_out, (x) => x === Infinity || x === -Infinity);
export const isposinf = applyFuncToArray_freeze(bool_out, (x) => x === Infinity);
export const isneginf = applyFuncToArray_freeze(bool_out, (x) => x === -Infinity);
export const isnan = applyFuncToArray_freeze(bool_out, (x) => Number.isNaN(x));
export const signbit = applyFuncToArray_freeze(bool_out, (x) => x < 0 || Object.is(x, -0));
export const iscomplex = applyFuncToArray_freeze(bool_out, (_x) => false);
export const isreal = applyFuncToArray_freeze(bool_out, (_x) => true);
export const reciprocal = applyFuncToArray_freeze(same_out, (x) => 1 / x);
export const positive = applyFuncToArray_freeze(numeric_out, (x) => +x);
export const angle = applyFuncToArray_freeze(float_out, (x) => Number.isNaN(x) ? NaN : (x < 0 ? Math.PI : 0));
export const real = applyFuncToArray_freeze(same_out, (x) => x);
export const imag = applyFuncToArray_freeze(same_out, (_x) => 0);
export const conj = applyFuncToArray_freeze(same_out, (x) => x);
export const conjugate = conj;
export const cbrt = applyFuncToArray_freeze(float_out, Math.cbrt);
export const real_if_close = applyFuncToArray_freeze(same_out, (x) => x);
export const negative = applyFuncToArray_freeze(numeric_out, x => -x);
const _bitwise_not = applyFuncToArray_freeze(integer_out, x => ~x);
/** ~x for integers; logical not for bool arrays (as numpy) */
export const bitwise_not: ElementwiseOp = function (A: NDArray, out: NDArray | DType = null) {
  A = asarray(A);
  if (A.dtype === bool) return applyFuncToArray(bool_out, (x) => !truthy(x), A, out);
  return _bitwise_not(A, out);
} as ElementwiseOp;
export const logical_not = applyFuncToArray_freeze(bool_out, x => !truthy(x));
export const valueOf = applyFuncToArray_freeze(same_out, x => +x);
export const abs = applyFuncToArray_freeze(same_out, Math.abs);

// sinc(x) = sin(pi x) / (pi x)
export const sinc = applyFuncToArray_freeze(float_out, (x) => {
  if (x === 0) return 1;
  const y = Math.PI * x;
  return Math.sin(y) / y;
});

// Modified Bessel function of order 0 (polynomial approximation of Abramowitz & Stegun + series for exactness)
export const i0 = applyFuncToArray_freeze(float_out, (x) => {
  if (!Number.isFinite(x)) return NaN; // as numpy
  x = Math.abs(x);
  // Power series: sum_k ((x/2)^k / k!)^2. Converges for all x; terms are positive.
  let term = 1, sum = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 500; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
});

/** Moves a float64 to the next/previous representable value in bit-pattern order (away from/towards zero for positive x) */
function f64_step(x: number, dir: 1 | -1): number {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  let hi = view.getUint32(0), lo = view.getUint32(4);
  if (dir > 0) { lo = (lo + 1) >>> 0; if (lo === 0) hi = (hi + 1) >>> 0; }
  else { if (lo === 0) hi = (hi - 1) >>> 0; lo = (lo - 1) >>> 0; }
  view.setUint32(0, hi); view.setUint32(4, lo);
  return view.getFloat64(0);
}

// Spacing between x and the next float of larger magnitude, as numpy.spacing
function spacing_scalar(x: number, dtype: DType) {
  if (!Number.isFinite(x)) return NaN;
  if (dtype === float32) {
    const f = Math.fround(Math.abs(x));
    const buf = new Float32Array([f]);
    const bits = new Int32Array(buf.buffer);
    bits[0] += 1;
    const next = buf[0];
    return x < 0 ? -(next - f) : next - f;
  }
  const next = f64_step(Math.abs(x), 1);
  const d = next - Math.abs(x);
  return x < 0 ? -d : d;
}
export function spacing(A: NDArray, out: NDArray | DType = null): NDArray {
  A = asarray(A);
  const dtype = float_out([A.dtype], out);
  return applyFuncToArray(float_out, (x) => spacing_scalar(x, dtype), A, out);
}

export function nextafter_scalar(x: number, y: number, dtype: DType): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (x === y) return y;
  if (x === 0) {
    const tiny = dtype === float32 ? 1.401298464324817e-45 : 5e-324;
    return y > 0 ? tiny : -tiny;
  }
  if (dtype === float32) {
    const buf = new Float32Array([x]);
    const bits = new Int32Array(buf.buffer);
    bits[0] += ((y > x) === (x > 0)) ? 1 : -1;
    return buf[0];
  }
  return f64_step(x, ((y > x) === (x > 0)) ? 1 : -1);
}

export function round(arr: NDArray, decimals: number = 0, out: NDArray | DType = null) {
  arr = asarray(arr);
  if (dtype_is_integer(arr.dtype) || arr.dtype === bool) {
    // Integers are unchanged for decimals >= 0
    if (decimals >= 0) return applyFuncToArray(same_out, (x) => x, arr, out);
    const f = Math.pow(10, -decimals);
    return applyFuncToArray(same_out, (x) => rint_scalar(x / f) * f, arr, out);
  }
  if (decimals === 0) return applyFuncToArray(same_out, rint_scalar, arr, out);
  if (decimals > 0) {
    const f = Math.pow(10, decimals);
    return applyFuncToArray(same_out, (x) => rint_scalar(x * f) / f, arr, out);
  }
  const f = Math.pow(10, -decimals);
  return applyFuncToArray(same_out, (x) => rint_scalar(x / f) * f, arr, out);
};

export function nan_to_num(a: NDArray, nan: number = 0, posinf: number | null = null, neginf: number | null = null, out: NDArray | DType = null) {
  a = asarray(a);
  const big = a.dtype === float32 ? 3.4028234663852886e+38 : Number.MAX_VALUE;
  const pos = posinf === null ? big : posinf;
  const neg = neginf === null ? -big : neginf;
  return applyFuncToArray(same_out, (x) => {
    if (Number.isNaN(x)) return nan;
    if (x === Infinity) return pos;
    if (x === -Infinity) return neg;
    return x;
  }, a, out);
}

/** modf: returns [fractional part, integral part] */
export function modf(a: NDArray): [NDArray, NDArray] {
  a = asarray(a);
  const ip = applyFuncToArray(float_out, (x) => Math.trunc(x), a);
  const fp = applyFuncToArray(float_out, (x) => Number.isFinite(x) ? x - Math.trunc(x) : (Number.isNaN(x) ? NaN : 0), a);
  return [fp, ip];
}

/** frexp: returns [mantissa, exponent] with x = mantissa * 2**exponent */
export function frexp(a: NDArray): [NDArray, NDArray] {
  a = asarray(a);
  const split = (x: number): [number, number] => {
    if (x === 0 || !Number.isFinite(x)) return [x, 0];
    let e = Math.max(-1023, Math.floor(Math.log2(Math.abs(x))) + 1);
    let m = x * Math.pow(2, -e);
    // Fix rounding in log2
    while (Math.abs(m) < 0.5) { m *= 2; e--; }
    while (Math.abs(m) >= 1) { m /= 2; e++; }
    return [m, e];
  };
  const mant = applyFuncToArray(float_out, (x) => split(x)[0], a);
  const exps = applyFuncToArray(() => int32, (x) => split(x)[1], a);
  return [mant, exps];
}

export const funcs = {
}

export const ops = {
  ...funcs,
  "~": bitwise_not,
  "not": logical_not,
  "+": valueOf,
  "-": negative,
  round,
}

/** Next representable floating-point value after x towards y */
export function nextafter(A: NDArray, B: NDArray, out: NDArray | DType = null): NDArray {
  A = asarray(A); B = asarray(B);
  const dtype = float_out([A.dtype, B.dtype]);
  return apply_binary_operation(float_out, (x, y) => nextafter_scalar(x, y, dtype), A, B, out);
}
