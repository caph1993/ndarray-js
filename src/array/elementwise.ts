import { is_cx } from '../dtypes';
//@ts-check

import { isarray } from '../NDArray';
import { asarray } from '../NDArray';
import NDArray from "../NDArray";

import {
  DType, new_buffer, float_out, bool_out, bool, DtypeResolver, same_out, numeric_out, integer_out, sum_out,
  dtype_is_float, dtype_is_integer, float32, float64, int32, dtype_max, truthy, unary_float_out, check_out, get_out_dtype, int8, TYPE_BY_NAME
} from "../dtypes";
import { finish_output, apply_binary_operation, operand_dtypes } from './operators';
import { Complex, is_complex } from '../complex';
import { complex128, realpart_out, convert_buffer } from '../dtypes';


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
  out: NDArray | DType = null,
  cfunc?: (z: Complex) => any,
): NDArray {
  A = asarray(A);
  const dtype = get_dtype([A.dtype], out);
  let flat = A.flat as any;
  if (is_cx(A.dtype)) {
    if (!cfunc) throw new Error(`ufunc not supported for the input types (complex128)`);
    func = cfunc;
  } else if (is_cx(dtype) && cfunc) {
    // real input, complex result (e.g. functions defined on the complex plane only)
    flat = convert_buffer(flat, complex128);
    func = cfunc;
  }
  const n = flat.length;
  const buffer = new_buffer(n, dtype);
  if (dtype === bool) for (let i = 0; i < n; i++) buffer[i] = truthy(func(flat[i])) ? 1 : 0;
  else for (let i = 0; i < n; i++) buffer[i] = func(flat[i]);
  return finish_output(buffer, A.shape, dtype, out);
}

function applyFuncToArray_freeze(get_dtype: DtypeResolver, func, cfunc?: (z: Complex) => any): ElementwiseOp {
  return function (A: NDArray, out: NDArray | DType = null): NDArray {
    return applyFuncToArray(get_dtype, func, A, out, cfunc);
  } as ElementwiseOp;
}

/** Result dtype of real, imag, abs...: the real counterpart of complex input, the same dtype otherwise */
function real_out(dtypes: DType[], out?: DType | any): DType {
  const m = dtype_max(...dtypes);
  return is_cx(m) ? realpart_out(dtypes, out) : same_out(dtypes, out);
}

/** Round half to even (like numpy.rint) */
export function rint_scalar(x: number) {
  if (!Number.isFinite(x)) return x;
  if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2);
  return Math.round(x);
}

const ONE_C = new Complex(1, 0);
export const sign = applyFuncToArray_freeze(numeric_out, Math.sign, (z)=>z.sign());
export const sqrt = applyFuncToArray_freeze(unary_float_out, Math.sqrt, (z)=>z.sqrt());
/** square keeps the dtype (integers wrap around like in numpy); bool gives int8 */
function square_out(dtypes: DType[], out?: DType | any): DType { return dtype_max(...dtypes) === bool ? check_out(int8, get_out_dtype(out)) : same_out(dtypes, out); }
export const square = applyFuncToArray_freeze(square_out, (a) => a * a, (z)=>z.square());
export const exp = applyFuncToArray_freeze(unary_float_out, Math.exp, (z)=>z.exp());
export const expm1 = applyFuncToArray_freeze(unary_float_out, Math.expm1, (z)=>z.exp().sub(ONE_C));
export const exp2 = applyFuncToArray_freeze(unary_float_out, (x) => Math.pow(2, x), (z)=>new Complex(2,0).pow(z));
export const log = applyFuncToArray_freeze(unary_float_out, Math.log, (z)=>z.log());
export const log2 = applyFuncToArray_freeze(unary_float_out, Math.log2, (z)=>z.log().div(new Complex(Math.LN2,0)));
export const log10 = applyFuncToArray_freeze(unary_float_out, Math.log10, (z)=>z.log().div(new Complex(Math.LN10,0)));
export const log1p = applyFuncToArray_freeze(unary_float_out, Math.log1p, (z)=>z.add(ONE_C).log());
export const sin = applyFuncToArray_freeze(unary_float_out, Math.sin, (z)=>z.sin());
export const cos = applyFuncToArray_freeze(unary_float_out, Math.cos, (z)=>z.cos());
export const tan = applyFuncToArray_freeze(unary_float_out, Math.tan, (z)=>z.tan());
export const asin = applyFuncToArray_freeze(unary_float_out, Math.asin, (z)=>z.asin());
export const acos = applyFuncToArray_freeze(unary_float_out, Math.acos, (z)=>z.acos());
export const atan = applyFuncToArray_freeze(unary_float_out, Math.atan, (z)=>z.atan());
export const cosh = applyFuncToArray_freeze(unary_float_out, Math.cosh, (z)=>z.cosh());
export const sinh = applyFuncToArray_freeze(unary_float_out, Math.sinh, (z)=>z.sinh());
export const tanh = applyFuncToArray_freeze(unary_float_out, Math.tanh, (z)=>z.tanh());
export const acosh = applyFuncToArray_freeze(unary_float_out, Math.acosh, (z)=>z.acosh());
export const asinh = applyFuncToArray_freeze(unary_float_out, Math.asinh, (z)=>z.asinh());
export const atanh = applyFuncToArray_freeze(unary_float_out, Math.atanh, (z)=>z.atanh());
export const degrees = applyFuncToArray_freeze(unary_float_out, (x) => x * 180 / Math.PI);
export const radians = applyFuncToArray_freeze(unary_float_out, (x) => x * Math.PI / 180);
export const deg2rad = radians;
export const rad2deg = degrees;
export const floor = applyFuncToArray_freeze(same_out, Math.floor);
export const ceil = applyFuncToArray_freeze(same_out, Math.ceil);
export const trunc = applyFuncToArray_freeze(same_out, Math.trunc);
export const fix = trunc;
export const rint = applyFuncToArray_freeze(unary_float_out, rint_scalar, (z) => new Complex(rint_scalar(z.re), rint_scalar(z.im)));
export const isfinite = applyFuncToArray_freeze(bool_out, (x) => Number.isFinite(+x), (z)=>z.isFinite);
export const isinf = applyFuncToArray_freeze(bool_out, (x) => x === Infinity || x === -Infinity, (z)=>z.isInf);
export const isposinf = applyFuncToArray_freeze(bool_out, (x) => x === Infinity, (z) => z.re === Infinity && z.im === 0);
export const isneginf = applyFuncToArray_freeze(bool_out, (x) => x === -Infinity, (z) => z.re === -Infinity && z.im === 0);
export const isnan = applyFuncToArray_freeze(bool_out, (x) => Number.isNaN(x), (z)=>z.isNaN);
export const signbit = applyFuncToArray_freeze(bool_out, (x) => x < 0 || Object.is(x, -0));
export const iscomplex = applyFuncToArray_freeze(bool_out, (_x) => false, (z) => z.im !== 0);
export const isreal = applyFuncToArray_freeze(bool_out, (_x) => true, (z) => z.im === 0);
function reciprocal_out(dtypes: DType[], out?: DType | any): DType { return dtype_max(...dtypes) === bool ? check_out(int8, get_out_dtype(out)) : same_out(dtypes, out); }
const _reciprocal = applyFuncToArray_freeze(reciprocal_out, (x) => 1 / x, (z) => (z.re === 0 && z.im === 0) ? new Complex(NaN, NaN) : z.recip());
/** 1/x; integers follow C integer division (1/0 gives the smallest integer, like numpy) */
export const reciprocal: ElementwiseOp = function (A: NDArray, out: NDArray | DType = null) {
  A = asarray(A);
  const t = TYPE_BY_NAME[A.dtype];
  if (t.kind === 'i' || t.kind === 'u' || t.kind === 'b') {
    const min = (t.kind === 'i' && t.bits >= 32) ? -Math.pow(2, t.bits - 1) : 0; // C overflow behaviour of numpy
    return applyFuncToArray(reciprocal_out, (x) => x === 0 ? min : Math.trunc(1 / x), A, out);
  }
  return _reciprocal(A, out);
} as ElementwiseOp;
export const positive = applyFuncToArray_freeze(numeric_out, (x) => +x, (z)=>z);
function angle_out(dtypes: DType[], out?: DType | any): DType { return is_cx(dtype_max(...dtypes)) ? realpart_out(dtypes, out) : unary_float_out(dtypes, out); }
const angle_ = applyFuncToArray_freeze(angle_out, (x) => Number.isNaN(x) ? NaN : (x < 0 ? Math.PI : 0), (z) => z.arg);
/** Angle of the complex argument (in radians, or degrees if deg is true) */
export const angle = (A: NDArray, deg: boolean = false) => {
  const r = angle_(A);
  return deg ? applyFuncToArray(unary_float_out, (x) => x * 180 / Math.PI, r) : r;
};
export const real = applyFuncToArray_freeze(real_out, (x) => x, (z) => z.re);
export const imag = applyFuncToArray_freeze(real_out, (_x) => 0, (z) => z.im);
export const conj = applyFuncToArray_freeze(square_out, (x) => x, (z)=>z.conj());
export const conjugate = conj;
export const cbrt = applyFuncToArray_freeze(unary_float_out, Math.cbrt);
export function real_if_close(a: NDArray, tol: number = 100): NDArray {
  a = asarray(a);
  if (!is_cx(a.dtype)) return a;
  const eps = 2.220446049250313e-16;
  const t = tol > 1 ? tol * eps : tol;
  const f = a.flat as any;
  for (let i = 0; i < f.length; i++) if (Math.abs(f[i].im) >= t) return a;
  return applyFuncToArray(real_out, (x) => x, a, null, (z) => z.re);
}
export const negative = applyFuncToArray_freeze(numeric_out, x => -x, (z)=>z.neg());
const _bitwise_not = applyFuncToArray_freeze(integer_out, x => ~x);
/** ~x for integers; logical not for bool arrays (as numpy) */
export const bitwise_not: ElementwiseOp = function (A: NDArray, out: NDArray | DType = null) {
  A = asarray(A);
  if (A.dtype === bool) return applyFuncToArray(bool_out, (x) => !truthy(x), A, out);
  return _bitwise_not(A, out);
} as ElementwiseOp;
export const logical_not = applyFuncToArray_freeze(bool_out, x => !truthy(x), (z)=>!truthy(z));
export const valueOf = applyFuncToArray_freeze(same_out, x => +x, (z)=>z);
export const abs = applyFuncToArray_freeze(real_out, Math.abs, (z) => z.abs);

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
  if (dtype === 'float16') {
    // numpy computes nextafter(x, +inf) - x for float16 (so negative powers of two give the finer spacing)
    const a = Math.abs(x) * (x < 0 ? 1 - 1e-6 : 1);
    return a < Math.pow(2, -14) ? Math.pow(2, -24) : Math.pow(2, Math.floor(Math.log2(a)) - 10);
  }
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
  const dtype = unary_float_out([A.dtype], out);
  return applyFuncToArray(unary_float_out, (x) => spacing_scalar(x, dtype), A, out);
}

export function nextafter_scalar(x: number, y: number, dtype: DType): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (x === y) return y;
  if (x === 0) {
    const tiny = dtype === float32 ? 1.401298464324817e-45 : dtype === 'float16' ? Math.pow(2, -24) : 5e-324;
    return y > 0 ? tiny : -tiny;
  }
  if (dtype === 'float16') {
    const a = Math.abs(x);
    const up = ((y > x) === (x > 0)); // away from zero?
    const ulp = (v: number) => v < Math.pow(2, -14) ? Math.pow(2, -24) : Math.pow(2, Math.floor(Math.log2(v)) - 10);
    // moving towards zero across a power of two uses the finer spacing below it
    const step = up ? ulp(a) : ulp(a * (1 - 1e-6) );
    const r = up ? a + step : a - step;
    return x < 0 ? -r : r;
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
  if (is_cx(arr.dtype)) {
    const f = Math.pow(10, Math.abs(decimals));
    const r = (x: number) => decimals >= 0 ? rint_scalar(x * f) / f : rint_scalar(x / f) * f;
    return applyFuncToArray(same_out, (x) => x, arr, out, (z) => new Complex(r(z.re), r(z.im)));
  }
  if (arr.dtype === bool) return applyFuncToArray(unary_float_out, (x) => x, arr, out); // numpy gives float16
  if (dtype_is_integer(arr.dtype)) {
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
  const fix = (x: number) => Number.isNaN(x) ? nan : (x === Infinity ? pos : (x === -Infinity ? neg : x));
  return applyFuncToArray(same_out, (x) => {
    if (Number.isNaN(x)) return nan;
    if (x === Infinity) return pos;
    if (x === -Infinity) return neg;
    return x;
  }, a, out, (z) => new Complex(fix(z.re), fix(z.im)));
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
  const dtype = unary_float_out(operand_dtypes(A, B));
  return apply_binary_operation(unary_float_out, (x, y) => nextafter_scalar(x, y, dtype), A, B, out);
}

/** |x| as float (numpy.fabs): integers give floats, complex is not supported */
export const fabs: ElementwiseOp = function (A: NDArray, out: NDArray | DType = null) {
  A = asarray(A);
  if (is_cx(A.dtype)) throw new Error("ufunc 'fabs' not supported for the input types");
  return applyFuncToArray(unary_float_out, Math.abs, A, out);
} as ElementwiseOp;
