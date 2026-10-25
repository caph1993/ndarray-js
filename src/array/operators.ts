import { is_cx } from '../dtypes';
//@ts-check
import * as indexes from './indexes';
import { asarray } from '../NDArray';
import { empty, shape_shifts, broadcast_strides } from "../NDArray";
import { bool, bool_out, DType, DtypeResolver, float_out, unary_float_out, check_out, get_out_dtype, int8, TYPE_BY_NAME, integer_out, numeric_out, same_out, weak_dtype, dtype_cmp, dtype_is_integer, dtype_is_float, dtype_is_boolean, dtype_max } from '../dtypes';
import { tolist } from './js-interface';

import { NDArray } from "../NDArray";
import { isarray } from '../NDArray';
import { extend } from '../utils-js';
import { Where } from './indexes';
import { concatenate } from './transform';
import { broadcast_n_shapes, broadcast_shapes } from "../NDArray";
import { new_buffer, fix_buffer, truthy, int32, int64, complex128, convert_buffer } from '../dtypes';
import { Complex, cmp_complex } from '../complex';

export type ArrayOrConstant = NDArray | number | boolean;


export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | DType | null) => NDArray
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

// ==============================
//   Broadcast loops
// ==============================

/**
 * Evaluates `fn(...values)` for every position of an output of shape `shape`, where the
 * inputs `flats[k]` are broadcast to `shape` with the given strides (see broadcast_strides).
 * The result is written into `out` (a buffer of length prod(shape)).
 */
export function map_broadcast(
  shape: number[], strides: number[][], flats: ArrayLike<any>[], fn: (...values: any[]) => any, out: any
) {
  const n = out.length;
  if (n == 0) return out;
  const nd = shape.length;
  const k = flats.length;
  if (nd == 0) {
    out[0] = fn(...flats.map(f => f[0]));
    return out;
  }
  const last = nd - 1;
  const lastN = shape[last];
  const idx = new Array(nd).fill(0);
  const offs = new Array(k).fill(0);
  const vals = new Array(k);
  for (let i = 0; i < n;) {
    for (let j = 0; j < lastN; j++, i++) {
      for (let q = 0; q < k; q++) vals[q] = flats[q][offs[q] + j * strides[q][last]];
      out[i] = fn(...vals);
    }
    let axis = last - 1;
    while (axis >= 0) {
      idx[axis]++;
      for (let q = 0; q < k; q++) offs[q] += strides[q][axis];
      if (idx[axis] < shape[axis]) break;
      for (let q = 0; q < k; q++) offs[q] -= strides[q][axis] * shape[axis];
      idx[axis] = 0;
      axis--;
    }
    if (axis < 0) break;
  }
  return out;
}

/** Specialized (faster) version of map_broadcast for two inputs */
function map_broadcast2(
  shape: number[], sa: number[], sb: number[], fa: ArrayLike<any>, fb: ArrayLike<any>, fn: (a: any, b: any) => any, out: any
) {
  const n = out.length;
  if (n == 0) return out;
  const nd = shape.length;
  if (nd == 0) { out[0] = fn(fa[0], fb[0]); return out; }
  const last = nd - 1;
  const lastN = shape[last];
  const sal = sa[last], sbl = sb[last];
  const idx = new Array(nd).fill(0);
  let ia = 0, ib = 0;
  for (let i = 0; i < n;) {
    for (let j = 0, pa = ia, pb = ib; j < lastN; j++, i++, pa += sal, pb += sbl) out[i] = fn(fa[pa], fb[pb]);
    let axis = last - 1;
    while (axis >= 0) {
      idx[axis]++;
      ia += sa[axis]; ib += sb[axis];
      if (idx[axis] < shape[axis]) break;
      ia -= sa[axis] * shape[axis]; ib -= sb[axis] * shape[axis];
      idx[axis] = 0;
      axis--;
    }
    if (axis < 0) break;
  }
  return out;
}

/** Wraps a function so that its output is a valid element of a bool buffer */
export function bool_wrap(fn: (...args: any[]) => any) {
  return (...args: any[]) => truthy(fn(...args)) ? 1 : 0;
}

/**
 * Stores `buffer` into `out` (if it is an array) or wraps it in a new array of the given shape and dtype.
 * 0-d results are returned as 0-d arrays.
 */
export function finish_output(buffer: any, shape: number[], dtype: DType, out: NDArray | DType | null): NDArray {
  fix_buffer(buffer, dtype);
  if (isarray(out)) {
    if (out.shape.length != shape.length || out.shape.some((v, i) => v !== shape[i])) {
      throw new Error(`Output array has shape [${out.shape}] but the result has shape [${shape}].`);
    }
    out.flat = buffer;
    return out;
  }
  return new NDArray(buffer, shape, dtype);
}

/** A python int combined with a smaller integer array must fit in it (numpy 2: OverflowError) */
function check_weak_int(w: NDArray, other: NDArray) {
  if (!w._weak || other._weak) return;
  const tw = TYPE_BY_NAME[w.dtype], to = TYPE_BY_NAME[other.dtype];
  if (tw.kind !== 'i' || (to.kind !== 'i' && to.kind !== 'u')) return;
  if (to.bits >= 64 && to.kind !== 'u') return;
  const lo = to.kind === 'u' ? 0 : -Math.pow(2, to.bits - 1), hi = to.bits >= 64 ? Infinity : to.kind === 'u' ? Math.pow(2, to.bits) - 1 : Math.pow(2, to.bits - 1) - 1;
  const f = w.flat as any;
  for (let i = 0; i < f.length; i++) if (f[i] < lo || f[i] > hi) throw new Error(`OverflowError: Python integer ${f[i]} out of bounds for ${other.dtype}`);
}

export function operand_dtypes(A: NDArray, B: NDArray): [DType, DType] {
  // Two python scalars: python ints are unbounded, use int64
  if (A._weak && B._weak) return [A.dtype === int32 ? int64 : A.dtype, B.dtype === int32 ? int64 : B.dtype];
  if (A._weak && !B._weak) return [weak_dtype(A.dtype, B.dtype), B.dtype];
  if (B._weak && !A._weak) return [A.dtype, weak_dtype(B.dtype, A.dtype)];
  return [A.dtype, B.dtype];
}

function freeze_args_bin_op(dtype_resolver: DtypeResolver, func, cfunc?: (a: Complex, b: Complex) => any): BinaryOperator {
  return function (A: NDArray, B: NDArray, out: NDArray | DType = null): NDArray {
    return apply_binary_operation(dtype_resolver, func, A, B, out, cfunc);
  };
}

export function apply_binary_operation(
  dtype_resolver: DtypeResolver,
  func,
  A: ArrayOrConstant,
  B: ArrayOrConstant,
  out: NDArray | DType | null = null,
  cfunc?: (a: Complex, b: Complex) => any,
): NDArray {
  A = asarray(A);
  B = asarray(B);
  const [shape, shapeA, shapeB] = broadcast_shapes(A.shape, B.shape);
  const dtype = dtype_resolver(operand_dtypes(A, B), out);
  if (dtype_resolver !== bool_out) { check_weak_int(A, B); check_weak_int(B, A); } // comparisons accept any python int
  const buffer = new_buffer(prod(shape), dtype);
  let flatA = A.flat, flatB = B.flat;
  if (is_cx(A.dtype) || is_cx(B.dtype)) {
    if (!cfunc) throw new Error(`ufunc not supported for the input types: complex numbers are not supported by this function`);
    func = cfunc;
    flatA = convert_buffer(flatA, complex128); flatB = convert_buffer(flatB, complex128);
  }
  if (dtype === bool) func = bool_wrap(func);
  const sameA = shapeA.length == shape.length && shapeA.every((v, i) => v == shape[i]);
  const sameB = shapeB.length == shape.length && shapeB.every((v, i) => v == shape[i]);
  const n = buffer.length;
  if (sameA && sameB) for (let i = 0; i < n; i++) buffer[i] = func(flatA[i], flatB[i]);
  else if (sameA && B.size == 1) { const b = flatB[0]; for (let i = 0; i < n; i++) buffer[i] = func(flatA[i], b); }
  else if (sameB && A.size == 1) { const a = flatA[0]; for (let i = 0; i < n; i++) buffer[i] = func(a, flatB[i]); }
  else map_broadcast2(shape, broadcast_strides(shape, shapeA), broadcast_strides(shape, shapeB), flatA, flatB, func, buffer);
  return finish_output(buffer, shape, dtype, out);
}

function fix_mod(a: number, b: number) {
  // Python/numpy modulo: the result has the sign of the divisor
  const r = a % b;
  if (r === 0) return b < 0 || Object.is(b, -0) ? -0 : 0; // signed zero, as numpy
  return (r < 0) !== (b < 0) ? r + b : r;
}
/** numpy's npy_divmod: floor division of floats */
function float_floor_div(a: number, b: number) {
  if (Number.isInteger(a) && Number.isInteger(b) && Math.abs(a) < 4503599627370496 && b !== 0) return Math.floor(a / b);
  if (b === 0) return a / b;
  const m = a % b;
  let div = (a - m) / b;
  if (m !== 0) { if ((b < 0) !== (m < 0)) div -= 1; }
  if (div !== 0) { const f = Math.floor(div); return div - f > 0.5 ? f + 1 : f; }
  const q = a / b;
  return q < 0 || Object.is(q, -0) ? -0 : 0;
}
/** bool inputs give int8 (as numpy), otherwise the common dtype */
const intish_out: DtypeResolver = (dtypes, out) => dtype_max(...dtypes) === bool ? check_out(int8, get_out_dtype(out)) : same_out(dtypes, out);
/** C pow semantics: 1**nan == 1 and (-1)**inf == 1 */
function c_pow(a: number, b: number) {
  if (a === 1 || b === 0) return 1;
  if (a === -1 && (b === Infinity || b === -Infinity)) return 1;
  return Math.pow(a, b);
}

export const add = freeze_args_bin_op(same_out, (a, b) => a + b, (a, b) => a.add(b));
export const subtract = freeze_args_bin_op(numeric_out, (a, b) => a - b, (a, b) => a.sub(b));
export const multiply = freeze_args_bin_op(same_out, (a, b) => a * b, (a, b) => a.mul(b));
export const divide = freeze_args_bin_op(float_out, (a, b) => a / b, (a, b) => a.div(b));
export const true_divide = divide;
export const mod = freeze_args_bin_op(intish_out, fix_mod);
export const remainder = mod;
export const fmod = freeze_args_bin_op(intish_out, (a, b) => a % b);
export const divide_int = freeze_args_bin_op(intish_out, float_floor_div);
export const floor_divide = divide_int;
const _pow = freeze_args_bin_op(intish_out, c_pow, (a, b) => a.pow(b));
export const pow: BinaryOperator = function (A, B, out = null) {
  A = asarray(A); B = asarray(B);
  const [dA, dB] = operand_dtypes(A as NDArray, B as NDArray);
  if ((dtype_is_integer(dA) || dtype_is_boolean(dA)) && (dtype_is_integer(dB) || dtype_is_boolean(dB))) {
    const fb = (B as NDArray).flat;
    for (let i = 0; i < fb.length; i++) if (fb[i] < 0) throw new Error("Integers to negative integer powers are not allowed.");
  }
  const dt = intish_out([dA, dB].map(d => d === bool ? int8 : d), out as any);
  const t = TYPE_BY_NAME[dt];
  if ((t.kind === 'i' || t.kind === 'u') && t.bits <= 32 && (dtype_is_integer(dA) || dtype_is_boolean(dA)) && (dtype_is_integer(dB) || dtype_is_boolean(dB))) {
    // exact modular exponentiation (integers wrap around like in C)
    const mod32 = (v: number) => t.kind === 'u' ? v >>> 0 : v | 0;
    const ipow = (a: number, b: number) => {
      let r = 1, x = a | 0;
      while (b > 0) { if (b & 1) r = Math.imul(r, x); x = Math.imul(x, x); b = Math.floor(b / 2); }
      return mod32(r);
    };
    return apply_binary_operation(intish_out, ipow, A, B, out);
  }
  return _pow(A, B, out);
};
export const power = pow;
export const bitwise_or = freeze_args_bin_op(integer_out, (a, b) => a | b);
export const bitwise_and = freeze_args_bin_op(integer_out, (a, b) => a & b);
export const bitwise_xor = freeze_args_bin_op(integer_out, (a, b) => a ^ b);
/** Shifts by the width of the type or more give 0 (or the sign for right shifts of negative numbers), as in numpy */
function shift_op(left: boolean): BinaryOperator {
  return function (A, B, out = null) {
    A = asarray(A); B = asarray(B);
    const [dA, dB] = operand_dtypes(A as NDArray, B as NDArray);
    integer_out([dA, dB], out as any); // validates the dtypes
    const dt = intish_out([dA, dB].map(d => d === bool ? int8 : d), out as any);
    const t = TYPE_BY_NAME[dt], bits = t.bits, signed = t.kind === 'i';
    const f = left
      ? (a: number, b: number) => (b < 0 || b >= bits) ? 0 : (bits > 32 ? a * Math.pow(2, b) : (a << b))
      : (a: number, b: number) => (b < 0 || b >= bits) ? (signed && a < 0 ? -1 : 0) : (bits > 32 ? Math.floor(a / Math.pow(2, b)) : (signed ? (a >> b) : (a >>> b)));
    return apply_binary_operation(intish_out, f, A, B, out);
  };
}
export const bitwise_shift_left = shift_op(true);
export const bitwise_shift_right = shift_op(false);
export const greater = freeze_args_bin_op(bool_out, (a, b) => a > b, (a, b) => cmp_complex(a, b) > 0 && !a.isNaN && !b.isNaN);
export const less = freeze_args_bin_op(bool_out, (a, b) => a < b, (a, b) => cmp_complex(a, b) < 0 && !a.isNaN && !b.isNaN);
export const greater_equal = freeze_args_bin_op(bool_out, (a, b) => a >= b, (a, b) => cmp_complex(a, b) >= 0 && !a.isNaN && !b.isNaN);
export const less_equal = freeze_args_bin_op(bool_out, (a, b) => a <= b, (a, b) => cmp_complex(a, b) <= 0 && !a.isNaN && !b.isNaN);
export const equal = freeze_args_bin_op(bool_out, (a, b) => a == b, (a, b) => a.eq(b));
export const not_equal = freeze_args_bin_op(bool_out, (a, b) => a != b, (a, b) => !a.eq(b));
export const maximum = freeze_args_bin_op(same_out, (a, b) => (a !== a || b !== b) ? NaN : Math.max(a, b), (a, b) => (a.isNaN ? a : b.isNaN ? b : (cmp_complex(a, b) >= 0 ? a : b)));
export const minimum = freeze_args_bin_op(same_out, (a, b) => (a !== a || b !== b) ? NaN : Math.min(a, b), (a, b) => (a.isNaN ? a : b.isNaN ? b : (cmp_complex(a, b) <= 0 ? a : b)));
export const fmax = freeze_args_bin_op(same_out, (a, b) => {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return Math.max(a, b);
}, (a, b) => (a.isNaN ? b : b.isNaN ? a : (cmp_complex(a, b) >= 0 ? a : b)));
export const fmin = freeze_args_bin_op(same_out, (a, b) => {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return Math.min(a, b);
}, (a, b) => (a.isNaN ? b : b.isNaN ? a : (cmp_complex(a, b) <= 0 ? a : b)));
export const logical_or = freeze_args_bin_op(bool_out, (a, b) => truthy(a) || truthy(b), (a, b) => truthy(a) || truthy(b));
export const logical_and = freeze_args_bin_op(bool_out, (a, b) => truthy(a) && truthy(b), (a, b) => truthy(a) && truthy(b));
export const logical_xor = freeze_args_bin_op(bool_out, (a, b) => truthy(a) != truthy(b), (a, b) => truthy(a) != truthy(b));
export const hypot = freeze_args_bin_op(unary_float_out, Math.hypot);
export const atan2 = freeze_args_bin_op(unary_float_out, Math.atan2);
export const copysign = freeze_args_bin_op(unary_float_out, (x, y) => {
  const neg = y < 0 || Object.is(y, -0);
  return neg ? -Math.abs(x) : Math.abs(x);
});
export const logaddexp = freeze_args_bin_op(unary_float_out, (x, y) => {
  if (x === y && !isFinite(x)) return x;
  const m = Math.max(x, y);
  return m + Math.log1p(Math.exp(-Math.abs(x - y)));
});
export const logaddexp2 = freeze_args_bin_op(unary_float_out, (x, y) => {
  if (x === y && !isFinite(x)) return x;
  const m = Math.max(x, y);
  return m + Math.log2(1 + Math.pow(2, -Math.abs(x - y)));
});
const _ldexp = freeze_args_bin_op((dtypes, out) => unary_float_out([dtypes[0]], out), (x, e) => {
  if (x === 0 || !Number.isFinite(x)) return x;
  const h = Math.trunc(e / 2);
  return x * Math.pow(2, h) * Math.pow(2, e - h);
});
/** x * 2**e; e must be an integer array (as numpy) */
export const ldexp: BinaryOperator = function (A, B, out = null) {
  A = asarray(A); B = asarray(B);
  const t = TYPE_BY_NAME[(B as NDArray).dtype];
  if (t.kind !== 'i' && t.kind !== 'u' && t.kind !== 'b') throw new Error("ufunc 'ldexp' not supported for the input types: the exponent must be an integer");
  return _ldexp(A, B, out);
};
export const heaviside = freeze_args_bin_op(unary_float_out, (x1, x2) => Number.isNaN(x1) ? NaN : (x1 > 0 ? 1 : (x1 === 0 ? x2 : 0)));
const _gcd = (a: number, b: number) => {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
};
const no_bool_int = (dtypes: DType[], out?: any) => { if (dtypes.every(d => d === bool)) throw new Error("ufunc did not contain a loop with signature matching types (bool)"); return integer_out(dtypes, out); };
export const gcd = freeze_args_bin_op(no_bool_int, _gcd);
export const lcm = freeze_args_bin_op(no_bool_int, (a, b) => {
  if (a === 0 || b === 0) return 0;
  return Math.abs(a / _gcd(a, b) * b);
});
export const float_power = freeze_args_bin_op((dtypes, out) => is_cx(dtype_max(...dtypes)) ? complex128 : float_out([], out), c_pow, (a, b) => a.pow(b));

export const op_binary = {
  "+": add,
  "-": subtract,
  "*": multiply,
  "/": divide,
  "%": mod,
  "|": bitwise_or,
  "&": bitwise_and,
  "^": bitwise_xor,
  "<<": bitwise_shift_left,
  ">>": bitwise_shift_right,
  "**": pow,
  "//": divide_int,

  "<": less,
  ">": greater,
  ">=": greater_equal,
  "<=": less_equal,
  "==": equal,
  "!=": not_equal,
  // Operators with custom ascii identifiers:
  "or": logical_or,
  "and": logical_and,
  "xor": logical_xor,
  "max": maximum,
  "min": minimum,
  "hypot": hypot,
  "fmax": fmax,
  "fmin": fmin,
  // "approx": ,
}

op_binary["↑"] = op_binary["max"];
op_binary["↓"] = op_binary["min"];
op_binary["≤"] = op_binary["<="];
op_binary["≥"] = op_binary[">="];
op_binary["≠"] = op_binary["!="];
op_binary["divide"] = op_binary["/"];
op_binary["true_divide"] = op_binary["/"];
op_binary["fmod"] = fmod;
op_binary["remainder"] = mod;


// ==============================
//   Assignment operators
// ==============================

/**
 * Applies `func` between the region `where` of `tgt` and `src` (broadcast to the region), writing the result
 * in `tgt` (a view or the array itself). `func == null` is plain assignment.
 */
export function assign_operation(dtype_resolver: DtypeResolver, func, tgt: NDArray, src: ArrayOrConstant, where: Where): NDArray {

  if (tgt['__warnAssignments']) {
    console.warn(`Warning: You are assigning on a copy that resulted from an advanced index (list of indices or boolean mask) on a source array.\nThe source array is NOT modified. To assign to the source array, use source.assign(values, ...yourIndex) or source.add_assign(values, ...yourIndex), etc.\n`);
    delete tgt['__warnAssignments'];
  }

  if (!(isarray(tgt))) {
    return _assign_operation_toJS(dtype_resolver, func, tgt as any, src, where) as any;
  }
  const source = asarray(src);
  if (!where || where.length == 0) {
    if (func == null) {
      const [shape] = broadcast_shapes(tgt.shape, source.shape);
      if (shape.length != tgt.shape.length || shape.some((v, i) => v != tgt.shape[i])) {
        throw new Error(`Could not broadcast input array from shape [${source.shape}] into shape [${tgt.shape}]`);
      }
      const dtype = tgt.dtype;
      const buffer = new_buffer(tgt.size, dtype);
      map_broadcast(shape, [broadcast_strides(shape, source.shape)], [source.flat], (b) => b, buffer);
      tgt.flat = buffer;
    } else {
      apply_binary_operation(dtype_resolver, func, tgt, source, tgt, (func as any).complex);
    }
    return tgt;
  }
  // Region of tgt: positions (in the apparent flat order of tgt) and shape
  const axesIndex = indexes.AxesIndex.prototype.parse(tgt.shape, where);
  const apparent = axesIndex.indices;
  const physical = tgt._simpleIndexes ? tgt._simpleIndexes.indices : null;
  const flat = tgt._flat;
  const positions = physical ? apparent.map(i => physical[i]) : apparent;
  const shape = axesIndex.shape;
  let result: any;
  if (func == null) {
    // Plain assignment: no need to read the target
    const [bshape] = broadcast_shapes(shape, source.shape);
    if (bshape.length != shape.length || bshape.some((v, i) => v != shape[i])) {
      throw new Error(`Could not broadcast input array from shape [${source.shape}] into shape [${shape}]`);
    }
    const values = new_buffer(positions.length, tgt.dtype);
    map_broadcast(shape, [broadcast_strides(shape, source.shape)], [source.flat], (b) => b, values);
    result = values;
  } else {
    const region = new NDArray(new_buffer(positions.map(p => flat[p]), tgt.dtype), shape, tgt.dtype);
    const res = apply_binary_operation(dtype_resolver, func, region, source, null, (func as any).complex);
    if (res.shape.length != shape.length || res.shape.some((v, i) => v != shape[i])) {
      throw new Error(`Non-broadcastable output operand with shape [${shape}] doesn't match the broadcast shape [${res.shape}]`);
    }
    if (dtype_cmp(res.dtype, tgt.dtype) > 0) {
      throw new Error(`Output array has dtype ${tgt.dtype}, which cannot hold all values of type ${res.dtype}.`);
    }
    result = res.flat;
  }
  for (let k = 0; k < positions.length; k++) flat[positions[k]] = result[k];
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


export function __make_assignment_operator(dtype_resolver: DtypeResolver, func) {
  return function operator(a: NDArray, values: NDArray, where: Where) {
    if (where && where.length === 1 && where[0] === false) return a;
    if (where && where.length === 1 && where[0] === true) where = [];
    return assign_operation(dtype_resolver, func, a, values, where);
  }
}

export const op_assign = {
  "=": __make_assignment_operator(same_out, null),
  "+=": __make_assignment_operator(same_out, Object.assign((a, b) => a + b, { complex: (a: Complex, b: Complex) => a.add(b) })),
  "-=": __make_assignment_operator(numeric_out, Object.assign((a, b) => a - b, { complex: (a: Complex, b: Complex) => a.sub(b) })),
  "*=": __make_assignment_operator(same_out, Object.assign((a, b) => a * b, { complex: (a: Complex, b: Complex) => a.mul(b) })),
  "/=": __make_assignment_operator(float_out, Object.assign((a, b) => a / b, { complex: (a: Complex, b: Complex) => a.div(b) })),
  "%=": __make_assignment_operator(same_out, fix_mod),
  "//=": __make_assignment_operator(same_out, (a, b) => Math.floor(a / b)),
  "**=": __make_assignment_operator(same_out, (a, b) => Math.pow(a, b)),
  "|=": __make_assignment_operator(integer_out, (a, b) => a | b),
  "&=": __make_assignment_operator(integer_out, (a, b) => a & b),
  "^=": __make_assignment_operator(integer_out, (a, b) => a ^ b),
  "<<=": __make_assignment_operator(integer_out, (a, b) => a << b),
  ">>=": __make_assignment_operator(integer_out, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": __make_assignment_operator(same_out, (a, b) => Math.max(a, b)),
  "min=": __make_assignment_operator(same_out, (a, b) => Math.min(a, b)),
  "or=": __make_assignment_operator(bool_out, (a, b) => truthy(a) || truthy(b)),
  "and=": __make_assignment_operator(bool_out, (a, b) => truthy(a) && truthy(b)),
  "xor=": __make_assignment_operator(bool_out, (a, b) => truthy(a) != truthy(b)),
};


// ====================================

/** abs(a - b) <= atol + rtol * abs(b), handling infinities and (optionally) nans like numpy */
export function isclose_scalar(a: number, b: number, rtol: number, atol: number, equal_nan: boolean) {
  if (a === b) return true;
  if (Number.isNaN(a) || Number.isNaN(b)) return equal_nan && Number.isNaN(a) && Number.isNaN(b);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

export function isclose(a: ArrayOrConstant, b: ArrayOrConstant, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  const cclose = (x: Complex, y: Complex) => {
    if (x.eq(y)) return true;
    if (x.isNaN || y.isNaN) return equal_nan && x.isNaN && y.isNaN;
    if (!x.isFinite || !y.isFinite) return false;
    return x.sub(y).abs <= atol + rtol * y.abs;
  };
  return apply_binary_operation(bool_out, (x, y) => isclose_scalar(x, y, rtol, atol, equal_nan), a, b, null, cclose);
}

export function allclose(a: ArrayOrConstant, b: ArrayOrConstant, rtol = 1.e-5, atol = 1.e-8, equal_nan = false): boolean {
  const res = isclose(a, b, rtol, atol, equal_nan);
  const flat = res.flat;
  for (let i = 0; i < flat.length; i++) if (!flat[i]) return false;
  return true;
}

export function array_equal(a1: ArrayOrConstant, a2: ArrayOrConstant, equal_nan = false): boolean {
  const A = asarray(a1);
  const B = asarray(a2);
  if (A.shape.length !== B.shape.length) return false;
  for (let i = 0; i < A.shape.length; i++) if (A.shape[i] !== B.shape[i]) return false;
  const flatA = A.flat;
  const flatB = B.flat;
  for (let i = 0; i < flatA.length; i++) {
    const a = flatA[i];
    const b = flatB[i];
    if (a instanceof Complex || b instanceof Complex) { const x = Complex.from(a), y = Complex.from(b); if (x.eq(y) || (equal_nan && x.isNaN && y.isNaN)) continue; return false; }
    if (a == b) continue;
    if (equal_nan && Number.isNaN(a) && Number.isNaN(b)) continue;
    return false;
  }
  return true;
}

export function array_equiv(a1: ArrayOrConstant, a2: ArrayOrConstant): boolean {
  const A = asarray(a1);
  const B = asarray(a2);
  let shape: number[], shapeA: number[], shapeB: number[];
  try {
    [shape, shapeA, shapeB] = broadcast_shapes(A.shape, B.shape);
  } catch (_err) {
    return false;
  }
  // Shapes must be broadcastable to the shape of the larger one: equivalent shapes
  const eq = apply_binary_operation(bool_out, (a, b) => a == b, A, B);
  const flat = eq.flat;
  for (let i = 0; i < flat.length; i++) if (!flat[i]) return false;
  return true;
}


export function divmod(x: any, y: any) {
  x = asarray(x);
  y = asarray(y);
  const quotient = floor_divide(x, y);
  const remainder = mod(x, y);
  return [quotient, remainder];
}


export function ternary_operation(dtype_resolver: DtypeResolver, func, A: ArrayOrConstant, B: ArrayOrConstant, C: ArrayOrConstant, out: NDArray | DType | null = null): NDArray {
  return n_ary_elementwise(dtype_resolver, func, [A, B, C], out);
}

/** Elementwise function of n broadcast arrays */
export function n_ary_elementwise(dtype_resolver: DtypeResolver, func, arrs: ArrayOrConstant[], out: NDArray | DType | null = null): NDArray {
  const arrays = arrs.map(a => asarray(a));
  const [shapes, shape] = broadcast_n_shapes(...arrays.map(a => a.shape));
  // Plain JS scalars are "weak": they do not upcast the other operands within the same kind
  const strong = arrays.filter(a => !a._weak).map(a => a.dtype);
  const base = strong.length && strong.length < arrays.length ? dtype_max(...strong) : null;
  const dtypes = arrays.map(a => (a._weak && base) ? weak_dtype(a.dtype, base) : a.dtype);
  const dtype = dtype_resolver(dtypes, out);
  if (dtype === bool) func = bool_wrap(func);
  const buffer = new_buffer(prod(shape), dtype);
  const strides = arrays.map((a, i) => broadcast_strides(shape as number[], a.shape));
  map_broadcast(shape as number[], strides, arrays.map(a => a.flat), func, buffer);
  return finish_output(buffer, shape as number[], dtype, out);
}


export function n_ary_operation<
  F extends (...args: any) => NDArray | number | number[] | boolean | boolean[]
>(
  arrs: NDArray[],
  elem_shape: number[],
  func: F,
): NDArray {
  // Find output shape and input broadcast shapes
  arrs = arrs.map((a) => asarray(a));

  const rawShapes = arrs.map(a => a.shape);
  // Broadcast shapes
  const [shapes, shape] = broadcast_n_shapes(...rawShapes);
  const size = prod(shape as number[]);
  if (shape.length == 0) {
    return asarray(func(...arrs.map((a) => a.flat[0])));
  }
  const strides = arrs.map(a => broadcast_strides(shape as number[], a.shape));
  const flats = arrs.map(a => a.flat);
  const values: any[] = new Array(size);
  map_broadcast(shape as number[], strides, flats, func as any, values);
  const funcOutIsNDArray = values.length > 0 && isarray(values[0]);
  let out = funcOutIsNDArray ? concatenate(values) : asarray(values);
  out = out.reshape([...(shape as number[]), ...out.shape.slice(1)]);
  return out;
}


export function clip(a: NDArray, a_min: NDArray = null, a_max: NDArray = null, out: any = null) {
  if (a_min === null && a_max === null) throw new Error('At least one of a_min or a_max must be provided');
  if (is_cx(asarray(a).dtype) || (a_min !== null && is_cx(asarray(a_min).dtype)) || (a_max !== null && is_cx(asarray(a_max).dtype))) {
    // complex numbers are ordered lexicographically
    let r: any = a;
    if (a_min !== null) r = maximum(r, a_min);
    if (a_max !== null) r = minimum(r, a_max);
    return r;
  }
  if (a_min !== null && a_max !== null) {
    // numpy: minimum(maximum(a, a_min), a_max)
    const func = (v, minVal, maxVal) => (v !== v || minVal !== minVal || maxVal !== maxVal) ? NaN : Math.min(maxVal, Math.max(minVal, v));
    return n_ary_elementwise(same_out, func, [a, a_min, a_max], out);
  } else if (a_min !== null) {
    return apply_binary_operation(same_out, (v, minVal) => (v !== v || minVal !== minVal) ? NaN : Math.max(minVal, v), a, a_min, out);
  } else {
    return apply_binary_operation(same_out, (v, maxVal) => (v !== v || maxVal !== maxVal) ? NaN : Math.min(maxVal, v), a, a_max, out);
  }
}
