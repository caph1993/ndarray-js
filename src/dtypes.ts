//@ts-check
import { Complex, ZERO, is_complex } from './complex';

// Mozilla docs:
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypedArray

// ==============================
//   Registry of data types
// ==============================

export type Kind = 'b' | 'i' | 'u' | 'f' | 'c' | 'O';
export interface TypeInfo { name: string; kind: Kind; bits: number; char: string; str: string; num: number }

const T = (name: string, kind: Kind, bits: number, char: string, str: string, num: number): TypeInfo => ({ name, kind, bits, char, str, num });
export const TYPES: TypeInfo[] = [
  T('bool', 'b', 8, '?', '|b1', 0),
  T('int8', 'i', 8, 'b', '|i1', 1), T('uint8', 'u', 8, 'B', '|u1', 2),
  T('int16', 'i', 16, 'h', '<i2', 3), T('uint16', 'u', 16, 'H', '<u2', 4),
  T('int32', 'i', 32, 'i', '<i4', 5), T('uint32', 'u', 32, 'I', '<u4', 6),
  T('int64', 'i', 64, 'l', '<i8', 7), T('uint64', 'u', 64, 'L', '<u8', 8),
  T('float16', 'f', 16, 'e', '<f2', 23), T('float32', 'f', 32, 'f', '<f4', 11), T('float64', 'f', 64, 'd', '<f8', 12),
  T('complex64', 'c', 64, 'F', '<c8', 14), T('complex128', 'c', 128, 'D', '<c16', 15),
  T('object', 'O', 64, 'O', '|O', 17),
];
export const TYPE_BY_NAME: { [name: string]: TypeInfo } = {};
TYPES.forEach(t => { TYPE_BY_NAME[t.name] = t; TYPE_BY_NAME[t.char] = t; TYPE_BY_NAME[t.str] = t; TYPE_BY_NAME[t.str.slice(1)] = t; });
Object.assign(TYPE_BY_NAME, {
  'bool_': TYPE_BY_NAME.bool, 'b1': TYPE_BY_NAME.bool, '?': TYPE_BY_NAME.bool, 'int': TYPE_BY_NAME.int64, 'int_': TYPE_BY_NAME.int64, 'intp': TYPE_BY_NAME.int64, 'float': TYPE_BY_NAME.float64, 'float_': TYPE_BY_NAME.float64, 'double': TYPE_BY_NAME.float64, 'single': TYPE_BY_NAME.float32, 'half': TYPE_BY_NAME.float16,
  'complex': TYPE_BY_NAME.complex128, 'complex_': TYPE_BY_NAME.complex128, 'cdouble': TYPE_BY_NAME.complex128, 'csingle': TYPE_BY_NAME.complex64, 'uint': TYPE_BY_NAME.uint64, 'uintp': TYPE_BY_NAME.uint64,
  'q': TYPE_BY_NAME.int64, 'Q': TYPE_BY_NAME.uint64, 'byte': TYPE_BY_NAME.int8, 'ubyte': TYPE_BY_NAME.uint8, 'short': TYPE_BY_NAME.int16, 'ushort': TYPE_BY_NAME.uint16, 'intc': TYPE_BY_NAME.int32, 'uintc': TYPE_BY_NAME.uint32, 'long': TYPE_BY_NAME.int64, 'ulong': TYPE_BY_NAME.uint64, 'longlong': TYPE_BY_NAME.int64, 'ulonglong': TYPE_BY_NAME.uint64,
  'i1': TYPE_BY_NAME.int8, 'u1': TYPE_BY_NAME.uint8, 'i2': TYPE_BY_NAME.int16, 'u2': TYPE_BY_NAME.uint16, 'i4': TYPE_BY_NAME.int32, 'u4': TYPE_BY_NAME.uint32, 'i8': TYPE_BY_NAME.int64, 'u8': TYPE_BY_NAME.uint64,
  'f2': TYPE_BY_NAME.float16, 'f4': TYPE_BY_NAME.float32, 'f8': TYPE_BY_NAME.float64, 'c8': TYPE_BY_NAME.complex64, 'c16': TYPE_BY_NAME.complex128, 'O': TYPE_BY_NAME.object, 'o': TYPE_BY_NAME.object,
});

export type DType = "bool" | "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "int64" | "uint64" | "float16" | "float32" | "float64" | "complex64" | "complex128" | "object";
export const ALL_DTYPES = TYPES.map(t => t.name) as DType[];

export const bool = "bool";
export const int8 = "int8";
export const uint8 = "uint8";
export const int16 = "int16";
export const uint16 = "uint16";
export const int32 = "int32";
export const uint32 = "uint32";
/** Integers stored in a Float64Array: exact up to 2**53. Values are truncated to integers when stored. */
export const int64 = "int64";
export const uint64 = "uint64";
export const float16 = "float16";
export const float32 = "float32";
export const float64 = "float64";
export const complex64 = "complex64";
/** Complex numbers: the buffer is an Array of Complex objects */
export const complex128 = "complex128";
export const object = "object";

/** Canonical dtype name for any accepted spelling ('float', 'i4', '<f8', ...). Throws for unknown names. */
export function canonical_dtype(spec: string): DType {
  const t = TYPE_BY_NAME[spec] || TYPE_BY_NAME[String(spec).replace(/^[<>=]/, '')] || TYPE_BY_NAME[String(spec).replace(/^[<>=|]/, '|')];
  if (!t) throw new TypeError(`data type '${String(spec)}' not understood`);
  return t.name as DType;
}
export const dtype_info = (d: DType): TypeInfo => TYPE_BY_NAME[d];

export interface HasDType {
  readonly dtype: DType
}

export type DTypeOf<T extends HasDType> = T["dtype"]

export const infer_dtype = {
  Float32Array: float32,
  Float64Array: float64,
  Int32Array: int32,
  Int16Array: int16,
  Int8Array: int8,
  Uint16Array: uint16,
  Uint32Array: uint32,
  Uint8Array: bool,
  Array: object,
}

export const dtype_constructors = {
  bool: Uint8Array,
  int8: Int8Array, uint8: Uint8Array, int16: Int16Array, uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array,
  int64: Float64Array, uint64: Float64Array,
  float16: Float32Array, float32: Float32Array, float64: Float64Array,
  complex64: Array, complex128: Array,
  object: Array,
}

export type Buffer = {
  length: number
  [i: number]: any
  [Symbol.iterator](): IterableIterator<any>
  forEach: any;
  map: any;
  slice: (start?: number, end?: number) => Buffer;
}

export function new_buffer(
  size_or_array: number | Buffer,
  dtype?: DType
): InstanceType<any> {

  dtype = dtype || float64;
  if (dtype === "complex128" || dtype === "complex64") {
    if (typeof size_or_array === "number") return new Array(size_or_array).fill(ZERO) as any;
    return Array.from(size_or_array as any, (x: any) => Complex.from(x)) as any;
  }
  if (dtype === "object") {
    // Handled separately because Array is not a typed array and doesn't have the same constructor signature.
    if (Array.isArray(size_or_array)) {
      return size_or_array as any;
    } else if (typeof size_or_array === "number") {
      return Array.from({ length: size_or_array }) as any;
    } else {
      return Array.from(size_or_array as any) as any;
    }
  }
  return new dtype_constructors[dtype](size_or_array as any) as InstanceType<any>;
}

// ==============================
//   Value fix-ups for emulated types
// ==============================

const F32 = new Float32Array(1);
/** Rounds to the nearest half precision (float16) value, ties to even */
export function fround16(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const a = Math.abs(x);
  if (a >= 65520) return x > 0 ? Infinity : -Infinity;
  let e = Math.floor(Math.log2(a));
  if (Math.pow(2, e) > a) e--;
  if (Math.pow(2, e + 1) <= a) e++;
  if (e < -14) e = -14;
  const q = Math.pow(2, e - 10);
  const r = a / q;
  let n = Math.floor(r);
  const diff = r - n;
  if (diff > 0.5 || (diff === 0.5 && n % 2 === 1)) n++;
  const out = n * q;
  return x < 0 ? -out : out;
}
const TWO64 = 18446744073709551616;

/** Makes sure that the values of a buffer are valid for the dtype (int64/uint64 values are stored as truncated floats, float16 as rounded floats) */
export function fix_buffer(buffer: any, dtype: DType): any {
  if (dtype === "int64") for (let i = 0; i < buffer.length; i++) buffer[i] = Number.isFinite(buffer[i]) ? Math.trunc(buffer[i]) : 0;
  else if (dtype === "uint64") for (let i = 0; i < buffer.length; i++) { const v = Math.trunc(buffer[i]); buffer[i] = Number.isFinite(v) ? (v < 0 ? v + TWO64 : v) : 0; }
  else if (dtype === "float16") for (let i = 0; i < buffer.length; i++) buffer[i] = fround16(buffer[i]);
  else if (dtype === "complex64") for (let i = 0; i < buffer.length; i++) { const z = buffer[i]; buffer[i] = new Complex(Math.fround(z.re), Math.fround(z.im)); }
  return buffer;
}

/** Copies (and casts) a buffer or array into a new buffer of the given dtype */
export function convert_buffer(src: ArrayLike<any>, dtype: DType): any {
  const n = src.length;
  const out = new_buffer(n, dtype);
  if (dtype === "complex128") for (let i = 0; i < n; i++) out[i] = Complex.from(src[i]);
  else if (dtype === "complex64") for (let i = 0; i < n; i++) { const z = Complex.from(src[i]); out[i] = new Complex(Math.fround(z.re), Math.fround(z.im)); }
  else if (n > 0 && is_complex(src[0]) && dtype !== "object") {
    // complex -> real: the imaginary part is discarded (as numpy does)
    const re = (v: any) => is_complex(v) ? v.re : +v;
    if (dtype === "bool") for (let i = 0; i < n; i++) out[i] = truthy(src[i]) ? 1 : 0;
    else { for (let i = 0; i < n; i++) out[i] = re(src[i]); fix_buffer(out, dtype); }
  }
  else if (dtype === "bool") for (let i = 0; i < n; i++) out[i] = truthy(src[i]) ? 1 : 0;
  else if (dtype === "object") for (let i = 0; i < n; i++) out[i] = src[i];
  else { for (let i = 0; i < n; i++) out[i] = +src[i]; fix_buffer(out, dtype); }
  return out;
}

/** Python-like truthiness, except that nan is true (as in numpy) */
export function truthy(x: any): boolean { return x !== x ? true : (x instanceof Complex ? (x.re !== 0 || x.im !== 0 || x.isNaN) : !!x); }

// ==============================
//   Promotion rules
// ==============================

const by = (kind: Kind, bits: number) => TYPES.find(t => t.kind === kind && t.bits === bits)!;
/** Smallest float that can hold an integer type of the given size */
const float_bits_for_int = (bits: number) => bits <= 8 ? 16 : (bits <= 16 ? 32 : 64);

export function promote_info(a: TypeInfo, b: TypeInfo): TypeInfo {
  if (a.name === b.name) return a;
  if (a.kind === 'O' || b.kind === 'O') return TYPE_BY_NAME.object;
  if (a.kind === 'b') return b;
  if (b.kind === 'b') return a;
  const ints = (t: TypeInfo) => t.kind === 'i' || t.kind === 'u';
  if (ints(a) && ints(b)) {
    if (a.kind === b.kind) return a.bits >= b.bits ? a : b;
    const s = a.kind === 'i' ? a : b, u = a.kind === 'u' ? a : b;
    if (s.bits > u.bits) return s;
    if (u.bits >= 64) return TYPE_BY_NAME.float64;
    return by('i', u.bits * 2);
  }
  const toFloatBits = (t: TypeInfo) => ints(t) ? float_bits_for_int(t.bits) : (t.kind === 'f' ? t.bits : t.bits / 2);
  const anyComplex = a.kind === 'c' || b.kind === 'c';
  const fb = Math.max(toFloatBits(a), toFloatBits(b));
  return anyComplex ? by('c', Math.max(fb, 32) * 2) : by('f', fb);
}

export function promote(a: DType, b: DType): DType { return promote_info(TYPE_BY_NAME[a], TYPE_BY_NAME[b]).name as DType; }

/** Ordering used to decide which dtype is "larger" (a result of promote(a, b) is the larger one) */
export function dtype_cmp(t1: DType, t2: DType): number {
  if (t1 === t2) return 0;
  const p = promote(t1, t2);
  if (p === t1) return 1;
  if (p === t2) return -1;
  return 0;
}
export function dtype_max(...dtypes: DType[]): DType {
  if (dtypes.length === 0) return float64;
  return dtypes.reduce((a, b) => promote(a, b));
}

/** Can values of dtype `from` be stored in `to` with the "same_kind" rule that numpy applies to ufunc outputs? */
export function same_kind_cast(from: DType, to: DType): boolean {
  const a = TYPE_BY_NAME[from], b = TYPE_BY_NAME[to];
  if (promote_info(a, b).name === b.name || a.kind === 'b') return true;
  if (a.kind === 'O' || b.kind === 'O') return a.kind === b.kind;
  const order: { [k: string]: number } = { b: 0, u: 1, i: 2, f: 3, c: 4 };
  return order[a.kind] <= order[b.kind];
}

/**
 * dtype of a plain python/JS scalar (weak) combined with another operand: it never upcasts within the same kind
 * (int < float < complex), as in numpy 2 (NEP 50).
 */
export function weak_dtype(weak: DType, other: DType): DType {
  const w = TYPE_BY_NAME[weak], o = TYPE_BY_NAME[other];
  if (o.kind === 'O' || w.kind === 'O') return object;
  if (w.kind === 'b') return other;
  if (o.kind === 'b') return (w.kind === 'i' || w.kind === 'u') ? int64 : weak;
  if (w.kind === 'i' || w.kind === 'u') return other;
  if (w.kind === 'f') return (o.kind === 'f' || o.kind === 'c') ? other : float64;
  // weak complex
  if (o.kind === 'c') return other;
  if (o.kind === 'f') return o.bits <= 32 ? complex64 : complex128;
  return complex128;
}

export function dtype_is_integer(dtype: DType): boolean {
  const k = TYPE_BY_NAME[dtype].kind; return k === 'i' || k === 'u';
}
export function dtype_is_boolean(dtype: DType): boolean {
  return dtype === "bool";
}
export function dtype_is_complex(dtype: DType): boolean {
  return TYPE_BY_NAME[dtype].kind === 'c';
}
export function dtype_is_float(dtype: DType): boolean {
  return TYPE_BY_NAME[dtype].kind === 'f';
}

// ==============================
//   Result dtype resolvers (they also validate `out`)
// ==============================

export type DtypeResolver = {
  (dtypes: DType[], out?: DType | HasDType): DType;
};

function get_out_dtype(out?: DType | HasDType) {
  if (!out) return null;
  if (typeof out === "string") return out;
  if ("dtype" in out) return out.dtype;
  return out as DType;
}

function check_out(inferred: DType, out: DType | null): DType {
  if (out && !same_kind_cast(inferred, out))
    throw new Error(`Output array has dtype ${out}, which cannot hold all values of type ${inferred}.`);
  return out || inferred;
}

const is_complex_dtype = (d: DType) => TYPE_BY_NAME[d].kind === 'c';

export function addition_out(dtypes: DType[], out?: DType | HasDType): DType {
  return check_out(dtype_max(int32, ...dtypes), get_out_dtype(out));
}
export function bitwise_out(dtypes: DType[], out?: DType | HasDType): DType {
  return check_out(dtype_max(...dtypes), get_out_dtype(out));
}

/** Same dtype as the inputs (bool stays bool). Used by add, multiply, maximum, ... */
export function same_out(dtypes: DType[], out?: DType | HasDType): DType {
  return check_out(dtype_max(...dtypes), get_out_dtype(out));
}

/** Like same_out but bool is not allowed (subtract, negative). */
export function numeric_out(dtypes: DType[], out?: DType | HasDType): DType {
  const inferred = dtype_max(...dtypes);
  if (inferred === bool) throw new Error(`Numpy boolean subtract, the \`-\` operator, is not supported, use the bitwise_xor, the \`^\` operator, or the logical_xor function instead.`);
  return check_out(inferred, get_out_dtype(out));
}

/** Integer or bool only (bitwise operators and shifts). */
export function integer_out(dtypes: DType[], out?: DType | HasDType): DType {
  const inferred = dtype_max(...dtypes);
  const k = TYPE_BY_NAME[inferred].kind;
  if (k === 'f' || k === 'c' || k === 'O') throw new Error(`Bitwise operations are not supported for dtype ${inferred}. Only bool and integer arrays are supported.`);
  return check_out(inferred, get_out_dtype(out));
}

/** dtype of sums and products: small signed ints and bool become int64, small unsigned ints uint64 (like numpy). */
export function sum_out(dtypes: DType[], out?: DType | HasDType): DType {
  const inferred = dtype_max(...dtypes);
  const t = TYPE_BY_NAME[inferred];
  let result: DType = inferred;
  if (t.kind === 'b' || (t.kind === 'i' && t.bits < 64)) result = int64;
  else if (t.kind === 'u' && t.bits < 64) result = uint64;
  return check_out(result, get_out_dtype(out));
}

/** Result of functions that return the real-valued magnitude of complex numbers (abs, real, imag, angle) */
export function realpart_out(dtypes: DType[], out?: DType | HasDType): DType {
  const m = dtype_max(...dtypes);
  const t = TYPE_BY_NAME[m];
  const result: DType = t.kind === 'c' ? (t.bits === 64 ? float32 : float64) : (t.kind === 'f' ? m : float64);
  return check_out(result, get_out_dtype(out));
}

/** Float result for true division and other functions that always compute in double precision for integers */
export function float_out(dtypes: DType[], out?: DType | HasDType): DType {
  const m = dtype_max(...dtypes);
  const t = TYPE_BY_NAME[m];
  const inferred: DType = (t.kind === 'f' || t.kind === 'c' || t.kind === 'O') ? m : float64;
  return check_out(inferred, get_out_dtype(out));
}

/** Result of the unary math functions (sqrt, exp, sin, ...): small ints give float16/float32 like numpy */
export function unary_float_out(dtypes: DType[], out?: DType | HasDType): DType {
  const m = dtype_max(...dtypes);
  const t = TYPE_BY_NAME[m];
  let inferred: DType = m;
  if (t.kind === 'b' || t.kind === 'i' || t.kind === 'u') inferred = (t.kind === 'b' ? float16 : (t.bits <= 8 ? float16 : t.bits <= 16 ? float32 : float64));
  return check_out(inferred, get_out_dtype(out));
}

export function bool_out<T extends DType>(dtypes: DType[], out?: DType | HasDType): T {
  const o = get_out_dtype(out);
  if (o && o !== bool)
    throw new Error(`Output array has dtype ${o} but expected bool.`);
  return bool as T;
}

export function argmax_out(dtypes: DType[], out?: DType | HasDType): DType {
  const o = get_out_dtype(out);
  const inferred = int32;
  if (o && !dtype_is_integer(o)) {
    throw new Error(`Output array has dtype ${o}, which is not an integer type.`);
  }
  return o || inferred;
}
