import { is_cx, TYPE_BY_NAME } from '../dtypes';
//@ts-check
/**
 * numpy dtype utilities: np.dtype, can_cast, promote_types, result_type, finfo, iinfo, issubdtype, ...
 * Arrays only store bool, int32, int64, float32, float64, complex128 and object, but these functions know all the
 * numpy scalar types (int8..uint64, float16, complex64...) as type descriptors.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { Complex } from "../complex";

// ==============================
//   Type registry (shared with the array storage, see dtypes.ts)
// ==============================
import { TYPES, TYPE_BY_NAME as BY_NAME, TypeInfo, Kind as K0, promote_info } from "../dtypes";
export type Kind = K0 | 'U' | 'S';
export type { TypeInfo };

export function lookup_type(spec: any): TypeInfo {
  if (spec instanceof DType_) return spec.info;
  if (spec && typeof spec === "object" && 'kind' in spec && 'bits' in spec && 'char' in spec) return spec as TypeInfo;
  if (isarray(spec)) return lookup_type(spec.dtype);
  if (spec && typeof spec === "object" && (spec as any).dtype !== undefined && typeof (spec as any).dtype === "string") return lookup_type((spec as any).dtype);
  if (typeof spec === "function") {
    if (spec === Number) return BY_NAME.float64;
    if (spec === Boolean) return BY_NAME.bool;
    if (spec === Object) return BY_NAME.object;
  }
  const key = typeof spec === "string" ? spec.replace(/^[<>=|]/, m => m) : spec;
  const t = BY_NAME[key] || BY_NAME[String(key).replace(/^[<>=]/, '')] || BY_NAME[String(key).replace(/^[<>=|]/, '|')];
  if (!t) throw new TypeError(`data type '${String(spec)}' not understood`);
  return t;
}

/** numpy.dtype: describes a data type */
export class DType_ {
  info: TypeInfo;
  constructor(spec: any) { this.info = lookup_type(spec); }
  get name() { return this.info.name; }
  get kind() { return this.info.kind; }
  get itemsize() { return this.info.kind === 'O' ? 8 : this.info.bits / 8; }
  get char() { return this.info.char; }
  get str() { return this.info.str; }
  get num() { return this.info.num; }
  get byteorder() { return this.itemsize === 1 || this.info.kind === 'O' ? '|' : '='; }
  get isnative() { return true; }
  get alignment() { return Math.min(this.itemsize, 8); }
  get hasobject() { return this.info.kind === 'O'; }
  get type() { return this.info.name; }
  equals(o: any) { try { return lookup_type(o).name === this.info.name; } catch { return false; } }
  toString() { return this.info.name; }
  valueOf() { return this.info.name; }
}

export function dtype(spec: any): DType_ { return new DType_(spec); }

const by = (kind: Kind, bits: number) => TYPES.find(t => t.kind === kind && t.bits === bits)!;

export function promote_types(type1: any, type2: any): DType_ { return new DType_(promote_info(lookup_type(type1), lookup_type(type2))); }

const KIND_ORDER: { [k: string]: number } = { b: 0, u: 1, i: 2, f: 3, c: 4 };

export function can_cast(from_: any, to: any, casting: 'no' | 'equiv' | 'safe' | 'same_kind' | 'unsafe' = 'safe'): boolean {
  if (typeof from_ === "number" || typeof from_ === "boolean" || from_ instanceof Complex) throw new TypeError(`can_cast() does not support Python ints, floats, and complex because the result used to depend on the value.`);
  const a = lookup_type(from_), b = lookup_type(to);
  switch (casting) {
    case 'unsafe': return true;
    case 'no': case 'equiv': return a.name === b.name;
    case 'safe': return promote_info(a, b).name === b.name || (a.kind === 'b');
    case 'same_kind': {
      if (promote_info(a, b).name === b.name || a.kind === 'b') return true;
      if (a.kind === 'O' || b.kind === 'O') return a.kind === b.kind;
      return KIND_ORDER[a.kind] <= KIND_ORDER[b.kind];
    }
  }
  throw new TypeError(`casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'`);
}

function value_info(v: any): TypeInfo | null {
  // python scalars are weak
  if (typeof v === "boolean") return null;
  if (typeof v === "number" || v instanceof Complex) return null;
  return lookup_type(v);
}

/** numpy.result_type: arrays and dtypes are promoted; python scalars are weak (NEP 50) */
export function result_type(...args: any[]): DType_ {
  if (!args.length) throw new Error("at least one array or dtype is required");
  let strong: TypeInfo | null = null;
  const weak: ('b' | 'i' | 'f' | 'c')[] = [];
  for (const a of args) {
    if (typeof a === "boolean") weak.push('b');
    else if (typeof a === "number") weak.push(Number.isInteger(a) ? 'i' : 'f');
    else if (a instanceof Complex) weak.push('c');
    else if (isarray(a) && a._weak) weak.push(is_cx(a.dtype) ? 'c' : (a.dtype === 'float64' ? 'f' : (a.dtype === 'bool' ? 'b' : 'i')));
    else {
      const t = lookup_type(a);
      strong = strong === null ? t : promote_info(strong, t);
    }
  }
  let result = strong;
  const order = { b: 0, i: 1, f: 2, c: 3 };
  const top = weak.sort((p, q) => order[q] - order[p])[0];
  if (result === null) {
    result = top === 'b' ? BY_NAME.bool : top === 'i' ? BY_NAME.int64 : top === 'f' ? BY_NAME.float64 : BY_NAME.complex128;
  } else if (top === 'c' && result.kind !== 'c') {
    result = result.kind === 'f' ? (result.bits <= 32 ? BY_NAME.complex64 : BY_NAME.complex128) : BY_NAME.complex128;
  } else if (top === 'f' && (result.kind === 'b' || result.kind === 'i' || result.kind === 'u')) result = BY_NAME.float64;
  else if (top === 'i' && result.kind === 'b') result = BY_NAME.int64;
  return new DType_(result);
}

export function min_scalar_type(a: any): DType_ {
  let kind: string | null = null;
  if (isarray(a)) { if (a.size !== 1 && a.shape.length) return new DType_(a.dtype); kind = a._weak && a.dtype !== 'float64' ? null : TYPE_BY_NAME[a.dtype].kind; a = a.item(); }
  if (kind === 'f' && typeof a === "number") { // a float (even with an integer value) needs a float type
    const abs = Math.abs(a);
    if (!Number.isFinite(a) || abs < 65520) return new DType_(BY_NAME.float16);
    return new DType_(abs < 3.4028235677973366e+38 ? BY_NAME.float32 : BY_NAME.float64);
  }
  if (typeof a === "boolean") return new DType_(BY_NAME.bool);
  if (a instanceof Complex) return new DType_(Math.max(Math.abs(a.re), Math.abs(a.im)) < 65504 ? BY_NAME.complex64 : BY_NAME.complex128);
  if (typeof a === "number") {
    if (Number.isInteger(a) && Math.abs(a) < 18446744073709551616) {
      if (a >= 0) return new DType_(a < 256 ? BY_NAME.uint8 : a < 65536 ? BY_NAME.uint16 : a < 4294967296 ? BY_NAME.uint32 : BY_NAME.uint64);
      return new DType_(a >= -128 ? BY_NAME.int8 : a >= -32768 ? BY_NAME.int16 : a >= -2147483648 ? BY_NAME.int32 : BY_NAME.int64);
    }
    const abs = Math.abs(a);
    if (!Number.isFinite(a) || abs < 65520) return new DType_(BY_NAME.float16);
    return new DType_(abs < 3.4028235677973366e+38 ? BY_NAME.float32 : BY_NAME.float64);
  }
  return new DType_(lookup_type(a));
}
function f16_exact(x: number): boolean {
  if (x === 0 || !Number.isFinite(x)) return true;
  const m = Math.abs(x);
  if (m < 6.103515625e-05) return Math.round(m / 5.960464477539063e-08) * 5.960464477539063e-08 === m;
  const e = Math.floor(Math.log2(m));
  const mant = m / Math.pow(2, e) * 1024;
  return Number.isInteger(mant) || Math.abs(mant - Math.round(mant)) < 1e-9;
}

export function common_type(...arrays: any[]): DType_ {
  let is_complex = false, precision = 0;
  for (const a of arrays) {
    const t = lookup_type(isarray(a) ? a : asarray(a));
    if (t.kind === 'c') { is_complex = true; precision = Math.max(precision, t.bits === 64 ? 1 : 2); }
    else if (t.kind === 'f') precision = Math.max(precision, t.bits === 16 ? 0 : t.bits === 32 ? 1 : 2);
    else if (t.kind === 'i' || t.kind === 'u') precision = Math.max(precision, 2);
    else throw new TypeError(`can't get common type for non-numeric array`);
  }
  const f = precision === 0 ? BY_NAME.float16 : precision === 1 ? BY_NAME.float32 : BY_NAME.float64;
  return new DType_(is_complex ? by('c', f.bits * 2) : f);
}

// ==============================
//   Abstract types
// ==============================

const ABSTRACT: { [name: string]: (t: TypeInfo) => boolean } = {
  generic: () => true, number: t => 'iufc'.includes(t.kind), integer: t => t.kind === 'i' || t.kind === 'u',
  signedinteger: t => t.kind === 'i', unsignedinteger: t => t.kind === 'u', inexact: t => t.kind === 'f' || t.kind === 'c',
  floating: t => t.kind === 'f', complexfloating: t => t.kind === 'c', bool_: t => t.kind === 'b', character: () => false, flexible: () => false,
};

export function issubdtype(arg1: any, arg2: any): boolean {
  const a = lookup_type(arg1);
  if (typeof arg2 === "string" && ABSTRACT[arg2]) return ABSTRACT[arg2](a);
  return lookup_type(arg2).name === a.name;
}

const ISDTYPE_KINDS: { [k: string]: (t: TypeInfo) => boolean } = {
  'bool': t => t.kind === 'b', 'signed integer': t => t.kind === 'i', 'unsigned integer': t => t.kind === 'u', 'integral': t => t.kind === 'i' || t.kind === 'u',
  'real floating': t => t.kind === 'f', 'complex floating': t => t.kind === 'c', 'numeric': t => 'iufc'.includes(t.kind),
};
export function isdtype(dtype_: any, kind: any): boolean {
  const t = lookup_type(dtype_);
  const kinds = Array.isArray(kind) ? kind : [kind];
  return kinds.some(k => {
    if (typeof k === "string" && ISDTYPE_KINDS[k]) return ISDTYPE_KINDS[k](t);
    if (typeof k === "string" && !BY_NAME[k]) throw new TypeError(`kind argument is a string, but '${k}' is not a known kind name.`);
    return lookup_type(k).name === t.name;
  });
}

const TYPENAMES: { [c: string]: string } = {
  'S1': 'character', '?': 'bool', 'B': 'unsigned char', 'D': 'complex double precision', 'G': 'complex long double precision', 'F': 'complex single precision',
  'I': 'unsigned integer', 'H': 'unsigned short', 'L': 'unsigned long integer', 'O': 'object', 'Q': 'unsigned long long integer', 'S': 'string', 'U': 'unicode', 'V': 'void',
  'b': 'signed char', 'd': 'double precision', 'g': 'long precision', 'f': 'single precision', 'i': 'integer', 'h': 'short', 'l': 'long integer', 'q': 'long long integer',
};
export function typename(char: string): string {
  const n = TYPENAMES[char];
  if (n === undefined) throw new KeyError(char);
  return n;
}
class KeyError extends Error { constructor(k: string) { super(`'${k}'`); this.name = "KeyError"; } }

const TYPECODES_BY_ELSIZE = 'GDFgdfQqLlIiHhBb?';
export function mintypecode(typechars: any, typeset: any = 'GDFgdf', dflt: string = 'd'): string {
  const chars: string[] = (typeof typechars === "string" ? typechars.split('') : Array.from(typechars)).map((t: any) => typeof t === "string" ? t : (isarray(t) ? lookup_type(t).char : String(t)));
  const set: string[] = typeof typeset === "string" ? typeset.split('') : Array.from(typeset);
  const inter = chars.filter(c => set.includes(c));
  if (!inter.length) return dflt;
  if (inter.includes('F') && inter.includes('d')) return 'D';
  const l = inter.map(t => [TYPECODES_BY_ELSIZE.indexOf(t), t] as [number, string]).sort((p, q) => p[0] - q[0]);
  return l[0][1];
}

// ==============================
//   finfo / iinfo
// ==============================

const FLOATS: { [bits: number]: { eps: number, epsneg: number, max: number, tiny: number, tiny_sub: number, nmant: number, nexp: number, minexp: number, maxexp: number, precision: number, resolution: number, machep: number, negep: number } } = {
  16: { eps: 0.0009765625, epsneg: 0.00048828125, max: 65504, tiny: 6.103515625e-05, tiny_sub: 5.960464477539063e-08, nmant: 10, nexp: 5, minexp: -14, maxexp: 16, precision: 3, resolution: 0.0010004043579101562, machep: -10, negep: -11 },
  32: { eps: 1.1920928955078125e-07, epsneg: 5.960464477539063e-08, max: 3.4028234663852886e+38, tiny: 1.1754943508222875e-38, tiny_sub: 1.401298464324817e-45, nmant: 23, nexp: 8, minexp: -126, maxexp: 128, precision: 6, resolution: 9.999999974752427e-07, machep: -23, negep: -24 },
  64: { eps: 2.220446049250313e-16, epsneg: 1.1102230246251565e-16, max: Number.MAX_VALUE, tiny: 2.2250738585072014e-308, tiny_sub: 5e-324, nmant: 52, nexp: 11, minexp: -1022, maxexp: 1024, precision: 15, resolution: 1e-15, machep: -52, negep: -53 },
};

export class finfo {
  dtype: DType_; bits: number; eps: number; epsneg: number; max: number; min: number; tiny: number; smallest_normal: number; smallest_subnormal: number;
  precision: number; resolution: number; nmant: number; nexp: number; minexp: number; maxexp: number; machep: number; negep: number; iexp: number;
  constructor(spec: any = 'float64') {
    let t = lookup_type(spec);
    if (t.kind === 'c') t = by('f', t.bits / 2);
    if (t.kind !== 'f') throw new Error(`data type ${t.name} not inexact`);
    const f = FLOATS[t.bits];
    this.dtype = new DType_(t); this.bits = t.bits; this.eps = f.eps; this.epsneg = f.epsneg; this.max = f.max; this.min = -f.max;
    this.tiny = f.tiny; this.smallest_normal = f.tiny; this.smallest_subnormal = f.tiny_sub; this.precision = f.precision; this.resolution = f.resolution;
    this.nmant = f.nmant; this.nexp = f.nexp; this.minexp = f.minexp; this.maxexp = f.maxexp; this.machep = f.machep; this.negep = f.negep; this.iexp = f.nexp;
  }
}

export class iinfo {
  dtype: DType_; bits: number; kind: string; min: number | bigint; max: number | bigint;
  constructor(spec: any = 'int64') {
    const t = lookup_type(spec);
    if (t.kind !== 'i' && t.kind !== 'u') throw new Error(`Invalid integer data type '${t.kind}'.`);
    this.dtype = new DType_(t); this.bits = t.bits; this.kind = t.kind;
    const big = t.bits >= 64;
    const BI = (s: string) => (globalThis as any).BigInt(s);
    if (t.kind === 'i') { this.min = big ? BI("-9223372036854775808") : -Math.pow(2, t.bits - 1); this.max = big ? BI("9223372036854775807") : Math.pow(2, t.bits - 1) - 1; }
    else { this.min = 0; this.max = big ? BI("18446744073709551615") : Math.pow(2, t.bits) - 1; }
  }
}

// ==============================
//   Floating point error handling (settings only)
// ==============================

export type ErrMode = 'ignore' | 'warn' | 'raise' | 'call' | 'print' | 'log';
let errstate_: { divide: ErrMode, over: ErrMode, under: ErrMode, invalid: ErrMode } = { divide: 'warn', over: 'warn', under: 'ignore', invalid: 'warn' };
let errcall: any = null;
export const get_errstate = () => errstate_;

/** Set how floating-point errors are handled. ndarray-js does not emit floating point warnings: the settings are stored for compatibility. */
export function seterr(all: ErrMode | null = null, divide: ErrMode | null = null, over: ErrMode | null = null, under: ErrMode | null = null, invalid: ErrMode | null = null) {
  const old = { ...errstate_ };
  const valid = ['ignore', 'warn', 'raise', 'call', 'print', 'log'];
  for (const m of [all, divide, over, under, invalid]) if (m !== null && !valid.includes(m)) throw new Error(`invalid error mode '${m}'`);
  errstate_ = {
    divide: (divide ?? all ?? errstate_.divide) as ErrMode, over: (over ?? all ?? errstate_.over) as ErrMode,
    under: (under ?? all ?? errstate_.under) as ErrMode, invalid: (invalid ?? all ?? errstate_.invalid) as ErrMode,
  };
  return old;
}
export const geterr = () => ({ ...errstate_ });
export function seterrcall(func: any) { const old = errcall; errcall = func; return old; }
export const geterrcall = () => errcall;

/** errstate(options) returns an object to use with `using`, or errstate(options, fn) runs fn with these settings */
export function errstate(options: { all?: ErrMode, divide?: ErrMode, over?: ErrMode, under?: ErrMode, invalid?: ErrMode } = {}, fn?: () => any): any {
  const old = geterr();
  const enter = () => { seterr(options.all ?? null, options.divide ?? null, options.over ?? null, options.under ?? null, options.invalid ?? null); };
  const exit = () => { seterr(null, old.divide, old.over, old.under, old.invalid); };
  if (fn) { enter(); try { return fn(); } finally { exit(); } }
  const ctx: any = { enter, exit, [Symbol.dispose]: exit };
  enter();
  return ctx;
}

// ==============================
//   Exceptions
// ==============================

function make_error(name: string, parent: any = Error) {
  const C = class extends parent { constructor(msg?: string) { super(msg); this.name = name; } };
  Object.defineProperty(C, 'name', { value: name });
  return C;
}
export class AxisError extends Error {
  axis: number | null; ndim: number | null;
  constructor(axis: any, ndim: number | null = null, msg_prefix: string | null = null) {
    const msg = ndim === null ? String(axis) : `axis ${axis} is out of bounds for array of dimension ${ndim}`;
    super(msg_prefix ? `${msg_prefix}: ${msg}` : msg);
    this.name = "AxisError"; this.axis = ndim === null ? null : axis; this.ndim = ndim;
  }
}
export const exceptions = {
  AxisError,
  ComplexWarning: make_error('ComplexWarning'),
  VisibleDeprecationWarning: make_error('VisibleDeprecationWarning'),
  RankWarning: make_error('RankWarning'),
  DTypePromotionError: make_error('DTypePromotionError', TypeError),
  TooHardError: make_error('TooHardError'),
  ModuleDeprecationWarning: make_error('ModuleDeprecationWarning'),
};
