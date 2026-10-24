//@ts-check
/**
 * Formatting: numpy-style array printing (array2string, array_repr, array_str), print options,
 * float formatting, binary/base representations and bit packing.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { DType, bool, int32, new_buffer, dtype_is_integer, complex128, float32 } from "../dtypes";
import { Complex } from "../complex";
import { normalize_axis, transpose, swapAxes } from "../array/transform";
import { reshape } from "../array/shape_operations";
import { fixed_exact, exp_exact } from "./exact-format";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

// ==============================
//   Print options
// ==============================

export interface PrintOptions {
  precision: number; threshold: number; edgeitems: number; linewidth: number; suppress: boolean; nanstr: string; infstr: string;
  formatter: { [key: string]: (x: any) => string } | null; sign: '-' | '+' | ' '; floatmode: 'fixed' | 'unique' | 'maxprec' | 'maxprec_equal'; legacy: any; override_repr: ((a: NDArray) => string) | null;
}
const DEFAULTS: PrintOptions = { precision: 8, threshold: 1000, edgeitems: 3, linewidth: 75, suppress: false, nanstr: 'nan', infstr: 'inf', formatter: null, sign: '-', floatmode: 'maxprec', legacy: false, override_repr: null };
let options: PrintOptions = { ...DEFAULTS };

export function set_printoptions(opts: Partial<PrintOptions>) {
  for (const [k, v] of Object.entries(opts)) {
    if (v === undefined || v === null && k !== 'formatter' && k !== 'legacy' && k !== 'override_repr') continue;
    if (!(k in DEFAULTS)) throw new Error(`Unknown print option ${k}`);
    if (k === 'floatmode' && !['fixed', 'unique', 'maxprec', 'maxprec_equal'].includes(v as string)) throw new ValueError(`floatmode option must be one of fixed, unique, maxprec, maxprec_equal`);
    if (k === 'sign' && !['-', '+', ' '].includes(v as string)) throw new ValueError(`sign option must be one of ' ', '+', or '-'`);
    if (k === 'precision' && (!Number.isInteger(v) || (v as number) < 0)) throw new ValueError(`precision must be a non-negative integer`);
    (options as any)[k] = v;
  }
}
export function get_printoptions(): PrintOptions { return { ...options, formatter: options.formatter }; }
class ValueError extends Error { constructor(m: string) { super(m); this.name = "ValueError"; } }

/** printoptions(opts, fn): runs fn with temporary options. printoptions(opts) returns an object to use with `using` */
export function printoptions(opts: Partial<PrintOptions>, fn?: () => any): any {
  const old = get_printoptions();
  const restore = () => { options = { ...old }; };
  set_printoptions(opts);
  if (fn) { try { return fn(); } finally { restore(); } }
  return { restore, [Symbol.dispose]: restore };
}

// ==============================
//   Float formatting (a port of numpy's dragon4 wrappers and FloatingFormat)
// ==============================

/** Exact decimal expansion of an integer-valued double */
function exact_integer_string(x: number): string { return (globalThis as any).BigInt(x).toString(); }

/** Plain (non exponential) decimal expansion of the shortest round-trip representation of a positive number */
function repr_plain(x: number): string {
  const s = String(x);
  if (!/e/i.test(s)) return s;
  const [mant, exp] = s.toLowerCase().split('e');
  const e = parseInt(exp, 10);
  const [ip, fp = ''] = mant.split('.');
  const digits = ip + fp;
  const point = ip.length + e;
  if (point <= 0) return '0.' + '0'.repeat(-point) + digits;
  if (point >= digits.length) return digits + '0'.repeat(point - digits.length);
  return digits.slice(0, point) + '.' + digits.slice(point);
}

function apply_trim(fp: string, trim: string): string {
  if (trim === 'k') return fp;
  fp = fp.replace(/0+$/, '');
  if (trim === '0' && fp === '') fp = '0';
  return fp;
}

export function format_float_positional(x: number, precision: number | null = null, unique: boolean = true, fractional: boolean = true, trim: 'k' | '.' | '0' | '-' = 'k', sign: boolean = false, pad_left: number | null = null, pad_right: number | null = null, min_digits: number | null = null): string {
  if (!['k', '.', '0', '-'].includes(trim)) throw new ValueError(`trim must be one of 'k', '.', '0', '-'`);
  if (!unique && precision === null) throw new ValueError(`precision must be given if unique is False`);
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return (x < 0 ? '-' : (sign ? '+' : '')) + 'inf';
  const neg = x < 0 || Object.is(x, -0);
  const a = Math.abs(x);
  let ip: string, fp: string;
  if (!fractional && precision !== null) {
    // `precision` counts significant digits
    const r = a === 0 ? { mantissa: '0', exponent: 0 } : exp_exact(a, Math.max(precision - 1, 0));
    const digits = r.mantissa.replace('.', '');
    const point = r.exponent + 1;
    const full = point <= 0 ? '0.' + '0'.repeat(-point) + digits : (point >= digits.length ? digits + '0'.repeat(point - digits.length) + '.' : digits.slice(0, point) + '.' + digits.slice(point));
    [ip, fp = ''] = full.split('.');
    if (unique) fp = fp.replace(/0+$/, '');
  } else if (unique) {
    const s = precision !== null && repr_plain(a).split('.')[1]?.length! > precision ? fixed_exact(a, precision) : repr_plain(a);
    [ip, fp = ''] = s.split('.');
  } else {
    [ip, fp = ''] = fixed_exact(a, precision as number).split('.');
  }
  if (min_digits !== null) fp = fp.padEnd(min_digits, '0');
  fp = apply_trim(fp, trim);
  let out = ip + ((trim === '-' && fp === '') ? '' : '.') + fp;
  if (pad_right !== null) { const d = out.indexOf('.'); out = out + ' '.repeat(Math.max(0, pad_right - (d < 0 ? 0 : out.length - d - 1))); }
  out = (neg ? '-' : (sign ? '+' : '')) + out;
  if (pad_left !== null) { const d = out.indexOf('.'); const left = d < 0 ? out.length : d; out = ' '.repeat(Math.max(0, pad_left - left)) + out; }
  return out;
}

export function format_float_scientific(x: number, precision: number | null = null, unique: boolean = true, trim: 'k' | '.' | '0' | '-' = 'k', sign: boolean = false, pad_left: number | null = null, exp_digits: number | null = null, min_digits: number | null = null): string {
  if (!['k', '.', '0', '-'].includes(trim)) throw new ValueError(`trim must be one of 'k', '.', '0', '-'`);
  if (!unique && precision === null) throw new ValueError(`precision must be given if unique is False`);
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return (x < 0 ? '-' : (sign ? '+' : '')) + 'inf';
  const neg = x < 0 || Object.is(x, -0);
  const a = Math.abs(x);
  let mant: string, exp: number;
  if (unique) {
    const [m, e] = a.toExponential().split('e');
    mant = m; exp = parseInt(e, 10);
    if (precision !== null && (m.split('.')[1] || '').length > precision) { const r = exp_exact(a, precision); mant = r.mantissa; exp = r.exponent; }
  } else { const r = exp_exact(a, precision as number); mant = r.mantissa; exp = r.exponent; }
  let [ip, fp = ''] = mant.split('.');
  if (min_digits !== null) fp = fp.padEnd(min_digits, '0');
  fp = apply_trim(fp, trim);
  const edigits = String(Math.abs(exp)).padStart(Math.max(exp_digits ?? 2, 1), '0');
  let out = ip + ((trim === '-' && fp === '') ? '' : '.') + fp + 'e' + (exp < 0 ? '-' : '+') + edigits;
  out = (neg ? '-' : (sign ? '+' : '')) + out;
  if (pad_left !== null) { const d = out.indexOf('.'); const left = d < 0 ? out.indexOf('e') : d; out = ' '.repeat(Math.max(0, pad_left - left)) + out; }
  return out;
}

// ==============================
//   Element formatters
// ==============================

type Formatter = (v: any) => string;

/** Port of numpy's FloatingFormat: decides, from all the values, the common layout of every element */
function float_formatter(values: number[], opt: PrintOptions, signOverride?: '-' | '+' | ' '): Formatter {
  const sign = signOverride ?? opt.sign;
  const plus = sign === '+';
  const finite = values.filter(Number.isFinite);
  const abs_non_zero = finite.filter(v => v !== 0).map(Math.abs);
  let exp_format = false;
  if (abs_non_zero.length) {
    const mx = Math.max(...abs_non_zero), mn = Math.min(...abs_non_zero);
    if (mx >= 1e8 || (!opt.suppress && (mn < 0.0001 || mx / mn > 1000))) exp_format = true;
  }
  const mode = opt.floatmode;
  let precision = opt.precision, unique = true, trim: 'k' | '.' = '.', min_digits: number | null = null;
  let pad_left = 0, pad_right = 0, exp_size = -1;
  if (finite.length === 0) { pad_left = 0; pad_right = 0; trim = '.'; unique = true; }
  else if (exp_format) {
    unique = true; trim = '.';
    if (mode === 'fixed') { trim = 'k'; unique = false; }
    const strs = finite.map(v => format_float_scientific(v, precision, unique, trim, plus));
    const parts = strs.map(s => { const [m, e] = s.split('e'); const [ip, fp = ''] = m.split('.'); return { ip, fp, e }; });
    exp_size = Math.max(...parts.map(p => p.e.length)) - 1;
    precision = Math.max(...parts.map(p => p.fp.length));
    min_digits = precision;
    trim = 'k';
    pad_left = Math.max(...parts.map(p => p.ip.length));
    pad_right = exp_size + 2 + precision;
  } else {
    unique = true; trim = '.';
    if (mode === 'fixed') { trim = 'k'; unique = false; }
    const strs = finite.map(v => format_float_positional(v, precision, unique, true, trim, plus));
    const parts = strs.map(s => { const [ip, fp = ''] = s.split('.'); return { ip, fp }; });
    pad_left = Math.max(...parts.map(p => p.ip.length));
    pad_right = Math.max(...parts.map(p => p.fp.length));
    if (mode === 'fixed' || mode === 'maxprec_equal') { precision = pad_right; min_digits = pad_right; trim = 'k'; }
    else { trim = '.'; min_digits = 0; }
  }
  if (sign === ' ' && !finite.some(v => v < 0 || Object.is(v, -0))) pad_left += 1;
  if (values.length !== finite.length) {
    const neginf = sign !== '-' || values.some(v => v === -Infinity);
    const offset = pad_right + 1;
    pad_left = Math.max(pad_left, opt.nanstr.length - offset, opt.infstr.length + (neginf ? 1 : 0) - offset);
  }
  return (x: number) => {
    if (!Number.isFinite(x)) {
      const ret = Number.isNaN(x) ? (plus ? '+' : '') + opt.nanstr : (x < 0 ? '-' : (plus ? '+' : '')) + opt.infstr;
      return ' '.repeat(Math.max(0, pad_left + pad_right + 1 - ret.length)) + ret;
    }
    const sgn = sign === ' ' ? false : plus;
    if (exp_format) return format_float_scientific(x, precision, unique, trim, sgn, pad_left, exp_size, min_digits);
    return format_float_positional(x, precision, unique, true, trim, sgn, pad_left, pad_right, min_digits);
  };
}

function make_formatter(a: NDArray, values: any[], opt: PrintOptions, summary: boolean): Formatter {
  const dt = a.dtype;
  const custom = opt.formatter;
  const pick = (names: string[]) => { if (!custom) return null; for (const n of ['all', ...names]) if (custom[n]) return custom[n]; return null; };
  if (dt === bool) {
    const f = pick(['bool']); if (f) return (v: any) => f(!!v);
    return (v: any) => (v ? (a.shape.length ? ' True' : 'True') : 'False');
  }
  if (dtype_is_integer(dt)) {
    const f = pick(['int', 'int_kind']); if (f) return f;
    const strs = values.map(v => String(v));
    const w = Math.max(0, ...strs.map(s => s.length));
    return (v: any) => String(v).padStart(w);
  }
  if (dt === 'float16' || dt === 'float32' || dt === 'float64') {
    const f = pick(['float', 'float_kind']); if (f) return f;
    const nums = values.map(Number);
    const ff = float_formatter(nums, opt);
    return (v: any) => ff(Number(v));
  }
  if (dt === complex128 || dt === 'complex64') {
    const f = pick(['complexfloat', 'complex_kind']); if (f) return f;
    const re = float_formatter(values.map((z: Complex) => z.re), opt);
    const im = float_formatter(values.map((z: Complex) => z.im), opt, '+');
    return (z: Complex) => {
      const r = re(z.re);
      let i = im(z.im);
      const sp = i.replace(/\s+$/, '').length;
      i = i.slice(0, sp) + 'j' + i.slice(sp);
      return r + i;
    };
  }
  const f = pick(['object', 'numpystr']); if (f) return f;
  return (v: any) => typeof v === 'string' ? `'${v}'` : String(v);
}

// ==============================
//   array2string
// ==============================

function extend_line(s: string, line: string, word: string, line_width: number, next_line_prefix: string): [string, string] {
  let needs_wrap = line.length + word.length > line_width;
  if (line.length <= next_line_prefix.length) needs_wrap = false; // wrapping would not help
  if (needs_wrap) { s += line.replace(/\s+$/, '') + '\n'; line = next_line_prefix; }
  return [s, line + word];
}

function format_array(a: NDArray, flat: any[], fmt: Formatter, line_width: number, next_line_prefix: string, separator: string, edge_items: number, summary_insert: string): string {
  const shape = a.shape, nd = shape.length;
  const strides: number[] = new Array(nd);
  let acc = 1;
  for (let i = nd - 1; i >= 0; i--) { strides[i] = acc; acc *= shape[i]; }
  const recurse = (axis: number, offset: number, hanging_indent: string, curr_width: number): string => {
    const axes_left = nd - axis;
    if (axes_left === 0) return fmt(flat[offset]);
    const next_hanging_indent = hanging_indent + ' ';
    const next_width = curr_width - 1;
    const a_len = shape[axis];
    const show_summary = summary_insert !== '' && 2 * edge_items < a_len;
    const leading = show_summary ? edge_items : 0;
    const trailing = show_summary ? edge_items : a_len;
    let s = '';
    if (axes_left === 1) {
      const elem_width = curr_width - Math.max(separator.replace(/\s+$/, '').length, 1);
      let line = hanging_indent;
      for (let i = 0; i < leading; i++) {
        const word = recurse(axis + 1, offset + i * strides[axis], next_hanging_indent, next_width);
        [s, line] = extend_line(s, line, word, elem_width, hanging_indent);
        line += separator;
      }
      if (show_summary) { [s, line] = extend_line(s, line, summary_insert, elem_width, hanging_indent); line += separator; }
      for (let i = trailing; i > 1; i--) {
        const word = recurse(axis + 1, offset + (a_len - i) * strides[axis], next_hanging_indent, next_width);
        [s, line] = extend_line(s, line, word, elem_width, hanging_indent);
        line += separator;
      }
      const word = recurse(axis + 1, offset + (a_len - 1) * strides[axis], next_hanging_indent, next_width);
      [s, line] = extend_line(s, line, word, curr_width - 1 > 0 ? elem_width + Math.max(separator.replace(/\s+$/, '').length, 1) - 1 : elem_width, hanging_indent);
      s += line;
    } else {
      const line_sep = separator.replace(/\s+$/, '') + '\n'.repeat(axes_left - 1);
      for (let i = 0; i < leading; i++) {
        const nested = recurse(axis + 1, offset + i * strides[axis], next_hanging_indent, next_width);
        s += hanging_indent + nested + line_sep;
      }
      if (show_summary) s += hanging_indent + summary_insert + line_sep;
      for (let i = trailing; i > 1; i--) {
        const nested = recurse(axis + 1, offset + (a_len - i) * strides[axis], next_hanging_indent, next_width);
        s += hanging_indent + nested + line_sep;
      }
      const nested = recurse(axis + 1, offset + (a_len - 1) * strides[axis], next_hanging_indent, next_width);
      s += hanging_indent + nested;
    }
    return '[' + s.slice(hanging_indent.length) + ']';
  };
  return recurse(0, 0, next_line_prefix, line_width);
}

export interface Array2StringOptions {
  max_line_width?: number | null; precision?: number | null; suppress_small?: boolean | null; separator?: string; prefix?: string; style?: any;
  formatter?: PrintOptions['formatter']; threshold?: number | null; edgeitems?: number | null; sign?: PrintOptions['sign'] | null; floatmode?: PrintOptions['floatmode'] | null; suffix?: string; legacy?: any;
}

export function array2string(a: NDArray, o: Array2StringOptions = {}): string {
  a = asarray(a);
  const opt: PrintOptions = { ...options };
  if (o.max_line_width != null) opt.linewidth = o.max_line_width;
  if (o.precision != null) opt.precision = o.precision;
  if (o.suppress_small != null) opt.suppress = o.suppress_small;
  if (o.formatter != null) opt.formatter = o.formatter;
  if (o.threshold != null) opt.threshold = o.threshold;
  if (o.edgeitems != null) opt.edgeitems = o.edgeitems;
  if (o.sign != null) opt.sign = o.sign;
  if (o.floatmode != null) opt.floatmode = o.floatmode;
  const separator = o.separator ?? ' ', prefix = o.prefix ?? '', suffix = o.suffix ?? '';
  if (a.shape.length === 0) {
    const flat = a.flat as any;
    const f = make_formatter(a, [flat[0]], opt, false);
    return f(flat[0]).trim().length ? f(flat[0]) : f(flat[0]);
  }
  if (a.size === 0) return '[]';
  const summary = a.size > opt.threshold;
  const summary_insert = summary ? '...' : '';
  // Values used to compute the common format: only the displayed ones when summarizing
  let shown: any[];
  if (summary) {
    const maps = a.shape.map(n => n > 2 * opt.edgeitems ? [...range(opt.edgeitems), ...range(opt.edgeitems).map(i => n - opt.edgeitems + i)] : range(n));
    shown = [];
    const strides = a.shape.map((_, i) => a.shape.slice(i + 1).reduce((p, q) => p * q, 1));
    const flat0 = a.flat as any;
    const rec = (ax: number, off: number) => { if (ax === maps.length) { shown.push(flat0[off]); return; } for (const i of maps[ax]) rec(ax + 1, off + i * strides[ax]); };
    rec(0, 0);
  } else shown = Array.from(a.flat as any);
  const flat = a.flat as any;
  const fmt = make_formatter(a, shown, opt, summary);
  const next_line_prefix = ' ' + ' '.repeat(prefix.length);
  return format_array(a, flat, fmt, opt.linewidth - suffix.length, next_line_prefix, separator, opt.edgeitems, summary_insert);
}

// dtypes that numpy omits from the repr (ndarray-js has int32 as the default integer)
const IMPLIED_REPR_DTYPES = new Set(['int32', 'int64', 'float64', 'bool', 'complex128']);

/** Python-like str() of a scalar */
function scalar_str(v: any, dt: string): string {
  if (dt === 'bool') return v ? 'True' : 'False';
  if (v instanceof Complex) {
    const f = (x: number) => Number.isNaN(x) ? 'nan' : (x === Infinity ? 'inf' : x === -Infinity ? '-inf' : (Number.isInteger(x) && Math.abs(x) < 1e16 ? String(x) : String(x)));
    const neg = v.im < 0 || Object.is(v.im, -0);
    const re = v.re === 0 && !Object.is(v.re, -0) ? '' : f(v.re);
    const im = f(Math.abs(v.im)) + 'j';
    return re === '' ? `${neg ? '-' : ''}${im}`.replace(/^/, '') : `(${re}${neg ? '-' : '+'}${im})`;
  }
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return v < 0 ? '-inf' : 'inf';
    if (dt.startsWith('float')) return Number.isInteger(v) && Math.abs(v) < 1e16 ? v.toFixed(1).replace(/\.0$/, '.0') : String(v);
    return String(v);
  }
  return String(v);
}

export function array_repr(arr: NDArray, max_line_width: number | null = null, precision: number | null = null, suppress_small: boolean | null = null): string {
  arr = asarray(arr);
  if (options.override_repr) return options.override_repr(arr);
  const prefix = 'array(';
  const lw = max_line_width ?? options.linewidth;
  let lst: string;
  if (arr.shape.length === 0) lst = array2string(arr, { max_line_width: lw, precision, suppress_small, separator: ', ', prefix, suffix: ')' });
  else if (arr.size > 0 || (arr.shape.length === 1 && arr.shape[0] === 0)) lst = array2string(arr, { max_line_width: lw, precision, suppress_small, separator: ', ', prefix, suffix: ')' });
  else lst = '[]';
  const extras: string[] = [];
  const shape_str = `(${arr.shape.join(', ')}${arr.shape.length === 1 ? ',' : ''})`;
  if ((arr.size === 0 && !(arr.shape.length === 1 && arr.shape[0] === 0)) || arr.size > options.threshold) extras.push(`shape=${shape_str}`);
  if (!IMPLIED_REPR_DTYPES.has(arr.dtype) || arr.size === 0) extras.push(`dtype=${arr.dtype}`);
  const arr_str = prefix + lst;
  if (!extras.length) return arr_str + ')';
  const extra_str = extras.join(', ') + ')';
  const last_line_len = arr_str.length - (arr_str.lastIndexOf('\n') + 1);
  let spacer = ' ';
  if (last_line_len + extra_str.length + 1 > lw) spacer = '\n' + ' '.repeat(prefix.length);
  return arr_str + ',' + spacer + extra_str;
}

export function array_str(a: NDArray, max_line_width: number | null = null, precision: number | null = null, suppress_small: boolean | null = null): string {
  a = asarray(a);
  if (a.shape.length === 0) return scalar_str((a.flat as any)[0], a.dtype);
  return array2string(a, { max_line_width, precision, suppress_small, separator: ' ', prefix: '' });
}

// ==============================
//   Numbers as text
// ==============================

export function binary_repr(num: number, width: number | null = null): string {
  if (!Number.isInteger(num)) throw new TypeError(`'float' object cannot be interpreted as an integer`);
  if (num >= 0) {
    const s = num.toString(2);
    if (width !== null && s.length > width) throw new Error(`Insufficient bit width=${width} provided for binwidth=${s.length}`);
    return width === null ? s : s.padStart(width, '0');
  }
  if (width === null) return '-' + (-num).toString(2);
  if (num < -Math.pow(2, width - 1)) throw new Error(`Insufficient bit width=${width} provided for binwidth=${(-num).toString(2).length + 1}`);
  return (Math.pow(2, width) + num).toString(2).padStart(width, '1');
}

export function base_repr(number: number, base: number = 2, padding: number = 0): string {
  const digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  if (base > digits.length) throw new ValueError(`Bases greater than 36 not handled in base_repr.`);
  if (base < 2) throw new ValueError(`Bases less than 2 not handled in base_repr.`);
  const num = Math.abs(number);
  const res: string[] = [];
  let n = num;
  if (n === 0) res.push('0');
  while (n > 0) { res.push(digits[n % base]); n = Math.floor(n / base); }
  for (let i = 0; i < padding; i++) res.push('0');
  if (number < 0) res.push('-');
  return res.reverse().join('');
}

export function packbits(a: NDArray, axis: number | null, bitorder: 'big' | 'little'): NDArray {
  a = asarray(a);
  if (a.dtype !== bool && !dtype_is_integer(a.dtype)) throw new TypeError(`Expected an input array of integer or boolean data type`);
  if (bitorder !== 'big' && bitorder !== 'little') throw new ValueError(`'bitorder' must be either 'little' or 'big'`);
  if (axis === null) { a = reshape(a, [a.size]); axis = 0; }
  const ax = normalize_axis(axis, a.shape.length);
  const moved = ax === a.shape.length - 1 ? a : swapAxes(a, ax, -1);
  const n = moved.shape[moved.shape.length - 1];
  const outLen = Math.ceil(n / 8);
  const lanes = n === 0 ? moved.shape.slice(0, -1).reduce((p, q) => p * q, 1) : moved.size / n;
  const f = moved.flat as any;
  const buffer = new_buffer(lanes * outLen, int32);
  for (let l = 0; l < lanes; l++) for (let b = 0; b < outLen; b++) {
    let v = 0;
    for (let k = 0; k < 8; k++) {
      const idx = b * 8 + k;
      const bit = idx < n && f[l * n + idx] ? 1 : 0;
      v |= bit << (bitorder === 'big' ? 7 - k : k);
    }
    buffer[l * outLen + b] = v;
  }
  const out = new NDArray(buffer, [...moved.shape.slice(0, -1), outLen], int32);
  return ax === a.shape.length - 1 ? out : swapAxes(out, ax, -1);
}

export function unpackbits(a: NDArray, axis: number | null, count: number | null, bitorder: 'big' | 'little'): NDArray {
  a = asarray(a);
  if (a.dtype !== 'uint8') throw new TypeError(`Expected an input array of unsigned byte data type`);
  if (bitorder !== 'big' && bitorder !== 'little') throw new ValueError(`'bitorder' must be either 'little' or 'big'`);
  if (axis === null) { a = reshape(a, [a.size]); axis = 0; }
  const ax = normalize_axis(axis, a.shape.length);
  const moved = ax === a.shape.length - 1 ? a : swapAxes(a, ax, -1);
  const n = moved.shape[moved.shape.length - 1];
  const total = n * 8;
  const outLen = count === null ? total : (count < 0 ? Math.max(0, total + count) : count);
  const lanes = n === 0 ? 0 : moved.size / n;
  const f = moved.flat as any;
  const buffer = new_buffer(lanes * outLen, int32);
  for (let l = 0; l < lanes; l++) for (let i = 0; i < outLen; i++) {
    if (i >= total) continue;
    const byte = f[l * n + (i >> 3)] & 0xff, k = i & 7;
    buffer[l * outLen + i] = (byte >> (bitorder === 'big' ? 7 - k : k)) & 1;
  }
  const out = new NDArray(buffer, [...moved.shape.slice(0, -1), outLen], int32);
  return ax === a.shape.length - 1 ? out : swapAxes(out, ax, -1);
}
