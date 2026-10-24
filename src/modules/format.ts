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
//   Float formatting
// ==============================

/** Digits of x (unique shortest representation, or rounded to `precision` fractional digits) */
function frac_digits(x: number, precision: number | null, unique: boolean): string {
  // returns the fractional digits (without trailing zeros if unique)
  let s = unique ? repr_plain(Math.abs(x)) : Math.abs(x).toFixed(precision ?? 0);
  if (!unique) return s.split('.')[1] || '';
  let frac = s.split('.')[1] || '';
  if (precision !== null && frac.length > precision) frac = Math.abs(x).toFixed(precision).split('.')[1] || '';
  return frac;
}
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

export function format_float_positional(x: number, precision: number | null = null, unique: boolean = true, fractional: boolean = true, trim: 'k' | '.' | '0' | '-' = 'k', sign: boolean = false, pad_left: number | null = null, pad_right: number | null = null, min_digits: number | null = null): string {
  if (!['k', '.', '0', '-'].includes(trim)) throw new ValueError(`trim must be one of 'k', '.', '0', '-'`);
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return (x < 0 ? '-' : (sign ? '+' : '')) + 'inf';
  const neg = x < 0 || Object.is(x, -0);
  let ip: string, fp: string;
  if (unique) {
    let v = Math.abs(x);
    if (precision !== null && !fractional) {
      // significant digits
      const e = v === 0 ? 0 : Math.floor(Math.log10(v));
      const dec = Math.max(0, precision - 1 - e);
      const t = frac_digits(v, dec, true);
      [ip, fp] = [repr_plain(Math.trunc(v)).split('.')[0], t];
    } else {
      const s = repr_plain(v);
      ip = s.split('.')[0];
      fp = frac_digits(v, precision, true);
    }
  } else {
    const p = precision ?? 8;
    const fixed = (v: number, d: number) => Math.abs(v) >= 1e21 ? exact_integer_string(Math.abs(v)) + (d > 0 ? '.' + '0'.repeat(d) : '') : Math.abs(v).toFixed(d);
    const s = fractional ? fixed(x, p) : fixed(Number(Math.abs(x).toPrecision(p)), Math.max(0, p - 1 - Math.floor(Math.log10(Math.abs(x) || 1))));
    [ip, fp = ''] = s.split('.');
  }
  if (min_digits !== null) fp = fp.padEnd(min_digits, '0');
  if (trim === '-' || trim === '.' || trim === '0') fp = fp.replace(/0+$/, '');
  if (trim === '0' && fp === '') fp = '0';
  let out = ip + (trim === '-' && fp === '' ? '' : '.') + fp;
  if (pad_right !== null) out = out.padEnd(out.indexOf('.') + 1 + pad_right, ' ');
  const signStr = neg ? '-' : (sign ? '+' : '');
  out = signStr + out;
  if (pad_left !== null) { const d = out.indexOf('.'); const left = d < 0 ? out.length : d; out = ' '.repeat(Math.max(0, pad_left - left)) + out; }
  return out;
}

export function format_float_scientific(x: number, precision: number | null = null, unique: boolean = true, trim: 'k' | '.' | '0' | '-' = 'k', sign: boolean = false, pad_left: number | null = null, exp_digits: number | null = null, min_digits: number | null = null): string {
  if (!['k', '.', '0', '-'].includes(trim)) throw new ValueError(`trim must be one of 'k', '.', '0', '-'`);
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return (x < 0 ? '-' : (sign ? '+' : '')) + 'inf';
  const neg = x < 0 || Object.is(x, -0);
  const a = Math.abs(x);
  let s: string;
  if (unique) {
    s = a.toExponential();                               // shortest representation
    if (precision !== null) { const f = (s.split('e')[0].split('.')[1] || '').length; if (f > precision) s = a.toExponential(precision); }
  } else s = a.toExponential(precision ?? 8);
  let [mant, exp] = s.split('e');
  let [ip, fp = ''] = mant.split('.');
  if (min_digits !== null) fp = fp.padEnd(min_digits, '0');
  if (unique || trim !== 'k') fp = (trim === 'k') ? fp : fp.replace(/0+$/, '');
  if (trim === '0' && fp === '') fp = '0';
  const en = parseInt(exp, 10);
  const edigits = String(Math.abs(en)).padStart(Math.max(exp_digits ?? 2, 1), '0');
  let out = ip + (trim === '-' && fp === '' ? '' : '.') + fp + 'e' + (en < 0 ? '-' : '+') + edigits;
  out = (neg ? '-' : (sign ? '+' : '')) + out;
  if (pad_left !== null) { const d = out.indexOf('.'); const left = d < 0 ? out.indexOf('e') : d; out = ' '.repeat(Math.max(0, pad_left - left)) + out; }
  return out;
}

// ==============================
//   Element formatters
// ==============================

type Formatter = (v: any) => string;

function float_formatter(values: number[], opt: PrintOptions): Formatter {
  const finite = values.filter(Number.isFinite);
  const abs = finite.map(Math.abs).filter(v => v !== 0);
  const maxv = abs.length ? Math.max(...abs) : 0, minv = abs.length ? Math.min(...abs) : 0;
  const mode = opt.floatmode;
  const fixed = mode === 'fixed', uniq = mode === 'unique';
  const precision = opt.precision;
  const useExp = finite.length > 0 && (maxv >= 1e8 || (!opt.suppress && (minv < 0.0001 || maxv / minv > 1000)));
  const signStr = (neg: boolean) => neg ? '-' : (opt.sign === '+' ? '+' : (opt.sign === ' ' ? ' ' : ''));
  const special = (v: number) => Number.isNaN(v) ? opt.nanstr : (v > 0 ? (opt.sign === '+' ? '+' : opt.sign === ' ' ? ' ' : '') + opt.infstr : '-' + opt.infstr);
  if (useExp) {
    const mants = finite.map(v => {
      const s = (uniq || !fixed) ? Math.abs(v).toExponential() : Math.abs(v).toExponential(precision);
      let m = s.split('e')[0];
      let frac = (m.split('.')[1] || '');
      if (!fixed && frac.length > precision) { frac = Math.abs(v).toExponential(precision).split('e')[0].split('.')[1] || ''; frac = frac.replace(/0+$/, ''); }
      return frac;
    });
    const fracLen = fixed ? precision : (mode === 'unique' ? 0 : Math.max(0, ...mants.map(f => f.length)));
    const exps = finite.map(v => Math.abs(parseInt(Math.abs(v).toExponential().split('e')[1], 10)));
    const expDigits = Math.max(2, ...exps.map(e => String(e).length));
    const body = (v: number) => {
      const a = Math.abs(v);
      let [m, e] = a.toExponential(Math.max(fracLen, 0) > 20 ? 20 : fracLen).split('e');
      if (uniq) { const sh = a.toExponential(); [m, e] = sh.split('e'); if ((m.split('.')[1] || '').length > precision) [m, e] = a.toExponential(precision).split('e'); }
      if (!m.includes('.')) m += '.';
      else m = m;
      const en = parseInt(e, 10);
      return m.replace(/^(\d)$/, '$1') + 'e' + (en < 0 ? '-' : '+') + String(Math.abs(en)).padStart(expDigits, '0');
    };
    const strs = values.map(v => Number.isFinite(v) ? signStr(v < 0 || Object.is(v, -0)) + body(v) : special(v));
    const w = Math.max(...strs.map(s => s.length));
    return (v: number) => (Number.isFinite(v) ? signStr(v < 0 || Object.is(v, -0)) + body(v) : special(v)).padStart(w);
  }
  // positional notation
  const fracs = finite.map(v => frac_digits(v, precision, !fixed));
  let fracLen: number;
  if (fixed) fracLen = precision;
  else if (uniq) fracLen = Math.max(0, ...fracs.map(f => f.length));
  else fracLen = Math.max(0, ...fracs.map(f => Math.min(f.length, precision)));
  const equalMode = mode === 'maxprec_equal';
  const strOf = (v: number) => {
    if (!Number.isFinite(v)) return special(v);
    const neg = v < 0 || Object.is(v, -0);
    const ip = repr_plain(Math.abs(v)).split('.')[0];
    let fp = frac_digits(v, precision, !fixed);
    if (fixed || equalMode) fp = (Math.abs(v).toFixed(fixed || equalMode ? Math.max(fracLen, 0) : fp.length).split('.')[1] || '');
    return signStr(neg) + ip + '.' + fp;
  };
  const strs = values.map(strOf);
  const wl = Math.max(0, ...strs.filter((_, i) => Number.isFinite(values[i])).map(s => s.indexOf('.')));
  const wr = fracLen;
  const pad = (s: string, v: number) => {
    if (!Number.isFinite(v)) return s;
    const d = s.indexOf('.');
    return s.padStart(s.length + (wl - d)).padEnd(d + (wl - d) + 1 + wr);
  };
  const padded = values.map((v, i) => pad(strs[i], v));
  const w = Math.max(...padded.map(s => s.length));
  return (v: number) => pad(strOf(v), v).padStart(w);
}

function make_formatter(a: NDArray, values: any[], opt: PrintOptions, summary: boolean): Formatter {
  const dt = a.dtype;
  const custom = opt.formatter;
  const pick = (names: string[]) => { if (!custom) return null; for (const n of ['all', ...names]) if (custom[n]) return custom[n]; return null; };
  if (dt === bool) {
    const f = pick(['bool']); if (f) return (v: any) => f(!!v);
    return (v: any) => (v ? (a.shape.length ? ' True' : 'True') : 'False');
  }
  if (dt === 'int32' || dt === 'int64') {
    const f = pick(['int', 'int_kind']); if (f) return f;
    const strs = values.map(v => String(v));
    const w = Math.max(0, ...strs.map(s => s.length));
    return (v: any) => String(v).padStart(w);
  }
  if (dt === 'float32' || dt === 'float64') {
    const f = pick(['float', 'float_kind']); if (f) return f;
    const nums = values.map(Number);
    const ff = float_formatter(nums, opt);
    return (v: any) => ff(Number(v));
  }
  if (dt === complex128) {
    const f = pick(['complexfloat', 'complex_kind']); if (f) return f;
    const re = float_formatter(values.map((z: Complex) => z.re), opt);
    const imOpt = { ...opt, sign: '+' as const };
    const im = float_formatter(values.map((z: Complex) => z.im), imOpt);
    return (z: Complex) => {
      const r = re(z.re).trim().padStart(0), i = im(z.im);
      const rs = re(z.re), is = i.replace(/^(\s*)([+-])/, '$2$1');
      return rs + is.replace(/\s+$/, m => m) + 'j';
    };
  }
  const f = pick(['object', 'numpystr']); if (f) return f;
  return (v: any) => typeof v === 'string' ? `'${v}'` : String(v);
}

// ==============================
//   array2string
// ==============================

function extend_line(s: string, line: string, word: string, line_width: number, next_line_prefix: string): [string, string] {
  const needs_wrap = line.length + word.length > line_width;
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

const DEFAULT_REPR_DTYPES = new Set(['int32', 'int64', 'float64', 'bool', 'complex128']);

export function array_repr(arr: NDArray, max_line_width: number | null = null, precision: number | null = null, suppress_small: boolean | null = null): string {
  arr = asarray(arr);
  if (options.override_repr) return options.override_repr(arr);
  const prefix = 'array(';
  const dtype = arr.dtype;
  const needsDtype = !DEFAULT_REPR_DTYPES.has(dtype) || (arr.size === 0 && arr.shape.length === 1);
  const lw = max_line_width ?? options.linewidth;
  let arr_str: string;
  if (arr.shape.length === 0) arr_str = prefix + array2string(arr, { max_line_width: lw, precision, suppress_small, separator: ', ', prefix }) + ')';
  else if (arr.size === 0 && arr.shape.length >= 1 && arr.shape.every(d => d === 0) === false && arr.shape.length === 1) arr_str = prefix + '[]';
  else arr_str = prefix + array2string(arr, { max_line_width: lw, precision, suppress_small, separator: ', ', prefix, suffix: ')' }) + (needsDtype ? '' : ')');
  if (!needsDtype) return arr_str;
  const dtype_str = `dtype=${dtype})`;
  const last_line_len = arr_str.length - (arr_str.lastIndexOf('\n') + 1);
  let spacer = ' ';
  if (last_line_len + dtype_str.length + 1 > lw) spacer = '\n' + ' '.repeat(prefix.length);
  return arr_str + ',' + spacer + dtype_str;
}

export function array_str(a: NDArray, max_line_width: number | null = null, precision: number | null = null, suppress_small: boolean | null = null): string {
  a = asarray(a);
  if (a.shape.length === 0) return array2string(a, {});
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
