//@ts-check
/**
 * numpy.strings (and the legacy numpy.char): elementwise string functions on arrays of strings.
 * String arrays are arrays with dtype "object" that hold JS strings; arguments broadcast like in any ufunc.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { DType, bool, int64, object, new_buffer } from "../dtypes";
import { n_ary_elementwise } from "../array/operators";
import { fixed_exact, exp_exact } from "./exact-format";

type Arr = any;
const str = (x: any): string => typeof x === "string" ? x : String(x);

function map(resolver: DType, fn: (...a: any[]) => any, ...args: Arr[]): NDArray {
  return n_ary_elementwise(() => resolver, fn, args.map(a => asarray(a)));
}
const asStr = (a: any) => asarray(a).dtype === 'object' ? asarray(a) : (() => { throw new TypeError(`string operation on non-string array`); })();

// ==============================
//   Python str helpers
// ==============================

const WS = ' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0                　';
const isWs = (c: string) => WS.includes(c);

function py_strip(s: string, chars: string | null, left: boolean, right: boolean): string {
  const test = chars === null ? isWs : (c: string) => chars.includes(c);
  let i = 0, j = s.length;
  if (left) while (i < j && test(s[i])) i++;
  if (right) while (j > i && test(s[j - 1])) j--;
  return s.slice(i, j);
}

const py_center = (s: string, width: number, fill: string) => {
  if (s.length >= width) return s;
  const marg = width - s.length;
  const left = Math.floor(marg / 2) + (marg & width & 1);
  return fill.repeat(left) + s + fill.repeat(marg - left);
};
function check_fill(fill: string) { if (typeof fill !== "string" || [...fill].length !== 1) throw new TypeError(`The fill character must be exactly one character long`); }

function py_title(s: string): string {
  let out = '', prevCased = false;
  for (const c of s) {
    const isLetter = U('\\p{L}').test(c);
    if (isLetter) out += prevCased ? c.toLowerCase() : c.toUpperCase();
    else out += c;
    prevCased = isLetter;
  }
  return out;
}
const py_capitalize = (s: string) => s.length ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s;
const py_swapcase = (s: string) => Array.from(s, c => c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase()).join('');

function py_expandtabs(s: string, tabsize: number): string {
  let out = '', col = 0;
  for (const c of s) {
    if (c === '\t') { if (tabsize > 0) { const n = tabsize - (col % tabsize); out += ' '.repeat(n); col += n; } }
    else { out += c; col = (c === '\n' || c === '\r') ? 0 : col + 1; }
  }
  return out;
}
function py_zfill(s: string, width: number): string {
  if (s.length >= width) return s;
  const fill = width - s.length;
  if (s[0] === '+' || s[0] === '-') return s[0] + '0'.repeat(fill) + s.slice(1);
  return '0'.repeat(fill) + s;
}

/** Python slice indices for start/end arguments of find/count/startswith */
function bounds(len: number, start: number | null, end: number | null): [number, number] {
  const fix = (v: number | null, dflt: number) => { if (v === null || v === undefined) return dflt; if (v < 0) v += len; return Math.min(Math.max(v, 0), len); };
  return [fix(start, 0), fix(end, len)];
}

function py_find(s: string, sub: string, start: number | null, end: number | null, right: boolean): number {
  const [a, b] = bounds(s.length, start, end);
  const sl = s.slice(a, b);
  const i = right ? sl.lastIndexOf(sub) : sl.indexOf(sub);
  if (a > s.length) return -1;
  return i < 0 ? -1 : i + a;
}
function py_count(s: string, sub: string, start: number | null, end: number | null): number {
  const [a, b] = bounds(s.length, start, end);
  const sl = s.slice(a, b);
  if (sub === '') return sl.length + 1;
  let n = 0, i = 0;
  while ((i = sl.indexOf(sub, i)) >= 0) { n++; i += sub.length; }
  return n;
}

// ---- printf-style formatting (str % value)
function py_format(fmt: string, value: any): string {
  const values = Array.isArray(value) ? value : [value];
  let k = 0;
  const next = () => { if (k >= values.length) throw new TypeError('not enough arguments for format string'); return values[k++]; };
  const out = fmt.replace(/%([#0\- +]*)(\*|\d+)?(?:\.(\*|\d+))?([diouxXeEfFgGcrsa%])/g, (_m, flags: string, w: string, p: string, conv: string) => {
    if (conv === '%') return '%';
    const width = w === '*' ? Number(next()) : (w ? parseInt(w, 10) : 0);
    const prec = p === undefined ? null : (p === '*' ? Number(next()) : parseInt(p, 10));
    const left = flags.includes('-'), zero = flags.includes('0') && !left, plus = flags.includes('+'), space = flags.includes(' '), alt = flags.includes('#');
    const v = next();
    let body: string;
    const signed = (n: number, digits: string) => (n < 0 || Object.is(n, -0) ? '-' : plus ? '+' : space ? ' ' : '') + digits;
    switch (conv) {
      case 's': body = str(v); if (prec !== null) body = body.slice(0, prec); break;
      case 'r': case 'a': body = typeof v === 'string' ? `'${v}'` : str(v); break;
      case 'c': body = typeof v === 'number' ? String.fromCodePoint(v) : str(v); break;
      case 'd': case 'i': case 'u': { const n = Math.trunc(Number(v)); let d = Math.abs(n).toString(); if (prec !== null) d = d.padStart(prec, '0'); body = signed(n, d); break; }
      case 'o': { const n = Math.trunc(Number(v)); body = signed(n, (alt ? '0o' : '') + Math.abs(n).toString(8)); break; }
      case 'x': case 'X': { const n = Math.trunc(Number(v)); let d = Math.abs(n).toString(16); if (conv === 'X') d = d.toUpperCase(); body = signed(n, (alt ? (conv === 'X' ? '0X' : '0x') : '') + d); break; }
      case 'e': case 'E': { const n = Number(v); const r = Number.isFinite(n) ? exp_exact(n, prec ?? 6) : null; const d = r ? `${r.mantissa}e${r.exponent < 0 ? '-' : '+'}${String(Math.abs(r.exponent)).padStart(2, '0')}` : (Number.isNaN(n) ? 'nan' : 'inf'); body = signed(n, conv === 'E' ? d.toUpperCase() : d); break; }
      case 'f': case 'F': { const n = Number(v); const d = Number.isFinite(n) ? fixed_exact(n, prec ?? 6) : (Number.isNaN(n) ? 'nan' : 'inf'); body = signed(n, d); break; }
      case 'g': case 'G': {
        const n = Number(v); const P = prec === null ? 6 : (prec === 0 ? 1 : prec);
        if (!Number.isFinite(n)) { body = signed(n, Number.isNaN(n) ? 'nan' : 'inf'); break; }
        const e = n === 0 ? 0 : exp_exact(n, P - 1).exponent;
        let d: string;
        if (e < -4 || e >= P) { const r = exp_exact(n, P - 1); let m = r.mantissa; if (!alt && m.includes('.')) m = m.replace(/0+$/, '').replace(/\.$/, ''); d = `${m}e${r.exponent < 0 ? '-' : '+'}${String(Math.abs(r.exponent)).padStart(2, '0')}`; }
        else { d = fixed_exact(n, Math.max(0, P - 1 - e)); if (!alt && d.includes('.')) d = d.replace(/0+$/, '').replace(/\.$/, ''); }
        body = signed(n, conv === 'G' ? d.toUpperCase() : d); break;
      }
      default: body = str(v);
    }
    if (body.length < width) {
      if (left) body = body.padEnd(width);
      else if (zero && 'diouxXeEfFgG'.includes(conv)) { const sgn = /^[+\- ]/.test(body) ? body[0] : ''; body = sgn + body.slice(sgn.length).padStart(width - sgn.length, '0'); }
      else body = body.padStart(width);
    }
    return body;
  });
  return out;
}

// ==============================
//   Elementwise functions
// ==============================

export const add = (a: Arr, b: Arr) => map(object, (x, y) => str(x) + str(y), a, b);
export const multiply = (a: Arr, i: Arr) => map(object, (x, n) => (n <= 0 ? '' : str(x).repeat(n)), a, i);
export const mod = (a: Arr, values: Arr) => map(object, (f, v) => py_format(str(f), v), a, values);
export const capitalize = (a: Arr) => map(object, (x) => py_capitalize(str(x)), a);
export const lower = (a: Arr) => map(object, (x) => str(x).toLowerCase(), a);
export const upper = (a: Arr) => map(object, (x) => str(x).toUpperCase(), a);
export const swapcase = (a: Arr) => map(object, (x) => py_swapcase(str(x)), a);
export const title = (a: Arr) => map(object, (x) => py_title(str(x)), a);
export const center = (a: Arr, width: Arr, fillchar: string = ' ') => { check_fill(fillchar); return map(object, (x, w) => py_center(str(x), w, fillchar), a, width); };
export const ljust = (a: Arr, width: Arr, fillchar: string = ' ') => { check_fill(fillchar); return map(object, (x, w) => { const s = str(x); return s.length >= w ? s : s + fillchar.repeat(w - s.length); }, a, width); };
export const rjust = (a: Arr, width: Arr, fillchar: string = ' ') => { check_fill(fillchar); return map(object, (x, w) => { const s = str(x); return s.length >= w ? s : fillchar.repeat(w - s.length) + s; }, a, width); };
export const zfill = (a: Arr, width: Arr) => map(object, (x, w) => py_zfill(str(x), w), a, width);
export const expandtabs = (a: Arr, tabsize: Arr = 8) => map(object, (x, t) => py_expandtabs(str(x), t), a, tabsize);
export const strip = (a: Arr, chars: Arr = null) => chars === null ? map(object, (x) => py_strip(str(x), null, true, true), a) : map(object, (x, c) => py_strip(str(x), c === null ? null : str(c), true, true), a, chars);
export const lstrip = (a: Arr, chars: Arr = null) => chars === null ? map(object, (x) => py_strip(str(x), null, true, false), a) : map(object, (x, c) => py_strip(str(x), c === null ? null : str(c), true, false), a, chars);
export const rstrip = (a: Arr, chars: Arr = null) => chars === null ? map(object, (x) => py_strip(str(x), null, false, true), a) : map(object, (x, c) => py_strip(str(x), c === null ? null : str(c), false, true), a, chars);
export const replace = (a: Arr, old: Arr, new_: Arr, count: Arr = -1) => map(object, (x, o, n, c) => {
  const s = str(x), os = str(o), ns = str(n);
  if (c < 0) return os === '' ? s.split('').reduce((acc, ch) => acc + ns + ch, '') + ns : s.split(os).join(ns);
  let out = '', i = 0, k = 0;
  if (os === '') { for (; k < c && i <= s.length; k++) { out += ns + (s[i] ?? ''); i++; } return out + s.slice(i); }
  while (k < c) { const j = s.indexOf(os, i); if (j < 0) break; out += s.slice(i, j) + ns; i = j + os.length; k++; }
  return out + s.slice(i);
}, a, old, new_, count);
export const NO_VALUE = '__novalue__';
export const slice = (a: Arr, start: Arr = null, stop: Arr = NO_VALUE, step: Arr = null) => {
  // like python: a[start:stop:step]; when only `start` is given it is the stop
  if (stop === NO_VALUE) { stop = start; start = null; }
  return map(object, (x, b, e, st) => {
    const s = [...str(x)], n = s.length, stp = st === null ? 1 : st;
    if (stp === 0) throw new Error("slice step cannot be zero");
    const idx = (v: number | null, dflt: number, lo: number, hi: number) => v === null ? dflt : Math.min(Math.max(v < 0 ? v + n : v, lo), hi);
    const out: string[] = [];
    if (stp > 0) for (let i = idx(b, 0, 0, n); i < idx(e, n, 0, n); i += stp) out.push(s[i]);
    else for (let i = idx(b, n - 1, -1, n - 1); i > idx(e, -1, -1, n - 1); i += stp) out.push(s[i]);
    return out.join('');
  }, a, start, stop, step);
};
export function translate(a: Arr, table: string | { [code: number]: string | null }, deletechars: string | null = null): NDArray {
  return map(object, (x) => {
    let out = '';
    for (const c of str(x)) {
      const code = c.codePointAt(0)!;
      let m: string;
      if (typeof table === "string") m = code < table.length ? table[code] : c;
      else { const t = table[code]; m = t === undefined ? c : (t === null ? '' : t); }
      // as numpy: the characters to delete are removed from the translated text
      if (deletechars !== null && deletechars.includes(m)) continue;
      out += m;
    }
    return out;
  }, a);
}
export const decode = (a: Arr, encoding: string = 'utf-8', errors: string = 'strict') => map(object, (x) => {
  if (typeof x === "string") return x;
  const bytes = x instanceof Uint8Array ? x : new Uint8Array(x);
  return new TextDecoder(encoding === 'utf8' ? 'utf-8' : encoding, { fatal: errors === 'strict' }).decode(bytes);
}, a);
export const encode = (a: Arr, encoding: string = 'utf-8', errors: string = 'strict') => map(object, (x) => {
  if (!/^utf-?8$/i.test(encoding)) throw new Error(`encoding '${encoding}' is not supported (only utf-8)`);
  return new TextEncoder().encode(str(x));
}, a);

export function partition(a: Arr, sep: Arr): NDArray[] {
  const parts = (x: any, s: any) => { const t = str(x), p = str(s); if (p === '') throw new Error('empty separator'); const i = t.indexOf(p); return i < 0 ? [t, '', ''] : [t.slice(0, i), p, t.slice(i + p.length)]; };
  const r = map(object, (x, s) => parts(x, s), a, sep);
  return [0, 1, 2].map(k => new NDArray(Array.from(r.flat as any, (p: any) => p[k]) as any, r.shape, object));
}
export function rpartition(a: Arr, sep: Arr): NDArray[] {
  const parts = (x: any, s: any) => { const t = str(x), p = str(s); if (p === '') throw new Error('empty separator'); const i = t.lastIndexOf(p); return i < 0 ? ['', '', t] : [t.slice(0, i), p, t.slice(i + p.length)]; };
  const r = map(object, (x, s) => parts(x, s), a, sep);
  return [0, 1, 2].map(k => new NDArray(Array.from(r.flat as any, (p: any) => p[k]) as any, r.shape, object));
}

// ---- comparison
const cmp = (x: any, y: any) => { const a = str(x), b = str(y); return a < b ? -1 : a > b ? 1 : 0; };
export const equal = (a: Arr, b: Arr) => map(bool, (x, y) => str(x) === str(y), a, b);
export const not_equal = (a: Arr, b: Arr) => map(bool, (x, y) => str(x) !== str(y), a, b);
export const greater = (a: Arr, b: Arr) => map(bool, (x, y) => cmp(x, y) > 0, a, b);
export const greater_equal = (a: Arr, b: Arr) => map(bool, (x, y) => cmp(x, y) >= 0, a, b);
export const less = (a: Arr, b: Arr) => map(bool, (x, y) => cmp(x, y) < 0, a, b);
export const less_equal = (a: Arr, b: Arr) => map(bool, (x, y) => cmp(x, y) <= 0, a, b);

// ---- search
export const count = (a: Arr, sub: Arr, start: Arr = 0, end: Arr = null) => map(int64, (x, s, b, e) => py_count(str(x), str(s), b, e), a, sub, start, end);
export const find = (a: Arr, sub: Arr, start: Arr = 0, end: Arr = null) => map(int64, (x, s, b, e) => py_find(str(x), str(s), b, e, false), a, sub, start, end);
export const rfind = (a: Arr, sub: Arr, start: Arr = 0, end: Arr = null) => map(int64, (x, s, b, e) => py_find(str(x), str(s), b, e, true), a, sub, start, end);
export const index = (a: Arr, sub: Arr, start: Arr = 0, end: Arr = null) => map(int64, (x, s, b, e) => { const i = py_find(str(x), str(s), b, e, false); if (i < 0) throw new Error('substring not found'); return i; }, a, sub, start, end);
export const rindex = (a: Arr, sub: Arr, start: Arr = 0, end: Arr = null) => map(int64, (x, s, b, e) => { const i = py_find(str(x), str(s), b, e, true); if (i < 0) throw new Error('substring not found'); return i; }, a, sub, start, end);
export const startswith = (a: Arr, prefix: Arr, start: Arr = 0, end: Arr = null) => map(bool, (x, p, b, e) => { const s = str(x); const [i, j] = bounds(s.length, b, e); return i <= s.length && s.slice(i, j).startsWith(str(p)) && (i + str(p).length <= j || str(p) === ''); }, a, prefix, start, end);
export const endswith = (a: Arr, suffix: Arr, start: Arr = 0, end: Arr = null) => map(bool, (x, p, b, e) => { const s = str(x); const [i, j] = bounds(s.length, b, e); return i <= j && s.slice(i, j).endsWith(str(p)); }, a, suffix, start, end);
export const str_len = (a: Arr) => map(int64, (x) => [...str(x)].length, a);

// ---- predicates
const U = (src: string) => new RegExp(src, 'u');
const pred = (re: RegExp, emptyOk = false) => (a: Arr) => map(bool, (x) => { const s = str(x); return s.length > 0 ? re.test(s) : emptyOk; }, a);
export const isalpha = pred(U('^\\p{L}+$'));
export const isalnum = pred(U('^[\\p{L}\\p{Nd}\\p{No}\\p{Nl}]+$'));
export const isdecimal = pred(U('^\\p{Nd}+$'));
export const isdigit = pred(U('^[\\p{Nd}\\u00b2\\u00b3\\u00b9\\u2070\\u2074-\\u2079\\u2080-\\u2089]+$'));
export const isnumeric = pred(U('^\\p{N}+$'));
export const isspace = (a: Arr) => map(bool, (x) => { const s = str(x); return s.length > 0 && Array.from(s).every(isWs); }, a);
export const islower = (a: Arr) => map(bool, (x) => { const s = str(x); return s !== s.toUpperCase() && s === s.toLowerCase() || (U('\\p{Ll}').test(s) && !U('\\p{Lu}|\\p{Lt}').test(s)); }, a);
export const isupper = (a: Arr) => map(bool, (x) => { const s = str(x); return U('\\p{Lu}').test(s) && !U('\\p{Ll}|\\p{Lt}').test(s); }, a);
export const istitle = (a: Arr) => map(bool, (x) => {
  const s = str(x); let prevCased = false, cased = false;
  for (const c of s) {
    const up = U('\\p{Lu}|\\p{Lt}').test(c), lo = U('\\p{Ll}').test(c);
    if (up) { if (prevCased) return false; prevCased = true; cased = true; }
    else if (lo) { if (!prevCased) return false; prevCased = true; cased = true; }
    else prevCased = false;
  }
  return cased;
}, a);

// ---- legacy numpy.char extras
export const join = (sep: Arr, seq: Arr) => map(object, (s, q) => (Array.isArray(q) || typeof q === 'string' ? Array.from(q as any) : [q]).map(str).join(str(s)), sep, seq);
export function split(a: Arr, sep: Arr = null, maxsplit: Arr = null): NDArray {
  return map(object, (x, s, m) => {
    const t = str(x);
    if (s === null) { const words = py_strip(t, null, true, true).split(/[\s\x1c-\x1f\x85]+/).filter(w => w !== ''); return m !== null && m >= 0 && words.length > m + 1 ? [...words.slice(0, m), py_strip(t, null, true, true).split(/\s+/).slice(m).join(' ')] : words; }
    const parts = t.split(str(s));
    return m !== null && m >= 0 && parts.length > m + 1 ? [...parts.slice(0, m), parts.slice(m).join(str(s))] : parts;
  }, a, sep, maxsplit);
}
export function rsplit(a: Arr, sep: Arr = null, maxsplit: Arr = null): NDArray {
  return map(object, (x, s, m) => {
    const t = str(x);
    if (s === null) { const words = t.trim().split(/\s+/).filter(w => w !== ''); return m !== null && m >= 0 && words.length > m + 1 ? [words.slice(0, words.length - m).join(' '), ...words.slice(words.length - m)] : words; }
    const parts = t.split(str(s));
    return m !== null && m >= 0 && parts.length > m + 1 ? [parts.slice(0, parts.length - m).join(str(s)), ...parts.slice(parts.length - m)] : parts;
  }, a, sep, maxsplit);
}
export const splitlines = (a: Arr, keepends: boolean = false) => map(object, (x) => {
  const t = str(x), out: string[] = [];
  const re = /\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]/g;
  let last = 0, m: RegExpExecArray | null;
  while ((m = re.exec(t)) !== null) { out.push(t.slice(last, keepends ? m.index + m[0].length : m.index)); last = m.index + m[0].length; }
  if (last < t.length) out.push(t.slice(last));
  return out;
}, a);

/** numpy.char.array / asarray: arrays of strings */
export function char_array(obj: any, itemsize: number | null = null, copy: boolean = true, unicode: boolean | null = null): NDArray {
  const a = asarray(typeof obj === "string" ? [obj] : obj);
  const flat = Array.from(a.flat as any, (v: any) => str(v));
  const out = new NDArray(flat as any, typeof obj === "string" ? [] : a.shape, object);
  return out;
}
