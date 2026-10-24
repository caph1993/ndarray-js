//@ts-check
/**
 * Reading and writing arrays: .npy / .npz files, text files (loadtxt, savetxt, genfromtxt, fromregex, fromstring),
 * raw binary files (fromfile, tofile), memory maps and the lib.format module.
 * File access needs Node.js (fs). The functions that accept `null` / byte arrays / text work everywhere.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { DType, TYPE_BY_NAME, new_buffer, convert_buffer, canonical_dtype, complex128, float64, int64, object } from "../dtypes";
import { Complex } from "../complex";
import { reshape } from "../array/shape_operations";
import { transpose } from "../array/transform";
import { to_bytes, from_bytes, as_bytes, itemsize } from "./binary";
import { py_format_values } from "./strings";

// ==============================
//   Node helpers
// ==============================

function node_module(name: string): any {
  const g: any = globalThis as any;
  const p = g.process;
  if (p && typeof p.getBuiltinModule === "function") { const m = p.getBuiltinModule(name); if (m) return m; }
  throw new Error(`Reading and writing files needs Node.js (module '${name}' is not available here)`);
}
const fs = () => node_module('fs');
const is_path = (f: any) => typeof f === "string";

// ==============================
//   .npy format
// ==============================

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
const TEXT = new TextEncoder();

export function dtype_to_descr(dtype: DType): string {
  const t = TYPE_BY_NAME[dtype];
  if (t.kind === 'O') throw new Error("Array can't be saved: object arrays need pickle, which is not supported");
  return t.str;
}
export function descr_to_dtype(descr: string): { dtype: DType, little: boolean } {
  const m = /^([<>|=]?)([a-zA-Z])(\d+)$/.exec(descr);
  if (!m) throw new Error(`descriptor '${descr}' is not supported`);
  const little = m[1] !== '>';
  const key = m[2] + m[3];
  const t = TYPE_BY_NAME[key] || TYPE_BY_NAME[m[2] === 'b' ? 'b1' : key];
  if (!t) throw new Error(`descriptor '${descr}' is not supported`);
  return { dtype: t.name as DType, little };
}

export function array_to_npy(a: NDArray): Uint8Array {
  const shape = `(${a.shape.join(', ')}${a.shape.length === 1 ? ',' : ''})`;
  let header = `{'descr': '${dtype_to_descr(a.dtype)}', 'fortran_order': False, 'shape': ${shape}, }`;
  // total length (magic 6 + version 2 + header length 2 + header + newline) must be a multiple of 64
  const pre = 10;
  const pad = (64 - ((pre + header.length + 1) % 64)) % 64;
  header = header + ' '.repeat(pad) + '\n';
  const data = to_bytes(a);
  const out = new Uint8Array(pre + header.length + data.length);
  out.set(MAGIC, 0); out[6] = 1; out[7] = 0;
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(TEXT.encode(header), pre);
  out.set(data, pre + header.length);
  return out;
}

export function npy_header(bytes: Uint8Array): { descr: string, fortran_order: boolean, shape: number[], data_offset: number } {
  for (let i = 0; i < 6; i++) if (bytes[i] !== MAGIC[i]) throw new Error("the file is not a valid .npy file (bad magic string)");
  const major = bytes[6];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let hlen: number, start: number;
  if (major === 1) { hlen = dv.getUint16(8, true); start = 10; }
  else if (major === 2 || major === 3) { hlen = dv.getUint32(8, true); start = 12; }
  else throw new Error(`unsupported .npy version ${major}`);
  const text = new TextDecoder(major === 3 ? 'utf-8' : 'latin1').decode(bytes.subarray(start, start + hlen));
  const descr = /'descr'\s*:\s*'([^']*)'/.exec(text);
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(text);
  const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(text);
  if (!descr || !fortran || !shape) throw new Error(`Header does not contain the correct keys: ${text}`);
  return { descr: descr[1], fortran_order: fortran[1] === 'True', shape: shape[1].split(',').map(s => s.trim()).filter(s => s).map(Number), data_offset: start + hlen };
}

export function npy_to_array(bytes: Uint8Array): NDArray {
  const h = npy_header(bytes);
  const { dtype, little } = descr_to_dtype(h.descr);
  const n = h.shape.reduce((p, q) => p * q, 1);
  const flat = from_bytes(bytes, dtype, n, h.data_offset, little);
  let out = new NDArray(flat, h.shape, dtype);
  if (h.fortran_order && h.shape.length > 1) {
    // data is stored column major: read it with reversed shape and transpose
    out = transpose(new NDArray(flat, [...h.shape].reverse(), dtype));
  }
  return out;
}

// ==============================
//   zip (.npz)
// ==============================

const CRC_TABLE = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function make_zip(files: { name: string, data: Uint8Array }[], compress: boolean): Uint8Array {
  let zlib: any = null;
  if (compress) { try { zlib = node_module('zlib'); } catch { zlib = null; } }
  const chunks: Uint8Array[] = [], central: Uint8Array[] = [];
  let offset = 0;
  for (const f of files) {
    const name = TEXT.encode(f.name);
    const crc = crc32(f.data);
    let body = f.data, method = 0;
    if (zlib) { const d: Uint8Array = zlib.deflateRawSync(f.data); if (d.length < f.data.length) { body = new Uint8Array(d.buffer, d.byteOffset, d.length); method = 8; } }
    const local = new Uint8Array(30 + name.length);
    const dv = new DataView(local.buffer);
    dv.setUint32(0, 0x04034b50, true); dv.setUint16(4, 20, true); dv.setUint16(6, 0, true); dv.setUint16(8, method, true);
    dv.setUint16(10, 0, true); dv.setUint16(12, 0x21, true); dv.setUint32(14, crc, true); dv.setUint32(18, body.length, true); dv.setUint32(22, f.data.length, true);
    dv.setUint16(26, name.length, true); dv.setUint16(28, 0, true);
    local.set(name, 30);
    chunks.push(local, body);
    const cd = new Uint8Array(46 + name.length);
    const cv = new DataView(cd.buffer);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true); cv.setUint16(8, 0, true); cv.setUint16(10, method, true);
    cv.setUint16(12, 0, true); cv.setUint16(14, 0x21, true); cv.setUint32(16, crc, true); cv.setUint32(20, body.length, true); cv.setUint32(24, f.data.length, true);
    cv.setUint16(28, name.length, true); cv.setUint32(42, offset, true);
    cd.set(name, 46);
    central.push(cd);
    offset += local.length + body.length;
  }
  const cdSize = central.reduce((s, c) => s + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, files.length, true); ev.setUint16(10, files.length, true); ev.setUint32(12, cdSize, true); ev.setUint32(16, offset, true);
  const all = [...chunks, ...central, end];
  const out = new Uint8Array(all.reduce((s, c) => s + c.length, 0));
  let p = 0; for (const c of all) { out.set(c, p); p += c.length; }
  return out;
}

export function read_zip(bytes: Uint8Array): { name: string, data: () => Uint8Array }[] {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("not a zip file");
  const n = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const entries: { name: string, data: () => Uint8Array }[] = [];
  for (let i = 0; i < n; i++) {
    const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true), elen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true);
    const local = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nlen));
    const dataStart = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true);
    entries.push({
      name, data: () => {
        const raw = bytes.subarray(dataStart, dataStart + csize);
        if (method === 0) return raw;
        if (method === 8) { const d = node_module('zlib').inflateRawSync(raw); return new Uint8Array(d.buffer, d.byteOffset, d.length); }
        throw new Error(`unsupported zip compression method ${method}`);
      }
    });
    p += 46 + nlen + elen + clen;
  }
  return entries;
}

/** The result of loading a .npz file: a lazy dictionary of arrays */
export class NpzFile {
  files: string[];
  f: { [name: string]: NDArray };
  private entries: { [name: string]: () => Uint8Array };
  constructor(bytes: Uint8Array) {
    const list = read_zip(bytes);
    this.entries = {};
    list.forEach(e => { this.entries[e.name.replace(/\.npy$/, '')] = e.data; });
    this.files = list.map(e => e.name.replace(/\.npy$/, ''));
    const self = this;
    this.f = new Proxy({}, { get: (_t, key: string) => self.get(key) }) as any;
  }
  get(name: string): NDArray {
    const e = this.entries[name] || this.entries[name.replace(/\.npy$/, '')];
    if (!e) throw new Error(`${name} is not a file in the archive`);
    return npy_to_array(e());
  }
  has(name: string) { return name in this.entries; }
  keys() { return this.files.slice(); }
  items(): [string, NDArray][] { return this.files.map(n => [n, this.get(n)]); }
  close() { }
  [Symbol.iterator]() { return this.files[Symbol.iterator](); }
}

// ==============================
//   save / load
// ==============================

function write_out(file: any, bytes: Uint8Array, ext: string): Uint8Array | null {
  if (file === null || file === undefined) return bytes;
  if (is_path(file)) { fs().writeFileSync(/\.[A-Za-z0-9]+$/.test(file) && file.endsWith(ext) ? file : file + ext, bytes); return null; }
  if (file && typeof file.write === "function") { file.write(bytes); return null; }
  throw new TypeError("file must be a path, null (return the bytes) or an object with a write method");
}

/** Saves an array in .npy format. With file=null the bytes are returned. */
export function save(file: any, arr: any, allow_pickle: boolean = true, fix_imports: boolean = true): Uint8Array | null {
  return write_out(file, array_to_npy(asarray(arr)), '.npy');
}

function savez_impl(file: any, args: any[], kwds: { [name: string]: any }, compress: boolean): Uint8Array | null {
  const files: { name: string, data: Uint8Array }[] = [];
  args.forEach((a, i) => files.push({ name: `arr_${i}.npy`, data: array_to_npy(asarray(a)) }));
  for (const [k, v] of Object.entries(kwds)) {
    if (files.some(f => f.name === `${k}.npy`)) throw new Error(`Cannot use un-named variables and keyword ${k}`);
    files.push({ name: `${k}.npy`, data: array_to_npy(asarray(v)) });
  }
  return write_out(file, make_zip(files, compress), '.npz');
}
export const savez = (file: any, args: any[], kwds: { [name: string]: any } = {}) => savez_impl(file, args, kwds, false);
export const savez_compressed = (file: any, args: any[], kwds: { [name: string]: any } = {}) => savez_impl(file, args, kwds, true);

export function load(file: any, mmap_mode: string | null = null, allow_pickle: boolean = false, fix_imports: boolean = true, encoding: string = 'ASCII', max_header_size: number = 10000): NDArray | NpzFile {
  let bytes: Uint8Array;
  if (is_path(file)) { const b: Uint8Array = fs().readFileSync(file); bytes = new Uint8Array(b.buffer, b.byteOffset, b.length); }
  else bytes = as_bytes(file);
  if (bytes[0] === 0x50 && bytes[1] === 0x4b) return new NpzFile(bytes);
  if (bytes[0] === 0x93) return npy_to_array(bytes);
  throw new Error("Cannot load file containing pickled data when allow_pickle=False");
}

// ==============================
//   Raw binary and text
// ==============================

export function tofile(a: NDArray, fid: any, sep: string = '', format: string = '%s'): void {
  a = asarray(a);
  if (sep === '') { write_out_raw(fid, to_bytes(a)); return; }
  const f = a.flat as any;
  const text = Array.from({ length: f.length }, (_, i) => format === '%s' ? scalar_text(f[i], a.dtype) : py_format_values(format, f[i])).join(sep);
  write_out_raw(fid, TEXT.encode(text));
}
function write_out_raw(fid: any, bytes: Uint8Array) {
  if (is_path(fid)) fs().writeFileSync(fid, bytes);
  else if (fid && typeof fid.write === "function") fid.write(bytes);
  else throw new TypeError("fid must be a path or an object with a write method");
}
const scalar_text = (v: any, dtype: string) => {
  if (dtype === 'bool') return v ? 'True' : 'False';
  if (v instanceof Complex) return `(${v.re}${v.im < 0 ? '-' : '+'}${Math.abs(v.im)}j)`;
  return String(v);
};

function parse_number(tok: string, dtype: DType): number | Complex | string {
  const t = tok.trim();
  const kind = TYPE_BY_NAME[dtype].kind;
  if (kind === 'O') return t;
  if (kind === 'c') {
    const m = /^\(?\s*([+-]?[\d.eE]+|nan|inf)?\s*(?:([+-]\s*[\d.eE]*|[+-]?nan|[+-]?inf)\s*[jJ])?\s*\)?$/i.exec(t);
    if (m) return new Complex(m[1] === undefined ? 0 : Number(m[1]), m[2] === undefined ? 0 : Number(m[2].replace(/\s+/g, '') === '+' ? 1 : m[2].replace(/\s+/g, '') === '-' ? -1 : m[2].replace(/\s+/g, '')));
  }
  if (t === '') throw new Error(`could not convert string '' to ${dtype}`);
  const lower = t.toLowerCase();
  if (lower === 'nan' || lower === '+nan' || lower === '-nan') return NaN;
  if (lower === 'inf' || lower === '+inf' || lower === 'infinity') return Infinity;
  if (lower === '-inf' || lower === '-infinity') return -Infinity;
  if (kind === 'b') { if (/^(true|1)$/i.test(t)) return 1; if (/^(false|0)$/i.test(t)) return 0; }
  const v = Number(t);
  if (Number.isNaN(v)) throw new Error(`could not convert string '${tok}' to ${dtype}`);
  if ((kind === 'i' || kind === 'u') && !Number.isInteger(v)) throw new Error(`invalid literal for int() with base 10: '${tok}'`);
  return v;
}

function read_text(source: any): string {
  if (Array.isArray(source)) return source.join('\n');
  if (typeof source === "string") {
    // a string is a file name if such a file exists, and the text itself otherwise
    if (!source.includes('\n')) { try { if (fs().existsSync(source)) return fs().readFileSync(source, 'utf8'); } catch { /* no file system */ } }
    return source;
  }
  if (source && typeof source.read === "function") return String(source.read());
  throw new TypeError("expected a file name, a list of lines or the text itself");
}

export interface LoadtxtOptions { dtype?: DType | string, comments?: string | string[] | null, delimiter?: string | null, converters?: any, skiprows?: number, usecols?: number | number[] | null, unpack?: boolean, ndmin?: number, max_rows?: number | null, quotechar?: string | null }

function split_line(line: string, delimiter: string | null, quotechar: string | null): string[] {
  if (delimiter === null || delimiter === '') return line.trim().split(/\s+/).filter(s => s !== '');
  if (!quotechar) return line.split(delimiter);
  const out: string[] = []; let cur = '', inq = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === quotechar) { if (inq && line[i + 1] === quotechar) { cur += c; i++; } else inq = !inq; }
    else if (!inq && line.startsWith(delimiter, i)) { out.push(cur); cur = ''; i += delimiter.length - 1; }
    else cur += c;
  }
  out.push(cur);
  return out;
}

export function loadtxt(fname: any, o: LoadtxtOptions = {}): NDArray {
  const dtype = canonical_dtype(String(o.dtype ?? 'float64'));
  const comments = o.comments === undefined ? ['#'] : (o.comments === null ? [] : (Array.isArray(o.comments) ? o.comments : [o.comments]));
  const lines = read_text(fname).split(/\r?\n/).slice(o.skiprows ?? 0);
  const rows: any[][] = [];
  const usecols = o.usecols === undefined || o.usecols === null ? null : (Array.isArray(o.usecols) ? o.usecols : [o.usecols]);
  let ncols = -1;
  for (let raw of lines) {
    for (const c of comments) { const i = raw.indexOf(c); if (i >= 0) raw = raw.slice(0, i); }
    if (raw.trim() === '') continue;
    const toks = split_line(raw, o.delimiter ?? null, o.quotechar ?? null);
    if (ncols < 0) ncols = toks.length;
    else if (toks.length !== ncols) throw new Error(`the number of columns changed from ${ncols} to ${toks.length} at row ${rows.length + 1}; use \`usecols\` to select a subset and avoid this error`);
    const sel = usecols ? usecols.map(c => toks[c < 0 ? c + toks.length : c]) : toks;
    rows.push(sel.map((t, j) => {
      const col = usecols ? usecols[j] : j;
      const conv = o.converters && (o.converters[col] || (typeof o.converters === "function" ? o.converters : undefined));
      return conv ? conv(t) : parse_number(t, dtype);
    }));
    if (o.max_rows != null && rows.length >= o.max_rows) break;
  }
  const nr = rows.length, nc = rows.length ? rows[0].length : 0;
  let arr = new NDArray(convert_buffer(rows.flat(), dtype), nr === 0 ? [0] : [nr, nc], dtype);
  let result = arr;
  if (nr > 0 && nc === 1 && (o.ndmin ?? 0) < 2) result = reshape(arr, [nr]);
  else if (nr === 1 && (o.ndmin ?? 0) < 2) result = reshape(arr, [nc]);
  const ndmin = o.ndmin ?? 0;
  if (ndmin === 2 && result.shape.length === 1) result = nr === 1 ? reshape(result, [1, result.shape[0]]) : reshape(result, [result.shape[0], 1]);
  if (ndmin === 2 && result.shape.length === 0) result = reshape(result, [1, 1]);
  if (o.unpack && result.shape.length >= 2) {
    const t = transpose(result);
    return Array.from({ length: t.shape[0] }, (_, i) => t.index(i)) as any;
  }
  return result;
}

export interface GenfromtxtOptions { dtype?: DType | string, comments?: string | null, delimiter?: string | null, skip_header?: number, skip_footer?: number, converters?: any, missing_values?: any, filling_values?: any, usecols?: number[] | null, unpack?: boolean, invalid_raise?: boolean, max_rows?: number | null, ndmin?: number }

/** Like loadtxt but empty fields and missing values become nan (or filling_values) */
export function genfromtxt(fname: any, o: GenfromtxtOptions = {}): NDArray {
  const dtype = canonical_dtype(String(o.dtype ?? 'float64'));
  const comments = o.comments === undefined ? '#' : o.comments;
  let lines = read_text(fname).split(/\r?\n/).slice(o.skip_header ?? 0);
  if (o.skip_footer) lines = lines.slice(0, lines.length - o.skip_footer);
  const missing = new Set<string>(['', ...(o.missing_values === undefined ? [] : (Array.isArray(o.missing_values) ? o.missing_values : [o.missing_values]).map(String))]);
  const filling = o.filling_values === undefined ? (TYPE_BY_NAME[dtype].kind === 'f' ? NaN : (TYPE_BY_NAME[dtype].kind === 'c' ? new Complex(NaN, 0) : -1)) : o.filling_values;
  const rows: any[][] = [];
  let ncols = -1;
  for (let raw of lines) {
    if (comments) { const i = raw.indexOf(comments); if (i >= 0) raw = raw.slice(0, i); }
    if (raw.trim() === '') continue;
    const toks = split_line(raw, o.delimiter ?? null, null);
    if (ncols < 0) ncols = toks.length;
    else if (toks.length !== ncols) { if (o.invalid_raise === false) continue; throw new Error(`Some errors were detected !\n    Line #${rows.length + 1} (got ${toks.length} columns instead of ${ncols})`); }
    const sel = o.usecols ? o.usecols.map(c => toks[c < 0 ? c + toks.length : c]) : toks;
    rows.push(sel.map((t, j) => {
      const col = o.usecols ? o.usecols[j] : j;
      if (missing.has(t.trim())) return filling;
      const conv = o.converters && o.converters[col];
      return conv ? conv(t) : parse_number(t, dtype);
    }));
    if (o.max_rows != null && rows.length >= o.max_rows) break;
  }
  const nr = rows.length, nc = rows.length ? rows[0].length : 0;
  let arr = new NDArray(convert_buffer(rows.flat(), dtype), nr === 0 ? [0] : [nr, nc], dtype);
  if (nr > 0 && nc === 1) arr = reshape(arr, [nr]);
  else if (nr === 1) arr = reshape(arr, [nc]);
  if (o.unpack && arr.shape.length >= 2) { const t = transpose(arr); return Array.from({ length: t.shape[0] }, (_, i) => t.index(i)) as any; }
  return arr;
}

export function savetxt(fname: any, X: any, fmt: string | string[] = '%.18e', delimiter: string = ' ', newline: string = '\n', header: string = '', footer: string = '', comments: string = '# ', encoding: string | null = null): string | null {
  const A = asarray(X);
  if (A.shape.length > 2) throw new Error("Expected 1D or 2D array, got " + A.shape.length + "D array instead");
  const rows = A.shape.length === 0 ? [[(A.flat as any)[0]]] : (A.shape.length === 1 ? Array.from(A.flat as any, (v: any) => [v]) : Array.from({ length: A.shape[0] }, (_, i) => Array.from((A.flat as any).slice(i * A.shape[1], (i + 1) * A.shape[1]))));
  const ncol = rows.length ? rows[0].length : 0;
  let formats: string[];
  if (Array.isArray(fmt)) { if (fmt.length !== ncol) throw new Error(`fmt has wrong number of % formats: ${fmt}`); formats = fmt; }
  else if ((fmt.match(/%/g) || []).length - 2 * (fmt.match(/%%/g) || []).length === ncol && ncol > 1 && fmt.includes('%')) { formats = [fmt]; }
  else formats = Array(ncol).fill(fmt);
  const isComplex = A.dtype === 'complex128' || A.dtype === 'complex64';
  const lines = rows.map((r: any[]) => {
    if (formats.length === 1 && Array.isArray(fmt) === false && formats[0] === fmt && ncol > 1 && (fmt.match(/%/g) || []).length > 1) return py_format_values(fmt, r);
    return r.map((v, j) => {
      if (isComplex) { const z = v as Complex; const [f] = [formats[j]]; return py_format_values(f, z.re) + py_format_values(f.replace('%', '%+'), z.im) + 'j'; }
      return py_format_values(formats[j], v);
    }).join(delimiter);
  });
  let text = '';
  const clean = (h: string) => h.split('\n').map(l => comments + l).join(newline) + newline;
  if (header) text += clean(header);
  text += lines.join(newline) + (lines.length ? newline : '');
  if (footer) text += clean(footer);
  if (fname === null || fname === undefined) return text;
  if (is_path(fname)) { fs().writeFileSync(fname, text); return null; }
  if (fname && typeof fname.write === "function") { fname.write(text); return null; }
  throw new TypeError("fname must be a path, null or an object with a write method");
}

/** Matches of a regular expression: one row per match, one column per group */
export function fromregex(file: any, regexp: string | RegExp, dtype: any): NDArray {
  const text = read_text(file);
  const re = typeof regexp === "string" ? new RegExp(regexp, 'g') : new RegExp(regexp.source, regexp.flags.includes('g') ? regexp.flags : regexp.flags + 'g');
  const dt = canonical_dtype(String(dtype));
  const rows: any[][] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) { rows.push(m.length > 1 ? m.slice(1).map(g => parse_number(g, dt)) : [parse_number(m[0], dt)]); if (m[0] === '') re.lastIndex++; }
  const nc = rows.length ? rows[0].length : 0;
  return new NDArray(convert_buffer(rows.flat(), dt), [rows.length, nc], dt);
}

export function fromstring(string: string, dtype: DType | string = 'float64', count: number = -1, sep: string = ''): NDArray {
  const dt = canonical_dtype(String(dtype));
  if (sep === '') throw new Error("The binary mode of fromstring is removed, use frombuffer instead");
  const toks = (sep === ' ' ? string.trim().split(/\s+/) : string.split(sep)).filter((t, i, all) => !(t.trim() === '' && i === all.length - 1));
  const vals = toks.map(t => parse_number(t, dt));
  const n = count >= 0 ? Math.min(count, vals.length) : vals.length;
  if (count >= 0 && count > vals.length) throw new Error("string size must be a multiple of element size");
  return new NDArray(convert_buffer(vals.slice(0, n), dt), [n], dt);
}

export function fromfile(file: any, dtype: DType | string = 'float64', count: number = -1, sep: string = '', offset: number = 0): NDArray {
  const dt = canonical_dtype(String(dtype));
  if (sep !== '') {
    const text = is_path(file) ? fs().readFileSync(file, 'utf8') : String(file);
    return fromstring(text.slice(offset), dt, count, sep);
  }
  const b: Uint8Array = is_path(file) ? fs().readFileSync(file) : as_bytes(file);
  const bytes = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  const sz = itemsize(dt);
  const n = count >= 0 ? count : Math.floor((bytes.length - offset) / sz);
  return new NDArray(from_bytes(bytes, dt, n, offset), [n], dt);
}

// ==============================
//   Memory maps (file contents are loaded in memory and written back on flush)
// ==============================

export function memmap(filename: string, dtype: DType | string = 'uint8', mode: 'r' | 'r+' | 'w+' | 'c' = 'r+', offset: number = 0, shape: number[] | number | null = null, order: string = 'C'): NDArray & { flush: () => void } {
  const dt = canonical_dtype(String(dtype));
  const sz = itemsize(dt);
  const f = fs();
  let arr: NDArray;
  if (mode === 'w+') {
    if (shape === null) throw new Error("shape must be given in 'w+' mode");
    const shp = typeof shape === "number" ? [shape] : shape;
    const n = shp.reduce((p, q) => p * q, 1);
    f.writeFileSync(filename, new Uint8Array(offset + n * sz));
    arr = new NDArray(new_buffer(n, dt), shp, dt);
  } else {
    const b: Uint8Array = f.readFileSync(filename);
    const bytes = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    const avail = Math.floor((bytes.length - offset) / sz);
    const shp = shape === null ? [avail] : (typeof shape === "number" ? [shape] : shape);
    const n = shp.reduce((p, q) => p * q, 1);
    if (n > avail) throw new Error("mmap length is greater than file size");
    arr = new NDArray(from_bytes(bytes, dt, n, offset), shp, dt);
  }
  const out: any = arr;
  out.flush = () => {
    if (mode === 'r' || mode === 'c') return;
    const data = to_bytes(arr);
    const fd = f.openSync(filename, 'r+');
    try { f.writeSync(fd, data, 0, data.length, offset); } finally { f.closeSync(fd); }
  };
  out.filename = filename; out.mode = mode; out.offset = offset;
  return out;
}

export function open_memmap(filename: string, mode: 'r' | 'r+' | 'w+' | 'c' = 'r+', dtype: DType | string | null = null, shape: number[] | null = null, fortran_order: boolean = false): NDArray & { flush: () => void } {
  const f = fs();
  if (mode === 'w+') {
    if (dtype === null || shape === null) throw new Error("dtype and shape must be given in 'w+' mode");
    const a = new NDArray(new_buffer(shape.reduce((p, q) => p * q, 1), canonical_dtype(String(dtype))), shape, canonical_dtype(String(dtype)));
    f.writeFileSync(filename, array_to_npy(a));
  }
  const b: Uint8Array = f.readFileSync(filename);
  const bytes = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  const arr = npy_to_array(bytes);
  const h = npy_header(bytes);
  const out: any = arr;
  out.flush = () => { if (mode === 'r' || mode === 'c') return; f.writeFileSync(filename, array_to_npy(arr)); };
  out.filename = filename; out.mode = mode; out.offset = h.data_offset;
  return out;
}

// ==============================
//   lib.format and lib.npyio helpers
// ==============================

export const lib_format = {
  magic: (major: number = 1, minor: number = 0) => new Uint8Array([...MAGIC, major, minor]),
  read_magic: (bytes: any) => { const b = as_bytes(bytes); return [b[6], b[7]]; },
  write_array: (arr: any) => array_to_npy(asarray(arr)),
  read_array: (bytes: any) => npy_to_array(as_bytes(bytes)),
  dtype_to_descr: (dtype: any) => dtype_to_descr(canonical_dtype(String(dtype))),
  descr_to_dtype: (descr: string) => descr_to_dtype(descr).dtype,
  header_data_from_array_1_0: (arr: any) => { const a = asarray(arr); return { shape: a.shape.slice(), fortran_order: false, descr: dtype_to_descr(a.dtype) }; },
  open_memmap,
  read_array_header_1_0: (bytes: any) => { const h = npy_header(as_bytes(bytes)); return [h.shape, h.fortran_order, descr_to_dtype(h.descr).dtype]; },
};

/** Minimal numpy.lib.npyio.DataSource: reads local files (and urls with fetch, asynchronously not supported) */
export class DataSource {
  destpath: string;
  constructor(destpath: string | null = '.') { this.destpath = destpath ?? '.'; }
  abspath(path: string) { return node_module('path').resolve(this.destpath, path); }
  exists(path: string): boolean { return fs().existsSync(path) || fs().existsSync(this.abspath(path)); }
  open(path: string, mode: string = 'r', encoding: string | null = null): any {
    const p = fs().existsSync(path) ? path : this.abspath(path);
    const data = fs().readFileSync(p);
    return mode.includes('b') ? new Uint8Array(data.buffer, data.byteOffset, data.length) : data.toString('utf8');
  }
}
