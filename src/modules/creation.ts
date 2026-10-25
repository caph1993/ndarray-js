//@ts-check
/** Index tricks (r_, c_, s_, mgrid, ogrid) and less common array creation functions */
import { NDArray, asarray, isarray } from "../NDArray";
import { DType, canonical_dtype, float64, int32, int64, new_buffer, convert_buffer, TYPE_BY_NAME, dtype_max, object } from "../dtypes";
import { concatenate, transpose, normalize_axis } from "../array/transform";
import { reshape } from "../array/shape_operations";
import { from_bytes, as_bytes, itemsize } from "./binary";
import { block, astype } from "./manipulation";
import { indices } from "./indexing-extra";
import { array } from "../array/js-interface";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Parses 'a:b:c' (or 'a:b:Nj') and returns the array (arange, or linspace for imaginary steps) */
function slice_to_array(spec: string): NDArray {
  const parts = spec.split(':').map(s => s.trim());
  if (parts.length < 2 || parts.length > 3) throw new Error(`Invalid slice '${spec}'`);
  const start = parts[0] === '' ? 0 : parseFloat(parts[0]);
  const stop = parts[1] === '' ? NaN : parseFloat(parts[1]);
  const stepSpec = parts[2] === undefined || parts[2] === '' ? '1' : parts[2];
  if (/[jJ]$/.test(stepSpec)) {
    const num = Math.round(parseFloat(stepSpec));
    const buffer = new_buffer(num, float64);
    for (let i = 0; i < num; i++) buffer[i] = num > 1 ? start + i * (stop - start) / (num - 1) : start;
    return new NDArray(buffer, [num], float64);
  }
  const step = parseFloat(stepSpec);
  const n = Math.max(0, Math.ceil((stop - start) / step));
  const isInt = [start, step].every(Number.isInteger) && !/[.eE]/.test(parts[0] + stepSpec);
  const dt: DType = isInt ? int32 : float64;
  const buffer = new_buffer(n, dt);
  for (let i = 0; i < n; i++) buffer[i] = start + i * step;
  return new NDArray(buffer, [n], dt);
}
const is_slice_spec = (x: any) => typeof x === "string" && /^[^a-zA-Z]*:/.test(x) && /^[-+\d.eEjJ\s]*:[-+\d.eEjJ\s]*(:[-+\d.eEjJ\s]*)?$/.test(x);

/** numpy.r_ / numpy.c_: build arrays by concatenation along an axis */
export function make_concatenator(axis0: number, ndmin0: number, trans1d0: number) {
  const run = (...items: any[]): NDArray => {
    let axis = axis0, ndmin = ndmin0, trans1d = trans1d0;
    if (items.length && typeof items[0] === "string" && !is_slice_spec(items[0])) {
      const spec = items.shift() as string;
      if (spec === 'r' || spec === 'c') { ndmin = 2; if (spec === 'c') { axis = 1; trans1d = 0; } else { axis = 0; trans1d = 1; } }
      else {
        const nums = spec.split(',').map(s => parseInt(s, 10));
        if (nums.some(Number.isNaN)) throw new Error(`unknown special directive '${spec}'`);
        axis = nums[0]; if (nums.length > 1) ndmin = nums[1]; if (nums.length > 2) trans1d = nums[2];
      }
    }
    const objs: NDArray[] = [];
    for (const item of items) {
      let arr: NDArray;
      if (is_slice_spec(item)) arr = slice_to_array(item as string);
      else arr = asarray(item);
      if (arr.shape.length === 0) arr = reshape(arr, [1]);
      let nd = arr.shape.length;
      if (nd < ndmin) {
        arr = reshape(arr, [...Array(ndmin - nd).fill(1), ...arr.shape]);   // numpy: array(item, ndmin=ndmin)
        if (trans1d !== -1) {
          const k2 = ndmin - nd;
          let k1 = trans1d;
          if (k1 < 0) k1 += k2 + 1;
          const def = range(ndmin);
          const axes = [...def.slice(0, k1), ...def.slice(k2), ...def.slice(k1, k2)];
          arr = transpose(arr, axes);
        }
      }
      objs.push(arr);
    }
    if (!objs.length) throw new Error("need at least one array to concatenate");
    return concatenate(objs, axis);
  };
  return Object.assign(run, { index: run });
}

export const r_ = make_concatenator(0, 1, -1);
export const c_ = make_concatenator(-1, 2, 0);

/** numpy.s_ / index_exp: build index tuples. With a single argument the argument itself is returned (s_), index_exp always returns a tuple */
export const s_ = Object.assign((...args: any[]) => args.length === 1 ? args[0] : args, { index: (...args: any[]) => args.length === 1 ? args[0] : args });
export const index_exp = Object.assign((...args: any[]) => args, { index: (...args: any[]) => args });

// ==============================
//   Construction from other objects
// ==============================

export function asmatrix(data: any, dtype: DType | null = null): NDArray {
  let a = asarray(data, dtype);
  if (a.shape.length === 0) a = reshape(a, [1, 1]);
  else if (a.shape.length === 1) a = reshape(a, [1, a.shape[0]]);
  else if (a.shape.length > 2) throw new Error("shape too large to be a matrix.");
  return a;
}

/** Block matrix from a string like '1 2; 3 4' (rows separated by ';') or a nested list of arrays (see block) */
export function bmat(obj: any, ldict: any = null, gdict: any = null): NDArray {
  if (typeof obj === "string") {
    const rows = obj.split(';').map(r => r.trim().split(/[\s,]+/).filter(t => t !== '').map(t => {
      if (ldict && t in ldict) return asarray(ldict[t]);
      if (gdict && t in gdict) return asarray(gdict[t]);
      const v = Number(t);
      if (Number.isNaN(v)) throw new Error(`name '${t}' is not defined`);
      return v;
    }));
    return block(rows);
  }
  return block(obj);
}

export function ascontiguousarray(a: any, dtype: DType | null = null): NDArray {
  let x = array(a, dtype);
  if (x.shape.length === 0) x = reshape(x, [1]);
  return x;
}
export const asfortranarray = (a: any, dtype: DType | null = null): NDArray => ascontiguousarray(a, dtype);

export function asarray_chkfinite(a: any, dtype: DType | null = null, order: string | null = null): NDArray {
  const x = asarray(a, dtype);
  const f = x.flat as any;
  const kind = TYPE_BY_NAME[x.dtype].kind;
  if (kind === 'f' || kind === 'c') for (let i = 0; i < f.length; i++) {
    const v = f[i];
    const bad = kind === 'c' ? !v.isFinite : !Number.isFinite(v);
    if (bad) throw new Error("array must not contain infs or NaNs");
  }
  return x;
}

export function frombuffer(buffer: any, dtype: DType | string = 'float64', count: number = -1, offset: number = 0): NDArray {
  const dt = canonical_dtype(String(dtype));
  const bytes = as_bytes(buffer);
  const sz = itemsize(dt);
  const n = count >= 0 ? count : Math.floor((bytes.byteLength - offset) / sz);
  return new NDArray(from_bytes(bytes, dt, count >= 0 ? count : -1, offset), [n], dt);
}

export function fromiter(iter: Iterable<any>, dtype: DType | string, count: number = -1): NDArray {
  const dt = canonical_dtype(String(dtype));
  const vals: any[] = [];
  for (const v of iter) { if (count >= 0 && vals.length >= count) break; vals.push(v); }
  if (count >= 0 && vals.length < count) throw new Error("iterator too short");
  return new NDArray(convert_buffer(vals, dt), [vals.length], dt);
}

/** Calls function with the coordinate arrays of a grid of the given shape */
export function fromfunction(func: (...idx: NDArray[]) => any, shape: number[], dtype: DType | string = 'float64', kwargs: any = {}): NDArray {
  const dt = canonical_dtype(String(dtype));
  const idx = indices(shape, dt, false) as NDArray;
  const coords = range(shape.length).map(k => idx.index(k) as NDArray);
  return asarray(func(...coords, ...(Object.keys(kwargs).length ? [kwargs] : [])));
}

export function from_dlpack(x: any): NDArray { throw new Error("from_dlpack is not supported in JavaScript"); }

const no_records = (name: string) => () => { throw new Error(`${name}: record / structured arrays are not supported by ndarray-js`); };
export const rec = {
  array: no_records('rec.array'), fromarrays: no_records('rec.fromarrays'), fromrecords: no_records('rec.fromrecords'), fromstring: no_records('rec.fromstring'), fromfile: no_records('rec.fromfile'),
  /** Parses the formats / names of a structured dtype (the dtype itself can not be used to create arrays) */
  format_parser: (formats: any, names: any, titles: any = null, aligned: boolean = false, byteorder: any = null) => new FormatParser(formats, names, titles, aligned, byteorder),
};
export class FormatParser {
  dtype: { names: string[], formats: string[], titles: (string | null)[] };
  constructor(formats: string | string[], names: string | string[] | null, titles: string | string[] | null = null, aligned: boolean = false, byteorder: string | null = null) {
      const fm = (typeof formats === "string" ? formats.split(',') : formats).map(s => s.trim());
      const nm = names === null ? fm.map((_, i) => `f${i}`) : (typeof names === "string" ? names.split(',') : names).map(s => s.trim());
      while (nm.length < fm.length) nm.push(`f${nm.length}`);
      const tt = titles === null ? fm.map(() => null) : (typeof titles === "string" ? titles.split(',') : titles);
      this.dtype = { names: nm, formats: fm, titles: tt as any };
    }
}
