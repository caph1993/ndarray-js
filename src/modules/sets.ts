//@ts-check
import { NDArray, asarray, isarray } from "../NDArray";
import { new_buffer, int32, bool, dtype_max, DType } from "../dtypes";
import { cmp_nan_at_the_end, concatenate, normalize_axis, swapAxes } from "../array/transform";
import { ravel, reshape } from "../array/shape_operations";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

type UniqueResult = { values: NDArray, index?: NDArray, inverse?: NDArray, counts?: NDArray };

/** Core of unique for a sequence of (comparable) keys. Returns groups of equal consecutive elements after a stable sort. */
function unique_groups(n: number, cmp: (i: number, j: number) => number, is_nan: (i: number) => boolean, equal_nan: boolean) {
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((i, j) => cmp(i, j) || (i - j));
  const groups: number[][] = [];
  for (const i of order) {
    const last = groups.length ? groups[groups.length - 1] : null;
    if (last) {
      const j = last[0];
      if (cmp(i, j) === 0 && !(is_nan(i) && !equal_nan)) { last.push(i); continue; }
      if (equal_nan && is_nan(i) && is_nan(j)) { last.push(i); continue; }
    }
    groups.push([i]);
  }
  return groups;
}

export function unique_full(ar: NDArray, axis: number | null, equal_nan: boolean): UniqueResult {
  ar = asarray(ar);
  if (axis == null) {
    const flat = ar.flat as any;
    const n = flat.length;
    const groups = unique_groups(n, (i, j) => cmp_nan_at_the_end(flat[i], flat[j]), (i) => flat[i] !== flat[i], equal_nan);
    const values = new_buffer(groups.length, ar.dtype);
    groups.forEach((g, k) => { values[k] = flat[g[0]]; });
    const inverse = new_buffer(n, int32);
    groups.forEach((g, k) => g.forEach(i => { inverse[i] = k; }));
    return {
      values: new NDArray(values, [groups.length], ar.dtype),
      index: new NDArray(new_buffer(groups.map(g => g[0]), int32), [groups.length], int32),
      inverse: new NDArray(inverse, ar.shape, int32),
      counts: new NDArray(new_buffer(groups.map(g => g.length), int32), [groups.length], int32),
    };
  }
  const nd = ar.shape.length;
  axis = normalize_axis(axis, nd);
  const moved = swapAxes(ar, axis, 0);
  const n = moved.shape[0];
  const rest = prod(moved.shape.slice(1));
  const flat = moved.flat as any;
  const cmpRows = (i: number, j: number) => {
    for (let k = 0; k < rest; k++) {
      const c = cmp_nan_at_the_end(flat[i * rest + k], flat[j * rest + k]);
      if (c) return c;
    }
    return 0;
  };
  const groups = unique_groups(n, cmpRows, () => false, equal_nan);
  const buffer = new_buffer(groups.length * rest, ar.dtype);
  groups.forEach((g, k) => { for (let r = 0; r < rest; r++) buffer[k * rest + r] = flat[g[0] * rest + r]; });
  const inverse = new_buffer(n, int32);
  groups.forEach((g, k) => g.forEach(i => { inverse[i] = k; }));
  const values = swapAxes(new NDArray(buffer, [groups.length, ...moved.shape.slice(1)], ar.dtype), axis, 0);
  return {
    values,
    index: new NDArray(new_buffer(groups.map(g => g[0]), int32), [groups.length], int32),
    inverse: new NDArray(inverse, [n], int32),
    counts: new NDArray(new_buffer(groups.map(g => g.length), int32), [groups.length], int32),
  };
}

/** numpy.unique: returns the array of unique values, or [values, index?, inverse?, counts?] when requested */
export function unique(ar: NDArray, return_index: boolean, return_inverse: boolean, return_counts: boolean, axis: number | null, equal_nan: boolean): NDArray | NDArray[] {
  const r = unique_full(ar, axis, equal_nan);
  const out: NDArray[] = [r.values];
  if (return_index) out.push(r.index!);
  if (return_inverse) out.push(r.inverse!);
  if (return_counts) out.push(r.counts!);
  return out.length == 1 ? out[0] : out;
}

export function unique_values(x: NDArray) { return unique_full(x, null, true).values; }
/** Arrays that can be destructured ([values, counts] = ...) and also have named fields */
function named(fields: { [name: string]: NDArray }) { return Object.assign(Object.values(fields), fields); }
export function unique_counts(x: NDArray) { const r = unique_full(x, null, true); return named({ values: r.values, counts: r.counts! }); }
export function unique_inverse(x: NDArray) { const r = unique_full(x, null, true); return named({ values: r.values, inverse_indices: r.inverse! }); }
export function unique_all(x: NDArray) { const r = unique_full(x, null, true); return named({ values: r.values, indices: r.index!, inverse_indices: r.inverse!, counts: r.counts! }); }

function sorted_unique_1d(ar: NDArray) { return unique_full(ravel(asarray(ar)), null, true); }

export function union1d(ar1: NDArray, ar2: NDArray): NDArray {
  return sorted_unique_1d(concatenate([ravel(asarray(ar1)), ravel(asarray(ar2))], 0)).values;
}

export function intersect1d(ar1: NDArray, ar2: NDArray, assume_unique: boolean, return_indices: boolean): NDArray | NDArray[] {
  const a = ravel(asarray(ar1)), b = ravel(asarray(ar2));
  const ua = unique_full(a, null, true), ub = unique_full(b, null, true);
  const fa = ua.values.flat as any, fb = ub.values.flat as any;
  const values: any[] = [], ia: number[] = [], ib: number[] = [];
  let i = 0, j = 0;
  while (i < fa.length && j < fb.length) {
    const c = cmp_nan_at_the_end(fa[i], fb[j]);
    if (c == 0 && fa[i] === fa[i]) { values.push(fa[i]); ia.push((ua.index!.flat as any)[i]); ib.push((ub.index!.flat as any)[j]); i++; j++; }
    else if (c <= 0) i++; else j++;
  }
  const dtype = dtype_max(a.dtype, b.dtype);
  const res = new NDArray(new_buffer(values, dtype), [values.length], dtype);
  if (!return_indices) return res;
  return [res, new NDArray(new_buffer(ia, int32), [ia.length], int32), new NDArray(new_buffer(ib, int32), [ib.length], int32)];
}

/** Boolean array (shape of `element`): is each element in test_elements? */
export function isin(element: NDArray, test_elements: NDArray, assume_unique: boolean, invert: boolean): NDArray {
  element = asarray(element);
  const tf = asarray(test_elements).flat as any;
  const set = new Set<any>();
  let hasNaN = false;
  for (let i = 0; i < tf.length; i++) { if (tf[i] !== tf[i]) hasNaN = true; else set.add(+tf[i]); }
  const f = element.flat as any;
  const buffer = new_buffer(f.length, bool);
  for (let i = 0; i < f.length; i++) {
    const v = f[i];
    const found = set.has(+v);   // NaN never equals NaN
    buffer[i] = (found !== invert) ? 1 : 0;
  }
  return new NDArray(buffer, element.shape, bool);
}

export function setdiff1d(ar1: NDArray, ar2: NDArray, assume_unique: boolean): NDArray {
  const a = assume_unique ? ravel(asarray(ar1)) : unique_full(ravel(asarray(ar1)), null, true).values;
  const mask = isin(a, ravel(asarray(ar2)), assume_unique, true);
  const f = a.flat as any, m = mask.flat as any;
  const out: any[] = [];
  for (let i = 0; i < f.length; i++) if (m[i]) out.push(f[i]);
  return new NDArray(new_buffer(out, a.dtype), [out.length], a.dtype);
}

export function setxor1d(ar1: NDArray, ar2: NDArray, assume_unique: boolean): NDArray {
  const a = assume_unique ? ravel(asarray(ar1)) : unique_full(ravel(asarray(ar1)), null, true).values;
  const b = assume_unique ? ravel(asarray(ar2)) : unique_full(ravel(asarray(ar2)), null, true).values;
  const all = unique_full(concatenate([a, b], 0), null, true);
  const f = all.values.flat as any, c = all.counts!.flat as any;
  const out: any[] = [];
  for (let i = 0; i < f.length; i++) if (c[i] == 1) out.push(f[i]);
  const dtype = dtype_max(a.dtype, b.dtype);
  return new NDArray(new_buffer(out, dtype), [out.length], dtype);
}
