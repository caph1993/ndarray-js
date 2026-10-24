import { NDArray } from "../NDArray";
import { asarray, broadcast_shapes } from "../NDArray";
import { Where, IndexSpec } from "../array/indexes";
import { new_buffer, int32, dtype_max, DType, truthy } from "../dtypes";
import { n_ary_elementwise } from "../array/operators";
import { normalize_axis } from "../array/transform";


export function take(a: NDArray, indices: NDArray, axis: number) {
  const where: Where = a.shape.map(() => ':' as IndexSpec);
  where[normalize_axis(axis, a.shape.length)] = indices;
  return a.index(...where);
}

/** where(condition, x, y): elements from x where condition is true, else from y (broadcasting all three) */
export function where(condition: NDArray, x: NDArray, y: NDArray, out: NDArray | null = null) {
  if (typeof condition === 'boolean') return condition ? asarray(x) : asarray(y);
  return n_ary_elementwise((dtypes) => dtype_max(dtypes[1], dtypes[2]), (c, a, b) => truthy(c) ? a : b, [condition, x, y], out);
}

/** Indices of the non-zero elements: one int array per axis */
export function nonzero(a: NDArray) {
  a = asarray(a);
  if (a.shape.length == 0) throw new Error(`Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.`);
  const flat = a.flat;
  const raw_indices: number[] = [];
  for (let i = 0; i < flat.length; i++) if (truthy(flat[i])) raw_indices.push(i);
  const indices = a.shape.map(() => new_buffer(raw_indices.length, 'int32'));
  raw_indices.forEach((v, pos) => {
    for (let i = a.shape.length - 1; i >= 0; i--) {
      indices[i][pos] = v % a.shape[i];
      v = Math.floor(v / a.shape[i]);
    }
  });
  return indices.map(buffer => new NDArray(buffer, [buffer.length], 'int32'));
}

/** Indices of the non-zero elements, grouped by element: shape (n, ndim) */
export function argwhere(a: NDArray) {
  a = asarray(a);
  const nd = a.shape.length;
  const flat = a.flat;
  const rows: number[] = [];
  for (let i = 0; i < flat.length; i++) if (truthy(flat[i])) rows.push(i);
  const buffer = new_buffer(rows.length * nd, int32);
  rows.forEach((v, r) => {
    for (let d = nd - 1; d >= 0; d--) { buffer[r * nd + d] = v % a.shape[d]; v = Math.floor(v / a.shape[d]); }
  });
  return new NDArray(buffer, [rows.length, nd], int32);
}

/** Indices of the non-zero elements of the flattened array */
export function flatnonzero(a: NDArray) {
  a = asarray(a);
  const flat = a.flat;
  const idx: number[] = [];
  for (let i = 0; i < flat.length; i++) if (truthy(flat[i])) idx.push(i);
  return new NDArray(new_buffer(idx, int32), [idx.length], int32);
}

/** Elements of `arr` (flattened) where `condition` (flattened) is true */
export function extract(condition: NDArray, arr: NDArray) {
  condition = asarray(condition); arr = asarray(arr);
  const c = condition.flat, f = arr.flat;
  const out: any[] = [];
  for (let i = 0; i < c.length && i < f.length; i++) if (truthy(c[i])) out.push(f[i]);
  return new NDArray(new_buffer(out, arr.dtype), [out.length], arr.dtype);
}
