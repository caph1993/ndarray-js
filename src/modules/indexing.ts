import NDArray from "../NDArray";
import { asarray } from "../array/basic";
import { Where, IndexSpec } from "../array/indexes";



export function take(a: NDArray, indices: NDArray, axis: number) {
  const where: Where = a.shape.map(() => ':' as IndexSpec);
  where[axis] = indices;
  return a.index(...where);
}

export function where(condition: NDArray, x: NDArray, y: NDArray, out: NDArray | null = null) {
  if (typeof condition === 'boolean') return condition ? x : y;
  condition = asarray(condition);
  if (out == null) out = condition.copy();
  else out = asarray(out);
  out.assign(x, condition);
  out.assign(y, condition.logical_not());
  return out;
}

export function nonzero(a: NDArray<any>) {
  const raw_indices = [...a.flat].map((v, i) => v ? i : null).filter((v) => v !== null);
  // Convert to many 1D arrays
  const indices = a.shape.map(() => new Int32Array(raw_indices.length));
  raw_indices.forEach((v, pos) => {
    for (let i = a.shape.length - 1; i >= 0; i--) {
      indices[i][pos] = v % a.shape[i];
      v = Math.floor(v / a.shape[i]);
    }
  });
  return indices.map(buffer => new NDArray<Int32ArrayConstructor>(buffer));
}
