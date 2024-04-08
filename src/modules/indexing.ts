import NDArray from "../NDArray";
import { asarray } from "../array/basic";
import { Where, indexSpec } from "../array/indexes";



export function take(a: NDArray, indices: NDArray, axis) {
  const where: Where = a.shape.map(() => ':' as indexSpec);
  where[axis] = indices;
  return a.index(...where);
}

export function where(condition: NDArray, x: NDArray, y: NDArray, out: NDArray = null) {
  if (typeof condition === 'boolean') return condition ? x : y;
  x = asarray(x);
  if (out == null) out = x.copy();
  else out = asarray(out);
  return out.assign(y, condition);
}
