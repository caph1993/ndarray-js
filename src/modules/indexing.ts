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
