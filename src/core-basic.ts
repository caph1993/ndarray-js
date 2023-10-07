//@ts-check

import { parse_shape, asarray, as_number, NDArray } from './casting';
type NDArrayClass = import("./core").default;


export function _new(shape, f, dtype) {
  shape = parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  const flat: number[] = Array.from({ length: size }, f);
  return new NDArray(flat, shape, dtype);
};

export function reshape(A: NDArrayClass, shape: number[], ...more_shape: number[]) {
  A = asarray(A);
  if (!more_shape.length) shape = parse_shape(shape);
  else shape = [shape, ...more_shape].map(as_number)
  const n = A.size;
  const inferredIndex = shape.indexOf(-1);
  if (inferredIndex !== -1) {
    const known = shape.filter((dim: number) => dim !== -1).reduce((acc: number, val: number) => acc * val, 1);
    if (n % known !== 0) {
      throw new Error("Invalid shape. The total number of elements must match the product of the known dimensions.");
    }
    shape[inferredIndex] = n / known;
  }
  return new NDArray(A.flat, shape, A.dtype);
};
export function ravel(A: NDArrayClass) {
  A = asarray(A);
  return new NDArray(A.flat, [A.size], A.dtype);
};


