// ====================
// Reshape and shape shifts for indexing
// ====================

import { NDArray, isarray, asarray, parse_shape, Shape } from "../NDArray";


export function reshape(A: NDArray, shape_or_first: Shape, ...more_shape: number[]) {
  A = asarray(A);
  let shape: number[];
  if (!more_shape.length) shape = parse_shape(shape_or_first);
  else shape = [shape_or_first, ...more_shape].map((obj) => {
    if (isarray(obj)) obj = obj.item();
    else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as number: ${obj}`);
    return parseFloat(obj as any);
  });
  const n = A.size;
  // Find -1
  const inferredIndex = shape.indexOf(-1);
  if (inferredIndex !== -1) {
    const known = shape.filter((dim: number) => dim !== -1).reduce((acc: number, val: number) => acc * val, 1);
    if (n % known !== 0) {
      throw new Error("Invalid shape. The total number of elements must match the product of the known dimensions.");
    }
    shape[inferredIndex] = n / known;
  } else {
    let m = shape.reduce((a, b) => a * b, 1);
    if (n !== m) {
      throw new Error(`Invalid target shape [${shape}] from source [${A.shape}]. Expected size ${n}. Found ${m}.`);
    }
  }
  return new NDArray(A.flat, shape);
}

export function ravel(A: NDArray) {
  A = asarray(A);
  return new NDArray(A.flat, [A.size]);
}

NDArray.prototype.reshape = function (shape, ...more_shape) {
  return reshape(this, shape, ...more_shape);
};

NDArray.prototype.ravel = function () {
  return ravel(this);
};

