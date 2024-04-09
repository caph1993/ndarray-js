//@ts-check
import type NDArray from "../NDArray";
import { DType } from "../NDArray";
import { GLOBALS } from '../_globals';
const { np, NDArray: __NDArray } = GLOBALS;
if (!__NDArray) throw new Error(`Programming error: NDArray not defined`);

// Functions to avoid importing NDArray (because if I import NDArray, I can't use it as a type annotation in the same file)
export const _NDArray = __NDArray;


export function isarray(A: any): A is NDArray {
  return A instanceof _NDArray;
}

export const new_NDArray = (flat: number[], shape: number[], dtype: DType) => new _NDArray(flat, shape, dtype);

export function asarray(A): NDArray {
  if (isarray(A)) return A;
  else return np.fromlist(A);
}

export function array(A) {
  if (isarray(A)) { // shallow copy of A
    let flat = A._simpleIndexes == null ? [...A.flat] : A.flat;
    return new_NDArray(flat, A.shape, A.dtype);
  }
  else return asarray(A);
}


export function broadcast_shapes(shapeA, shapeB) {
  const shape = [];
  const maxDim = Math.max(shapeA.length, shapeB.length);
  shapeA = [...Array.from({ length: maxDim - shapeA.length }, () => 1), ...shapeA];
  shapeB = [...Array.from({ length: maxDim - shapeB.length }, () => 1), ...shapeB];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dim1 = shapeA[axis];
    const dim2 = shapeB[axis];
    if (dim1 !== 1 && dim2 !== 1 && dim1 !== dim2)
      throw new Error(`Can not broadcast axis ${axis} with sizes ${dim1} and ${dim2}`);
    shape.unshift(Math.max(dim1, dim2));
  }
  return [shape, shapeA, shapeB];
}


export function broadcast_n_shapes(...shapes: number[][]) {
  const maxDim = Math.max(...shapes.map(shape => shape.length));
  const broadcastShapes = shapes.map(shape => {
    return [...Array.from({ length: maxDim - shape.length }, () => 1), ...shape];
  });
  const outputShape = [];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dims = broadcastShapes.map(shape => shape[axis]);
    const dim = Math.max(...dims);
    if (dims.some(d => d !== 1 && d !== dim)) throw new Error(`Can not broadcast axis ${axis} with sizes ${dims}`);
    outputShape.unshift(dim);
  }
  return [broadcastShapes, outputShape];
}


