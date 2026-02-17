//@ts-check
import NDArray from "../NDArray";
import { DType, Buffer, new_buffer } from "../dtypes";
import { fromlist } from "./js-interface";


// Types used everywhere
export type ArrOrAny = NDArray | number | boolean | any[];
export type ArrOrConst = NDArray | number | boolean;

export type AxisArg = number | null;
export type ArrOrNull = NDArray | null;


// TODO: move to NDArray
export function isarray(A: any): A is NDArray {
  return A instanceof NDArray;
}

// TODO: move to NDArray
export function new_NDArray(flat: Buffer, shape: number[], dtype?: DType): NDArray {
  return new NDArray(flat, shape, dtype);
}

// TODO: move to js-interface
export function asarray(A: NDArray | any, dtype: DType = null): NDArray {
  if (isarray(A)) return A as NDArray;
  else return fromlist(A, dtype) as NDArray;
}

// TODO: move to js-interface
export function array(A: NDArray | any, dtype: DType = null) {
  if (isarray(A)) { // copy if it's a view
    let flat = A._simpleIndexes == null ? new_buffer(A.flat, A.dtype) : A.flat;
    if (dtype && dtype !== A.dtype) {
      flat = new_buffer(flat, dtype);
    }
    return new_NDArray(flat, A.shape);
  }
  else return asarray(A, dtype);
}


export function broadcast_shapes(shapeA: number[], shapeB: number[]) {
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

// TODO: move to shape-utils
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


