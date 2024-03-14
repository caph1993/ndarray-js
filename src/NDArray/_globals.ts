//@ts-check
import type NDArray from "../NDArray-class";
import { DType } from "../NDArray-class";
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


