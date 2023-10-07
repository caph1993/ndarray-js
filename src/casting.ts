//@ts-check

import { GLOBALS } from './globals';
const { np, NDArray: _NDArray } = GLOBALS;
type NDArrayClass = import("./core").default;

if (!_NDArray) throw new Error(`Programming error: NDArray not defined`);

export const NDArray = _NDArray;


/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 */
export function number_collapse(arr: NDArrayClass, expect = false): NDArrayClass | number {
  if (!arr.shape.length) return arr.flat[0];
  if (expect) throw new Error(`Expected constant. Got array with shape ${arr.shape}`);
  return arr;
}

export function as_boolean(obj) {
  if (obj instanceof np.NDArray) obj = number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return !!(0 + obj);
}
export function as_number(obj) {
  if (obj instanceof np.NDArray) obj = number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return parseFloat(obj);
}

export function asarray(A) {
  if (A instanceof np.NDArray) return A;
  else return np.fromJS(A);
}

export function array(A) {
  // @ts-ignore
  if (A instanceof np.NDArray) {
    let flat = A._simpleIndexes == null ? [...A.flat] : A.flat;
    return new np.NDArray(flat, A.shape, A.dtype);
  }
  else return np.fromJS(A);
}

export function parse_shape(list) {
  if (Array.isArray(list)) return list;
  if (list instanceof np.NDArray) {
    if (list.shape.length > 1) {
      throw new Error(`Expected flat list. Got array with shape ${list.shape}`);
    }
    return list.flat;
  }
  if (typeof list == "number") return [list];
  throw new Error(`Expected list. Got ${list}`);
}




