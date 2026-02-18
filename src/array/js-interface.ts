//@ts-check

import { bool, DType, float64, int32, new_buffer, object } from '../dtypes';
import { asarray, isarray, NDArray, Shape, parse_shape } from "../NDArray";



export function fromlist(arr: any, dtype?: DType) {
  if (isarray(arr)) return arr;
  if (typeof arr === "number") {
    if (!dtype && Number.isInteger(arr)) dtype = int32;
    else if (!dtype) dtype = float64;
    return new NDArray(new_buffer([arr], dtype), [], dtype);
  }
  if (typeof arr === "boolean") {
    return new NDArray(new_buffer([arr ? 1 : 0], bool), [], bool);
  }
  if (!Array.isArray(arr)) throw new Error(`Can't parse input of type ${typeof arr}: ${arr}`);
  const shape: number[] = [];
  let root = arr;
  while (Array.isArray(root)) {
    shape.push(root.length);
    root = root[0];
    if (shape.length > 256) throw new Error(`Circular reference or excessive array depth`);
  }
  const flat: number[] = [];
  const pushToFlat = (arr, axis) => {
    // Check consistency
    if (axis == shape.length - 1) {
      for (let elem of arr) {
        if (Array.isArray(elem)) throw new Error(`Inconsistent shape`);
        flat.push(elem);
        // Update dtype
      }
    } else {
      if (!Array.isArray(arr)) throw new Error(`Inconsistent shape`);
      for (let sub of arr) {
        if (sub.length != shape[axis + 1])
          throw new Error(`Inconsistent shape: found sibling arrays of lengths ${sub.length} and ${shape[axis + 1]}`);
        pushToFlat(sub, axis + 1);
      }
    }
  }
  pushToFlat(arr, 0);
  // Infer dtype
  if (!dtype) {
    if (flat.every(x => typeof x === "boolean")) dtype = bool;
    else if (flat.every(x => typeof x === "number" && Number.isInteger(x))) dtype = int32;
    else if (flat.every(x => typeof x === "number")) dtype = float64;
    else dtype = object;
  }
  return new NDArray(new_buffer(flat, dtype), shape, dtype);
}

NDArray.prototype.fromlist = fromlist;

export function tolist(arr) {
  if (arr === null || typeof arr == "number" || typeof arr == "boolean") return arr;
  if (Array.isArray(arr)) return arr.map(tolist);
  if (!(isarray(arr))) throw new Error(`Expected MyArray. Got ${typeof arr}: ${arr}`);
  arr = number_collapse(arr);
  if (!(isarray(arr))) return arr;

  // let out = [], top;
  // let q = /**@type {[MyArray, any][]}*/([[arr, out]])
  // while (top = q.pop()) {
  //   let [arr, out] = top;
  //   if (arr.shape.length <= 1) {
  //     out.push(...arr.flat);
  //   } else {
  //     for (let i = 0; i < arr.shape[0]; i++) {
  //       let l = []
  //       out.push(l);
  //       q.push([arr.index(i), l]);
  //     }
  //   }
  // }
  // return out;
  function recursiveReshape(flatArr, shapeArr) {
    if (shapeArr.length === 0) {
      return flatArr.shift();
    }
    const innerShape = shapeArr.slice(1);
    const outerSize = shapeArr[0];
    const innerArray: number[] = [];
    for (let i = 0; i < outerSize; i++) {
      innerArray.push(recursiveReshape(flatArr, innerShape));
    }
    return innerArray;
  }
  const out = recursiveReshape([...arr.flat], arr.shape);
  return out;
}

NDArray.prototype.tolist = function () {
  return tolist(this);
}

export function array(A: NDArray | any, dtype: DType = null) {
  if (isarray(A)) { // copy if it's a view
    let flat = A._simpleIndexes == null ? new_buffer(A.flat, A.dtype) : A.flat;
    if (dtype && dtype !== A.dtype) {
      flat = new_buffer(flat, dtype);
    }
    return new NDArray(flat, A.shape);
  }
  else return asarray(A, dtype);
}

/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 */
export function number_collapse(arr: NDArray, expect = false): NDArray | number {
  if (!arr.shape.length) return arr.item() as NDArray | number;
  if (expect) throw new Error(`Expected constant. Got array with shape ${arr.shape}`);
  return arr;
}

export function as_boolean(obj) {
  if (isarray(obj)) obj = number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return !!(0 + obj);
}
export function as_number(obj) {
  if (isarray(obj)) obj = number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return parseFloat(obj);
}
// Types used everywhere

export type Arr = NDArray;
export type ArrOrAny = NDArray | number | boolean | any[];
export type ArrOrConst = NDArray | number | boolean;
export function new_array(shape: Shape, dtype?: DType, fill?: Function | number | boolean) {
  const size = parse_shape(shape).reduce((a, b) => a * b, 1);
  const buffer = new_buffer(size, dtype);
  // If callable, call fill for each element. Otherwise, fill with the value.
  if (typeof fill == 'function') {
    for (let i = 0; i < size; i++) {
      buffer[i] = fill(i);
    }
  } else if (fill !== undefined) {
    for (let i = 0; i < size; i++) {
      buffer[i] = fill;
    }
  }
  return new NDArray(buffer, parse_shape(shape), dtype);
}

