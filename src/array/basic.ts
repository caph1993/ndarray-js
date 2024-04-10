//@ts-check
import type NDArray from "../NDArray";
import type { TypedArrayConstructor, TypedArray } from "../dtypes";
import { isarray, asarray, array, new_NDArray, _NDArray } from "./_globals";

export {
  isarray,
  asarray,
  array,
  new_NDArray,
  _NDArray
}
// Functions to avoid importing NDArray (because if I import NDArray, I can't use it as a type annotation in the same file)


/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 */
export function number_collapse(arr: NDArray<any>, expect = false): NDArray | number {
  if (!arr.shape.length) return arr.flat[0];
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


// ====================
// Reshape and shape shifts for indexing
// ====================


export function shape_shifts(shape) {
  // increasing one by one on a given axis is increasing by shifts[axis] in flat representation
  const shifts = Array.from({ length: shape.length }, (_) => 0);
  shifts[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) shifts[i] = shifts[i + 1] * shape[i + 1];
  return shifts;
}


export type Shape = number | number[] | NDArray;

export function parse_shape(list: Shape) {
  if (typeof list == "number") return [list];
  if (isarray(list)) {
    if (list.shape.length > 1) {
      throw new Error(`Expected flat list. Got array with shape ${list.shape}`);
    }
    return [...list.flat];
  }
  if (Array.isArray(list)) return list;
  throw new Error(`Expected list. Got ${list}`);
}


export function reshape(A: NDArray, shape_or_first: Shape, ...more_shape: number[]) {
  A = asarray(A);
  let shape: number[];
  if (!more_shape.length) shape = parse_shape(shape_or_first);
  else shape = [shape_or_first, ...more_shape].map(as_number);
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
  return new_NDArray(A.flat, shape);
};
export function ravel(A: NDArray) {
  A = asarray(A);
  return new_NDArray(A.flat, [A.size]);
};


// ====================
// Constructors
// ====================

export function new_from(shape: Shape, f: any = undefined, dtype: TypedArrayConstructor = Float32Array) {
  shape = parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  //@ts-ignore
  const buffer = f === undefined ? new dtype(size) : dtype.from({ length: size }, f)
  return new_NDArray(buffer, shape);
};

export function empty(shape: Shape, dtype: TypedArrayConstructor = Float32Array) {
  return new_from(shape, undefined, dtype)
};

export function copy(A: NDArray) {
  return new_NDArray(A.dtype.from(A.flat), A.shape);
}

