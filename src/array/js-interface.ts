//@ts-check

import { bool, DType, float64, int32, new_buffer, object } from '../dtypes';
import { isarray, number_collapse, new_NDArray, _NDArray } from './basic';


export function fromlist(arr: any, dtype?: DType) {
  if (isarray(arr)) return arr;
  if (typeof arr === "number") {
    if (!dtype && Number.isInteger(arr)) dtype = int32;
    else if (!dtype) dtype = float64;
    return new_NDArray(new_buffer([arr], dtype), [], dtype);
  }
  if (typeof arr === "boolean") {
    return new_NDArray(new_buffer([arr ? 1 : 0], bool), [], bool);
  }
  if (arr === _NDArray.prototype) throw new Error(`Programming error`);
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
  return new_NDArray(new_buffer(flat, dtype), shape, dtype);
}

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





