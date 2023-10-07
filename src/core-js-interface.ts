//@ts-check

import { number_collapse, NDArray } from './casting';
type NDArrayClass = import("./core").default;


export function fromJS(arr) {
  if (arr instanceof NDArray) return arr;
  if (typeof arr === "number") return new NDArray([arr], [], Number);
  if (typeof arr === "boolean") return new NDArray([arr ? 1 : 0], [], Boolean);
  if (arr === NDArray.prototype) throw new Error(`Programming error`);
  if (!Array.isArray(arr)) throw new Error(`Can't parse input of type ${typeof arr}: ${arr}`);
  const shape: number[] = [];
  let root = arr;
  while (Array.isArray(root)) {
    shape.push(root.length);
    root = root[0];
    if (shape.length > 256) throw new Error(`Circular reference or excessive array depth`);
  }
  let dtype = typeof root === "boolean" ? Boolean : Number;
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
  return new NDArray(flat, shape, dtype)
}

export function toJS(arr) {
  if (this instanceof NDArray) return toJS;
  if (arr === null || typeof arr == "number" || typeof arr == "boolean") return arr;
  if (Array.isArray(arr)) return arr.map(toJS);
  if (!(arr instanceof NDArray)) throw new Error(`Expected MyArray. Got ${typeof arr}: ${arr}`);
  arr = number_collapse(arr);
  if (!(arr instanceof NDArray)) return arr;

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





