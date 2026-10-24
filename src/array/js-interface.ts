//@ts-check
import { bool, DType, float64, int32, int64, complex128, new_buffer, object, convert_buffer } from '../dtypes';
import { Complex } from '../complex';
import { asarray, isarray, NDArray, Shape, parse_shape } from "../NDArray";



const TYPED_ARRAY_DTYPES: { [name: string]: DType } = {
  Float64Array: float64, Float32Array: "float32", Int32Array: int32, Uint8Array: bool,
  Int8Array: int32, Int16Array: int32, Uint16Array: int32, Uint8ClampedArray: int32, Uint32Array: float64,
};

export function fromlist(arr: any, dtype?: DType) {
  if (isarray(arr)) return dtype && arr.dtype !== dtype ? new NDArray(convert_buffer(arr.flat, dtype), arr.shape, dtype) : arr;
  if (typeof arr === "number") {
    const weak = !dtype;
    if (!dtype && Number.isInteger(arr)) dtype = Math.abs(arr) <= 2147483647 ? int32 : (Number.isSafeInteger(arr) ? int64 : float64);
    else if (!dtype) dtype = float64;
    const out = new NDArray(convert_buffer([arr], dtype), [], dtype);
    if (weak) out._weak = true;
    return out;
  }
  if (arr instanceof Complex) {
    const out = new NDArray(convert_buffer([arr], dtype || complex128), [], dtype || complex128);
    if (!dtype) out._weak = true;
    return out;
  }
  if (typeof arr === "boolean") {
    const out = new NDArray(convert_buffer([arr], dtype || bool), [], dtype || bool);
    if (!dtype) out._weak = true;
    return out;
  }
  if (ArrayBuffer.isView(arr) && !(arr instanceof DataView)) {
    const own: DType = TYPED_ARRAY_DTYPES[arr.constructor.name] || float64;
    const flat = new_buffer(arr as any, own === float64 && arr.constructor.name === "Uint32Array" ? float64 : own);
    return new NDArray(dtype && dtype !== own ? convert_buffer(flat, dtype) : flat, [(arr as any).length], dtype || own);
  }
  if (!Array.isArray(arr)) throw new Error(`Can't parse input of type ${typeof arr}: ${arr}`);
  const shape: number[] = [];
  let root = arr;
  while (Array.isArray(root)) {
    shape.push(root.length);
    root = root[0];
    if (shape.length > 256) throw new Error(`Circular reference or excessive array depth`);
  }
  const flat: any[] = [];
  const pushToFlat = (arr, axis) => {
    if (!Array.isArray(arr)) throw new Error(`Inconsistent shape`);
    if (arr.length != shape[axis]) throw new Error(`Inconsistent shape: found sibling arrays of lengths ${arr.length} and ${shape[axis]}`);
    if (axis == shape.length - 1) {
      for (let elem of arr) {
        if (Array.isArray(elem)) throw new Error(`Inconsistent shape`);
        flat.push(elem);
      }
    } else {
      for (let sub of arr) pushToFlat(sub, axis + 1);
    }
  }
  pushToFlat(arr, 0);
  // Infer dtype
  const float_flag = (arr as any).__float === true || (shape.length > 1 && arr.some((x: any) => x.__float === true));
  if (!dtype) {
    let n_bool = 0, n_int = 0, n_num = 0, n_safe = 0, n_complex = 0;
    for (const x of flat) {
      if (x instanceof Complex) n_complex++;
      else if (typeof x === "boolean") n_bool++;
      else if (typeof x === "number") {
        n_num++;
        if (Number.isInteger(x) && Math.abs(x) <= 2147483647) n_int++;
        if (Number.isSafeInteger(x)) n_safe++;
      }
    }
    if (flat.length == 0) dtype = float64;
    else if (n_complex > 0 && n_complex + n_bool + n_num == flat.length) dtype = complex128;
    else if (n_bool == flat.length) dtype = bool;
    else if (n_bool + n_num < flat.length) dtype = object;
    else if (n_bool + n_int == flat.length && !float_flag) dtype = int32;
    else if (n_bool + n_safe == flat.length && !float_flag) dtype = int64;
    else dtype = float64;
  }
  return new NDArray(convert_buffer(flat, dtype), shape, dtype);
}


export function tolist(arr) {
  if (arr === null || typeof arr == "number" || typeof arr == "boolean") return arr;
  if (Array.isArray(arr)) return arr.map(tolist);
  if (!isarray(arr)) throw new Error(`Expected MyArray. Got ${typeof arr}: ${arr}`);
  if (!arr.shape.length) return arr.item();
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
  const flat = arr.flat;
  const isBool = arr.dtype === bool;
  let pos = 0;
  function build(axis: number): any {
    const n = arr.shape[axis];
    const out = new Array(n);
    if (axis == arr.shape.length - 1) {
      if (isBool) for (let i = 0; i < n; i++) out[i] = !!flat[pos++];
      else for (let i = 0; i < n; i++) out[i] = flat[pos++];
    } else {
      for (let i = 0; i < n; i++) out[i] = build(axis + 1);
    }
    return out;
  }
  return build(0);
}


export function array(A: NDArray | any, dtype: DType = null, copy: boolean = true, ndmin: number = 0): NDArray {
  let out: NDArray;
  if (isarray(A)) {
    const target = dtype || A.dtype;
    // Views are always copied; plain arrays only if requested (or if the dtype changes)
    out = (copy || A._simpleIndexes || target !== A.dtype)
      ? new NDArray(convert_buffer(A.flat, target), A.shape, target)
      : A;
  } else {
    out = asarray(A, dtype);
    if (out._weak) out = new NDArray(out._flat, [], out.dtype); // np.array(5) is a regular (strong) 0-d array
  }
  if (out.shape.length < ndmin) out = new NDArray(out.flat, [...Array(ndmin - out.shape.length).fill(1), ...out.shape], out.dtype);
  return out;
}


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

