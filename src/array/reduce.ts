//@ts-check
import { asarray } from '../NDArray';
import { shape_shifts } from "../NDArray";
import { multiply, divide, subtract, pow } from './operators';
import NDArray from "../NDArray";
import { AxisArg } from '../kwargs/kwargs';
import { addition_out, argmax_out, bitwise_out, bool_out, DType, DtypeResolver, float_out, new_buffer } from '../dtypes';



/**
 * This function reduces an array along an axis
 */
function apply_on_axis(
  dtype_resolver: DtypeResolver,
  func: (arr: any[]) => number | boolean,
  arr: NDArray,
  axis: AxisArg,
  keepdims: boolean): NDArray {

  arr = asarray(arr);
  if (axis == null) return func(arr.flat as any) as any as NDArray;
  if (axis < 0) axis = arr.shape.length - 1;

  let m = arr.shape[axis];
  let shift = shape_shifts(arr.shape)[axis];
  const groups = Array.from({ length: m }, (_) => []);
  arr.flat.forEach((value, i) => groups[(Math.floor(i / shift)) % m].push(value));
  // Transpose it:
  let nCols = arr.size / m;
  const groupsT = [];
  for (let j = 0; j < nCols; j++) {
    const newRow = [];
    for (let i = 0; i < m; i++) newRow.push(groups[i][j]);
    groupsT.push(newRow);
  }
  //@ts-ignore
  const flat: number[] | boolean[] = groupsT.map(func);
  let shape = [...arr.shape];
  if (keepdims) shape[axis] = 1;
  else shape = shape.filter((_, i) => i != axis);
  let dtype = dtype_resolver([arr.dtype], null);
  const out = new NDArray(new_buffer(flat, dtype), shape);
  return out.item_if_scalar() as NDArray;
};



// ==============================
//       Reducing functions
// ==============================



function mk_reducer(dtype_resolver: DtypeResolver, flat_reduce: (arr: any[]) => number | boolean) {
  return (arr: NDArray, axis: AxisArg | null, keepdims: boolean): NDArray => {
    return apply_on_axis(dtype_resolver, flat_reduce, arr, axis, keepdims);
  }
}

export const sum = mk_reducer(addition_out, (arr: any[]) => arr.reduce((a, b) => a + b, 0));
export const product = mk_reducer(addition_out, (arr: any[]) => arr.reduce((a, b) => a * b, 1));
export const mean = mk_reducer(float_out, (arr: any[]) => arr.reduce((a, b) => a + b, 0) / arr.length);
export const max = mk_reducer(bitwise_out, (arr: any[]) => Math.max(...arr));
export const min = mk_reducer(bitwise_out, (arr: any[]) => Math.min(...arr));
export const argmax = mk_reducer(argmax_out, (arr: any[]) => arr.indexOf(Math.max(...arr)));
export const argmin = mk_reducer(argmax_out, (arr: any[]) => arr.indexOf(Math.min(...arr)));
export const len = mk_reducer(argmax_out, (arr: any[]) => arr.length);
export const any = mk_reducer(bool_out, (arr: any[]) => {
  for (let x of arr) if (x) return true;
  return false;
});
export const all = mk_reducer(bool_out, (arr: any[]) => {
  for (let x of arr) if (!x) return false;
  return true;
});

export const norm = (arr: NDArray, ord: number, axis: AxisArg, keepdims: boolean) => {
  if (ord % 2 != 0) arr = arr.abs();
  if (ord == Infinity) return max(arr, axis, keepdims);
  if (ord == 1) return sum(arr, axis, keepdims);
  return pow(sum(pow(arr, ord), axis, keepdims), 1 / ord);
}
export const variance = (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean) => {
  arr = subtract(arr, mean(arr, axis, true));
  arr = multiply(arr, arr);
  if (ddof == 0) return mean(arr, axis, keepdims);
  return divide(sum(arr, axis, keepdims), arr.shape[axis] - ddof);
}
export const std = (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean) => {
  return pow(variance(arr, axis, ddof, keepdims), 0.5);
}

