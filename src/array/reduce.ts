//@ts-check
import { asarray } from '../NDArray';
import { as_boolean, number_collapse } from "./js-interface";
import { shape_shifts } from "../NDArray";
import { op_binary } from './operators';
import NDArray from "../NDArray";
import { AxisArg, Func_a_axis_ddof_keepdims, Func_a_axis_keepdims, Func_a_ord_axis_keepdims, Method_a_axis_ddof_keepdims, Method_a_axis_keepdims, Method_a_ord_axis_keepdims } from './kwargs';
import { addition_out, argmax_out, bitwise_out, bool_out, DType, DtypeResolver, float_out, new_buffer } from '../dtypes';

const multiply = op_binary["*"];
const divide = op_binary["/"];
const subtract = op_binary["-"];
const pow = op_binary["**"];


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
  return out.size == 1 ? (out.flat[0] as any) : out;
};



// ==============================
//       Reducing functions
// ==============================



function mk_reducer(dtype_resolver: DtypeResolver, flat_reduce: (arr: any[]) => number | boolean) {
  return (arr: NDArray, axis: AxisArg | null, keepdims: boolean): NDArray => {
    return apply_on_axis(dtype_resolver, flat_reduce, arr, axis, keepdims);
  }
}

export const reducers = {
  sum: mk_reducer(addition_out, (arr: any[]) => arr.reduce((a, b) => a + b, 0)),
  product: mk_reducer(addition_out, (arr: any[]) => arr.reduce((a, b) => a * b, 1)),
  mean: mk_reducer(float_out, (arr: any[]) => arr.reduce((a, b) => a + b, 0) / arr.length),
  max: mk_reducer(bitwise_out, (arr: any[]) => Math.max(...arr)),
  min: mk_reducer(bitwise_out, (arr: any[]) => Math.min(...arr)),
  argmax: mk_reducer(argmax_out, (arr: any[]) => arr.indexOf(Math.max(...arr))),
  argmin: mk_reducer(argmax_out, (arr: any[]) => arr.indexOf(Math.min(...arr))),
  len: mk_reducer(argmax_out, (arr: any[]) => arr.length),
  any: mk_reducer(bool_out, (arr: any[]) => {
    for (let x of arr) if (x) return true;
    return false;
  }),
  all: mk_reducer(bool_out, (arr: any[]) => {
    for (let x of arr) if (!x) return false;
    return true;
  }),
  norm: (arr: NDArray, ord: number, axis: AxisArg, keepdims: boolean) => {
    if (ord % 2 != 0) arr = arr.abs();
    if (ord == Infinity) return reducers.max(arr, axis, keepdims);
    if (ord == 1) return reducers.sum(arr, axis, keepdims);
    return pow(reducers.sum(pow(arr, ord), axis, keepdims), 1 / ord);
  },
  var: (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean) => {
    arr = subtract(arr, reducers.mean(arr, axis, true));
    arr = multiply(arr, arr);
    if (ddof == 0) return reducers.mean(arr, axis, keepdims);
    return divide(reducers.sum(arr, axis, keepdims), arr.shape[axis] - ddof);
  },
  std: (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean) => {
    return pow(reducers.var(arr, axis, ddof, keepdims), 0.5);
  },
}


export const kw_reducers = {
  sum: Func_a_axis_keepdims.defaultDecorator(reducers.sum),
  product: Func_a_axis_keepdims.defaultDecorator(reducers.product),
  mean: Func_a_axis_keepdims.defaultDecorator(reducers.mean),
  max: Func_a_axis_keepdims.defaultDecorator(reducers.max),
  min: Func_a_axis_keepdims.defaultDecorator(reducers.min),

  argmax: Func_a_axis_keepdims.defaultDecorator(reducers.argmax),
  argmin: Func_a_axis_keepdims.defaultDecorator(reducers.argmin),
  len: Func_a_axis_keepdims.defaultDecorator(reducers.len),

  any: Func_a_axis_keepdims.defaultDecorator(reducers.any),
  all: Func_a_axis_keepdims.defaultDecorator(reducers.all),

  norm: Func_a_ord_axis_keepdims.defaultDecorator(reducers.norm),

  var: Func_a_axis_ddof_keepdims.defaultDecorator(reducers.var),
  std: Func_a_axis_ddof_keepdims.defaultDecorator(reducers.std),
};



NDArray.prototype.any = Method_a_axis_keepdims.defaultDecorator(reducers.any);
NDArray.prototype.all = Method_a_axis_keepdims.defaultDecorator(reducers.all);

NDArray.prototype.sum = Method_a_axis_keepdims.defaultDecorator(reducers.sum);
NDArray.prototype.product = Method_a_axis_keepdims.defaultDecorator(reducers.product);
NDArray.prototype.max = Method_a_axis_keepdims.defaultDecorator(reducers.max);
NDArray.prototype.min = Method_a_axis_keepdims.defaultDecorator(reducers.min);
NDArray.prototype.argmax = Method_a_axis_keepdims.defaultDecorator(reducers.argmax);
NDArray.prototype.argmin = Method_a_axis_keepdims.defaultDecorator(reducers.argmin);
NDArray.prototype.mean = Method_a_axis_keepdims.defaultDecorator(reducers.mean);

NDArray.prototype.var = Method_a_axis_ddof_keepdims.defaultDecorator(reducers.var);

NDArray.prototype.std = Method_a_axis_ddof_keepdims.defaultDecorator(reducers.std);
NDArray.prototype.norm = Method_a_ord_axis_keepdims.defaultDecorator(reducers.norm);