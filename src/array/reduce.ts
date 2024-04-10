//@ts-check
import { asarray, new_NDArray, as_boolean, number_collapse, shape_shifts } from './basic';
import { ArrayOrConstant, op_binary } from './operators';
import NDArray from "../NDArray";
import { AxisArg, ReduceNormParsedKwargs, ReduceNormSignature, ReduceParsedKwargs, ReduceSignature, ReduceStdParsedKwargs, ReduceStdSignature, kwDecorators } from './kwargs';
import { TypedArray, TypedArrayConstructor, new_buffer } from '../dtypes';

const multiply = op_binary["*"];
const subtract = op_binary["-"];
const pow = op_binary["**"];


/**
 * This function reduces an array along an axis
 */
function apply_on_axis<
  T extends TypedArrayConstructor = Float64ArrayConstructor,
>(func: (arr: any[]) => number | boolean, dtype: T, arr: NDArray, axis: AxisArg, keepdims: boolean): NDArray<T> {
  if (axis == null) return func(arr.flat as any) as any as NDArray<T>;
  if (axis < 0) axis = arr.shape.length - 1;
  let m = arr.shape[axis];
  let shift = shape_shifts(arr.shape)[axis];
  const groups = Array.from({ length: m }, (_) => [] as T[]);
  arr.flat.forEach((value, i) => groups[(Math.floor(i / shift)) % m].push(value as any as T));
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
  const out = new_NDArray(new_buffer(flat, dtype), shape);
  return out.size == 1 ? (out.flat[0] as any) : out;
};



// ==============================
//       Reducing functions
// ==============================



function mk_reducer<
  T extends TypedArrayConstructor = Float64ArrayConstructor,
>(flat_reduce: (arr: any[]) => number | boolean, dtype?: T) {
  if (!dtype) dtype = Float64Array as any as T;
  return (arr: NDArray, axis: AxisArg | null, keepdims: boolean): NDArray<T> => {
    return apply_on_axis(flat_reduce, dtype, arr, axis, keepdims);
  }
}

export const reducers = {
  sum: mk_reducer((arr: any[]) => arr.reduce((a, b) => a + b, 0)),
  product: mk_reducer((arr: any[]) => arr.reduce((a, b) => a * b, 1)),
  mean: mk_reducer((arr: any[]) => arr.reduce((a, b) => a + b, 0) / arr.length),
  max: mk_reducer((arr: any[]) => Math.max(...arr)),
  min: mk_reducer((arr: any[]) => Math.min(...arr)),
  argmax: mk_reducer((arr: any[]) => arr.indexOf(Math.max(...arr)), Int32Array),
  argmin: mk_reducer((arr: any[]) => arr.indexOf(Math.min(...arr)), Int32Array),
  len: mk_reducer((arr: any[]) => arr.length, Int32Array),
  any: mk_reducer((arr: any[]) => {
    for (let x of arr) if (x) return true;
    return false;
  }, Uint8Array),
  all: mk_reducer((arr) => {
    for (let x of arr) if (!x) return false;
    return true;
  }, Uint8Array),
  norm: (arr: NDArray, axis: AxisArg, keepdims: boolean, ord: number) => {
    if (ord % 2 != 0) arr = arr.abs();
    if (ord == Infinity) return reducers.max(arr, axis, keepdims);
    if (ord == 1) return reducers.sum(arr, axis, keepdims);
    return pow(reducers.sum(pow(arr, ord), axis, keepdims), 1 / ord);
  },
  var: (arr: NDArray, axis: AxisArg, keepdims: boolean) => {
    arr = subtract(arr, reducers.mean(arr, axis, true));
    arr = multiply(arr, arr);
    return arr.mean({ axis, keepdims });
  },
  std: (arr: NDArray, axis: AxisArg, keepdims: boolean, ddof: number) => {
    if (ddof == 0) return pow(arr.var(axis, keepdims), 0.5);
    const _sum = reducers.sum(pow(arr, 2), axis, keepdims);
    const _len = reducers.len(arr, axis, keepdims);
    return pow(op_binary["/"](_sum, op_binary["-"](_len, ddof)), 0.5);
  },
}

export const kw_reducers = {
  sum: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.sum,
  }),
  product: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.product,
  }),
  mean: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.mean,
  }),
  max: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.max,
  }),
  min: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.min,
  }),
  argmax: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.argmax,
  }),
  argmin: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.argmin,
  }),
  len: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.len,
  }),
  any: kwDecorators<ReduceSignature<boolean>, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.any,
  }),
  all: kwDecorators<ReduceSignature<boolean>, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.all,
  }),
  norm: kwDecorators<ReduceNormSignature, ReduceNormParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false], ["ord", 2]],
    func: reducers.norm,
  }),
  var: kwDecorators<ReduceSignature, ReduceParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false]],
    func: reducers.var,
  }),
  std: kwDecorators<ReduceStdSignature, ReduceStdParsedKwargs>({
    defaults: [["axis", null], ["keepdims", false], ["ddof", 0]],
    func: reducers.std,
  }),
};

