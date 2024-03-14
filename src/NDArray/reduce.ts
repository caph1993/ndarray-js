//@ts-check
import { asarray, new_NDArray, as_boolean, number_collapse, shape_shifts } from './basic';
import { ArrayOrConstant, op_binary } from './operators';
import type NDArray from "../NDArray-class";


export type AxisArg = null | number;


// ==============================
//       Reducing functions
// ==============================


export function reduce(arr, axis, keepdims, reducer, dtype = Number): ArrayOrConstant {
  keepdims = as_boolean(keepdims);
  // console.log({ arr })
  arr = asarray(arr);
  if (axis == null) return reducer(arr.flat);
  if (axis < 0) axis = arr.shape.length - 1;
  let m = arr.shape[axis];
  let shift = shape_shifts(arr.shape)[axis];
  const groups = Array.from({ length: m }, (_) =>/**@type {number[]}*/([]));
  arr.flat.forEach((value, i) => groups[(Math.floor(i / shift)) % m].push(value));
  // Transpose it:
  let nCols = arr.size / m;
  const groupsT = [];
  for (let j = 0; j < nCols; j++) {
    const newRow = [];
    for (let i = 0; i < m; i++) newRow.push(groups[i][j]);
    groupsT.push(newRow);
  }
  const flat: number[] = groupsT.map(reducer);
  let shape = [...arr.shape];
  if (keepdims) shape[axis] = 1;
  else shape = shape.filter((_, i) => i != axis);
  const out = new_NDArray(flat, shape, dtype)
  return number_collapse(out);
};

type ReduceKwArgs = { axis?: number, keepdims?: boolean };

function __make_reducer(dtype, reducer) {
  return function (arr: NDArray, axis: AxisArg | ReduceKwArgs = null, keepdims: boolean | ReduceKwArgs = false) {
    // Parse kwargs from right to left to overwrite (right has priority)
    if (keepdims instanceof Object) ({ axis, keepdims } = Object.assign({ axis, keepdims }, keepdims));
    if (axis instanceof Object) ({ axis, keepdims } = Object.assign({ axis, keepdims }, axis));
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    return reduce(arr, axis, keepdims, reducer, dtype);
  };
}

export const reducers = {
  sum: __make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0)),
  product: __make_reducer(Number, (arr) => arr.reduce((a, b) => a * b, 1)),
  any: __make_reducer(Boolean, (arr) => {
    for (let x of arr) if (x) return true;
    return false;
  }),
  all: __make_reducer(Boolean, (arr) => {
    for (let x of arr) if (!x) return false;
    return true;
  }),
  max: __make_reducer(Number, (arr) => Math.max(...arr)),
  min: __make_reducer(Number, (arr) => Math.min(...arr)),
  argmax: __make_reducer(Number, (arr) => arr.indexOf(Math.max(...arr))),
  argmin: __make_reducer(Number, (arr) => arr.indexOf(Math.min(...arr))),
  mean: __make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0) / arr.length),
  var: function (arr, axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    const arrMean = reducers.mean.bind({ axis, keepdims: true })(arr);
    arr = op_binary["-"](arr, arrMean);
    arr = op_binary["*"](arr, arr);
    return reducers.mean.bind({ axis, keepdims })(arr);
  },
  std: function (arr, axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    const variance = reducers.var.bind({ axis, keepdims })(arr);
    return op_binary["**"](variance, 0.5);
  },
};
