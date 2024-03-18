//@ts-check
import { asarray, new_NDArray, as_boolean, number_collapse, shape_shifts } from './basic';
import { ArrayOrConstant, op_binary, op_unary } from './operators';
import NDArray from "../NDArray-class";
import { DType } from '../NDArray-class';
import { ReduceStdParsedKwargs, ReduceStdSignature } from './kwargs';

const multiply = op_binary["*"];
const subtract = op_binary["-"];
const pow = op_binary["**"];


export type AxisArg = null | number;


export type ReduceKwArgs = { axis?: number, keepdims?: boolean };
function _NDReducer_parse_args(arr: NDArray, axis: AxisArg | ReduceKwArgs = null, keepdims: boolean | ReduceKwArgs = false) {
  if (keepdims instanceof Object) ({ axis, keepdims } = Object.assign({ axis, keepdims }, keepdims));
  if (axis instanceof Object) ({ axis, keepdims } = Object.assign({ axis, keepdims }, axis));
  ({ axis, keepdims } = Object.assign({ axis, keepdims }, arr.__popKwArgs()));
  keepdims = as_boolean(keepdims) as boolean;
  axis = axis as AxisArg;
  return [axis, keepdims] as [AxisArg, boolean];
}

export class NDReducer<T extends number | boolean, Kw extends ReduceKwArgs>{
  reducer: (arr: any[]) => T;
  self_reduce: (axis?: AxisArg | ReduceKwArgs, keepdims?: boolean | ReduceKwArgs) => T | NDArray;
  dtype: DType;

  constructor(reducer: (x: any[]) => T, dtype: any, _reduce = null) {
    this.reducer = reducer;
    this.dtype = dtype;
    let _this = this;
    this.self_reduce = function (axis = null, keepdims = false) { return _this.reduce(this, axis, keepdims); }
    if (_reduce) this._reduce = _reduce;
  }

  reduce(arr: NDArray, axis: AxisArg | ReduceKwArgs = null, keepdims: boolean | ReduceKwArgs = false): NDArray | T {
    return this._reduce(arr, ..._NDReducer_parse_args(arr, axis, keepdims));
  }

  _reduce(arr: NDArray, axis: AxisArg, keepdims: boolean): NDArray | T {
    if (axis == null) return this.reducer(arr.flat);
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
    const flat: T[] = groupsT.map(this.reducer);
    let shape = [...arr.shape];
    if (keepdims) shape[axis] = 1;
    else shape = shape.filter((_, i) => i != axis);
    const out = new_NDArray((flat as any[] as number[]), shape, this.dtype);
    return out.size == 1 ? (out.flat[0] as any as T) : out;
  };
}



function apply_on_axis<T>(func: (arr: any[]) => T, dtype, arr: NDArray, axis: AxisArg, keepdims: boolean): NDArray | T {
  if (axis == null) return func(arr.flat);
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
  const flat: T[] = groupsT.map(func);
  let shape = [...arr.shape];
  if (keepdims) shape[axis] = 1;
  else shape = shape.filter((_, i) => i != axis);
  const out = new_NDArray((flat as any[] as number[]), shape, dtype);
  return out.size == 1 ? (out.flat[0] as any as T) : out;
};


interface ABC_ReducerGeneric<Parsed extends any[], Signature extends (...args: any[]) => any> {
  self_reduce: (...args: Parameters<Signature>) => ReturnType<Signature>;
  reduce: (arr: NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;
  kwParse: (arr: NDArray, ...args: Parameters<Signature>) => Parsed;
  _reduce: (arr: NDArray, ...args: Parsed) => ReturnType<Signature>;
}

class New_ReducerGeneric<Signature extends (...args: any[]) => any, Parsed extends any[]> implements ABC_ReducerGeneric<Parsed, Signature>{

  dtype: any = Number;

  defaults: [string, any][] = [
    ["axis", null],
    ["keepdims", false],
  ];

  reduce: (arr: NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;
  kwParse: (arr: NDArray, ...args: Parameters<Signature>) => Parsed;
  self_reduce: (...args: Parameters<Signature>) => ReturnType<Signature>;
  constructor(flat_reduce: (flat: any[]) => any = null, extras: [string, any][] = []) {
    let self = this;
    // Define self_reduce (reduce with this as first argument)
    this.self_reduce = function (...args: Parameters<Signature>) {
      if (!(this instanceof NDArray)) throw 'self_reduce must be called from an NDArray instance';
      return self._reduce(this, ...self.kwParse(this, ...args));
    };
    // Define _reduce (reduce implementation after parsing kwargs)
    if (flat_reduce) {
      const dtype = this.dtype;
      this._reduce = function (arr: NDArray, ...args: Parsed) {
        const [axis, keepdims] = args;
        return apply_on_axis(flat_reduce, dtype, arr, axis, keepdims) as ReturnType<Signature>;
      }
    }
    this.reduce = function (arr: NDArray, ...args: Parameters<Signature>): ReturnType<Signature> {
      const parsedKwargs = self.kwParse.bind(this)(arr, ...args);
      return self._reduce(arr, ...parsedKwargs);
    }
    let defaults = [...this.defaults, ...extras];
    this.kwParse = function (arr: NDArray, ...args: Parameters<Signature>): Parsed {
      let kwargs = Object.assign(Object.fromEntries(defaults), arr.__popKwArgs(), this);
      for (let i = 0; i < args.length; i++) {
        let value = args[i];
        if (value instanceof Object) Object.assign(kwargs, value);
        else if (value !== undefined) kwargs[defaults[i][0]] = value;
      }
      let out = defaults.map(([key, _]) => kwargs[key]);
      return out as Parsed;
    }
  }
  _reduce(arr: NDArray, ...args: Parsed): ReturnType<Signature> {
    throw 'Programming error: _reduce was not defined';
  }
}

type KwargsBasic = { axis?: number, keepdims?: boolean };
type SignatureNumeric = (axis?: AxisArg | KwargsBasic, keepdims?: boolean | KwargsBasic) => NDArray | number;
type ParsedNumeric = [number, boolean];
class New_ReducerNumeric extends New_ReducerGeneric<SignatureNumeric, ParsedNumeric>{ }


const sum = new New_ReducerNumeric((arr: any[]) => arr.reduce((a, b) => a + b, 0));

const variance = new class extends New_ReducerNumeric {
  _reduce(arr: NDArray, ...args: ParsedNumeric) {
    let [axis, keepdims] = args;
    arr = subtract(arr, arr.mean({ axis, keepdims: true }));
    arr = multiply(arr, arr);
    return arr.mean({ axis, keepdims });
  }
}();


type KwargsOrd = { axis?: number, keepdims?: boolean, ord?: number };
type ParsedOrd = [number, boolean, number];
type SignatureOrd = (arr: NDArray, axis?: AxisArg | KwargsOrd, keepdims?: boolean | KwargsOrd, ord?: number | KwargsOrd) => NDArray | number;
const norm = new class extends New_ReducerGeneric<SignatureOrd, ParsedOrd>{
  _reduce(arr: NDArray, ...args: ParsedOrd) {
    let [axis, keepdims, ord] = args;
    if (ord % 2 != 0) arr = arr.abs();
    if (ord == Infinity) return arr.max(axis, keepdims);
    if (ord == 1) return arr.sum(axis, keepdims);
    return pow(pow(arr, ord).sum(axis, keepdims), 1 / ord);
  }
}(null, [["ord", 2]]);


const standard_deviation = new class extends New_ReducerGeneric<ReduceStdSignature, ReduceStdParsedKwargs>{
  _reduce(arr: NDArray, ...args: ReduceStdParsedKwargs) {
    let [axis, keepdims, ddof] = args;
    console.log('args', args);
    console.log('arr', arr);
    if (ddof == 0) return pow(arr.var(axis, keepdims), 0.5);
    const _sum = _reducers.sum.reduce(op_binary["**"](arr, 2), { axis, keepdims });
    const _len = _reducers.len.reduce(arr, { axis, keepdims });
    return op_binary["**"](op_binary["/"](_sum, op_binary["-"](_len, ddof)), 0.5);
  }
}(null, [["ddof", 0]]);


// ==============================
//       Reducing functions
// ==============================


export const _reducers = {
  sum,
  // sum: new NDReducer<number, ReduceKwArgs>((arr) => arr.reduce((a, b) => a + b, 0), Number),
  product: new NDReducer<number, ReduceKwArgs>((arr) => arr.reduce((a, b) => a * b, 1), Number),
  mean: new NDReducer<number, ReduceKwArgs>((arr) => arr.reduce((a, b) => a + b, 0) / arr.length, Number),
  max: new New_ReducerNumeric((arr: any[]) => Math.max(...arr)),
  // max: new NDReducer<number, ReduceKwArgs>((arr) => Math.max(...arr), Number),
  min: new NDReducer<number, ReduceKwArgs>((arr) => Math.min(...arr), Number),
  argmax: new NDReducer<number, ReduceKwArgs>((arr) => arr.indexOf(Math.max(...arr)), Number),
  argmin: new NDReducer<number, ReduceKwArgs>((arr) => arr.indexOf(Math.min(...arr)), Number),
  len: new NDReducer<number, ReduceKwArgs>((arr) => arr.length, Number),
  any: new NDReducer<boolean, ReduceKwArgs>((arr) => {
    for (let x of arr) if (x) return true;
    return false;
  }, Boolean),
  all: new NDReducer<boolean, ReduceKwArgs>((arr) => {
    for (let x of arr) if (!x) return false;
    return true;
  }, Boolean),
  norm,
  var: variance,
  std: standard_deviation,
};

export const self_reducers = {
  sum: _reducers.sum.self_reduce,
  product: _reducers.product.self_reduce,
  mean: _reducers.mean.self_reduce,
  max: _reducers.max.self_reduce,
  min: _reducers.min.self_reduce,
  argmax: _reducers.argmax.self_reduce,
  argmin: _reducers.argmin.self_reduce,
  len: _reducers.len.self_reduce,
  any: _reducers.any.self_reduce,
  all: _reducers.all.self_reduce,
  norm: _reducers.norm.self_reduce,
  std: standard_deviation.self_reduce,
  var: _reducers.var.self_reduce,
}
export const reducers = {
  sum: _reducers.sum.reduce,
  product: _reducers.product.reduce,
  mean: _reducers.mean.reduce,
  max: _reducers.max.reduce,
  min: _reducers.min.reduce,
  argmax: _reducers.argmax.reduce,
  argmin: _reducers.argmin.reduce,
  len: _reducers.len.reduce,
  any: _reducers.any.reduce,
  all: _reducers.all.reduce,
  norm: _reducers.norm.reduce,
  std: _reducers.std.reduce,
  var: _reducers.var.reduce,
}
