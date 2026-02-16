//@ts-check
import NDArray from '../NDArray';
import { Shape } from '../array/basic';
import { float64, DType, int32 } from '../dtypes';

const { basic, elementwise } = NDArray.prototype.modules;

/**
 * Creates a new array with the specified shape and type, without initializing entries.
 */
export function empty(shape: Shape, dtype: DType = float64) {
  return basic.empty(shape, dtype);
};

/**
 * Creates a new array of zeros with the specified shape and dtype.
 */
export function zeros(shape: Shape, dtype: DType = float64) {
  return basic.new_array(shape, dtype, (_) => 0)
};

export function ones(shape: Shape, dtype: DType = float64) {
  return basic.new_array(shape, dtype, (_) => 1)
};

export function arange(arg0, arg1 = null, dtype: DType = int32) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return basic.new_array(end - start, dtype, (i) => start + i)
};

export function linspace(start: number, stop: number, steps = 50, endpoint = true) {
  start = basic.as_number(start);
  stop = basic.as_number(stop);
  let n = (steps - (endpoint ? 1 : 0));
  let arr = arange(steps).divide(n).multiply(stop - start).add(start);
  return arr;
}

export function geomspace(start: number, stop: number, steps = 50, endpoint = true) {
  start = Math.log(start);
  stop = Math.log(stop);
  return elementwise.funcs.exp(linspace(start, stop, steps, endpoint));
}
