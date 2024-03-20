//@ts-check
import NDArray from '../NDArray';
import { KwParser } from '../array/kwargs';

const { basic, elementwise } = NDArray.prototype.modules;

/**
 * Creates a new array with the specified shape and type, without initializing entries.
 */
export function empty(shape: number[], dtype: import('../NDArray').DType = Number) {
  return basic.new_from(shape, undefined, dtype)
};

/**
 * Creates a new array of zeros with the specified shape and dtype.
 */
export function zeros(shape: number[], dtype: import('../NDArray').DType = Number) {
  const c = dtype == Boolean ? false : 0;
  return basic.new_from(shape, (_) => c, dtype)
};

export function ones(shape: number[], dtype: import('../NDArray').DType = Number) {
  const c = dtype == Boolean ? true : 1;
  return basic.new_from(shape, (_) => c, dtype)
};

export function arange(arg0, arg1 = null) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return basic.new_from(end - start, (_, i) => start + i, Number)
};

export function linspace(start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  start = basic.as_number(start);
  stop = basic.as_number(stop);
  let n = (num - (endpoint ? 1 : 0))
  let arr = arange(num).multiply((stop - start) / n).add(start);
  return arr;
}

export function geomspace(start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  start = elementwise.funcs.log(start);
  stop = elementwise.funcs.log(stop);
  return elementwise.funcs.exp(linspace(start, stop, num, endpoint));
}
