//@ts-check
import { modules } from '../core-modules';
import type NDArray from '../core';

const { basic, elementwise } = modules;

export function empty(shape, dtype: import('../core').DType = Number) {
  return basic.new_from(shape, undefined, dtype)
};

export function zeros(shape, dtype: import('../core').DType = Number) {
  const c = dtype == Boolean ? false : 0;
  return basic.new_from(shape, (_) => c, dtype)
};

export function ones(shape, dtype: import('../core').DType = Number) {
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
  start = elementwise.ops.log(start);
  stop = elementwise.ops.log(stop);
  return elementwise.ops.exp(linspace(start, stop, num, endpoint));
}
