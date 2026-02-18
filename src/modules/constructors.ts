//@ts-check
import { Shape, empty } from '../NDArray';
import { float64, DType, int32 } from '../dtypes';
import { new_array } from '../array/js-interface';
import { exp } from '../array/elementwise';

export { empty };
/**
 * Creates a new array of zeros with the specified shape and dtype.
 */
export function zeros(shape: Shape, dtype: DType = float64) {
  return new_array(shape, dtype, (_) => 0)
};

export function ones(shape: Shape, dtype: DType = float64) {
  return new_array(shape, dtype, (_) => 1)
};

export function arange(arg0, arg1 = null, dtype: DType = int32) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return new_array(end - start, dtype, (i) => start + i)
};

export function linspace(start: number, stop: number, steps = 50, endpoint = true) {
  let n = (steps - (endpoint ? 1 : 0));
  let arr = arange(steps).divide(n).multiply(stop - start).add(start);
  return arr;
}

export function geomspace(start: number, stop: number, steps = 50, endpoint = true) {
  start = Math.log(start);
  stop = Math.log(stop);
  return exp(linspace(start, stop, steps, endpoint));
}
