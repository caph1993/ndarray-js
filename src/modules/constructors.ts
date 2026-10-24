//@ts-check
import { Shape, empty, NDArray, asarray } from '../NDArray';
import { float64, DType, int32, int64, new_buffer, fix_buffer, dtype_is_integer } from '../dtypes';
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

/**
 * arange(stop), arange(start, stop), arange(start, stop, step).
 * The dtype is inferred from the arguments (int32 if they are all integers, float64 otherwise) unless given.
 */
export function arange(arg0: number, arg1: number | null = null, step: number = 1, dtype: DType | null = null) {
  const is_weak_float = (v: any) => v instanceof NDArray && v._weak && v.dtype === float64;
  const values = [arg0, arg1, step].filter(v => v !== null);
  const any_float = values.some(v => is_weak_float(v) || !Number.isInteger(+(v as any)));
  const num = (v: any) => (v instanceof NDArray ? v.item() : v) as number;
  let start = 0, stop: number;
  if (arg1 === null) stop = num(arg0);
  else { start = num(arg0); stop = num(arg1); }
  const st = num(step);
  if (st === 0) throw new Error(`Maximum allowed size exceeded: step must not be zero`);
  if (!Number.isFinite(start) || !Number.isFinite(stop)) throw new Error(`arange: cannot compute length`);
  const n = Math.max(0, Math.ceil((stop - start) / st));
  const dt: DType = dtype || (any_float ? float64 : ((Math.abs(start) > 2147483647 || Math.abs(stop) > 2147483647) ? int64 : int32));
  return new_array(n, dt, (i) => start + i * st);
};

/** num evenly spaced samples over [start, stop] */
export function linspace(start: number, stop: number, num = 50, endpoint = true, dtype: DType | null = null) {
  if (!Number.isInteger(num) || num < 0) throw new Error(`Number of samples, ${num}, must be non-negative.`);
  const div = endpoint ? num - 1 : num;
  const delta = stop - start;
  const step = div > 0 ? delta / div : NaN;
  const buffer = new_buffer(num, float64);
  for (let i = 0; i < num; i++) buffer[i] = div > 0 ? start + i * step : start;
  if (endpoint && num > 1) buffer[num - 1] = stop;
  let out = new NDArray(buffer, [num], float64);
  if (dtype && dtype !== float64) {
    out = new NDArray(fix_buffer(new_buffer(Array.from(buffer as any).map((v: any) => dtype_is_integer(dtype) ? Math.floor(v) : v), dtype), dtype), [num], dtype);
  }
  return out;
}

/** num samples spaced evenly on a log scale (a geometric progression) from start to stop */
export function geomspace(start: number, stop: number, num = 50, endpoint = true, dtype: DType | null = null) {
  if (start === 0 || stop === 0) throw new Error(`Geometric sequence cannot include zero`);
  if (start < 0 !== stop < 0) throw new Error(`Geometric sequence cannot mix signs (complex numbers are not supported)`);
  const sign = start < 0 ? -1 : 1;
  const lin = linspace(Math.log10(Math.abs(start)), Math.log10(Math.abs(stop)), num, endpoint);
  const buffer = new_buffer(num, float64);
  const f = lin.flat as any;
  for (let i = 0; i < num; i++) buffer[i] = sign * Math.pow(10, f[i]);
  // Exact endpoints
  if (num > 0) buffer[0] = start;
  if (endpoint && num > 1) buffer[num - 1] = stop;
  return new NDArray(buffer, [num], float64);
}
