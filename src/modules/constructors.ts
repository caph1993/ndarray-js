//@ts-check
import { Shape, empty, NDArray, asarray, broadcast_shapes } from '../NDArray';
import { np } from '../np_circular_import';
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

/** Samples of linspace between two numbers */
function lin_samples(start: number, stop: number, num: number, endpoint: boolean): number[] {
  const div = endpoint ? num - 1 : num;
  const step = div > 0 ? (stop - start) / div : NaN;
  const out = new Array(num);
  for (let i = 0; i < num; i++) out[i] = div > 0 ? start + i * step : start;
  if (endpoint && num > 1) out[num - 1] = stop;
  return out;
}
function geom_samples(start: number, stop: number, num: number, endpoint: boolean): number[] {
  if (start === 0 || stop === 0) throw new Error(`Geometric sequence cannot include zero`);
  if (start < 0 !== stop < 0) throw new Error(`Geometric sequence cannot mix signs (complex numbers are not supported)`);
  const sign = start < 0 ? -1 : 1;
  const lin = lin_samples(Math.log10(Math.abs(start)), Math.log10(Math.abs(stop)), num, endpoint);
  const out = lin.map(v => sign * Math.pow(10, v));
  if (num > 0) out[0] = start;
  if (endpoint && num > 1) out[num - 1] = stop;
  return out;
}

/**
 * Builds an array with `num` samples between every (start, stop) pair (start and stop broadcast against each other);
 * the samples run along `axis` of the result.
 */
function space_nd(start: any, stop: any, num: number, endpoint: boolean, axis: number, gen: (a: number, b: number, num: number, endpoint: boolean) => number[], dtype: DType | null) {
  if (!Number.isInteger(num) || num < 0) throw new Error(`Number of samples, ${num}, must be non-negative.`);
  const sa = asarray(start), sb = asarray(stop);
  const shape = broadcast_shapes(sa.shape, sb.shape)[0] as number[];
  const fa = asarray(broadcast_to_(sa, shape)).flat as any, fb = asarray(broadcast_to_(sb, shape)).flat as any;
  const m = fa.length;
  const buffer = new_buffer(m * num, float64);
  for (let i = 0; i < m; i++) { const row = gen(+fa[i], +fb[i], num, endpoint); for (let j = 0; j < num; j++) buffer[i * num + j] = row[j]; }
  let out = new NDArray(buffer, [...shape, num], float64);
  const nd = shape.length + 1;
  if (axis < -nd || axis >= nd) throw new Error(`axis ${axis} is out of bounds for array of dimension ${nd}`);
  axis = axis < 0 ? axis + nd : axis;
  if (axis !== nd - 1) out = np.copy(np.moveaxis(out, -1, axis));
  if (dtype && dtype !== float64) {
    const f = out.flat as any;
    out = new NDArray(fix_buffer(new_buffer(Array.from(f).map((v: any) => dtype_is_integer(dtype) ? Math.floor(v) : v), dtype), dtype), out.shape, dtype);
  }
  return out;
}
function broadcast_to_(a: NDArray, shape: number[]) { return a.shape.length === shape.length && a.shape.every((d, i) => d === shape[i]) ? a : np.broadcast_to(a, shape); }

/** num evenly spaced samples over [start, stop] (start and stop can be arrays; retstep also returns the step) */
export function linspace(start: any, stop: any, num = 50, endpoint = true, retstep = false, dtype: DType | null = null, axis = 0): any {
  const out = space_nd(start, stop, num, endpoint, axis, lin_samples, dtype);
  if (!retstep) return out;
  const div = endpoint ? num - 1 : num;
  const delta = np.subtract(stop, start);
  const step = div > 0 ? np.divide(delta, div) : np.multiply(delta, NaN);
  return Object.assign([out, step], { samples: out, step });
}

/** num samples spaced evenly on a log scale (a geometric progression) from start to stop */
export function geomspace(start: any, stop: any, num = 50, endpoint = true, dtype: DType | null = null, axis = 0) {
  return space_nd(start, stop, num, endpoint, axis, geom_samples, dtype);
}

/** num samples spaced evenly on a log scale: base ** linspace(start, stop) */
export function logspace(start: any, stop: any, num = 50, endpoint = true, base = 10.0, dtype: DType | null = null, axis = 0) {
  return space_nd(start, stop, num, endpoint, axis, (a, b, n, e) => lin_samples(a, b, n, e).map(v => Math.pow(base, v)), dtype);
}
