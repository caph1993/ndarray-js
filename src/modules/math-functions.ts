import { is_cx } from '../dtypes';
//@ts-check
import { NDArray, asarray, isarray, broadcast_shapes } from "../NDArray";
import {
  DType, DtypeResolver, bool, float_out, same_out, sum_out, new_buffer, int32, float64, dtype_max,
  dtype_is_float, dtype_is_integer, numeric_out, complex128
} from "../dtypes";
import { apply_along_axis, concatenate, normalize_axis, swapAxes } from "../array/transform";
import { ravel, reshape } from "../array/shape_operations";
import { finish_output, apply_binary_operation, n_ary_elementwise } from "../array/operators";
import { AxisArg, normalize_axes } from "../array/reduce";
import { Complex, ZERO, ONE } from "../complex";
import { convert_buffer } from "../dtypes";
import { np } from "../np_circular_import";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

function store(res: NDArray, out: NDArray | null): NDArray {
  return out ? finish_output(res.flat, res.shape, out.dtype, out) : res;
}

// ==============================
//   Cumulative operations
// ==============================

function accumulate(
  a: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null,
  resolver: DtypeResolver, init: number, step: (acc: number, x: number) => number, skip_nan = false, include_initial = false
): NDArray {
  a = asarray(a);
  if (dtype && dtype !== a.dtype && is_cx(a.dtype) && !is_cx(dtype)) a = new NDArray(convert_buffer(a.flat, dtype), a.shape, dtype); // complex -> real drops the imaginary part
  if (axis == null) { a = ravel(a); axis = 0; }
  if (a.shape.length == 0) a = reshape(a, [1]);
  if (a.size === 0 && !include_initial) {
    const dt0 = is_cx(a.dtype) ? complex128 : resolver([dtype || a.dtype], out || dtype);
    return store(new NDArray(new_buffer(0, dt0), a.shape.slice(), dt0), out);
  }
  if (is_cx(a.dtype)) {
    if (init !== 0 && init !== 1) throw new Error(`this cumulative operation is not supported for complex numbers`);
    const cstep = init === 0 ? (x: Complex, y: Complex) => x.add(y) : (x: Complex, y: Complex) => x.mul(y);
    const res = apply_along_axis(() => complex128, (lane) => {
      const o = new Array(lane.length + (include_initial ? 1 : 0));
      let acc = init === 0 ? ZERO : ONE, k = 0;
      if (include_initial) o[k++] = acc;
      for (let i = 0; i < lane.length; i++) { if (!(skip_nan && lane[i].isNaN)) acc = cstep(acc, lane[i]); o[k++] = acc; }
      return o;
    }, a, axis);
    return store(res, out);
  }
  const dt = resolver([dtype || a.dtype], out || dtype);
  const res = apply_along_axis(() => dt, (lane) => {
    const n = lane.length;
    const o = new Array(n + (include_initial ? 1 : 0));
    let acc = init, k = 0;
    if (include_initial) o[k++] = acc;
    for (let i = 0; i < n; i++) {
      const v = +lane[i];
      if (!(skip_nan && v !== v)) acc = step(acc, v);
      o[k++] = acc;
    }
    return o;
  }, a, axis);
  return store(res, out);
}

export function cumsum(a: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null) {
  return accumulate(a, axis, dtype, out, sum_out, 0, (s, x) => s + x);
}
export function cumprod(a: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null) {
  return accumulate(a, axis, dtype, out, sum_out, 1, (s, x) => s * x);
}
export function nancumsum(a: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null) {
  return accumulate(a, axis, dtype, out, sum_out, 0, (s, x) => s + x, true);
}
export function nancumprod(a: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null) {
  return accumulate(a, axis, dtype, out, sum_out, 1, (s, x) => s * x, true);
}
export function cumulative_sum(x: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null, include_initial: boolean) {
  x = asarray(x);
  if (axis == null && x.shape.length > 1) throw new Error(`axis must be specified for arrays with more than one dimension`);
  return accumulate(x, axis, dtype, out, sum_out, 0, (s, v) => s + v, false, include_initial);
}
export function cumulative_prod(x: NDArray, axis: number | null, dtype: DType | null, out: NDArray | null, include_initial: boolean) {
  x = asarray(x);
  if (axis == null && x.shape.length > 1) throw new Error(`axis must be specified for arrays with more than one dimension`);
  return accumulate(x, axis, dtype, out, sum_out, 1, (s, v) => s * v, false, include_initial);
}

// ==============================
//   Differences
// ==============================

export function diff(a: NDArray, n: number, axis: number, prepend: NDArray | null, append: NDArray | null): NDArray {
  a = asarray(a);
  if (a.shape.length == 0) throw new Error(`diff requires input that is at least one dimensional`);
  if (n < 0) throw new Error(`order must be non-negative but got ${n}`);
  if (n === 0 && prepend == null && append == null) return a;
  axis = normalize_axis(axis, a.shape.length);
  const parts: NDArray[] = [];
  const as_part = (p: NDArray) => {
    p = asarray(p);
    if (p.shape.length == 0) {
      const shape = a.shape.slice(); shape[axis] = 1;
      return apply_binary_operation(same_out, (_x, y) => y, new NDArray(new_buffer(prod(shape), p.dtype), shape, p.dtype), p);
    }
    return p;
  };
  if (prepend != null) parts.push(as_part(prepend));
  parts.push(a);
  if (append != null) parts.push(as_part(append));
  if (parts.length > 1) a = concatenate(parts, axis);
  if (n == 0) return a;
  const isBool = a.dtype === bool, cx = is_cx(a.dtype);
  return apply_along_axis(isBool ? () => bool : cx ? () => a.dtype : numeric_out, (lane) => {
    let cur: any = Array.from(lane);
    for (let k = 0; k < n && cur.length > 0; k++) {
      const next = new Array(cur.length - 1);
      if (isBool) for (let i = 0; i < next.length; i++) next[i] = (!!cur[i + 1]) !== (!!cur[i]);
      else if (cx) for (let i = 0; i < next.length; i++) next[i] = Complex.from(cur[i + 1]).sub(Complex.from(cur[i]));
      else for (let i = 0; i < next.length; i++) next[i] = cur[i + 1] - cur[i];
      cur = next;
    }
    return cur;
  }, a, axis);
}

export function ediff1d(ary: NDArray, to_end: NDArray | null, to_begin: NDArray | null): NDArray {
  ary = ravel(asarray(ary));
  if (ary.dtype === bool) throw new TypeError('numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.');
  const d = diff(ary, 1, 0, null, null);
  const parts: NDArray[] = [];
  if (to_begin != null) parts.push(ravel(asarray(to_begin)));
  parts.push(d);
  if (to_end != null) parts.push(ravel(asarray(to_end)));
  return parts.length == 1 ? d : concatenate(parts, 0);
}

function gradient_lane(f: ArrayLike<number>, coords: ArrayLike<number> | number, edge_order: number): number[] {
  const n = f.length;
  if (n < edge_order + 1) throw new Error(`Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.`);
  const out = new Array(n);
  const uniform = typeof coords === "number";
  const x = (i: number) => uniform ? i * (coords as number) : (coords as ArrayLike<number>)[i];
  for (let i = 1; i < n - 1; i++) {
    const dx1 = x(i) - x(i - 1), dx2 = x(i + 1) - x(i);
    if (uniform) out[i] = (f[i + 1] - f[i - 1]) / (2 * (coords as number));
    else {
      const a = -dx2 / (dx1 * (dx1 + dx2)), b = (dx2 - dx1) / (dx1 * dx2), c = dx1 / (dx2 * (dx1 + dx2));
      out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
    }
  }
  if (edge_order == 1) {
    out[0] = (f[1] - f[0]) / (x(1) - x(0));
    out[n - 1] = (f[n - 1] - f[n - 2]) / (x(n - 1) - x(n - 2));
  } else {
    let dx1 = x(1) - x(0), dx2 = x(2) - x(1);
    let a = -(2 * dx1 + dx2) / (dx1 * (dx1 + dx2)), b = (dx1 + dx2) / (dx1 * dx2), c = -dx1 / (dx2 * (dx1 + dx2));
    out[0] = a * f[0] + b * f[1] + c * f[2];
    dx1 = x(n - 2) - x(n - 3); dx2 = x(n - 1) - x(n - 2);
    a = dx2 / (dx1 * (dx1 + dx2)); b = -(dx2 + dx1) / (dx1 * dx2); c = (2 * dx2 + dx1) / (dx2 * (dx1 + dx2));
    out[n - 1] = a * f[n - 3] + b * f[n - 2] + c * f[n - 1];
  }
  return out;
}

/** Gradient of an N-dimensional array. `varargs` are spacings (numbers) or coordinate arrays, one per axis (or a single one for all). */
export function gradient(f: NDArray, varargs: any[], axis: AxisArg, edge_order: number): NDArray | NDArray[] {
  f = asarray(f);
  if (is_cx(f.dtype)) { // linear: gradient(re) + 1j * gradient(im)
    const re = gradient(np.real(f), varargs, axis, edge_order), im = gradient(np.imag(f), varargs, axis, edge_order);
    const comb = (r: any, i: any) => np.add(r, np.multiply(i, new Complex(0, 1) as any));
    return Array.isArray(re) ? (re as NDArray[]).map((r, k) => comb(r, (im as NDArray[])[k])) : comb(re, im);
  }
  if (f.dtype === bool) throw new TypeError('numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.');
  const nd = f.shape.length;
  if (edge_order != 1 && edge_order != 2) throw new Error(`'edge_order' greater than 2 not supported`);
  const axes = normalize_axes(axis, nd);
  let spacings: any[];
  if (varargs.length == 0) spacings = axes.map(() => 1);
  else if (varargs.length == 1) spacings = axes.map(() => varargs[0]);
  else if (varargs.length == axes.length) spacings = varargs;
  else throw new Error(`Invalid number of arguments`);
  const dt = float_out([f.dtype]);
  const results = axes.map((ax, k) => {
    let sp = spacings[k];
    if (isarray(sp) && sp.shape.length == 0) sp = sp.item();
    else if (Array.isArray(sp) || isarray(sp)) {
      sp = asarray(sp);
      if (sp.shape.length != 1 || sp.shape[0] != f.shape[ax]) throw new Error(`Coordinates must be 1-D arrays with length matching the axis`);
      sp = Array.from(sp.flat as any) as number[];
      // Uniform coordinates are treated as a scalar spacing (as numpy)
      const d0 = sp[1] - sp[0];
      if (sp.every((v: number, i: number) => i == 0 || Math.abs((v - sp[i - 1]) - d0) <= 1e-12 * Math.max(1, Math.abs(d0)))) sp = d0 === d0 && sp.length > 1 ? d0 : sp;
    }
    return apply_along_axis(() => dt, (lane) => gradient_lane(lane, sp, edge_order), f, ax);
  });
  return axis == null && nd > 1 || (axis != null && Array.isArray(axis)) ? results : results[0];
}

export function cross(a: NDArray, b: NDArray, axisa: number, axisb: number, axisc: number, axis: number | null): NDArray {
  a = asarray(a); b = asarray(b);
  if (axis != null) { axisa = axisb = axisc = axis; }
  const na = a.shape[normalize_axis(axisa, a.shape.length)], nb = b.shape[normalize_axis(axisb, b.shape.length)];
  if (na != 3 || nb != 3) throw new Error(`Both input arrays must be (arrays of) 3-dimensional vectors, but they are ${na} and ${nb} dimensional instead.`);
  // Move the vector axis last and take the three components
  const A = swapAxes(a, axisa, -1), B = swapAxes(b, axisb, -1);
  const comp = (X: NDArray, k: number) => {
    const sel = indexingTake(X, k);
    return sel;
  };
  const mul = (u: NDArray, v: NDArray) => apply_binary_operation(same_out, (p, q) => p * q, u, v);
  const sub = (u: NDArray, v: NDArray) => apply_binary_operation(numeric_out, (p, q) => p - q, u, v);
  const [a0, a1, a2, b0, b1, b2] = [comp(A, 0), comp(A, 1), comp(A, 2), comp(B, 0), comp(B, 1), comp(B, 2)];
  const c0 = sub(mul(a1, b2), mul(a2, b1));
  const c1 = sub(mul(a2, b0), mul(a0, b2));
  const c2 = sub(mul(a0, b1), mul(a1, b0));
  const res = concatenate([c0, c1, c2].map(c => reshape(c, [...c.shape, 1])), c0.shape.length);
  return res.shape.length == 1 ? res : swapAxes(res, -1, normalize_axis(axisc, res.shape.length));
}

/** Component k of the last axis (keeps the dtype) */
function indexingTake(X: NDArray, k: number): NDArray {
  const flat = X.flat;
  const n = X.shape[X.shape.length - 1];
  const out = new_buffer(flat.length / n, X.dtype);
  for (let i = 0; i < out.length; i++) out[i] = flat[i * n + k];
  return new NDArray(out, X.shape.slice(0, -1), X.dtype);
}

export function trapezoid(y: NDArray, x: NDArray | null, dx: number, axis: number): NDArray {
  y = asarray(y);
  axis = normalize_axis(axis, y.shape.length);
  const dt = float_out([y.dtype]);
  let xs: NDArray | null = x == null ? null : asarray(x);
  if (xs && xs.shape.length > 1) {
    // x with the same shape as y: integrate pairing lanes
    if (xs.shape.length != y.shape.length) throw new Error(`x and y must have the same number of dimensions or x must be 1D`);
    const lanes = apply_along_axis(() => dt, (lane) => lane, y, axis);
    const xl = apply_along_axis(() => dt, (lane) => lane, xs, axis);
    const yy = swapAxes(lanes, axis, -1), xx = swapAxes(xl, axis, -1);
    const n = yy.shape[yy.shape.length - 1];
    const fy = yy.flat, fx = xx.flat;
    const nLanes = n == 0 ? 0 : fy.length / n;
    const buffer = new_buffer(nLanes, dt);
    for (let l = 0; l < nLanes; l++) {
      let s = 0;
      for (let i = 1; i < n; i++) s += (fx[l * n + i] - fx[l * n + i - 1]) * (y.dtype === bool ? (fy[l * n + i] || fy[l * n + i - 1] ? 1 : 0) : fy[l * n + i] + fy[l * n + i - 1]) / 2;
      buffer[l] = s;
    }
    return new NDArray(buffer, yy.shape.slice(0, -1), dt);
  }
  const xv = xs ? Array.from(xs.flat as any) as number[] : null;
  if (xv && xv.length != y.shape[axis]) throw new Error(`operands could not be broadcast together: x has ${xv.length} elements, y axis has ${y.shape[axis]}`);
  const lanes = apply_along_axis(() => dt, (lane) => {
    let s = 0;
    for (let i = 1; i < lane.length; i++) s += (xv ? xv[i] - xv[i - 1] : dx) * (y.dtype === bool ? (lane[i] || lane[i - 1] ? 1 : 0) : lane[i] + lane[i - 1]) / 2; // bool + bool is a logical or
    return [s];
  }, y, axis);
  // Remove the reduced axis
  const shape = lanes.shape.filter((_, i) => i != axis);
  return new NDArray(lanes.flat, shape, dt);
}

// ==============================
//   Convolution / correlation / interpolation
// ==============================

export type ConvolveMode = 'full' | 'same' | 'valid';

function check_mode(mode: string) {
  if (!['full', 'same', 'valid'].includes(mode)) throw new Error(`mode must be one of 'full', 'same', or 'valid' (got '${mode}')`);
}

/** Full discrete convolution of two lists */
function convolve_full(a: ArrayLike<any>, v: ArrayLike<any>): any[] {
  if (a.length && (a[0] instanceof Complex || v[0] instanceof Complex)) {
    const out: Complex[] = new Array(a.length + v.length - 1).fill(ZERO);
    for (let i = 0; i < a.length; i++) for (let j = 0; j < v.length; j++) out[i + j] = out[i + j].add(Complex.from(a[i]).mul(Complex.from(v[j])));
    return out;
  }
  const out = new Array(a.length + v.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < v.length; j++) out[i + j] += a[i] * v[j];
  return out;
}

function slice_mode(full: any[], na: number, nv: number, mode: string): any[] {
  if (mode == 'full') return full;
  const n = Math.max(na, nv), m = Math.min(na, nv);
  if (mode == 'valid') return full.slice(m - 1, n);
  const first = Math.floor((full.length - n) / 2);
  return full.slice(first, first + n);
}

export function convolve(a: NDArray, v: NDArray, mode: ConvolveMode): NDArray {
  check_mode(mode);
  a = asarray(a); v = asarray(v);
  if (a.shape.length == 0) a = reshape(a, [1]);
  if (v.shape.length == 0) v = reshape(v, [1]);
  if (a.shape.length != 1 || v.shape.length != 1) throw new Error(`object too deep for desired array / convolve expects 1-D inputs`);
  if (a.size == 0 || v.size == 0) throw new Error(`${a.size == 0 ? 'a' : 'v'} cannot be empty`);
  const dtype = same_out([a.dtype === bool ? int32 : a.dtype, v.dtype === bool ? int32 : v.dtype].map(d => d));
  const res = slice_mode(convolve_full(a.flat as any, v.flat as any), a.size, v.size, mode);
  return new NDArray(new_buffer(res, dtype), [res.length], dtype);
}

export function correlate(a: NDArray, v: NDArray, mode: ConvolveMode): NDArray {
  check_mode(mode);
  a = asarray(a); v = asarray(v);
  if (a.shape.length != 1 || v.shape.length != 1) throw new Error(`object too deep for desired array / correlate expects 1-D inputs`);
  if (a.size == 0 || v.size == 0) throw new Error(`${a.size == 0 ? 'a' : 'v'} cannot be empty`);
  const na = a.size, nv = v.size;
  const fa = a.flat as any, fv = v.flat as any;
  // full[k + nv - 1] = sum_n a[n + k] v[n],  k = -(nv-1) .. na-1
  const cx = is_cx(a.dtype) || is_cx(v.dtype);
  const full: any[] = new Array(na + nv - 1).fill(cx ? ZERO : 0);
  for (let k = -(nv - 1); k < na; k++) {
    let s: any = cx ? ZERO : 0;
    // numpy correlates with the conjugate of v
    if (cx) { for (let n = Math.max(0, -k); n < nv && n + k < na; n++) s = s.add(Complex.from(fa[n + k]).mul(Complex.from(fv[n]).conj())); }
    else for (let n = Math.max(0, -k); n < nv && n + k < na; n++) s += fa[n + k] * fv[n];
    full[k + nv - 1] = s;
  }
  const dtype = same_out([a.dtype === bool ? int32 : a.dtype, v.dtype === bool ? int32 : v.dtype]);
  // numpy's slicing when v is longer than a: result is computed as the full correlation, trimmed around the zero-lag center
  let res: any[];
  if (mode == 'full') res = full;
  else if (mode == 'valid') {
    const n = Math.abs(na - nv) + 1;
    // zero-lag index is nv-1. valid keeps lags between (na>=nv ? 0 : na-nv) .. (na>=nv ? na-nv : 0)
    const lo = na >= nv ? 0 : na - nv;
    res = full.slice(lo + nv - 1, lo + nv - 1 + n);
  } else {
    const n = Math.max(na, nv);
    const first = (nv > na ? Math.ceil : Math.floor)((full.length - n) / 2);
    res = full.slice(first, first + n);
  }
  return new NDArray(new_buffer(res, dtype), [res.length], dtype);
}

export function interp(x: NDArray, xp: NDArray, fp: NDArray, left: number | null, right: number | null, period: number | null): NDArray {
  x = asarray(x); xp = asarray(xp); fp = asarray(fp);
  if (xp.shape.length != 1 || fp.shape.length != 1) throw new Error(`xp and fp must be 1D arrays`);
  if (xp.size != fp.size) throw new Error(`fp and xp are not of the same length`);
  if (xp.size == 0) throw new Error(`array of sample points is empty`);
  let xs = Array.from(xp.flat as any) as number[];
  let fs = Array.from(fp.flat as any) as number[];
  let l = left, r = right;
  if (period != null) {
    if (period == 0) throw new Error(`period must be a non-zero value`);
    period = Math.abs(period);
    l = null; r = null;
    const mod = (v: number) => ((v % period!) + period!) % period!;
    const idx = xs.map((v, i) => i).sort((i, j) => mod(xs[i]) - mod(xs[j]));
    const xs2 = idx.map(i => mod(xs[i])), fs2 = idx.map(i => fs[i]);
    xs = [xs2[xs2.length - 1] - period, ...xs2, xs2[0] + period];
    fs = [fs2[fs2.length - 1], ...fs2, fs2[0]];
    const xin = x;
    x = apply_binary_operation(() => float64, (v) => mod(v), xin, 0);
  }
  const n = xs.length;
  const lv = l === null ? fs[0] : l, rv = r === null ? fs[n - 1] : r;
  const buffer = new_buffer(x.size, float64);
  const flat = x.flat;
  for (let k = 0; k < flat.length; k++) {
    const v = +flat[k];
    if (Number.isNaN(v)) { buffer[k] = v; continue; }
    if (v < xs[0]) { buffer[k] = lv; continue; }
    if (v > xs[n - 1]) { buffer[k] = rv; continue; }
    if (v === xs[n - 1]) { buffer[k] = fs[n - 1]; continue; }
    let lo = 0, hi = n - 1; // find lo with xs[lo] <= v < xs[lo+1]
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= v) lo = mid; else hi = mid; }
    const slope = (fs[lo + 1] - fs[lo]) / (xs[lo + 1] - xs[lo]);
    let res = slope * (v - xs[lo]) + fs[lo];
    if (Number.isNaN(res)) {
      res = slope * (v - xs[lo + 1]) + fs[lo + 1];
      if (Number.isNaN(res) && fs[lo] === fs[lo + 1]) res = fs[lo];
    }
    buffer[k] = res;
  }
  return new NDArray(buffer, x.shape, float64);
}

export function unwrap(p: NDArray, discont: number | null, axis: number, period: number): NDArray {
  p = asarray(p);
  const dt = float_out([p.dtype]);
  const half = period / 2;
  if (discont === null) discont = half;
  discont = Math.max(discont, half);
  return apply_along_axis(() => dt, (lane) => {
    const n = lane.length;
    const out = new Array(n);
    if (n == 0) return out;
    out[0] = lane[0];
    let correction = 0;
    for (let i = 1; i < n; i++) {
      const dd = lane[i] - lane[i - 1];
      let ddmod = ((dd + half) % period + period) % period - half;
      if (ddmod === -half && dd > 0) ddmod = half;
      let corr = ddmod - dd;
      if (Math.abs(dd) < discont) corr = 0;
      correction += corr;
      out[i] = lane[i] + correction;
    }
    return out;
  }, p, axis);
}

// ==============================
//   Window functions
// ==============================
function window_fn(M: number, f: (n: number, M: number) => number): NDArray {
  if (!Number.isInteger(M)) throw new TypeError(`M must be an integer`);
  if (M < 1) return new NDArray(new_buffer(0, float64), [0], float64);
  if (M === 1) return new NDArray(new_buffer([1], float64), [1], float64);
  const buffer = new_buffer(M, float64);
  for (let i = 0; i < M; i++) buffer[i] = f(1 - M + 2 * i, M);
  return new NDArray(buffer, [M], float64);
}
export const hanning = (M: number) => window_fn(M, (n, M) => 0.5 + 0.5 * Math.cos(Math.PI * n / (M - 1)));
export const hamming = (M: number) => window_fn(M, (n, M) => 0.54 + 0.46 * Math.cos(Math.PI * n / (M - 1)));
export const blackman = (M: number) => window_fn(M, (n, M) => 0.42 + 0.5 * Math.cos(Math.PI * n / (M - 1)) + 0.08 * Math.cos(2 * Math.PI * n / (M - 1)));
export const bartlett = (M: number) => window_fn(M, (n, M) => n <= 0 ? 1 + n / (M - 1) : 1 - n / (M - 1));
export function kaiser(M: number, beta: number): NDArray {
  if (M < 1) return new NDArray(new_buffer(0, float64), [0], float64);
  if (M === 1) return new NDArray(new_buffer([1], float64), [1], float64);
  const i0 = (x: number) => { let term = 1, sum = 1; const q = x * x / 4; for (let k = 1; k < 500; k++) { term *= q / (k * k); sum += term; if (term < sum * 1e-17) break; } return sum; };
  const alpha = (M - 1) / 2, buffer = new_buffer(M, float64);
  for (let n = 0; n < M; n++) buffer[n] = i0(beta * Math.sqrt(1 - Math.pow((n - alpha) / alpha, 2))) / i0(beta);
  return new NDArray(buffer, [M], float64);
}
