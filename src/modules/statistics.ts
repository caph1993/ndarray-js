//@ts-check
import { NDArray, asarray, isarray, broadcast_shapes } from "../NDArray";
import { AxisArg, reduce_lanes, normalize_axes, lane_sum, lane_max, lane_min, clane_mean } from "../array/reduce";
import { Complex, cmp_complex } from "../complex";
import { cmp_nan_at_the_end, normalize_axis, swapAxes, concatenate, transpose } from "../array/transform";
import { ravel, reshape } from "../array/shape_operations";
import { complex128, DType, bool, float32, float64, int32, float_out, new_buffer, dtype_is_integer, dtype_is_float } from "../dtypes";
import { finish_output } from "../array/operators";
import { rint_scalar } from "../array/elementwise";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

// ==============================
//   Quantiles
// ==============================

export type QuantileMethod =
  | 'inverted_cdf' | 'averaged_inverted_cdf' | 'closest_observation' | 'interpolated_inverted_cdf'
  | 'hazen' | 'weibull' | 'linear' | 'median_unbiased' | 'normal_unbiased'
  | 'lower' | 'higher' | 'midpoint' | 'nearest';

/** Methods that always return one of the observed values (keep integer dtypes) */
const DISCRETE_METHODS = ['inverted_cdf', 'closest_observation', 'lower', 'higher', 'nearest'];

const virtual_index = (n: number, q: number, alpha: number, beta: number) => n * q + (alpha + q * (1 - alpha - beta)) - 1;
const continuous = (alpha: number, beta: number) => ({
  index: (n: number, q: number) => virtual_index(n, q, alpha, beta),
  gamma: (g: number, _v: number) => g,
});
/**
 * Hyndman & Fan / numpy quantile methods.
 * Continuous methods interpolate between sorted[floor(index)] and the next one with weight gamma.
 * Discrete methods return `pick`: the position of the observation.
 */
const QUANTILE_METHODS: { [name: string]: { index?: (n: number, q: number) => number, gamma?: (g: number, v: number) => number, pick?: (n: number, q: number) => number } } = {
  inverted_cdf: { pick: (n, q) => { const v = n * q - 1, prev = Math.floor(v); return (v - prev === 0) ? prev : prev + 1; } },
  averaged_inverted_cdf: { index: (n, q) => n * q - 1, gamma: (g) => g == 0 ? 0.5 : 1 },
  closest_observation: { pick: (n, q) => { const v = n * q - 1.5, prev = Math.floor(v); return (v - prev === 0 && Math.floor(v) % 2 !== 0 && Math.abs(Math.floor(v) % 2) === 1) ? prev : prev + 1; } },
  interpolated_inverted_cdf: continuous(0, 1),
  hazen: continuous(0.5, 0.5),
  weibull: continuous(0, 0),
  linear: { index: (n, q) => (n - 1) * q, gamma: (g) => g },
  median_unbiased: continuous(1 / 3, 1 / 3),
  normal_unbiased: continuous(3 / 8, 3 / 8),
  lower: { pick: (n, q) => Math.floor((n - 1) * q) },
  higher: { pick: (n, q) => Math.ceil((n - 1) * q) },
  midpoint: { index: (n, q) => 0.5 * (Math.floor((n - 1) * q) + Math.ceil((n - 1) * q)), gamma: (g) => g },
  nearest: { pick: (n, q) => rint_scalar((n - 1) * q) },
};

function lerp(a: number, b: number, t: number) {
  const d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

/** q-th quantile (q in [0,1]) of a lane; `x` must be sorted (nans at the end) and n is the number of valid values */
function quantile_sorted(x: ArrayLike<number>, n: number, q: number, method: string): number {
  const m = QUANTILE_METHODS[method];
  if (n == 0) return NaN;
  if (m.pick) return x[Math.min(n - 1, Math.max(0, m.pick(n, q)))];
  const v = m.index!(n, q);
  let prev = Math.floor(v), next = prev + 1;
  if (v >= n - 1) { prev = n - 1; next = n - 1; }
  else if (v < 0) { prev = 0; next = 0; }
  if (prev === next) return x[prev];
  return lerp(x[prev], x[next], m.gamma!(v - Math.floor(v), v));
}

function quantile_lane(lane: any, q: number, method: string, skip_nan: boolean): number {
  const x = Float64Array.from(lane as ArrayLike<number>, (v) => +v);
  x.sort(); // numeric order, nans at the end
  let n = x.length;
  while (n > 0 && x[n - 1] !== x[n - 1]) n--;
  if (n < x.length && !skip_nan) return NaN;
  return quantile_sorted(x, n, q, method);
}

/**
 * Compute the q-th quantile of the data along the specified axis. q must be in [0, 1].
 * If q is an array, its shape is prepended to the shape of the result.
 */
export function quantile(a: NDArray, q: NDArray | number, axis: AxisArg, keepdims: boolean = false, method: QuantileMethod = 'linear', out: NDArray | null = null, skip_nan = false, scale = 1): NDArray {
  a = asarray(a);
  const Q = asarray(q);
  if (!QUANTILE_METHODS[method]) throw new Error(`${method} is not a valid method. Use one of: ${Object.keys(QUANTILE_METHODS).join(', ')}`);
  const qs = Array.from(Q.flat as any).map(v => +v / scale);
  for (const v of qs) if (!(v >= 0 && v <= 1)) throw new Error(scale == 1 ? `Quantiles must be in the range [0, 1]` : `Percentiles must be in the range [0, 100]`);
  const integer = DISCRETE_METHODS.includes(method) && (dtype_is_integer(a.dtype) || a.dtype === bool);
  const dtype: DType = integer ? (a.dtype === bool ? int32 : a.dtype) : float_out([a.dtype]);
  const parts = qs.map(qv => reduce_lanes(a, axis, keepdims, dtype, (lane) => quantile_lane(lane, qv, method, skip_nan)));
  const inner = parts[0] ? parts[0].shape : reduce_lanes(a, axis, keepdims, dtype, () => 0).shape;
  const shape = [...Q.shape, ...inner];
  const buffer = new_buffer(prod(shape), dtype);
  const innerSize = prod(inner);
  parts.forEach((p, k) => { const f = p.flat; for (let i = 0; i < innerSize; i++) buffer[k * innerSize + i] = f[i]; });
  return finish_output(buffer, shape, dtype, out);
}

export function nanquantile(a: NDArray, q: NDArray | number, axis: AxisArg, keepdims: boolean = false, method: QuantileMethod = 'linear', out: NDArray | null = null) {
  return quantile(a, q, axis, keepdims, method, out, true);
}
export function percentile(a: NDArray, q: NDArray | number, axis: AxisArg, keepdims: boolean = false, method: QuantileMethod = 'linear', out: NDArray | null = null) {
  return quantile(a, q, axis, keepdims, method, out, false, 100);
}
export function nanpercentile(a: NDArray, q: NDArray | number, axis: AxisArg, keepdims: boolean = false, method: QuantileMethod = 'linear', out: NDArray | null = null) {
  return quantile(a, q, axis, keepdims, method, out, true, 100);
}

/** Compute the median along the specified axis. */
/** median of complex numbers: lexicographic order, mean of the middle values */
function complex_median(a: NDArray, axis: AxisArg, keepdims: boolean, out: NDArray | null, skip_nan: boolean) {
  return reduce_lanes(a, axis, keepdims, complex128, (lane: any) => {
    const l: Complex[] = Array.from(lane as ArrayLike<Complex>);
    const v = skip_nan ? l.filter(z => !z.isNaN) : l;
    if (!skip_nan && v.some(z => z.isNaN)) return new Complex(NaN, NaN);
    if (v.length === 0) return new Complex(NaN, NaN);
    v.sort(cmp_complex);
    const n = v.length;
    return n % 2 ? v[(n - 1) / 2] : v[n / 2 - 1].add(v[n / 2]).div(new Complex(2, 0));
  }, out);
}
export function median(a: NDArray, axis: AxisArg, keepdims: boolean = false, out: NDArray | null = null) {
  if (asarray(a).dtype === complex128) return complex_median(asarray(a), axis, keepdims, out, false);
  return quantile(a, 0.5, axis, keepdims, 'linear', out, false);
}
export function nanmedian(a: NDArray, axis: AxisArg, keepdims: boolean = false, out: NDArray | null = null) {
  if (asarray(a).dtype === complex128) return complex_median(asarray(a), axis, keepdims, out, true);
  return quantile(a, 0.5, axis, keepdims, 'linear', out, true);
}

/**
 * Compute the weighted average along the specified axis.
 * Returns [average, sum_of_weights] if `returned`, else the average.
 */
export function average(a: NDArray, axis: AxisArg, weights: NDArray | null, returned: boolean, keepdims: boolean): NDArray | NDArray[] {
  a = asarray(a);
  const dtype = float_out([a.dtype]);
  if (weights === null && a.dtype === complex128) {
    const avg = reduce_lanes(a, axis, keepdims, complex128, (lane: any) => clane_mean(lane));
    return returned ? [avg, reduce_lanes(a, axis, keepdims, float64, (lane: any) => lane.length)] : avg;
  }
  if (weights === null) {
    const avg = reduce_lanes(a, axis, keepdims, dtype, (lane) => lane_sum(lane) / lane.length);
    if (!returned) return avg;
    const cnt = a.size / Math.max(1, avg.size);
    const scl = reduce_lanes(a, axis, keepdims, dtype, (lane) => lane.length);
    return [avg, scl];
  }
  weights = asarray(weights);
  const nd = a.shape.length;
  let w: NDArray;
  if (weights.shape.length === a.shape.length && weights.shape.every((d, i) => d === a.shape[i])) {
    w = weights;
  } else {
    if (axis == null) throw new Error(`Axis must be specified when shapes of a and weights differ.`);
    if (Array.isArray(axis)) throw new Error(`Axis must be an integer when shapes of a and weights differ.`);
    if (weights.shape.length !== 1) throw new Error(`1D weights expected when shapes of a and weights differ.`);
    const ax = normalize_axis(axis, nd);
    if (weights.shape[0] !== a.shape[ax]) throw new Error(`Length of weights not compatible with specified axis.`);
    const wshape = a.shape.map((_, i) => i == ax ? weights.shape[0] : 1);
    // broadcast weights to the shape of a
    const wflat = weights.flat;
    const buffer = new_buffer(a.size, float64);
    const inner = prod(a.shape.slice(ax + 1));
    for (let i = 0; i < a.size; i++) buffer[i] = wflat[Math.floor(i / inner) % weights.shape[0]];
    w = new NDArray(buffer, a.shape, float64);
  }
  const num = reduce_lanes(
    new NDArray(new_buffer(Array.from(a.flat as any).map((v: any, i) => (+v) * (+w.flat[i]) ), float64), a.shape, float64),
    axis, keepdims, dtype, lane_sum);
  const den = reduce_lanes(w, axis, keepdims, dtype, lane_sum);
  const nf = num.flat, df = den.flat;
  for (let i = 0; i < df.length; i++) if (df[i] === 0) throw new Error(`Weights sum to zero, can't be normalized`);
  const avg = new NDArray(new_buffer(Array.from(nf as any).map((v: any, i) => v / df[i]), dtype), num.shape, dtype);
  return returned ? [avg, den] : avg;
}

// ==============================
//   Covariance and correlation
// ==============================

function as_2d_rows(m: NDArray, rowvar: boolean): NDArray {
  m = asarray(m);
  if (m.shape.length > 2) throw new Error(`m has more than 2 dimensions`);
  if (m.shape.length < 2) m = reshape(m, [1, m.size]);
  if (!rowvar && m.shape[0] !== 1) m = transpose(m);
  return m;
}

export function cov(m: NDArray, y: NDArray | null, rowvar: boolean, bias: boolean, ddof: number | null, fweights: NDArray | null, aweights: NDArray | null): NDArray {
  if (ddof !== null && !Number.isInteger(ddof)) throw new Error(`ddof must be integer`);
  let X = as_2d_rows(m, rowvar);
  if (y !== null) {
    const Y = as_2d_rows(y, rowvar);
    X = concatenate([X, Y], 0);
  }
  if (ddof === null) ddof = bias ? 0 : 1;
  const [nv, N] = X.shape;
  let w: number[] | null = null;
  if (fweights !== null) {
    const f = Array.from(asarray(fweights).flat as any) as number[];
    if (f.length !== N) throw new Error(`incompatible numbers of samples and fweights`);
    if (f.some(v => v < 0)) throw new Error(`fweights cannot be negative`);
    w = f;
  }
  if (aweights !== null) {
    const aw = Array.from(asarray(aweights).flat as any) as number[];
    if (aw.length !== N) throw new Error(`incompatible numbers of samples and aweights`);
    if (aw.some(v => v < 0)) throw new Error(`aweights cannot be negative`);
    w = w === null ? aw : w.map((v, i) => v * aw[i]);
  }
  const x = X.flat as any;
  let wsum = N;
  if (w) wsum = w.reduce((s, v) => s + v, 0);
  const avg = new Array(nv).fill(0);
  for (let i = 0; i < nv; i++) {
    let s = 0;
    for (let k = 0; k < N; k++) s += x[i * N + k] * (w ? w[k] : 1);
    avg[i] = s / wsum;
  }
  let fact: number;
  if (w === null) fact = N - ddof;
  else if (ddof === 0) fact = wsum;
  else if (aweights === null) fact = wsum - ddof;
  else {
    const aw = Array.from(asarray(aweights).flat as any) as number[];
    let s = 0; for (let k = 0; k < N; k++) s += w[k] * aw[k];
    fact = wsum - ddof * s / wsum;
  }
  fact = Math.max(fact, 0);
  const out = new_buffer(nv * nv, float64);
  for (let i = 0; i < nv; i++) for (let j = i; j < nv; j++) {
    let s = 0;
    for (let k = 0; k < N; k++) s += (x[i * N + k] - avg[i]) * (x[j * N + k] - avg[j]) * (w ? w[k] : 1);
    out[i * nv + j] = out[j * nv + i] = s / fact;
  }
  return new NDArray(out, nv == 1 ? [] : [nv, nv], float64);
}

export function corrcoef(x: NDArray, y: NDArray | null, rowvar: boolean): NDArray {
  const c = cov(x, y, rowvar, false, null, null, null);
  if (c.shape.length == 0) return new NDArray(new_buffer([c.flat[0] / c.flat[0]], float64), [], float64);
  const n = c.shape[0];
  const f = c.flat as any;
  const sd = new Array(n);
  for (let i = 0; i < n; i++) sd[i] = Math.sqrt(f[i * n + i]);
  const out = new_buffer(n * n, float64);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = f[i * n + j] / sd[i] / sd[j];
    out[i * n + j] = Math.min(1, Math.max(-1, v));
  }
  return new NDArray(out, [n, n], float64);
}

// ==============================
//   Histograms
// ==============================

export type BinsMethod = 'auto' | 'fd' | 'doane' | 'scott' | 'stone' | 'rice' | 'sturges' | 'sqrt';
const BIN_METHODS = ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'];

function finite_range(x: ArrayLike<number>, range: [number, number] | null): [number, number] {
  let lo: number, hi: number;
  if (range) {
    [lo, hi] = range;
    if (lo > hi) throw new Error(`max must be larger than min in range parameter.`);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new Error(`supplied range of [${lo}, ${hi}] is not finite`);
  } else if (x.length == 0) {
    lo = 0; hi = 1;
  } else {
    lo = Infinity; hi = -Infinity;
    for (let i = 0; i < x.length; i++) { const v = +x[i]; if (v < lo) lo = v; if (v > hi) hi = v; if (v !== v) { lo = NaN; hi = NaN; break; } }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new Error(`autodetected range of [${lo}, ${hi}] is not finite`);
  }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  return [lo, hi];
}

function sorted_copy(x: ArrayLike<number>): number[] {
  const out = Array.from(x as any).map(v => +(v as any));
  return out.sort(cmp_nan_at_the_end);
}
function lane_quantile_sorted(x: number[], q: number) { return quantile_sorted(x, x.length, q, 'linear'); }
function lane_std(x: ArrayLike<number>) {
  const n = x.length; let s = 0; for (let i = 0; i < n; i++) s += x[i];
  const m = s / n; let ss = 0; for (let i = 0; i < n; i++) ss += (x[i] - m) * (x[i] - m);
  return Math.sqrt(ss / n);
}

/** Optimal bin width for each estimator (0 if undefined) */
function bin_width(method: string, x: number[], ptp: number): number {
  const n = x.length;
  if (n == 0) return 0;
  const sturges = () => ptp / (Math.log2(n) + 1);
  const fd = () => {
    const s = sorted_copy(x);
    const iqr = lane_quantile_sorted(s, 0.75) - lane_quantile_sorted(s, 0.25);
    return 2 * iqr * Math.pow(n, -1 / 3);
  };
  switch (method) {
    case 'sqrt': return ptp / Math.sqrt(n);
    case 'sturges': return sturges();
    case 'rice': return ptp / (2 * Math.pow(n, 1 / 3));
    case 'scott': return Math.pow(24 * Math.PI ** 0.5 / n, 1 / 3) * lane_std(x);
    case 'fd': return fd();
    case 'auto': { const f = fd(); return f > 0 ? Math.min(f, sturges()) : sturges(); }
    case 'doane': {
      if (n <= 2) return 0;
      const sd = Math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)));
      const m = x.reduce((s, v) => s + v, 0) / n;
      const sigma = lane_std(x);
      if (sigma <= 0) return 0;
      const g1 = Math.abs(x.reduce((s, v) => s + Math.pow((v - m) / sigma, 3), 0) / n);
      return ptp / (1 + Math.log2(n) + Math.log2(1 + g1 / sd));
    }
    case 'stone': {
      const nbins_upper = Math.round(Math.sqrt(n));
      const s = sorted_copy(x);
      const first = s[0], last = s[n - 1];
      let best = Infinity, bestW = 0;
      for (let nb = 1; nb <= nbins_upper; nb++) {
        const hh = ptp / nb;
        const counts = new Array(nb).fill(0);
        for (const v of s) { let i = Math.floor((v - first) / hh); if (i >= nb) i = nb - 1; counts[i]++; }
        const pn = counts.map(c => c / n);
        const sumsq = pn.reduce((s2, p) => s2 + p * p, 0);
        const score = (2 - (n + 1) * sumsq) / hh;
        if (score < best) { best = score; bestW = hh; }
      }
      return bestW;
    }
  }
  throw new Error(`'${method}' is not a valid estimator for \`bins\`. Options: ${BIN_METHODS.join(', ')}`);
}

/** Edges of the bins used by the histogram function */
export function histogram_bin_edges(a: NDArray, bins: number | string | NDArray | number[], range: [number, number] | null, weights: NDArray | null = null): NDArray {
  a = asarray(a);
  let x = Array.from(a.flat as any).map(v => +(v as any));
  if (typeof bins === "string") {
    if (!BIN_METHODS.includes(bins)) throw new Error(`'${bins}' is not a valid estimator for \`bins\`. Options: ${BIN_METHODS.join(', ')}`);
    if (weights !== null) throw new Error(`weights are not supported for automated bin estimators`);
    const [lo, hi] = finite_range(x, range);
    if (range) x = x.filter(v => v >= lo && v <= hi);
    let n_equal = 1;
    if (x.length > 0) {
      const w = bin_width(bins, x, hi - lo);
      n_equal = w > 0 ? Math.max(1, Math.ceil((hi - lo) / w)) : 1;
    }
    return linspace_edges(lo, hi, n_equal);
  }
  if (typeof bins === "number") {
    if (!Number.isInteger(bins) || bins < 1) throw new Error(`\`bins\` must be a positive integer, when an integer`);
    const [lo, hi] = finite_range(x, range);
    return linspace_edges(lo, hi, bins);
  }
  const given = asarray(bins);
  const edges = Array.from(given.flat as any).map(v => +(v as any));
  if (given.shape.length != 1) throw new Error(`\`bins\` must be 1d, when an array`);
  for (let i = 1; i < edges.length; i++) if (edges[i] < edges[i - 1]) throw new Error(`\`bins\` must increase monotonically, when an array`);
  return new NDArray(new_buffer(edges, given.dtype), [edges.length], given.dtype);
}

function linspace_edges(lo: number, hi: number, n: number): NDArray {
  const edges = new_buffer(n + 1, float64);
  const step = (hi - lo) / n;
  for (let i = 0; i <= n; i++) edges[i] = lo + i * step;
  edges[n] = hi;
  return new NDArray(edges, [n + 1], float64);
}

/** Index of the bin of x in sorted `edges` (last bin closed), -1 if out of range */
function bin_index(edges: ArrayLike<number>, x: number): number {
  const n = edges.length;
  if (!(x >= edges[0] && x <= edges[n - 1])) return -1;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (edges[mid] <= x) lo = mid; else hi = mid; }
  return lo;
}

export function histogram(a: NDArray, bins: number | string | NDArray | number[], range: [number, number] | null, density: boolean | null, weights: NDArray | null): NDArray[] {
  a = asarray(a);
  if (weights !== null) {
    weights = asarray(weights);
    if (weights.size !== a.size) throw new Error(`weights should have the same shape as a.`);
  }
  const edges = histogram_bin_edges(a, bins, range, weights);
  const nb = edges.size - 1;
  const e = edges.flat as any;
  const x = a.flat as any;
  const w = weights ? weights.flat as any : null;
  const counts = new Array(nb).fill(0);
  for (let i = 0; i < x.length; i++) {
    const k = bin_index(e, +x[i]);
    if (k >= 0) counts[Math.min(k, nb - 1)] += w ? +w[i] : 1;
  }
  let dtype: DType = w ? float_out([weights!.dtype]) : int32;
  let values = counts;
  if (density) {
    const total = counts.reduce((s, v) => s + v, 0);
    values = counts.map((c, i) => c / total / (e[i + 1] - e[i]));
    dtype = float64;
  }
  return [new NDArray(new_buffer(values, dtype), [nb], dtype), edges];
}

export function histogramdd(sample: NDArray, bins: any, range: any[] | null, density: boolean | null, weights: NDArray | null): [NDArray, NDArray[]] {
  sample = asarray(sample);
  if (sample.shape.length == 1) sample = reshape(sample, [sample.size, 1]);
  if (sample.shape.length != 2) throw new Error(`The dimension of bins must be equal to the dimension of the sample x.`);
  const [N, D] = sample.shape;
  const x = sample.flat as any;
  const edges: NDArray[] = [];
  for (let d = 0; d < D; d++) {
    const col = new Array(N);
    for (let i = 0; i < N; i++) col[i] = x[i * D + d];
    const b = Array.isArray(bins) && bins.length == D && !(bins.every(v => typeof v === 'number') && D == 1 && false) ? bins[d] : (isarray(bins) ? bins : bins);
    const colArr = new NDArray(new_buffer(col, float64), [N], float64);
    edges.push(histogram_bin_edges(colArr, typeof bins === 'number' ? bins : b, range ? range[d] : null));
  }
  const nbins = edges.map(e => e.size - 1);
  const H = new Array(prod(nbins)).fill(0);
  const w = weights ? asarray(weights).flat as any : null;
  const strides = nbins.map((_, d) => prod(nbins.slice(d + 1)));
  for (let i = 0; i < N; i++) {
    let flat = 0, ok = true;
    for (let d = 0; d < D; d++) {
      const e = edges[d].flat as any;
      const k = bin_index(e, +x[i * D + d]);
      if (k < 0) { ok = false; break; }
      flat += Math.min(k, nbins[d] - 1) * strides[d];
    }
    if (ok) H[flat] += w ? +w[i] : 1;
  }
  let dtype: DType = float64;
  let values = H;
  if (density) {
    const total = H.reduce((s, v) => s + v, 0);
    values = H.map((c, flat) => {
      let vol = 1;
      for (let d = 0; d < D; d++) { const k = Math.floor(flat / strides[d]) % nbins[d]; const e = edges[d].flat as any; vol *= e[k + 1] - e[k]; }
      return c / total / vol;
    });
    dtype = float64;
  }
  return [new NDArray(new_buffer(values, dtype), nbins, dtype), edges];
}

export function histogram2d(x: NDArray, y: NDArray, bins: any, range: any[] | null, density: boolean | null, weights: NDArray | null): NDArray[] {
  x = asarray(x); y = asarray(y);
  if (x.size !== y.size) throw new Error(`x and y must have the same length.`);
  let b: any = bins;
  if (!(typeof bins === 'number') && Array.isArray(bins) && bins.length == 2 && (Array.isArray(bins[0]) || isarray(bins[0]) || typeof bins[0] === 'number') && !(bins.every((v: any) => typeof v === 'number' && false))) b = bins;
  const sample = new NDArray(new_buffer([...Array.from(x.flat as any).flatMap((v, i) => [v, (y.flat as any)[i]])], float64), [x.size, 2], float64);
  const perDim = typeof b === 'number' ? [b, b] : b;
  const [H, edges] = histogramdd(sample, perDim, range, density, weights);
  return [H, edges[0], edges[1]];
}

export function bincount(x: NDArray, weights: NDArray | null, minlength: number): NDArray {
  x = asarray(x);
  if (x.shape.length != 1) throw new Error(`object too deep for desired array`);
  if (!dtype_is_integer(x.dtype) && x.dtype !== bool && x.size > 0) {
    throw new Error(`Cannot cast array data from dtype('${x.dtype}') to dtype('int64') according to the rule 'safe'. Convert with astype or np.array(x, dtype='int32') first.`);
  }
  if (minlength < 0) throw new Error(`'minlength' must not be negative`);
  const f = x.flat as any;
  let max = -1;
  for (let i = 0; i < f.length; i++) { if (f[i] < 0) throw new Error(`'list' argument must have no negative elements`); if (f[i] > max) max = f[i]; }
  const n = Math.max(max + 1, minlength);
  const w = weights ? asarray(weights) : null;
  if (w && w.size != x.size) throw new Error(`The weights and list don't have the same length.`);
  const dtype: DType = w ? float64 : int32;
  const buffer = new_buffer(n, dtype);
  for (let i = 0; i < f.length; i++) buffer[f[i]] += w ? +(w.flat as any)[i] : 1;
  return new NDArray(buffer, [n], dtype);
}

export function searchsorted(a: NDArray, v: NDArray, side: 'left' | 'right', sorter: NDArray | null): NDArray {
  a = asarray(a); v = asarray(v);
  if (a.shape.length != 1) throw new Error(`object too deep for desired array`);
  if (side !== 'left' && side !== 'right') throw new Error(`side must be 'left' or 'right' (got '${side}')`);
  const af = a.flat as any;
  const sf = sorter ? asarray(sorter).flat as any : null;
  const at = (i: number) => sf ? af[sf[i]] : af[i];
  const n = a.size;
  const vf = v.flat as any;
  const buffer = new_buffer(vf.length, int32);
  const less = side == 'left'
    ? (x: number, y: number) => cmp_nan_at_the_end(x, y) < 0   // x < y
    : (x: number, y: number) => cmp_nan_at_the_end(x, y) <= 0; // x <= y
  for (let k = 0; k < vf.length; k++) {
    let lo = 0, hi = n;
    const val = +vf[k];
    while (lo < hi) { const mid = (lo + hi) >> 1; if (less(+at(mid), val)) lo = mid + 1; else hi = mid; }
    buffer[k] = lo;
  }
  return new NDArray(buffer, v.shape, int32);
}

export function digitize(x: NDArray, bins: NDArray, right: boolean): NDArray {
  x = asarray(x); bins = asarray(bins);
  if (bins.shape.length != 1) throw new Error(`object too deep for desired array`);
  const b = Array.from(bins.flat as any).map(v => +(v as any));
  const n = b.length;
  let increasing = true, decreasing = true;
  for (let i = 1; i < n; i++) { if (b[i] < b[i - 1]) increasing = false; if (b[i] > b[i - 1]) decreasing = false; }
  if (!increasing && !decreasing) throw new Error(`bins must be monotonically increasing or decreasing`);
  const xf = x.flat as any;
  const buffer = new_buffer(xf.length, int32);
  for (let k = 0; k < xf.length; k++) {
    const v = +xf[k];
    let lo = 0, hi = n;
    if (increasing) {
      // right=False: i such that bins[i-1] <= x < bins[i]  -> searchsorted(bins, x, side='right')
      while (lo < hi) { const mid = (lo + hi) >> 1; if (right ? b[mid] < v : b[mid] <= v) lo = mid + 1; else hi = mid; }
      buffer[k] = lo;
    } else {
      // reversed bins searched, result inverted (as numpy)
      while (lo < hi) { const mid = (lo + hi) >> 1; if (right ? b[n - 1 - mid] < v : b[n - 1 - mid] <= v) lo = mid + 1; else hi = mid; }
      buffer[k] = n - lo;
    }
  }
  return new NDArray(buffer, x.shape, int32);
}
