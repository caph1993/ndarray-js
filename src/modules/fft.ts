//@ts-check
/** numpy.fft: discrete Fourier transforms (radix-2 FFT, Bluestein's algorithm for other lengths) */
import { NDArray, asarray } from "../NDArray";
import { DType, complex128, float64, new_buffer, convert_buffer } from "../dtypes";
import { Complex } from "../complex";
import { apply_along_axis, normalize_axis } from "../array/transform";
import { reshape } from "../array/shape_operations";
import { reindex } from "./manipulation";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

/** In-place iterative radix-2 FFT (n must be a power of two). sign = -1 forward, +1 inverse (unscaled) */
function fft_pow2(re: Float64Array, im: Float64Array, sign: number) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = sign * 2 * Math.PI / len;
    const half = len >> 1;
    // twiddles computed directly for accuracy
    const wr = new Float64Array(half), wi = new Float64Array(half);
    for (let k = 0; k < half; k++) { wr[k] = Math.cos(ang * k); wi[k] = Math.sin(ang * k); }
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = re[b] * wr[k] - im[b] * wi[k], ti = re[b] * wi[k] + im[b] * wr[k];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

/** Unscaled DFT of any length: X_k = sum x_j exp(sign 2 pi i jk / n) */
function dft(re: Float64Array, im: Float64Array, sign: number): [Float64Array, Float64Array] {
  const n = re.length;
  if (n <= 1) return [re.slice(), im.slice()];
  if ((n & (n - 1)) === 0) { const r = re.slice(), i = im.slice(); fft_pow2(r, i, sign); return [r, i]; }
  // Bluestein: chirp w_k = exp(sign i pi k^2 / n)
  let m = 1; while (m < 2 * n - 1) m <<= 1;
  const cr = new Float64Array(n), ci = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const ang = sign * Math.PI * ((k * k) % (2 * n)) / n;
    cr[k] = Math.cos(ang); ci[k] = Math.sin(ang);
  }
  const ar = new Float64Array(m), ai = new Float64Array(m), br = new Float64Array(m), bi = new Float64Array(m);
  for (let k = 0; k < n; k++) { ar[k] = re[k] * cr[k] - im[k] * ci[k]; ai[k] = re[k] * ci[k] + im[k] * cr[k]; }
  br[0] = cr[0]; bi[0] = -ci[0];
  for (let k = 1; k < n; k++) { br[k] = br[m - k] = cr[k]; bi[k] = bi[m - k] = -ci[k]; }
  fft_pow2(ar, ai, -1); fft_pow2(br, bi, -1);
  for (let k = 0; k < m; k++) { const r = ar[k] * br[k] - ai[k] * bi[k], i = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = r; ai[k] = i; }
  fft_pow2(ar, ai, +1);
  const outR = new Float64Array(n), outI = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const xr = ar[k] / m, xi = ai[k] / m;
    outR[k] = xr * cr[k] - xi * ci[k]; outI[k] = xr * ci[k] + xi * cr[k];
  }
  return [outR, outI];
}

type Norm = 'backward' | 'ortho' | 'forward' | null;
function check_norm(norm: Norm) {
  if (norm !== null && norm !== 'backward' && norm !== 'ortho' && norm !== 'forward') throw new Error(`Invalid norm value ${norm}; should be "backward", "ortho" or "forward".`);
}
function scale_for(norm: Norm, n: number, inverse: boolean): number {
  const mode = norm || 'backward';
  if (mode === 'ortho') return 1 / Math.sqrt(n);
  if (mode === 'backward') return inverse ? 1 / n : 1;
  return inverse ? 1 : 1 / n;
}

function lane_parts(lane: any, n: number): [Float64Array, Float64Array] {
  const re = new Float64Array(n), im = new Float64Array(n);
  const m = Math.min(n, lane.length);
  for (let i = 0; i < m; i++) { const v = lane[i]; if (v instanceof Complex) { re[i] = v.re; im[i] = v.im; } else re[i] = +v; }
  return [re, im];
}

function check_n(n: number | null, len: number): number {
  const m = n === null ? len : n;
  if (!Number.isInteger(m) || m < 1) throw new Error(`Invalid number of FFT data points (${m}) specified.`);
  return m;
}

/** Transform along one axis of complex data. Output lanes have `outLen(n)` values */
function along(a: NDArray, axis: number, n: number | null, fn: (lane: any, n: number) => Complex[] | number[], dtype: DType): NDArray {
  a = asarray(a);
  if (a.shape.length == 0) throw new Error(`axis ${axis} is out of bounds for array of dimension 0`);
  const ax = normalize_axis(axis, a.shape.length);
  const len = check_n(n, a.shape[ax]);
  return apply_along_axis(() => dtype, (lane) => fn(lane, len), a, ax);
}

function c_fft(a: NDArray, n: number | null, axis: number, norm: Norm, inverse: boolean): NDArray {
  check_norm(norm);
  return along(a, axis, n, (lane, len) => {
    const [re, im] = lane_parts(lane, len);
    const [r, i] = dft(re, im, inverse ? 1 : -1);
    const s = scale_for(norm, len, inverse);
    return range(len).map(k => new Complex(r[k] * s, i[k] * s));
  }, complex128);
}

export const fft = (a: NDArray, n: number | null, axis: number, norm: Norm) => c_fft(a, n, axis, norm, false);
export const ifft = (a: NDArray, n: number | null, axis: number, norm: Norm) => c_fft(a, n, axis, norm, true);

export function rfft(a: NDArray, n: number | null, axis: number, norm: Norm): NDArray {
  a = asarray(a);
  if (a.dtype === complex128) throw new Error(`rfft requires real input`);
  check_norm(norm);
  return along(a, axis, n, (lane, len) => {
    const [re, im] = lane_parts(lane, len);
    const [r, i] = dft(re, im, -1);
    const s = scale_for(norm, len, false);
    return range(Math.floor(len / 2) + 1).map(k => new Complex(r[k] * s, i[k] * s));
  }, complex128);
}

/** Real signal from a half spectrum (Hermitian symmetry). `n` is the output length */
export function irfft(a: NDArray, n: number | null, axis: number, norm: Norm): NDArray {
  check_norm(norm);
  a = asarray(a);
  const ax = normalize_axis(axis, a.shape.length);
  const outLen = check_n(n === null ? 2 * (a.shape[ax] - 1) : n, 0);
  if (outLen < 1) throw new Error(`Invalid number of FFT data points (${outLen}) specified.`);
  const half = Math.floor(outLen / 2) + 1;
  return apply_along_axis(() => float64, (lane) => {
    const re = new Float64Array(outLen), im = new Float64Array(outLen);
    for (let k = 0; k < half && k < lane.length; k++) {
      const v = lane[k] instanceof Complex ? lane[k] : new Complex(+lane[k], 0);
      re[k] = v.re; im[k] = v.im;
    }
    // Imaginary parts of the zero and Nyquist frequencies are ignored
    im[0] = 0; if (outLen % 2 == 0) im[outLen / 2] = 0;
    for (let k = 1; k < outLen - half + 1; k++) { re[outLen - k] = re[k]; im[outLen - k] = -im[k]; }
    const [r] = dft(re, im, +1);
    const s = scale_for(norm, outLen, true);
    return Array.from(r, v => v * s);
  }, a, ax);
}

export function hfft(a: NDArray, n: number | null, axis: number, norm: Norm): NDArray {
  // forward transform of a signal with Hermitian symmetry: irfft(conj(a)) with swapped normalisation
  check_norm(norm);
  a = asarray(a);
  const conj = new NDArray(Array.from(convert_buffer(a.flat as any, complex128), (z: Complex) => z.conj()) as any, a.shape, complex128);
  const swapped: Norm = norm === 'ortho' ? 'ortho' : (norm === 'forward' ? 'backward' : 'forward');
  return irfft(conj, n, axis, swapped);
}

export function ihfft(a: NDArray, n: number | null, axis: number, norm: Norm): NDArray {
  check_norm(norm);
  const swapped: Norm = norm === 'ortho' ? 'ortho' : (norm === 'forward' ? 'backward' : 'forward');
  const r = rfft(a, n, axis, swapped);
  return new NDArray(Array.from(r.flat as any, (z: any) => z.conj()) as any, r.shape, complex128);
}

// ==============================
//   Multi-dimensional transforms
// ==============================

function axes_and_shape(a: NDArray, s: number[] | null, axes: number[] | null, defaultAxes: number[] | null): { axes: number[], s: (number | null)[] } {
  const nd = a.shape.length;
  let ax: number[];
  if (axes !== null) ax = axes.map(x => normalize_axis(x, nd));
  else if (s !== null) ax = range(s.length).map(i => nd - s.length + i);
  else ax = defaultAxes ? defaultAxes.map(x => normalize_axis(x, nd)) : range(nd);
  if (s !== null && s.length != ax.length) throw new Error(`Shape and axes have different lengths.`);
  return { axes: ax, s: s === null ? ax.map(() => null) : s };
}

function fftn_(a: NDArray, s: number[] | null, axes: number[] | null, norm: Norm, inverse: boolean, defaultAxes: number[] | null = null): NDArray {
  a = asarray(a);
  const { axes: ax, s: sizes } = axes_and_shape(a, s, axes, defaultAxes);
  let out = a;
  for (let i = ax.length - 1; i >= 0; i--) out = c_fft(out, sizes[i], ax[i], norm, inverse);
  if (ax.length == 0) out = c_fft(out, null, -1, norm, inverse);
  return out;
}
export const fftn = (a: NDArray, s: number[] | null, axes: number[] | null, norm: Norm) => fftn_(a, s, axes, norm, false);
export const ifftn = (a: NDArray, s: number[] | null, axes: number[] | null, norm: Norm) => fftn_(a, s, axes, norm, true);
export const fft2 = (a: NDArray, s: number[] | null, axes: number[], norm: Norm) => fftn_(a, s, axes, norm, false, [-2, -1]);
export const ifft2 = (a: NDArray, s: number[] | null, axes: number[], norm: Norm) => fftn_(a, s, axes, norm, true, [-2, -1]);

export function rfftn(a: NDArray, s: number[] | null, axes: number[] | null, norm: Norm): NDArray {
  a = asarray(a);
  const { axes: ax, s: sizes } = axes_and_shape(a, s, axes, null);
  const last = ax.length - 1;
  let out = rfft(a, sizes[last], ax[last], norm);
  for (let i = last - 1; i >= 0; i--) out = c_fft(out, sizes[i], ax[i], norm, false);
  return out;
}
export const rfft2 = (a: NDArray, s: number[] | null, axes: number[], norm: Norm) => rfftn(a, s, axes, norm);

export function irfftn(a: NDArray, s: number[] | null, axes: number[] | null, norm: Norm): NDArray {
  a = asarray(a);
  const { axes: ax, s: sizes } = axes_and_shape(a, s, axes, null);
  const last = ax.length - 1;
  let out = a;
  for (let i = 0; i < last; i++) out = c_fft(out, sizes[i], ax[i], norm, true);
  return irfft(out, sizes[last], ax[last], norm);
}
export const irfft2 = (a: NDArray, s: number[] | null, axes: number[], norm: Norm) => irfftn(a, s, axes, norm);

// ==============================
//   Helpers
// ==============================

export function fftfreq(n: number, d: number): NDArray {
  if (!Number.isInteger(n) || n < 1) throw new Error(`n should be an integer greater than 0`);
  const buffer = new_buffer(n, float64);
  const val = 1 / (n * d), half = Math.floor((n - 1) / 2) + 1;
  for (let i = 0; i < n; i++) buffer[i] = (i < half ? i : i - n) * val;
  return new NDArray(buffer, [n], float64);
}
export function rfftfreq(n: number, d: number): NDArray {
  if (!Number.isInteger(n) || n < 1) throw new Error(`n should be an integer greater than 0`);
  const m = Math.floor(n / 2) + 1;
  const buffer = new_buffer(m, float64);
  for (let i = 0; i < m; i++) buffer[i] = i / (n * d);
  return new NDArray(buffer, [m], float64);
}

function shift(x: NDArray, axes: number | number[] | null, inverse: boolean): NDArray {
  x = asarray(x);
  const nd = x.shape.length;
  const ax = axes === null ? range(nd) : (Array.isArray(axes) ? axes : [axes]).map(a => normalize_axis(a, nd));
  const maps = x.shape.map((n, d) => {
    if (!ax.includes(d)) return null;
    const k = inverse ? -Math.floor(n / 2) : Math.floor(n / 2);
    return range(n).map(i => (((i - k) % n) + n) % n);
  });
  return reindex(x, maps);
}
export const fftshift = (x: NDArray, axes: number | number[] | null) => shift(x, axes, false);
export const ifftshift = (x: NDArray, axes: number | number[] | null) => shift(x, axes, true);
