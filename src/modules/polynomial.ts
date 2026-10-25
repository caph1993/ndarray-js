import { is_cx } from '../dtypes';
//@ts-check
/**
 * numpy.polynomial: power series and the orthogonal-polynomial bases (Chebyshev, Hermite, HermiteE, Laguerre, Legendre).
 * Every basis is described by its three-term recurrence  x p_n = a_n p_{n+1} + b_n p_n + c_n p_{n-1}  and
 * all the operations (mulx, mul, div, val, vander, companion, fit, ...) are derived from it.
 * Coefficients are 1-D real arrays (lowest degree first). Evaluation points may be complex.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { DType, float64, complex128, new_buffer, convert_buffer } from "../dtypes";
import { Complex, ZERO, ONE } from "../complex";
import * as la from "./linalg";
import { reshape } from "../array/shape_operations";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const EPS = 2.220446049250313e-16;

export class PolyError extends Error {
  constructor(message: string) { super(message); this.name = "PolyError"; }
}

// ==============================
//   polyutils
// ==============================

const as_numbers = (c: any): number[] => {
  const a = asarray(c);
  if (is_cx(a.dtype)) throw new PolyError("complex coefficients are not supported");
  return Array.from(a.flat as any, (v: any) => +v);
};
export const arr = (v: number[], dtype: DType = float64) => new NDArray(new_buffer(v, dtype), [v.length], dtype);

/** Removes trailing zeros (always keeps at least one coefficient) */
export function trimseq(seq: number[]): number[] {
  let n = seq.length;
  while (n > 1 && seq[n - 1] === 0) n--;
  return seq.slice(0, n);
}

export function trimcoef(c: number[], tol: number = 0): number[] {
  if (tol < 0) throw new PolyError("tol must be non-negative");
  const keep = c.map((v, i) => Math.abs(v) > tol ? i : -1).filter(i => i >= 0);
  if (!keep.length) return [0];
  return c.slice(0, keep[keep.length - 1] + 1);
}

/** Returns a list of 1-D float arrays of the same dtype (trailing zeros removed if trim) */
export function as_series(alist: any[], trim: boolean = true): number[][] {
  const arrays = alist.map(a => {
    const x = asarray(a);
    if (x.shape.length > 1) throw new PolyError("Coefficient array is not 1-d");
    if (x.size === 0) throw new PolyError("Coefficient array is empty");
    if (is_cx(x.dtype)) throw new PolyError("complex coefficients are not supported");
    return Array.from(reshape(x, [x.size]).flat as any, (v: any) => +v) as number[];
  });
  return trim ? arrays.map(trimseq) : arrays;
}

export function getdomain(x: any): number[] {
  const v = as_numbers(x);
  return [Math.min(...v), Math.max(...v)];
}
export function mapparms(old: number[], nw: number[]): [number, number] {
  const oldlen = old[1] - old[0], newlen = nw[1] - nw[0];
  const off = (old[1] * nw[0] - old[0] * nw[1]) / oldlen;
  const scl = newlen / oldlen;
  return [off, scl];
}
export function mapdomain(x: any, old: number[], nw: number[]): NDArray {
  const [off, scl] = mapparms(old, nw);
  const a = asarray(x);
  return la_map(a, (v) => off + scl * v);
}
function la_map(a: NDArray, f: (v: number) => number): NDArray {
  return new NDArray(new_buffer(Array.from(a.flat as any, (v: any) => f(+v)), float64), a.shape, float64);
}

// ==============================
//   Basis specifications
// ==============================

export interface Basis {
  name: string; prefix: string; domain: number[];
  a: (n: number) => number; b: (n: number) => number; c: (n: number) => number;
  weight: (x: number) => number;
}

export const POWER: Basis = { name: 'Polynomial', prefix: 'poly', domain: [-1, 1], a: () => 1, b: () => 0, c: () => 0, weight: () => 1 };
export const CHEBYSHEV: Basis = { name: 'Chebyshev', prefix: 'cheb', domain: [-1, 1], a: n => n == 0 ? 1 : 0.5, b: () => 0, c: n => n == 0 ? 0 : 0.5, weight: x => 1 / (Math.sqrt(1 + x) * Math.sqrt(1 - x)) };
export const LEGENDRE: Basis = { name: 'Legendre', prefix: 'leg', domain: [-1, 1], a: n => (n + 1) / (2 * n + 1), b: () => 0, c: n => n / (2 * n + 1), weight: () => 1 };
export const HERMITE: Basis = { name: 'Hermite', prefix: 'herm', domain: [-1, 1], a: () => 0.5, b: () => 0, c: n => n, weight: x => Math.exp(-x * x) };
export const HERMITE_E: Basis = { name: 'HermiteE', prefix: 'herme', domain: [-1, 1], a: () => 1, b: () => 0, c: n => n, weight: x => Math.exp(-0.5 * x * x) };
export const LAGUERRE: Basis = { name: 'Laguerre', prefix: 'lag', domain: [0, 1], a: n => -(n + 1), b: n => 2 * n + 1, c: n => -n, weight: x => Math.exp(-x) };

/** Leading monomial coefficient of the n-th basis polynomial */
function leading(B: Basis, n: number): number {
  let k = 1;
  for (let i = 0; i < n; i++) k /= B.a(i);
  return k;
}

// ==============================
//   Series algebra (1-D real coefficients)
// ==============================

const trimmed = (c: number[]) => trimseq(c.slice());

export function add(B: Basis, c1: number[], c2: number[]): number[] {
  const n = Math.max(c1.length, c2.length);
  return trimseq(range(n).map(i => (c1[i] || 0) + (c2[i] || 0)));
}
export function sub(B: Basis, c1: number[], c2: number[]): number[] {
  const n = Math.max(c1.length, c2.length);
  return trimseq(range(n).map(i => (c1[i] || 0) - (c2[i] || 0)));
}

/** Multiply by x */
export function mulx(B: Basis, c: number[]): number[] {
  if (c.length === 1 && c[0] === 0) return c.slice();
  const out = new Array(c.length + 1).fill(0);
  c.forEach((v, n) => {
    out[n + 1] += B.a(n) * v;
    out[n] += B.b(n) * v;
    if (n >= 1) out[n - 1] += B.c(n) * v;
  });
  return trimseq(out);
}

export function mul(B: Basis, c1: number[], c2: number[]): number[] {
  if (B === POWER) {
    const out = new Array(c1.length + c2.length - 1).fill(0);
    for (let i = 0; i < c1.length; i++) for (let j = 0; j < c2.length; j++) out[i + j] += c1[i] * c2[j];
    return trimseq(out);
  }
  // sum_k c2_k * (p_k * c1), with p_{k+1} * c1 = (x p_k c1 - b_k p_k c1 - c_k p_{k-1} c1) / a_k
  let prev: number[] = [0], cur = c1.slice();
  let result: number[] = [0];
  for (let k = 0; k < c2.length; k++) {
    if (c2[k] !== 0) result = add(B, result, cur.map(v => v * c2[k]));
    if (k + 1 < c2.length) {
      const x = mulx(B, cur);
      let next = sub(B, x, cur.map(v => v * B.b(k)));
      if (k >= 1) next = sub(B, next, prev.map(v => v * B.c(k)));
      prev = cur; cur = next.map(v => v / B.a(k));
    }
  }
  return trimseq(result);
}

export function pow(B: Basis, c: number[], p: number, maxpower: number = 16): number[] {
  if (!Number.isInteger(p) || p < 0) throw new PolyError("Power must be a non-negative integer.");
  if (maxpower !== null && p > maxpower) throw new PolyError("Power is too large");
  let res: number[] = [1];
  for (let i = 0; i < p; i++) res = mul(B, res, c);
  return res;
}

export function div(B: Basis, c1: number[], c2: number[]): [number[], number[]] {
  if (c2[c2.length - 1] === 0) throw new PolyError("Division by zero");
  const n1 = c1.length, n2 = c2.length;
  if (n1 < n2) return [[0], c1.slice()];
  if (n2 === 1) return [c1.map(v => v / c2[0]), [0]];
  let rem = c1.slice();
  const quo = new Array(n1 - n2 + 1).fill(0);
  const lead2 = c2[n2 - 1], k2 = leading(B, n2 - 1);
  for (let d = n1 - n2; d >= 0; d--) {
    // leading coefficient of p_d * c2 is c2_top * k_d * k_{n2-1} / k_{d+n2-1}
    const ratio = leading(B, d) * k2 / leading(B, d + n2 - 1);
    const q = rem[d + n2 - 1] / (lead2 * ratio);
    quo[d] = q;
    if (q !== 0) {
      const unit = new Array(d + 1).fill(0); unit[d] = 1;
      const prod = mul(B, unit, c2);
      for (let i = 0; i < prod.length; i++) rem[i] -= q * prod[i];
      rem[d + n2 - 1] = 0;
    }
  }
  return [trimseq(quo), trimseq(rem.slice(0, n2 - 1).length ? rem.slice(0, n2 - 1) : [0])];
}

/** Evaluates sum c_k p_k(x) with Clenshaw's recurrence */
function clenshaw(B: Basis, c: number[], x: number): number {
  const n = c.length;
  if (n === 0) return 0;
  // b_k = c_k + (x - beta_k)/alpha_k b_{k+1} - gamma_{k+1}/alpha_{k+1} b_{k+2}
  let b1 = 0, b2 = 0;
  for (let k = n - 1; k >= 1; k--) {
    const b0 = c[k] + (x - B.b(k)) / B.a(k) * b1 - (k + 1 < n ? B.c(k + 1) / B.a(k + 1) * b2 : 0);
    b2 = b1; b1 = b0;
  }
  // p_0 = 1, p_1 = (x - b_0)/a_0
  return c[0] + (n > 1 ? (x - B.b(0)) / B.a(0) * b1 - (n > 2 ? B.c(1) / B.a(1) * b2 : 0) : 0);
}
function clenshawC(B: Basis, c: number[], x: Complex): Complex {
  const n = c.length;
  let b1: Complex = ZERO, b2: Complex = ZERO;
  const real = (v: number) => new Complex(v, 0);
  for (let k = n - 1; k >= 1; k--) {
    let b0 = real(c[k]).add(x.sub(real(B.b(k))).mul(b1).mul(real(1 / B.a(k))));
    if (k + 1 < n) b0 = b0.sub(b2.mul(real(B.c(k + 1) / B.a(k + 1))));
    b2 = b1; b1 = b0;
  }
  if (n === 0) return ZERO;
  let r = real(c[0]);
  if (n > 1) r = r.add(x.sub(real(B.b(0))).mul(b1).mul(real(1 / B.a(0))));
  if (n > 2) r = r.sub(b2.mul(real(B.c(1) / B.a(1))));
  return r;
}

/** x: numbers or complex numbers. Returns an array with the shape of x (tensor=true: shape c.shape[1:] + x.shape for 1-D c) */
export function val(B: Basis, x: any, c: number[]): NDArray {
  const X = asarray(x);
  if (is_cx(X.dtype)) return new NDArray((X.flat as any).map((z: Complex) => clenshawC(B, c, z)), X.shape, complex128);
  return new NDArray(new_buffer(Array.from(X.flat as any, (v: any) => clenshaw(B, c, +v)), float64), X.shape, float64);
}

/** Vandermonde-like matrix: V[..., k] = p_k(x), k = 0..deg */
export function vander(B: Basis, x: any, deg: number): NDArray {
  if (!Number.isInteger(deg) || deg < 0) throw new PolyError("deg must be non-negative integer");
  const X = asarray(x);
  const xs = Array.from(X.flat as any, (v: any) => +v) as number[];
  const m = deg + 1, n = xs.length;
  const buffer = new_buffer(n * m, float64);
  xs.forEach((xv, i) => {
    buffer[i * m] = 1;
    if (deg >= 1) buffer[i * m + 1] = (xv - B.b(0)) / B.a(0);
    for (let k = 1; k < deg; k++) buffer[i * m + k + 1] = ((xv - B.b(k)) * buffer[i * m + k] - B.c(k) * buffer[i * m + k - 1]) / B.a(k);
  });
  return new NDArray(buffer, [...X.shape, m], float64);
}

export function line(B: Basis, off: number, scl: number): number[] {
  if (scl === 0) return [off];
  const x = mulx(B, [1]);
  return trimseq([off + scl * (x[0] || 0), scl * (x[1] || 0)]);
}

export function fromroots(B: Basis, roots: number[]): number[] {
  if (roots.length === 0) return [1];
  const sorted = roots.slice().sort((p, q) => p - q);
  let p: number[][] = sorted.map(r => line(B, -r, 1));
  while (p.length > 1) {
    const next: number[][] = [];
    for (let i = 0; i + 1 < p.length; i += 2) next.push(mul(B, p[i], p[i + 1]));
    if (p.length % 2) next.push(p[p.length - 1]);
    p = next;
  }
  return p[0];
}

/** Derivative in the basis via the exact power-basis conversion */
export function toPower(B: Basis, c: number[]): number[] {
  if (B === POWER) return c.slice();
  // p_0 = 1, p_{k+1} = ((x - b_k) p_k - c_k p_{k-1}) / a_k  in the power basis
  const n = c.length;
  const out = new Array(n).fill(0);
  let prev: number[] = [], cur: number[] = [1];
  for (let k = 0; k < n; k++) {
    cur.forEach((v, i) => { out[i] += c[k] * v; });
    if (k + 1 < n) {
      const next = new Array(cur.length + 1).fill(0);
      cur.forEach((v, i) => { next[i + 1] += v; next[i] -= B.b(k) * v; });
      prev.forEach((v, i) => { next[i] -= B.c(k) * v; });
      prev = cur; cur = next.map(v => v / B.a(k));
    }
  }
  return trimseq(out);
}
export function fromPower(B: Basis, p: number[]): number[] {
  if (B === POWER) return p.slice();
  // Horner: p(x) = (...(p_n x + p_{n-1}) x + ...) with x multiplication in the basis
  let res: number[] = [0];
  for (let i = p.length - 1; i >= 0; i--) {
    res = mulx(B, res);
    res = add(B, res, [p[i]]);
  }
  return trimseq(res);
}

export function der(B: Basis, c: number[], m: number = 1, scl: number = 1): number[] {
  if (!Number.isInteger(m) || m < 0) throw new PolyError("The order of derivation must be integer and non-negative");
  let cur = c.slice();
  for (let i = 0; i < m; i++) {
    if (cur.length <= 1) { cur = [0]; break; }
    let pw = toPower(B, cur);
    pw = range(pw.length - 1).map(j => pw[j + 1] * (j + 1) * scl);
    cur = fromPower(B, pw.length ? pw : [0]);
    if (!cur.length) cur = [0];
  }
  return m === 0 ? c.slice() : cur;
}

function polyval_scalar(pw: number[], x: number) { let r = 0; for (let i = pw.length - 1; i >= 0; i--) r = r * x + pw[i]; return r; }

export function integ(B: Basis, c: number[], m: number = 1, k: number[] = [], lbnd: number = 0, scl: number = 1): number[] {
  if (!Number.isInteger(m) || m < 0) throw new PolyError("The order of integration must be integer and non-negative");
  if (k.length > m) throw new PolyError("Too many integration constants");
  if (m === 0) return c.slice();
  let cur = c.slice();
  for (let i = 0; i < m; i++) {
    const ki = k[i] || 0;
    if (cur.length === 1 && cur[0] === 0) { cur = [ki]; continue; }
    const pw = toPower(B, cur);
    const ip = [0, ...pw.map((v, j) => v * scl / (j + 1))];
    ip[0] = ki - polyval_scalar(ip, lbnd);
    cur = fromPower(B, ip);
  }
  return cur;
}

export function compan(B: Basis, c: number[]): NDArray {
  if (c.length < 2) throw new PolyError("Series must have maximum degree of at least 1.");
  const n = c.length - 1;
  if (c.length === 2) {
    // 1x1 companion
    const root = B.b(0) - B.a(0) * c[0] / c[1];
    return new NDArray(new_buffer([root], float64), [1, 1], float64);
  }
  const mat = range(n).map(() => new Array(n).fill(0));
  let last: number[];
  switch (B.prefix) {
    case 'poly': {
      for (let i = 1; i < n; i++) mat[i][i - 1] = 1;
      for (let i = 0; i < n; i++) mat[i][n - 1] = -c[i] / c[n];
      break;
    }
    case 'cheb': {
      const scl = [1, ...Array(n - 1).fill(Math.SQRT1_2)];
      for (let i = 0; i < n - 1; i++) { mat[i][i + 1] = i === 0 ? Math.SQRT1_2 : 0.5; mat[i + 1][i] = i === 0 ? Math.SQRT1_2 : 0.5; }
      for (let i = 0; i < n; i++) mat[i][n - 1] -= (c[i] / c[n]) * (scl[i] / scl[n - 1]) * 0.5;
      break;
    }
    case 'leg': {
      const scl = range(n).map(i => 1 / Math.sqrt(2 * i + 1));
      for (let i = 0; i < n - 1; i++) { const v = (i + 1) * scl[i] * scl[i + 1]; mat[i][i + 1] = v; mat[i + 1][i] = v; }
      for (let i = 0; i < n; i++) mat[i][n - 1] -= (c[i] / c[n]) * (scl[i] / scl[n - 1]) * (n / (2 * n - 1));
      break;
    }
    case 'herm': {
      let sc = [1, ...range(n - 1).map(i => 1 / Math.sqrt(2 * (n - 1 - i)))];
      for (let i = 1; i < sc.length; i++) sc[i] *= sc[i - 1];
      sc = sc.reverse();
      for (let i = 0; i < n - 1; i++) { const v = Math.sqrt(0.5 * (i + 1)); mat[i][i + 1] = v; mat[i + 1][i] = v; }
      for (let i = 0; i < n; i++) mat[i][n - 1] -= sc[i] * c[i] / (2 * c[n]);
      break;
    }
    case 'herme': {
      let sc = [1, ...range(n - 1).map(i => 1 / Math.sqrt(n - 1 - i))];
      for (let i = 1; i < sc.length; i++) sc[i] *= sc[i - 1];
      sc = sc.reverse();
      for (let i = 0; i < n - 1; i++) { const v = Math.sqrt(i + 1); mat[i][i + 1] = v; mat[i + 1][i] = v; }
      for (let i = 0; i < n; i++) mat[i][n - 1] -= sc[i] * c[i] / c[n];
      break;
    }
    case 'lag': {
      for (let i = 0; i < n - 1; i++) { mat[i][i + 1] = -(i + 1); mat[i + 1][i] = -(i + 1); }
      for (let i = 0; i < n; i++) mat[i][i] = 2 * i + 1;
      for (let i = 0; i < n; i++) mat[i][n - 1] += (c[i] / c[n]) * n;
      break;
    }
    default: throw new PolyError("unknown basis");
  }
  return new NDArray(new_buffer(mat.flat(), float64), [n, n], float64);
}

/** Roots (real or complex) of the series, sorted like numpy (by real part then imaginary part) */
export function roots(B: Basis, c: number[]): NDArray {
  c = trimseq(c);
  if (c.length < 2) return new NDArray(new_buffer(0, float64), [0], float64);
  if (c.length === 2) return arr([B.b(0) - B.a(0) * c[0] / c[1]]);
  const m = compan(B, c);
  const w = la.eigvals(m);
  const vals = Array.from(w.flat as any) as Complex[];
  vals.sort((p, q) => p.re - q.re || p.im - q.im);
  // numpy returns complex128 for these bases (eigvals of the companion matrix) and real roots for the others
  if (B.prefix !== 'cheb' && B.prefix !== 'leg' && B.prefix !== 'herme' && vals.every(z => z.im === 0)) return arr(vals.map(z => z.re));
  return new NDArray(vals, [vals.length], complex128);
}

// ==============================
//   Fitting
// ==============================

export function fit(B: Basis, x: any, y: any, deg: number | number[], rcond: number | null, full: boolean, w: any): NDArray | any[] {
  const X = Array.from(asarray(x).flat as any, (v: any) => +v) as number[];
  let Y = asarray(y);
  if (is_cx(Y.dtype)) throw new PolyError("complex data is not supported");
  const degs = Array.isArray(deg) ? deg.slice().sort((p, q) => p - q) : null;
  if (degs === null && (!Number.isInteger(deg) || (deg as number) < 0)) throw new PolyError("expected deg >= 0");
  if (X.length === 0) throw new PolyError("expected non-empty vector for x");
  if (asarray(x).shape.length !== 1) throw new PolyError("expected 1D vector for x");
  if (Y.shape.length < 1 || Y.shape.length > 2) throw new PolyError("expected 1D or 2D array for y");
  if (X.length !== Y.shape[0]) throw new PolyError("expected x and y to have same length");
  const lmax = degs ? degs[degs.length - 1] : (deg as number);
  const order = degs ? degs.length : lmax + 1;
  const V = vander(B, asarray(x), lmax);                                  // (n, lmax+1)
  const n = X.length;
  const vf = V.flat as any;
  const cols = degs ? degs : range(lmax + 1);
  let lhs: number[][] = range(n).map(i => cols.map(c => vf[i * (lmax + 1) + c]));  // (n, order)
  const k = Y.shape.length === 2 ? Y.shape[1] : 1;
  let rhs: number[][] = range(n).map(i => range(k).map(j => +(Y.flat as any)[i * k + j]));
  if (w !== null && w !== undefined) {
    const wv = Array.from(asarray(w).flat as any, (v: any) => +v) as number[];
    if (asarray(w).shape.length !== 1) throw new PolyError("expected 1D vector for w");
    if (wv.length !== n) throw new PolyError("expected x and w to have same length");
    lhs = lhs.map((r, i) => r.map(v => v * wv[i]));
    rhs = rhs.map((r, i) => r.map(v => v * wv[i]));
  }
  const rc = rcond === null ? n * EPS : rcond;
  const scl = range(order).map(j => { const s = Math.sqrt(lhs.reduce((acc, r) => acc + r[j] * r[j], 0)); return s === 0 ? 1 : s; });
  const A = new NDArray(new_buffer(lhs.flatMap(r => r.map((v, j) => v / scl[j])), float64), [n, order], float64);
  const Bm = new NDArray(new_buffer(rhs.flat(), float64), [n, k], float64);
  const [sol, resids, rank, s] = la.lstsq(A, Bm, rc);
  const sf = sol.flat as any;
  const full_c = range(lmax + 1).map(() => new Array(k).fill(0));
  cols.forEach((col, j) => { for (let t = 0; t < k; t++) full_c[col][t] = sf[j * k + t] / scl[j]; });
  const c = new NDArray(new_buffer(full_c.flat(), float64), Y.shape.length === 2 ? [lmax + 1, k] : [lmax + 1], float64);
  if (full) return [c, [resids, rank, s, rc]];
  return c;
}

// ==============================
//   Quadrature
// ==============================

export function gauss(B: Basis, deg: number): [NDArray, NDArray] {
  if (!Number.isInteger(deg) || deg <= 0) throw new PolyError("deg must be a positive integer");
  if (B === CHEBYSHEV) {
    const x = range(deg).map(i => Math.cos(Math.PI * (2 * i + 1) / (2 * deg)));
    return [arr(x), arr(x.map(() => Math.PI / deg))];
  }
  const c = [...Array(deg).fill(0), 1];
  const m = compan(B, c);
  let x = Array.from((la.eigh(m, 'L')[0] as NDArray).flat as any, (v: any) => +v) as number[];
  const ds = der(B, c);
  const dy = x.map(v => clenshaw(B, c, v)), df = x.map(v => clenshaw(B, ds, v));
  x = x.map((v, i) => v - dy[i] / df[i]);
  const fm = x.map(v => clenshaw(B, c.slice(1), v)), df2 = x.map(v => clenshaw(B, ds, v));
  const fmax = Math.max(...fm.map(Math.abs)), dmax = Math.max(...df2.map(Math.abs));
  let w = fm.map((v, i) => 1 / ((v / fmax) * (df2[i] / dmax)));
  if (B === LAGUERRE) {
    const s = w.reduce((p, q) => p + q, 0);
    return [arr(x), arr(w.map(v => v / s))];
  }
  w = w.map((v, i) => (v + w[w.length - 1 - i]) / 2);
  x = x.map((v, i) => (v - x[x.length - 1 - i]) / 2);
  const total = w.reduce((p, q) => p + q, 0);
  const target = B === LEGENDRE ? 2 : (B === HERMITE ? Math.sqrt(Math.PI) : Math.sqrt(2 * Math.PI));
  return [arr(x), arr(w.map(v => v * target / total))];
}

export function pts1(npts: number): NDArray {
  if (!Number.isInteger(npts) || npts < 1) throw new PolyError("npts must be a positive integer");
  return arr(range(npts).map(i => Math.cos(-Math.PI + i * Math.PI / npts + Math.PI / (2 * npts))));
}
export function pts2(npts: number): NDArray {
  if (!Number.isInteger(npts) || npts < 2) throw new PolyError("npts must be integer greater than 1");
  return arr(range(npts).map(i => Math.cos(-Math.PI + i * Math.PI / (npts - 1))));
}
export function chebinterpolate(func: (x: NDArray, ...args: any[]) => any, deg: number, args: any[] = []): NDArray {
  if (!Number.isInteger(deg) || deg < 0) throw new PolyError("deg must be an integer");
  const order = deg + 1;
  const xcheb = pts1(order);
  const yfun = asarray(func(xcheb, ...args));
  const m = Array.from(vander(CHEBYSHEV, xcheb, deg).flat as any) as number[];
  const y = Array.from(yfun.flat as any, (v: any) => +v) as number[];
  const c = range(order).map(k => range(order).reduce((s, i) => s + m[i * order + k] * y[i], 0));
  c[0] /= order;
  for (let k = 1; k < order; k++) c[k] /= 0.5 * order;
  return arr(c);
}

// ==============================
//   Multi-dimensional evaluation
// ==============================

export function val2d(B: Basis, x: any, y: any, c: NDArray): NDArray {
  const X = asarray(x), Y = asarray(y);
  if (X.shape.length !== Y.shape.length || X.shape.some((d, i) => d !== Y.shape[i])) throw new PolyError("x, y are incompatible");
  const [m, n] = c.shape;
  const cf = c.flat as any;
  const xs = Array.from(X.flat as any, (v: any) => +v) as number[], ys = Array.from(Y.flat as any, (v: any) => +v) as number[];
  const out = xs.map((xv, i) => {
    // evaluate along x for every column then along y
    const col = range(n).map(j => clenshaw(B, range(m).map(k => cf[k * n + j]), xv));
    return clenshaw(B, col, ys[i]);
  });
  return new NDArray(new_buffer(out, float64), X.shape, float64);
}
export function val3d(B: Basis, x: any, y: any, z: any, c: NDArray): NDArray {
  const X = asarray(x), Y = asarray(y), Z = asarray(z);
  if ([Y, Z].some(A => A.shape.length !== X.shape.length || A.shape.some((d, i) => d !== X.shape[i]))) throw new PolyError("x, y, z are incompatible");
  const [l, m, n] = c.shape;
  const cf = c.flat as any;
  const xs = Array.from(X.flat as any, (v: any) => +v) as number[], ys = Array.from(Y.flat as any, (v: any) => +v) as number[], zs = Array.from(Z.flat as any, (v: any) => +v) as number[];
  const out = xs.map((xv, i) => {
    const plane = range(m).map(j => clenshaw(B, range(n).map(t => clenshaw(B, range(l).map(k => cf[(k * m + j) * n + t]), xv)), zs[i]));
    return clenshaw(B, plane, ys[i]);
  });
  return new NDArray(new_buffer(out, float64), X.shape, float64);
}
export function grid2d(B: Basis, x: any, y: any, c: NDArray): NDArray {
  const xs = Array.from(asarray(x).flat as any, (v: any) => +v) as number[], ys = Array.from(asarray(y).flat as any, (v: any) => +v) as number[];
  const [m, n] = c.shape, cf = c.flat as any;
  const xshape = asarray(x).shape, yshape = asarray(y).shape;
  const out: number[] = [];
  for (const xv of xs) {
    const col = range(n).map(j => clenshaw(B, range(m).map(k => cf[k * n + j]), xv));
    for (const yv of ys) out.push(clenshaw(B, col, yv));
  }
  return new NDArray(new_buffer(out, float64), [...xshape, ...yshape], float64);
}
export function grid3d(B: Basis, x: any, y: any, z: any, c: NDArray): NDArray {
  const xs = Array.from(asarray(x).flat as any, (v: any) => +v) as number[], ys = Array.from(asarray(y).flat as any, (v: any) => +v) as number[], zs = Array.from(asarray(z).flat as any, (v: any) => +v) as number[];
  const [l, m, n] = c.shape, cf = c.flat as any;
  const out: number[] = [];
  for (const xv of xs) {
    const plane = range(m).map(j => range(n).map(t => clenshaw(B, range(l).map(k => cf[(k * m + j) * n + t]), xv)));
    for (const yv of ys) {
      const row = range(n).map(t => clenshaw(B, range(m).map(j => plane[j][t]), yv));
      for (const zv of zs) out.push(clenshaw(B, row, zv));
    }
  }
  return new NDArray(new_buffer(out, float64), [...asarray(x).shape, ...asarray(y).shape, ...asarray(z).shape], float64);
}
export function vander2d(B: Basis, x: any, y: any, deg: number[]): NDArray {
  const [dx, dy] = deg;
  const X = asarray(x), Y = asarray(y);
  if (X.shape.length !== Y.shape.length || X.shape.some((d, i) => d !== Y.shape[i])) throw new PolyError("x and y must have the same shape");
  const vx = vander(B, X, dx).flat as any, vy = vander(B, Y, dy).flat as any;
  const n = X.size, mx = dx + 1, my = dy + 1;
  const buffer = new_buffer(n * mx * my, float64);
  for (let i = 0; i < n; i++) for (let a = 0; a < mx; a++) for (let b = 0; b < my; b++) buffer[i * mx * my + a * my + b] = vx[i * mx + a] * vy[i * my + b];
  return new NDArray(buffer, [...X.shape, mx * my], float64);
}
export function vander3d(B: Basis, x: any, y: any, z: any, deg: number[]): NDArray {
  const [dx, dy, dz] = deg;
  const X = asarray(x), Y = asarray(y), Z = asarray(z);
  const vx = vander(B, X, dx).flat as any, vy = vander(B, Y, dy).flat as any, vz = vander(B, Z, dz).flat as any;
  const n = X.size, mx = dx + 1, my = dy + 1, mz = dz + 1;
  const buffer = new_buffer(n * mx * my * mz, float64);
  for (let i = 0; i < n; i++) for (let a = 0; a < mx; a++) for (let b = 0; b < my; b++) for (let d = 0; d < mz; d++)
    buffer[((i * mx + a) * my + b) * mz + d] = vx[i * mx + a] * vy[i * my + b] * vz[i * mz + d];
  return new NDArray(buffer, [...X.shape, mx * my * mz], float64);
}

export function valfromroots(x: any, r: any, tensor: boolean = true): NDArray {
  const R = as_numbers(r);
  const X = asarray(x);
  const xs = Array.from(X.flat as any, (v: any) => +v) as number[];
  const prodAt = (xv: number) => R.reduce((p, rt) => p * (xv - rt), 1);
  return new NDArray(new_buffer(xs.map(prodAt), float64), X.shape, float64);
}
