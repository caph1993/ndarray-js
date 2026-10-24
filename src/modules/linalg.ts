//@ts-check
/**
 * Matrix products and linear algebra (numpy.dot, numpy.matmul, numpy.einsum, numpy.linalg.*).
 * Everything works on stacks of matrices (leading dimensions are batch dimensions) and on real numbers only.
 */
import { NDArray, asarray, isarray, shape_shifts, broadcast_shapes, broadcast_n_shapes, broadcast_strides } from "../NDArray";
import { DType, bool, int32, float32, float64, new_buffer, convert_buffer, fix_buffer, dtype_max, dtype_is_integer, dtype_is_float, float_out, same_out, sum_out, truthy } from "../dtypes";
import { normalize_axis, transpose, swapAxes, concatenate } from "../array/transform";
import { ravel, reshape } from "../array/shape_operations";
import { finish_output, map_broadcast } from "../array/operators";
import { broadcast_to, diag, eye } from "./manipulation";
import { diagonal } from "./indexing-extra";
import { Complex, ZERO, ONE } from "../complex";
import { complex128 } from "../dtypes";

const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export class LinAlgError extends Error {
  constructor(message: string) { super(message); this.name = "LinAlgError"; }
}

type Mat = number[][];

function to_mat(flat: ArrayLike<number>, rows: number, cols: number, offset = 0): Mat {
  const m: Mat = new Array(rows);
  for (let i = 0; i < rows; i++) { m[i] = new Array(cols); for (let j = 0; j < cols; j++) m[i][j] = +flat[offset + i * cols + j]; }
  return m;
}
function from_mats(mats: Mat[], rows: number, cols: number, batch: number[], dtype: DType): NDArray {
  const buffer = new_buffer(mats.length * rows * cols, dtype);
  mats.forEach((m, b) => { for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) buffer[(b * rows + i) * cols + j] = m[i][j]; });
  return new NDArray(fix_buffer(buffer, dtype), [...batch, rows, cols], dtype);
}
function from_vecs(vecs: number[][], len: number, batch: number[], dtype: DType): NDArray {
  const buffer = new_buffer(vecs.length * len, dtype);
  vecs.forEach((v, b) => { for (let i = 0; i < len; i++) buffer[b * len + i] = v[i]; });
  return new NDArray(fix_buffer(buffer, dtype), [...batch, len], dtype);
}
function from_scalars(vals: number[], batch: number[], dtype: DType): NDArray {
  return new NDArray(fix_buffer(new_buffer(vals, dtype), dtype), batch, dtype);
}

/** Splits an array of shape (..., m, n) into a list of matrices */
function matrices(a: NDArray, name = 'array'): { mats: Mat[], batch: number[], rows: number, cols: number } {
  a = asarray(a);
  if (a.dtype === complex128) throw new LinAlgError(`complex matrices are not supported by this linear algebra function`);
  if (a.shape.length < 2) throw new LinAlgError(`${a.shape.length}-dimensional array given. Array must be at least two-dimensional`);
  const [rows, cols] = a.shape.slice(-2);
  const batch = a.shape.slice(0, -2);
  const flat = a.flat as any;
  const mats = range(prod(batch)).map(b => to_mat(flat, rows, cols, b * rows * cols));
  return { mats, batch, rows, cols };
}
function square_matrices(a: NDArray) {
  const r = matrices(a);
  if (r.rows != r.cols) throw new LinAlgError(`Last 2 dimensions of the array must be square`);
  return r;
}
function check_finite(m: Mat) {
  for (const row of m) for (const v of row) if (!Number.isFinite(v)) throw new LinAlgError(`Array must not contain infs or NaNs`);
}
const float_dtype = (a: NDArray): DType => (a.dtype === float32 ? float32 : float64);

// ==============================
//   Products
// ==============================

function product_dtype(a: NDArray, b: NDArray): DType {
  return dtype_max(a.dtype, b.dtype);
}

/** Core matrix product of (n,k) x (k,m) stored in flat arrays */
function matmul_flat(A: any, ao: number, B: any, bo: number, n: number, k: number, m: number, out: any, oo: number, isBool: boolean, isComplex = false) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s: any = 0;
      if (isComplex) { let re = 0, im = 0; for (let t = 0; t < k; t++) { const a = A[ao + i * k + t], b = B[bo + t * m + j]; re += a.re * b.re - a.im * b.im; im += a.re * b.im + a.im * b.re; } s = new Complex(re, im); }
      else if (isBool) { for (let t = 0; t < k; t++) if (A[ao + i * k + t] && B[bo + t * m + j]) { s = 1; break; } }
      else for (let t = 0; t < k; t++) s += A[ao + i * k + t] * B[bo + t * m + j];
      out[oo + i * m + j] = s;
    }
  }
}

/** Flat values of an array converted to the dtype of the product (complex values need Complex objects) */
const operand = (x: NDArray, dtype: DType) => dtype === complex128 ? convert_buffer(x.flat as any, complex128) : x.flat as any;

export function matmul(a: NDArray, b: NDArray, out: NDArray | null = null): NDArray {
  a = asarray(a); b = asarray(b);
  if (a.shape.length == 0 || b.shape.length == 0) throw new Error(`matmul: Input operand does not have enough dimensions (scalars are not allowed, use multiply)`);
  const a1 = a.shape.length == 1, b1 = b.shape.length == 1;
  const A = a1 ? reshape(a, [1, a.shape[0]]) : a, B = b1 ? reshape(b, [b.shape[0], 1]) : b;
  const [n, k] = A.shape.slice(-2), [k2, m] = B.shape.slice(-2);
  if (k != k2) throw new Error(`matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size ${k2} is different from ${k})`);
  const [, batchShape] = broadcast_n_shapes(A.shape.slice(0, -2), B.shape.slice(0, -2));
  const batch = batchShape as number[];
  const dtype = product_dtype(a, b);
  const Ab = operand(broadcast_to(A, [...batch, n, k]), dtype), Bb = operand(broadcast_to(B, [...batch, k, m]), dtype);
  const nb = prod(batch);
  const buffer = new_buffer(nb * n * m, dtype);
  for (let t = 0; t < nb; t++) matmul_flat(Ab, t * n * k, Bb, t * k * m, n, k, m, buffer, t * n * m, dtype === bool, dtype === complex128);
  fix_buffer(buffer, dtype);
  let shape = [...batch, n, m];
  if (a1) shape = shape.filter((_, i) => i != shape.length - 2);
  if (b1) shape = shape.slice(0, -1);
  // when both are vectors: shape is []
  if (a1 && b1) shape = [];
  else if (a1) shape = [...batch, m];
  else if (b1) shape = [...batch, n];
  return finish_output(buffer, shape, dtype, out);
}

export function dot(a: NDArray, b: NDArray, out: NDArray | null = null): NDArray {
  a = asarray(a); b = asarray(b);
  if (a.shape.length == 0 || b.shape.length == 0) {
    const dtype = product_dtype(a, b);
    const [shape] = broadcast_shapes(a.shape, b.shape);
    const buffer = new_buffer(prod(shape), dtype);
    map_broadcast(shape, [broadcast_strides(shape, a.shape), broadcast_strides(shape, b.shape)], [operand(a, dtype), operand(b, dtype)], dtype === complex128 ? (x: Complex, y: Complex) => x.mul(y) : (x: number, y: number) => x * y, buffer);
    return finish_output(fix_buffer(buffer, dtype), shape, dtype, out);
  }
  if (a.shape.length <= 2 && b.shape.length <= 2) return matmul(a, b, out);
  // dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
  const kA = a.shape[a.shape.length - 1];
  const kB = b.shape.length == 1 ? b.shape[0] : b.shape[b.shape.length - 2];
  if (kA != kB) throw new Error(`shapes (${a.shape}) and (${b.shape}) not aligned: ${kA} (dim ${a.shape.length - 1}) != ${kB} (dim ${Math.max(0, b.shape.length - 2)})`);
  const bm = b.shape.length == 1 ? reshape(b, [kB, 1]) : swapAxes(b, -2, -1);       // (..., m, k)
  const bshape = b.shape.length == 1 ? [] : [...b.shape.slice(0, -2), b.shape[b.shape.length - 1]];
  const lead = a.shape.slice(0, -1);
  const rowsA = prod(lead);
  const A2 = reshape(a, [rowsA, kA]);
  // B as (batch..., m) rows of length k
  const rowsB = b.shape.length == 1 ? 1 : prod(b.shape.slice(0, -2)) * b.shape[b.shape.length - 1];
  const B2 = reshape(bm, [rowsB, kB]);
  const dtype = product_dtype(a, b);
  const fa = operand(A2, dtype), fb = operand(B2, dtype);
  const buffer = new_buffer(rowsA * rowsB, dtype);
  for (let i = 0; i < rowsA; i++) for (let j = 0; j < rowsB; j++) {
    if (dtype === complex128) {
      let re = 0, im = 0;
      for (let t = 0; t < kA; t++) { const x = fa[i * kA + t], y = fb[j * kB + t]; re += x.re * y.re - x.im * y.im; im += x.re * y.im + x.im * y.re; }
      buffer[i * rowsB + j] = new Complex(re, im);
      continue;
    }
    let s = 0;
    for (let t = 0; t < kA; t++) s += fa[i * kA + t] * fb[j * kB + t];
    buffer[i * rowsB + j] = dtype === bool ? (s ? 1 : 0) : s;
  }
  fix_buffer(buffer, dtype);
  return finish_output(buffer, [...lead, ...bshape], dtype, out);
}

export function vdot(a: NDArray, b: NDArray): NDArray {
  a = ravel(asarray(a)); b = ravel(asarray(b));
  if (a.size != b.size) throw new Error(`vectors have different lengths`);
  if (a.dtype === complex128) a = new NDArray(Array.from(a.flat as any, (z: any) => z.conj()) as any, a.shape, complex128);
  return dot(a, b);
}

export function inner(a: NDArray, b: NDArray): NDArray {
  a = asarray(a); b = asarray(b);
  if (a.shape.length == 0 || b.shape.length == 0) return dot(a, b);
  if (a.shape[a.shape.length - 1] != b.shape[b.shape.length - 1]) throw new Error(`shapes (${a.shape}) and (${b.shape}) not aligned: ${a.shape[a.shape.length - 1]} (dim ${a.shape.length - 1}) != ${b.shape[b.shape.length - 1]} (dim ${b.shape.length - 1})`);
  return tensordot(a, b, [[-1], [-1]]);
}

export function outer(a: NDArray, b: NDArray, out: NDArray | null = null): NDArray {
  const x = ravel(asarray(a)), y = ravel(asarray(b));
  const dtype = product_dtype(x, y);
  const buffer = new_buffer(x.size * y.size, dtype);
  const fx = operand(x, dtype), fy = operand(y, dtype);
  for (let i = 0; i < x.size; i++) for (let j = 0; j < y.size; j++) buffer[i * y.size + j] = dtype === complex128 ? fx[i].mul(fy[j]) : fx[i] * fy[j];
  return finish_output(fix_buffer(buffer, dtype), [x.size, y.size], dtype, out);
}

export function matvec(x1: NDArray, x2: NDArray): NDArray {
  x1 = asarray(x1); x2 = asarray(x2);
  if (x2.shape.length < 1) throw new Error(`matvec: Input operand 1 does not have enough dimensions`);
  const r = matmul(x1, reshape(x2, [...x2.shape, 1]));
  return reshape(r, r.shape.slice(0, -1));
}
export function vecmat(x1: NDArray, x2: NDArray): NDArray {
  x1 = asarray(x1); x2 = asarray(x2);
  const r = matmul(reshape(x1, [...x1.shape.slice(0, -1), 1, x1.shape[x1.shape.length - 1]]), x2);
  return reshape(r, [...r.shape.slice(0, -2), r.shape[r.shape.length - 1]]);
}

export function vecdot(x1: NDArray, x2: NDArray, axis: number): NDArray {
  x1 = asarray(x1); x2 = asarray(x2);
  const a = swapAxes(x1, axis, -1), b = swapAxes(x2, axis, -1);
  const [, shape] = broadcast_n_shapes(a.shape, b.shape);
  const bs = shape as number[];
  const A = broadcast_to(a, bs).flat as any, B = broadcast_to(b, bs).flat as any;
  const k = bs[bs.length - 1] ?? 1;
  const dtype = product_dtype(x1, x2);
  const n = prod(bs.slice(0, -1));
  const buffer = new_buffer(n, dtype);
  if (dtype === complex128) {
    const CA = convert_buffer(A, complex128), CB = convert_buffer(B, complex128);
    for (let i = 0; i < n; i++) { let s = ZERO; for (let t = 0; t < k; t++) s = s.add(CA[i * k + t].conj().mul(CB[i * k + t])); buffer[i] = s; }
  } else for (let i = 0; i < n; i++) { let s = 0; for (let t = 0; t < k; t++) s += A[i * k + t] * B[i * k + t]; buffer[i] = s; }
  return new NDArray(fix_buffer(buffer, dtype), bs.slice(0, -1), dtype);
}

export function tensordot(a: NDArray, b: NDArray, axes: number | [number[] | number, number[] | number]): NDArray {
  a = asarray(a); b = asarray(b);
  let axa: number[], axb: number[];
  if (typeof axes === "number") {
    if (axes < 0 || axes > a.shape.length || axes > b.shape.length) throw new Error(`tensordot: axes=${axes} out of range`);
    axa = range(axes).map(i => a.shape.length - axes + i);
    axb = range(axes);
  } else {
    const [p, q] = axes;
    axa = (Array.isArray(p) ? p : [p]).map(v => normalize_axis(v, a.shape.length));
    axb = (Array.isArray(q) ? q : [q]).map(v => normalize_axis(v, b.shape.length));
  }
  if (axa.length != axb.length) throw new Error(`shape-mismatch for sum`);
  axa.forEach((x, i) => { if (a.shape[x] != b.shape[axb[i]]) throw new Error(`shape-mismatch for sum`); });
  const freeA = range(a.shape.length).filter(i => !axa.includes(i)), freeB = range(b.shape.length).filter(i => !axb.includes(i));
  const k = prod(axa.map(x => a.shape[x]));
  const A2 = reshape(transpose(a, [...freeA, ...axa]), [prod(freeA.map(i => a.shape[i])), k]);
  const B2 = reshape(transpose(b, [...axb, ...freeB]), [k, prod(freeB.map(i => b.shape[i]))]);
  const r = matmul(A2, B2);
  return reshape(r, [...freeA.map(i => a.shape[i]), ...freeB.map(i => b.shape[i])]);
}

export function kron(a: NDArray, b: NDArray): NDArray {
  a = asarray(a); b = asarray(b);
  const nd = Math.max(a.shape.length, b.shape.length);
  const as = [...Array(nd - a.shape.length).fill(1), ...a.shape], bs = [...Array(nd - b.shape.length).fill(1), ...b.shape];
  const shape = as.map((d, i) => d * bs[i]);
  const dtype = product_dtype(a, b);
  const buffer = new_buffer(prod(shape), dtype);
  const fa = operand(a, dtype), fb = operand(b, dtype);
  const sa = shape_shifts(as), sb = shape_shifts(bs), so = shape_shifts(shape);
  const nA = prod(as), nB = prod(bs);
  for (let i = 0; i < nA; i++) {
    const ia = as.map((d, ax) => Math.floor(i / sa[ax]) % d);
    for (let j = 0; j < nB; j++) {
      const ib = bs.map((d, ax) => Math.floor(j / sb[ax]) % d);
      let pos = 0;
      for (let ax = 0; ax < nd; ax++) pos += (ia[ax] * bs[ax] + ib[ax]) * so[ax];
      buffer[pos] = dtype === complex128 ? fa[i].mul(fb[j]) : fa[i] * fb[j];
    }
  }
  return new NDArray(fix_buffer(buffer, dtype), shape, dtype);
}

export function matrix_power(a: NDArray, n: number): NDArray {
  const { rows, cols } = square_matrices(a);
  a = asarray(a);
  if (!Number.isInteger(n)) throw new TypeError(`exponent must be an integer`);
  if (n === 0) return broadcast_to(eye(rows, null, 0, a.dtype === bool ? int32 : a.dtype), a.shape);
  let base = a;
  if (n < 0) { base = inv(a); n = -n; }
  let result: NDArray | null = null;
  while (n > 0) {
    if (n & 1) result = result ? matmul(result, base) : base;
    n >>= 1;
    if (n) base = matmul(base, base);
  }
  return result!;
}

export function multi_dot(arrays: NDArray[]): NDArray {
  if (arrays.length < 2) throw new Error(`Expecting at least two arrays.`);
  const arrs = arrays.map(a => asarray(a));
  arrs.forEach((a, i) => { if ((i == 0 || i == arrs.length - 1) ? a.shape.length > 2 || a.shape.length < 1 : a.shape.length != 2) throw new LinAlgError(`${a.shape.length}-dimensional array given. Array must be two-dimensional`); });
  return arrs.slice(1).reduce((acc, a) => dot(acc, a), arrs[0]);
}

export function trace(a: NDArray, offset: number, axis1: number, axis2: number, dtype: DType | null, out: NDArray | null): NDArray {
  a = asarray(a);
  const d = diagonal(a, offset, axis1, axis2);
  const dt = dtype || sum_out([a.dtype]);
  const n = d.shape[d.shape.length - 1];
  const lead = prod(d.shape.slice(0, -1));
  const f = d.flat as any;
  const buffer = new_buffer(lead, dt);
  if (dt === complex128) for (let l = 0; l < lead; l++) { let s = ZERO; for (let i = 0; i < n; i++) s = s.add(f[l * n + i]); buffer[l] = s; }
  else for (let l = 0; l < lead; l++) { let s = 0; for (let i = 0; i < n; i++) s += +f[l * n + i]; buffer[l] = s; }
  return finish_output(fix_buffer(buffer, dt), d.shape.slice(0, -1), dt, out);
}

// ==============================
//   einsum
// ==============================

function parse_einsum(subscripts: string, operands: NDArray[]) {
  const s = subscripts.replace(/\s+/g, '');
  const [lhs, rhs] = s.includes('->') ? s.split('->') : [s, null];
  const inputs = lhs.split(',');
  if (inputs.length != operands.length) throw new Error(`more operands provided to einstein sum function than specified in the subscripts string`);
  // expand ellipsis into upper-case-free private labels
  const ellLabels = (n: number) => range(n).map(i => String.fromCharCode(0x2460 + i));
  const maxEll = Math.max(0, ...inputs.map((inp, k) => inp.includes('...') ? operands[k].shape.length - (inp.replace('...', '').length) : 0));
  const expanded = inputs.map((inp, k) => {
    if (!inp.includes('...')) {
      if (inp.length != operands[k].shape.length) throw new Error(`operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions.`);
      return inp.split('');
    }
    const nEll = operands[k].shape.length - inp.replace('...', '').length;
    const labels = ellLabels(maxEll).slice(maxEll - nEll);
    return inp.replace('...', '\0').split('').flatMap(c => c === '\0' ? labels : [c]);
  });
  let out: string[];
  if (rhs === null) {
    const count: { [c: string]: number } = {};
    expanded.flat().forEach(c => { count[c] = (count[c] || 0) + 1; });
    out = [...ellLabels(maxEll), ...Object.keys(count).filter(c => count[c] == 1 && !/[①-⒀]/.test(c)).sort()];
  } else out = rhs.replace('...', '\0').split('').flatMap(c => c === '\0' ? ellLabels(maxEll) : [c]);
  return { expanded, out };
}

export function einsum(subscripts: string, operands: NDArray[], dtype: DType | null = null, out: NDArray | null = null): NDArray {
  const ops = operands.map(o => asarray(o));
  const { expanded, out: outLabels } = parse_einsum(subscripts, ops);
  const sizes: { [c: string]: number } = {};
  expanded.forEach((labels, k) => labels.forEach((c, ax) => {
    const d = ops[k].shape[ax];
    if (sizes[c] === undefined || sizes[c] === 1) sizes[c] = d;
    else if (d !== 1 && d !== sizes[c]) throw new Error(`operands could not be broadcast together with remapped shapes [original->remapped]: dimension mismatch for label ${c}`);
  }));
  for (const c of outLabels) if (sizes[c] === undefined) throw new Error(`output subscript ${c} does not appear in the equation for any input operand`);
  const summed = Object.keys(sizes).filter(c => !outLabels.includes(c));
  const all = [...outLabels, ...summed];
  const dims = all.map(c => sizes[c]);
  const outShape = outLabels.map(c => sizes[c]);
  const strides = ops.map((o, k) => {
    const st = shape_shifts(o.shape);
    // a label repeated in one operand (like 'ii') walks the diagonal: its strides add up
    return all.map(c => expanded[k].reduce((sum, label, ax) => label === c && o.shape[ax] !== 1 ? sum + st[ax] : sum, 0));
  });
  const dt = dtype || dtype_max(...ops.map(o => o.dtype));
  const flats = ops.map(o => operand(o, dt));
  const nOut = prod(outShape), nSum = prod(dims.slice(outLabels.length));
  const buffer = new_buffer(nOut, dt);
  const nOutAxes = outLabels.length;
  const idx = new Array(all.length).fill(0);
  const odd = (pos: number[]) => 0;
  for (let o = 0; o < nOut; o++) {
    let total: any = dt === complex128 ? ZERO : 0;
    // iterate over summed labels
    for (let s = 0; s < nSum; s++) {
      let p: any = dt === complex128 ? ONE : 1;
      for (let k = 0; k < ops.length; k++) {
        let pos = 0;
        for (let d = 0; d < all.length; d++) pos += idx[d] * strides[k][d];
        if (dt === complex128) p = p.mul(flats[k][pos]); else p *= +flats[k][pos];
      }
      total = dt === complex128 ? total.add(p) : total + p;
      for (let d = all.length - 1; d >= nOutAxes; d--) { if (++idx[d] < dims[d]) break; idx[d] = 0; }
    }
    buffer[o] = dt === bool ? (total ? 1 : 0) : total;
    for (let d = nOutAxes - 1; d >= 0; d--) { if (++idx[d] < dims[d]) break; idx[d] = 0; }
  }
  return finish_output(fix_buffer(buffer, dt), outShape, dt, out);
}

export function einsum_path(subscripts: string, operands: NDArray[], optimize: any = 'greedy'): [any[], string] {
  const ops = operands.map(o => asarray(o));
  const { expanded, out } = parse_einsum(subscripts, ops);
  const path: any[] = ['einsum_path'];
  if (ops.length > 2) { path.push(range(ops.length)); } else path.push(range(ops.length));
  const info = `  Complete contraction:  ${subscripts}\n         Naive scaling:  ${new Set(expanded.flat()).size}\n     Optimized scaling:  ${new Set(expanded.flat()).size}`;
  return [path, info];
}

// ==============================
//   Decompositions and solvers
// ==============================

/** LU decomposition with partial pivoting. Returns null when the matrix is exactly singular. */
function lu(m: Mat): { lu: Mat, piv: number[], sign: number, singular: boolean } {
  const n = m.length;
  const a = m.map(r => r.slice());
  const piv = range(n);
  let sign = 1, singular = false;
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(a[k][k]);
    for (let i = k + 1; i < n; i++) if (Math.abs(a[i][k]) > best) { best = Math.abs(a[i][k]); p = i; }
    if (best === 0) { singular = true; continue; }
    if (p != k) { [a[p], a[k]] = [a[k], a[p]]; [piv[p], piv[k]] = [piv[k], piv[p]]; sign = -sign; }
    for (let i = k + 1; i < n; i++) {
      a[i][k] /= a[k][k];
      for (let j = k + 1; j < n; j++) a[i][j] -= a[i][k] * a[k][j];
    }
  }
  return { lu: a, piv, sign, singular };
}

function lu_solve(f: { lu: Mat, piv: number[] }, B: Mat): Mat {
  const n = f.lu.length, k = B[0] ? B[0].length : 0;
  const X: Mat = f.piv.map(p => B[p].slice());
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) X[i][c] -= f.lu[i][j] * X[j][c];
    for (let i = n - 1; i >= 0; i--) {
      for (let j = i + 1; j < n; j++) X[i][c] -= f.lu[i][j] * X[j][c];
      X[i][c] /= f.lu[i][i];
    }
  }
  return X;
}

export function det(a: NDArray): NDArray {
  const { mats, batch } = square_matrices(a);
  const vals = mats.map(m => { check_finite(m); const f = lu(m); if (f.singular) return 0; let d = f.sign; for (let i = 0; i < m.length; i++) d *= f.lu[i][i]; return d; });
  return from_scalars(vals, batch, float_dtype(asarray(a)));
}

export function slogdet(a: NDArray): [NDArray, NDArray] {
  const { mats, batch } = square_matrices(a);
  const sign: number[] = [], logdet: number[] = [];
  mats.forEach(m => {
    const f = lu(m);
    if (f.singular) { sign.push(0); logdet.push(-Infinity); return; }
    let s = f.sign, l = 0;
    for (let i = 0; i < m.length; i++) { const d = f.lu[i][i]; if (d < 0) s = -s; l += Math.log(Math.abs(d)); }
    sign.push(s); logdet.push(l);
  });
  const dt = float_dtype(asarray(a));
  return [from_scalars(sign, batch, dt), from_scalars(logdet, batch, dt)];
}

export function inv(a: NDArray): NDArray {
  const { mats, batch, rows } = square_matrices(a);
  const res = mats.map(m => {
    check_finite(m);
    const f = lu(m);
    if (f.singular) throw new LinAlgError(`Singular matrix`);
    return lu_solve(f, range(rows).map(i => range(rows).map(j => i == j ? 1 : 0)));
  });
  return from_mats(res, rows, rows, batch, float_dtype(asarray(a)));
}

export function solve(a: NDArray, b: NDArray): NDArray {
  a = asarray(a); b = asarray(b);
  const { rows } = square_matrices(a);
  const vector = b.shape.length == 1;
  const B = vector ? reshape(b, [b.shape[0], 1]) : b;
  if (B.shape.length < 2 || B.shape[B.shape.length - 2] != rows) throw new Error(`solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size ${B.shape[B.shape.length - 2]} is different from ${rows})`);
  const [, batchShape] = broadcast_n_shapes(a.shape.slice(0, -2), B.shape.slice(0, -2));
  const batch = batchShape as number[];
  const k = B.shape[B.shape.length - 1];
  const Ab = matrices(broadcast_to(a, [...batch, rows, rows])), Bb = matrices(broadcast_to(B, [...batch, rows, k]));
  const res = Ab.mats.map((m, i) => {
    const f = lu(m);
    if (f.singular) throw new LinAlgError(`Singular matrix`);
    return lu_solve(f, Bb.mats[i]);
  });
  const dt = float_dtype(dtype_max(a.dtype, b.dtype) === float32 ? asarray([0.5], 'float32') : asarray([0.5]));
  const out = from_mats(res, rows, k, batch, dt);
  return vector ? reshape(out, [...batch, rows]) : out;
}

export function cholesky(a: NDArray, upper: boolean = false): NDArray {
  const { mats, batch, rows } = square_matrices(a);
  const res = mats.map(m => {
    check_finite(m);
    const n = m.length;
    const L: Mat = range(n).map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = m[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i == j) { if (!(s > 0)) throw new LinAlgError(`Matrix is not positive definite`); L[i][j] = Math.sqrt(s); }
      else L[i][j] = s / L[j][j];
    }
    return upper ? range(n).map(i => range(n).map(j => L[j][i])) : L;
  });
  return from_mats(res, rows, rows, batch, float_dtype(asarray(a)));
}

/** Householder QR as in LAPACK (geqrf): R has the sign conventions of numpy */
function qr_mat(A: Mat, m: number, n: number): { Q: Mat, R: Mat } {
  const R = A.map(r => r.slice());
  const Q: Mat = range(m).map(i => range(m).map(j => i == j ? 1 : 0));
  const k = Math.min(m, n);
  for (let j = 0; j < k; j++) {
    let tail = 0;
    for (let i = j + 1; i < m; i++) tail += R[i][j] * R[i][j];
    if (tail === 0) continue; // H = I
    const alpha = R[j][j];
    const norm = Math.sqrt(alpha * alpha + tail);
    const beta = alpha >= 0 ? -norm : norm;
    const tau = (beta - alpha) / beta;
    const v = new Array(m).fill(0);
    v[j] = 1;
    for (let i = j + 1; i < m; i++) v[i] = R[i][j] / (alpha - beta);
    // R = H R, Q = Q H with H = I - tau v v^T
    for (let c = j; c < n; c++) {
      let s = 0; for (let i = j; i < m; i++) s += v[i] * R[i][c];
      s *= tau; for (let i = j; i < m; i++) R[i][c] -= s * v[i];
    }
    for (let r = 0; r < m; r++) {
      let s = 0; for (let i = j; i < m; i++) s += Q[r][i] * v[i];
      s *= tau; for (let i = j; i < m; i++) Q[r][i] -= s * v[i];
    }
    R[j][j] = beta;
    for (let i = j + 1; i < m; i++) R[i][j] = 0;
  }
  return { Q, R };
}

export function qr(a: NDArray, mode: 'reduced' | 'complete' | 'r' | 'raw'): NDArray | NDArray[] {
  const { mats, batch, rows: m, cols: n } = matrices(a);
  if (!['reduced', 'complete', 'r', 'raw'].includes(mode)) throw new Error(`Unrecognized mode '${mode}'`);
  if (mode === 'raw') throw new Error(`qr mode 'raw' is not supported`);
  const dt = float_dtype(asarray(a));
  const k = Math.min(m, n);
  const results = mats.map(M => { check_finite(M); return qr_mat(M, m, n); });
  if (mode === 'r') return from_mats(results.map(r => r.R.slice(0, k)), k, n, batch, dt);
  if (mode === 'complete') return [from_mats(results.map(r => r.Q), m, m, batch, dt), from_mats(results.map(r => r.R), m, n, batch, dt)];
  return [from_mats(results.map(r => r.Q.map(row => row.slice(0, k))), m, k, batch, dt), from_mats(results.map(r => r.R.slice(0, k)), k, n, batch, dt)];
}

function norm2(v: number[]) { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); }

/** One-sided Jacobi SVD of an m x n matrix with m >= n. Returns U (m x n), s (n), V (n x n) sorted by decreasing s */
function svd_jacobi(A: Mat, m: number, n: number): { U: Mat, s: number[], V: Mat } {
  const U = A.map(r => r.slice());
  const V: Mat = range(n).map(i => range(n).map(j => i == j ? 1 : 0));
  const eps = 1e-15;
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let alpha = 0, beta = 0, gamma = 0;
      for (let i = 0; i < m; i++) { alpha += U[i][p] * U[i][p]; beta += U[i][q] * U[i][q]; gamma += U[i][p] * U[i][q]; }
      if (gamma === 0 || Math.abs(gamma) <= eps * Math.sqrt(alpha * beta)) continue;
      rotated = true;
      const zeta = (beta - alpha) / (2 * gamma);
      const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const c = 1 / Math.sqrt(1 + t * t), s = c * t;
      for (let i = 0; i < m; i++) { const up = U[i][p], uq = U[i][q]; U[i][p] = c * up - s * uq; U[i][q] = s * up + c * uq; }
      for (let i = 0; i < n; i++) { const vp = V[i][p], vq = V[i][q]; V[i][p] = c * vp - s * vq; V[i][q] = s * vp + c * vq; }
    }
    if (!rotated) break;
  }
  const s = range(n).map(j => norm2(U.map(r => r[j])));
  const order = range(n).sort((i, j) => s[j] - s[i]);
  const Us: Mat = U.map(r => order.map(j => r[j]));
  const Vs: Mat = V.map(r => order.map(j => r[j]));
  const ss = order.map(j => s[j]);
  const tol = ss[0] * 1e-300;
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) Us[i][j] = ss[j] > tol && ss[j] > 0 ? Us[i][j] / ss[j] : 0;
  return { U: Us, s: ss, V: Vs };
}

/** Completes the orthonormal columns of Q (m x k) to an orthogonal m x m matrix */
function complete_basis(Q: Mat, m: number, k: number): Mat {
  const cols: number[][] = range(k).map(j => Q.map(r => r[j]));
  // columns of zero norm (rank deficiency) are replaced as well
  const good: number[][] = cols.filter(c => norm2(c) > 0.5);
  for (let e = 0; e < m && good.length < m; e++) {
    const v = new Array(m).fill(0); v[e] = 1;
    for (let pass = 0; pass < 2; pass++) for (const b of good) { let d = 0; for (let i = 0; i < m; i++) d += v[i] * b[i]; for (let i = 0; i < m; i++) v[i] -= d * b[i]; }
    const nv = norm2(v);
    if (nv > 1e-8) good.push(v.map(x => x / nv));
  }
  return range(m).map(i => good.map(c => c[i]));
}

function svd_mat(M: Mat, m: number, n: number) {
  if (m >= n) { const r = svd_jacobi(M, m, n); return { U: r.U, s: r.s, V: r.V }; }
  const T = range(n).map(i => range(m).map(j => M[j][i]));
  const r = svd_jacobi(T, n, m);
  return { U: r.V, s: r.s, V: r.U };   // M = (T^T) = V s U^T
}

export function svd(a: NDArray, full_matrices: boolean, compute_uv: boolean, hermitian: boolean): NDArray | NDArray[] {
  const { mats, batch, rows: m, cols: n } = matrices(a);
  const dt = float_dtype(asarray(a));
  const k = Math.min(m, n);
  const res = mats.map(M => { check_finite(M); return svd_mat(M, m, n); });
  const S = from_vecs(res.map(r => r.s.slice(0, k)), k, batch, dt);
  if (!compute_uv) return S;
  const U = res.map(r => { const u = r.U.map(row => row.slice(0, k)); return full_matrices ? complete_basis(u, m, k) : u; });
  const Vt = res.map(r => {
    const v = r.V; // n x k  (columns are right singular vectors)
    const V = full_matrices ? complete_basis(v.map(row => row.slice(0, k)), n, k) : v.map(row => row.slice(0, k));
    const cols = V[0] ? V[0].length : 0;
    return range(cols).map(j => range(n).map(i => V[i][j]));
  });
  const uc = full_matrices ? m : k, vr = full_matrices ? n : k;
  return [from_mats(U, m, uc, batch, dt), S, from_mats(Vt, vr, n, batch, dt)];
}
export const svdvals = (a: NDArray) => svd(a, false, false, false) as NDArray;

/** Symmetric eigenproblem with cyclic Jacobi rotations. Uses the lower triangle like numpy. */
function eigh_mat(M: Mat, upper: boolean) {
  const n = M.length;
  const A: Mat = range(n).map(i => range(n).map(j => (upper ? (j >= i ? M[i][j] : M[j][i]) : (j <= i ? M[i][j] : M[j][i]))));
  const V: Mat = range(n).map(i => range(n).map(j => i == j ? 1 : 0));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    let diag_norm = 0; for (let i = 0; i < n; i++) diag_norm += A[i][i] * A[i][i];
    if (off <= 1e-30 * Math.max(diag_norm, 1e-300)) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (A[p][q] === 0) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
      const c = 1 / Math.sqrt(1 + t * t), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const w = range(n).map(i => A[i][i]);
  const order = range(n).sort((i, j) => w[i] - w[j]);
  return { w: order.map(i => w[i]), V: V.map(r => order.map(j => r[j])) };
}

export function eigh(a: NDArray, UPLO: string): NDArray[] {
  if (UPLO !== 'L' && UPLO !== 'U') throw new Error(`UPLO argument must be 'L' or 'U'`);
  const { mats, batch, rows } = square_matrices(a);
  const dt = float_dtype(asarray(a));
  const res = mats.map(m => { check_finite(m); return eigh_mat(m, UPLO === 'U'); });
  return [from_vecs(res.map(r => r.w), rows, batch, dt), from_mats(res.map(r => r.V), rows, rows, batch, dt)];
}
export const eigvalsh = (a: NDArray, UPLO: string) => eigh(a, UPLO)[0];

/** Eigenvalues of a general real matrix by shifted QR iterations on its Hessenberg form */
function hessenberg(M: Mat): Mat {
  const n = M.length;
  const H = M.map(r => r.slice());
  for (let k = 0; k < n - 2; k++) {
    const x = range(n - k - 1).map(i => H[k + 1 + i][k]);
    const alpha = norm2(x);
    if (alpha === 0 || norm2(x.slice(1)) === 0) continue;
    const v = x.slice();
    v[0] += x[0] >= 0 ? alpha : -alpha;
    const vn = norm2(v);
    const u = v.map(t => t / vn);
    // H = P H P with P = I - 2 u u^T acting on rows/cols k+1..n-1
    for (let c = 0; c < n; c++) { let s = 0; for (let i = 0; i < u.length; i++) s += u[i] * H[k + 1 + i][c]; for (let i = 0; i < u.length; i++) H[k + 1 + i][c] -= 2 * u[i] * s; }
    for (let r = 0; r < n; r++) { let s = 0; for (let i = 0; i < u.length; i++) s += H[r][k + 1 + i] * u[i]; for (let i = 0; i < u.length; i++) H[r][k + 1 + i] -= 2 * s * u[i]; }
  }
  return H;
}

function eigvals_mat(M: Mat): Complex[] {
  const n = M.length;
  let H = hessenberg(M);
  const out: Complex[] = [];
  let m = n, iter = 0;
  const eps = 2.2e-16;
  while (m > 0) {
    if (m == 1) { out.push(new Complex(H[0][0], 0)); break; }
    const small = (i: number) => Math.abs(H[i][i - 1]) <= eps * (Math.abs(H[i][i]) + Math.abs(H[i - 1][i - 1]) + 1e-300);
    if (small(m - 1)) { out.push(new Complex(H[m - 1][m - 1], 0)); H = H.slice(0, m - 1).map(r => r.slice(0, m - 1)); m--; iter = 0; continue; }
    if (m == 2 || small(m - 2)) {
      const a = H[m - 2][m - 2], b = H[m - 2][m - 1], c = H[m - 1][m - 2], d = H[m - 1][m - 1];
      const tr = a + d, dt = a * d - b * c, disc = tr * tr / 4 - dt;
      if (disc < 0) { const r = Math.sqrt(-disc); out.push(new Complex(tr / 2, r), new Complex(tr / 2, -r)); }
      else { const r = Math.sqrt(disc); out.push(new Complex(tr / 2 + r, 0), new Complex(tr / 2 - r, 0)); }
      H = H.slice(0, m - 2).map(row => row.slice(0, m - 2)); m -= 2; iter = 0; continue;
    }
    if (++iter > 500) throw new LinAlgError(`Eigenvalues did not converge`);
    // explicit double-shift QR step on the active block
    const a = H[m - 2][m - 2], b = H[m - 2][m - 1], c = H[m - 1][m - 2], d = H[m - 1][m - 1];
    let t = a + d, dd = a * d - b * c;
    if (iter % 11 == 10) { t = 1.5 * Math.abs(H[m - 1][m - 2]) + t; dd += Math.abs(H[m - 2][m - 3] || 0); }
    const H2: Mat = range(m).map(i => range(m).map(j => { let s = 0; for (let k = 0; k < m; k++) s += H[i][k] * H[k][j]; return s; }));
    const Mx: Mat = H2.map((r, i) => r.map((v, j) => v - t * H[i][j] + (i == j ? dd : 0)));
    const { Q } = qr_mat(Mx, m, m);
    const QT = range(m).map(i => range(m).map(j => Q[j][i]));
    const HQ: Mat = range(m).map(i => range(m).map(j => { let s = 0; for (let k = 0; k < m; k++) s += H[i][k] * Q[k][j]; return s; }));
    H = range(m).map(i => range(m).map(j => { let s = 0; for (let k = 0; k < m; k++) s += QT[i][k] * HQ[k][j]; return s; }));
  }
  return out;
}

/** Solves A x = b for complex A (Gaussian elimination with partial pivoting); a zero pivot is replaced by a tiny number (inverse iteration) */
function complex_solve(A: Complex[][], b: Complex[]): Complex[] {
  const n = A.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (M[i][k].abs > M[p][k].abs) p = i;
    [M[k], M[p]] = [M[p], M[k]];
    if (M[k][k].abs < 1e-300) M[k][k] = new Complex(1e-300, 0);
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k].div(M[k][k]);
      for (let j = k; j <= n; j++) M[i][j] = M[i][j].sub(f.mul(M[k][j]));
    }
  }
  const x: Complex[] = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s = s.sub(M[i][j].mul(x[j]));
    x[i] = s.div(M[i][i]);
  }
  return x;
}

/** Unit-norm eigenvector of M for the eigenvalue lambda by inverse iteration. The largest component is made real, like LAPACK. */
function eigenvector(M: Mat, lambda: Complex, seed: number): Complex[] {
  const n = M.length;
  const scale = Math.max(1, ...M.flat().map(Math.abs));
  const shift = new Complex(lambda.re + scale * 1e-10, lambda.im + scale * 1e-11);
  const A = M.map((r, i) => r.map((v, j) => new Complex(v, 0).sub(i == j ? shift : ZERO)));
  let x: Complex[] = range(n).map(i => new Complex(1 + 0.1 * Math.sin(i + 1 + seed), 0.05 * Math.cos(i + seed)));
  for (let it = 0; it < 6; it++) {
    const y = complex_solve(A, x);
    const nv = Math.sqrt(y.reduce((s, z) => s + z.re * z.re + z.im * z.im, 0));
    x = y.map(z => new Complex(z.re / nv, z.im / nv));
  }
  // rotate so that the component with the largest modulus is real
  let big = 0; x.forEach((z, i) => { if (z.abs > x[big].abs) big = i; });
  const phase = x[big].abs === 0 ? ONE : new Complex(x[big].re / x[big].abs, -x[big].im / x[big].abs);
  x = x.map(z => z.mul(phase));
  return x.map(z => lambda.im === 0 ? new Complex(z.re, 0) : z);
}

function from_complex_vecs(vecs: Complex[][], len: number, batch: number[]): NDArray {
  const buffer = vecs.flat();
  return new NDArray(buffer, [...batch, len], complex128);
}
function from_complex_mats(mats: Complex[][][], n: number, batch: number[]): NDArray {
  const buffer: Complex[] = [];
  mats.forEach(m => m.forEach(r => buffer.push(...r)));
  return new NDArray(buffer, [...batch, n, n], complex128);
}

/** numpy.linalg.eig. Eigenvalues and eigenvectors are complex128 arrays (as in numpy) */
export function eig(a: NDArray): NDArray[] {
  const { mats, batch, rows } = square_matrices(a);
  const W: Complex[][] = [], V: Complex[][][] = [];
  mats.forEach(m => {
    check_finite(m);
    const w = eigvals_mat(m);
    const vecs = w.map((lam, i) => eigenvector(m, lam, i));
    // repeated eigenvalues: orthogonalise against earlier vectors of the same eigenvalue
    vecs.forEach((v, i) => {
      for (let j = 0; j < i; j++) if (w[j].sub(w[i]).abs <= 1e-8 * Math.max(1, w[i].abs)) {
        let d: Complex = ZERO;
        v.forEach((z, t) => { d = d.add(vecs[j][t].conj().mul(z)); });
        vecs[i] = v.map((z, t) => z.sub(d.mul(vecs[j][t])));
        const nv = Math.sqrt(vecs[i].reduce((s, z) => s + z.re * z.re + z.im * z.im, 0));
        vecs[i] = vecs[i].map(z => new Complex(z.re / nv, z.im / nv));
      }
    });
    W.push(w); V.push(range(rows).map(i => vecs.map(v => v[i])));
  });
  return [from_complex_vecs(W, rows, batch), from_complex_mats(V, rows, batch)];
}
export function eigvals(a: NDArray): NDArray {
  const { mats, batch, rows } = square_matrices(a);
  return from_complex_vecs(mats.map(m => { check_finite(m); return eigvals_mat(m); }), rows, batch);
}

export function pinv(a: NDArray, rcond: number | NDArray | null, hermitian: boolean, rtol: number | null): NDArray {
  const { mats, batch, rows: m, cols: n } = matrices(a);
  const dt = float_dtype(asarray(a));
  const res = mats.map(M => {
    check_finite(M);
    const { U, s, V } = svd_mat(M, m, n);
    const k = s.length;
    const cutoff = (rtol !== null ? rtol : (rcond !== null ? +asarray(rcond).item() : Math.max(m, n) * 2.220446049250313e-16)) * (s[0] || 0);
    // pinv = V diag(1/s) U^T
    return range(n).map(i => range(m).map(j => { let t = 0; for (let q = 0; q < k; q++) if (s[q] > cutoff) t += V[i][q] * U[j][q] / s[q]; return t; }));
  });
  return from_mats(res, n, m, batch, dt);
}

export function matrix_rank(A: NDArray, tol: number | null, hermitian: boolean, rtol: number | null): NDArray {
  A = asarray(A);
  if (A.shape.length < 2) return new NDArray(new_buffer([Array.from(asarray(A).flat as any).some((x: any) => x != 0) ? 1 : 0], int32), [], int32);
  const S = svdvals(A);
  const k = S.shape[S.shape.length - 1];
  const f = S.flat as any;
  const batch = S.shape.slice(0, -1);
  const [m, n] = A.shape.slice(-2);
  const out = range(prod(batch)).map(b => {
    const smax = k ? f[b * k] : 0;
    const cut = tol !== null ? tol : (rtol !== null ? rtol : Math.max(m, n) * 2.220446049250313e-16) * smax;
    let r = 0; for (let i = 0; i < k; i++) if (f[b * k + i] > cut) r++;
    return r;
  });
  return from_scalars(out, batch, int32);
}

export function cond(x: NDArray, p: any): NDArray {
  x = asarray(x);
  if (p === null || p === 2 || p === -2) {
    const S = svdvals(x);
    const k = S.shape[S.shape.length - 1], f = S.flat as any;
    const batch = S.shape.slice(0, -1);
    return from_scalars(range(prod(batch)).map(b => {
      const mx = f[b * k], mn = f[b * k + k - 1];
      return p === -2 ? mn / mx : mx / mn;
    }), batch, float_dtype(x));
  }
  const n1 = matrix_norm(x, p, false), n2 = matrix_norm(inv(x), p, false);
  return from_scalars(Array.from((n1.flat as any)).map((v: any, i) => v * (n2.flat as any)[i]), n1.shape, float_dtype(x));
}

export function lstsq(a: NDArray, b: NDArray, rcond: number | null): NDArray[] {
  a = asarray(a); b = asarray(b);
  if (a.shape.length != 2) throw new LinAlgError(`${a.shape.length}-dimensional array given. Array must be two-dimensional`);
  const [m, n] = a.shape;
  const vector = b.shape.length == 1;
  const B = vector ? reshape(b, [b.shape[0], 1]) : b;
  if (B.shape[0] != m) throw new LinAlgError(`Incompatible dimensions`);
  const cutoffRel = rcond === null || rcond < 0 ? Math.max(m, n) * 2.220446049250313e-16 : rcond;
  const P = pinv(a, cutoffRel, false, null);
  const x = matmul(P, B);
  const S = svdvals(a);
  const sv = S.flat as any;
  const rank = Array.from(sv as ArrayLike<number>).filter((s: number) => s > cutoffRel * sv[0]).length;
  const dt = float_dtype(a);
  // residuals: only when rank == n and m > n
  const k = B.shape[1];
  let residuals: NDArray;
  if (rank == n && m > n) {
    const R = matmul(a, x), rb = R.flat as any, bf = B.flat as any;
    const res = range(k).map(c => { let s = 0; for (let i = 0; i < m; i++) { const d = bf[i * k + c] - rb[i * k + c]; s += d * d; } return s; });
    residuals = from_vecs([res], k, [], dt);
    if (vector) residuals = reshape(residuals, [1]);
  } else residuals = new NDArray(new_buffer(0, dt), [0], dt);
  return [vector ? reshape(x, [n]) : x, residuals, new NDArray(new_buffer([rank], int32), [], int32), S];
}

export function tensorsolve(a: NDArray, b: NDArray, axes: number[] | null): NDArray {
  a = asarray(a); b = asarray(b);
  const an = a.shape.length;
  if (axes !== null) {
    const ax = axes.map(x => normalize_axis(x, an));
    a = transpose(a, [...range(an).filter(i => !ax.includes(i)), ...ax]);
  }
  const oldshape = a.shape.slice(b.shape.length);
  const p = prod(oldshape);
  const x = solve(reshape(a, [-1, p]), ravel(b));
  return reshape(x, oldshape);
}

export function tensorinv(a: NDArray, ind: number): NDArray {
  a = asarray(a);
  if (ind <= 0) throw new Error(`Invalid ind argument.`);
  const oldshape = a.shape;
  const invshape = [...oldshape.slice(ind), ...oldshape.slice(0, ind)];
  const n = prod(oldshape.slice(ind));
  const a2 = reshape(a, [n, prod(oldshape.slice(0, ind))]);
  return reshape(inv(a2), invshape);
}

// ==============================
//   Norms
// ==============================

export function vector_norm(x: NDArray, axis: number | number[] | null, keepdims: boolean, ord: number): NDArray {
  x = asarray(x);
  const dt = float_dtype(x);
  const nd = x.shape.length;
  const axes = axis === null ? range(nd) : (Array.isArray(axis) ? axis : [axis]).map(a => normalize_axis(a, nd));
  const kept = range(nd).filter(i => !axes.includes(i));
  const t = transpose(x, [...kept, ...axes]);
  const nl = prod(axes.map(a => x.shape[a]));
  const nOut = prod(kept.map(i => x.shape[i]));
  const f = t.flat as any;
  const buffer = new_buffer(nOut, dt);
  for (let o = 0; o < nOut; o++) {
    const lane = Array.from(f.slice(o * nl, (o + 1) * nl) as ArrayLike<any>, (v: any) => v instanceof Complex ? v.abs : Math.abs(+v));
    let r: number;
    if (ord === Infinity) r = lane.length ? Math.max(...lane.map(v => v !== v ? NaN : v)) : 0;
    else if (ord === -Infinity) r = lane.length ? Math.min(...lane) : Infinity;
    else if (ord === 0) r = lane.filter(v => v !== 0).length;
    else if (ord === 1) r = lane.reduce((s, v) => s + v, 0);
    else if (ord === 2) r = Math.sqrt(lane.reduce((s, v) => s + v * v, 0));
    else r = Math.pow(lane.reduce((s, v) => s + Math.pow(v, ord), 0), 1 / ord);
    buffer[o] = r;
  }
  const shape = keepdims ? x.shape.map((d, i) => axes.includes(i) ? 1 : d) : kept.map(i => x.shape[i]);
  return new NDArray(buffer, shape, dt);
}

export function matrix_norm(x: NDArray, ord: number | string, keepdims: boolean): NDArray {
  x = asarray(x);
  if (x.shape.length < 2) throw new Error(`Improper number of dimensions to norm.`);
  const dt = float_dtype(x);
  const { mats, batch, rows, cols } = matrices(x);
  const vals = mats.map(m => {
    const abssum = (rowsFirst: boolean) => {
      const sums = rowsFirst ? m.map(r => r.reduce((s, v) => s + Math.abs(v), 0)) : range(cols).map(j => m.reduce((s, r) => s + Math.abs(r[j]), 0));
      return sums;
    };
    switch (ord) {
      case 'fro': return Math.sqrt(m.flat().reduce((s, v) => s + v * v, 0));
      case 'nuc': return (svdvals(from_mats([m], rows, cols, [], dt)).flat as any).reduce((s: number, v: number) => s + v, 0);
      case 1: return Math.max(...abssum(false));
      case -1: return Math.min(...abssum(false));
      case Infinity: return Math.max(...abssum(true));
      case -Infinity: return Math.min(...abssum(true));
      case 2: case -2: { const sv = Array.from(svdvals(from_mats([m], rows, cols, [], dt)).flat as any) as number[]; return ord === 2 ? sv[0] : sv[sv.length - 1]; }
      default: throw new Error(`Invalid norm order for matrices.`);
    }
  });
  let out = from_scalars(vals, batch, dt);
  if (keepdims) out = reshape(out, [...batch, 1, 1]);
  return out;
}

/** numpy.linalg.norm: vector norms, and matrix norms for 2-D input with ord given or axis a pair */
export function norm(x: NDArray, ord: number | string | null, axis: number | number[] | null, keepdims: boolean): NDArray {
  x = asarray(x);
  const nd = x.shape.length;
  if (axis === null) {
    if (ord === null) return vector_norm(x, null, keepdims, 2);
    if (nd === 1) {
      if (typeof ord === "string") throw new Error(`Invalid norm order '${ord}' for vectors`);
      return vector_norm(x, 0, keepdims, ord as number);
    }
    if (nd === 2) return matrix_norm(x, ord as any, keepdims);
    throw new Error(`Improper number of dimensions to norm.`);
  }
  const axes = Array.isArray(axis) ? axis : [axis];
  if (axes.length === 1) {
    if (typeof ord === "string") throw new Error(`Invalid norm order '${ord}' for vectors`);
    return vector_norm(x, axes[0], keepdims, ord === null ? 2 : ord);
  }
  if (axes.length === 2) {
    const [a0, a1] = axes.map(a => normalize_axis(a, nd));
    if (a0 === a1) throw new Error(`Duplicate axes given.`);
    const moved = transpose(x, [...range(nd).filter(i => i != a0 && i != a1), a0, a1]);
    const r = matrix_norm(moved, ord === null ? 'fro' : ord, false);
    if (!keepdims) return r;
    const shape = x.shape.slice(); shape[a0] = 1; shape[a1] = 1;
    return reshape(r, shape);
  }
  throw new Error(`Improper number of dimensions to norm.`);
}
