//@ts-check
/**
 * numpy.random: legacy `RandomState` (Mersenne Twister, bit-exact streams for the common
 * methods) and the new `Generator` on top of `PCG64` + `SeedSequence`.
 * Distributions whose numpy algorithm is not ported (e.g. BTPE binomial) are statistically
 * correct but do not reproduce numpy's stream.
 */
import { NDArray } from "../NDArray";
import { np } from "../np_circular_import";
import { new_buffer, DType, TYPE_BY_NAME, canonical_dtype } from "../dtypes";

const B = (n: any) => (globalThis as any).BigInt(n);
const M64 = () => B("0xffffffffffffffff");
const M128 = () => B("0xffffffffffffffffffffffffffffffff");

// ---------------------------------------------------------------- helpers
function toShape(size: any): number[] | null {
  if (size === undefined || size === null) return null;
  if (typeof size === 'number') return [size];
  if (size instanceof NDArray) return size.tolist() as any;
  return Array.from(size);
}
function prod(shape: number[]) { let n = 1; for (const s of shape) n *= s; return n; }
function build(size: any, draw: () => number, dtype: DType = 'float64'): any {
  const shape = toShape(size);
  if (shape === null) return draw();
  const n = prod(shape);
  const buf = new_buffer(n, dtype);
  for (let i = 0; i < n; i++) buf[i] = draw();
  return new NDArray(buf, shape, dtype);
}
function arr(a: any): NDArray { return a instanceof NDArray ? a : np.array(a); }
function require_np(): any { return np; }
function isArrayLike(x: any) { return x instanceof NDArray || Array.isArray(x) || ArrayBuffer.isView(x); }
/** Broadcasted scalar-or-array parameter reader */
function param(x: any): (i: number) => number {
  if (isArrayLike(x)) {
    const a = arr(x); const f = a.flat as any;
    return (i) => Number(f[f.length === 1 ? 0 : i % f.length]);
  }
  return () => Number(x);
}
function paramShape(size: any, ...ps: any[]): any {
  if (size !== undefined && size !== null) return size;
  let best: number[] | null = null;
  for (const p of ps) if (isArrayLike(p)) { const s = arr(p).shape; if (best === null || prod(s) > prod(best)) best = s; }
  return best === null ? size : best;
}
/** Draw with array parameters: parameter i-th element for i-th output */
function dist(size: any, params: any[], f: (...p: number[]) => number, dtype: DType = 'float64'): any {
  size = paramShape(size, ...params);
  const rd = params.map(param);
  let i = 0;
  return build(size, () => { const k = i++; return f(...rd.map(r => r(k))); }, dtype);
}

// ---------------------------------------------------------------- MT19937
export class MT19937 {
  key = new Uint32Array(624);
  pos = 624;
  constructor(seed?: any) { this.seed(seed); }
  seed(seed?: any) {
    if (seed === undefined || seed === null) seed = Math.floor(Math.random() * 4294967296);
    if (typeof seed === 'number') {
      if (seed < 0 || seed !== Math.floor(seed)) throw new Error("Seed must be between 0 and 2**32 - 1");
      if (seed >= 4294967296) this.init_by_array([seed >>> 0, Math.floor(seed / 4294967296)]);
      else this.init_genrand(seed >>> 0);
    } else {
      const a = Array.from(arr(seed).flat as any).map(Number);
      this.init_by_array(a);
    }
  }
  init_genrand(s: number) {
    const k = this.key;
    k[0] = s >>> 0;
    for (let i = 1; i < 624; i++) {
      const p = k[i - 1] ^ (k[i - 1] >>> 30);
      k[i] = (Math.imul(1812433253, p) + i) >>> 0;
    }
    this.pos = 624;
  }
  init_by_array(init: number[]) {
    this.init_genrand(19650218);
    const k = this.key, n = init.length;
    let i = 1, j = 0;
    for (let c = Math.max(624, n); c > 0; c--) {
      const p = k[i - 1] ^ (k[i - 1] >>> 30);
      k[i] = ((k[i] ^ Math.imul(p, 1664525)) + init[j] + j) >>> 0;
      i++; j++;
      if (i >= 624) { k[0] = k[623]; i = 1; }
      if (j >= n) j = 0;
    }
    for (let c = 623; c > 0; c--) {
      const p = k[i - 1] ^ (k[i - 1] >>> 30);
      k[i] = ((k[i] ^ Math.imul(p, 1566083941)) - i) >>> 0;
      i++;
      if (i >= 624) { k[0] = k[623]; i = 1; }
    }
    k[0] = 0x80000000;
    this.pos = 624;
  }
  private gen() {
    const k = this.key;
    let kk = 0, y = 0;
    for (; kk < 624 - 397; kk++) {
      y = (k[kk] & 0x80000000) | (k[kk + 1] & 0x7fffffff);
      k[kk] = k[kk + 397] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
    }
    for (; kk < 623; kk++) {
      y = (k[kk] & 0x80000000) | (k[kk + 1] & 0x7fffffff);
      k[kk] = k[kk + (397 - 624)] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
    }
    y = (k[623] & 0x80000000) | (k[0] & 0x7fffffff);
    k[623] = k[396] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
    this.pos = 0;
  }
  next32(): number {
    if (this.pos >= 624) this.gen();
    let y = this.key[this.pos++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  nextDouble(): number {
    const a = this.next32() >>> 5, b = this.next32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }
  hi = 0; lo = 0;
  next64() { this.hi = this.next32(); this.lo = this.next32(); }
}

// ---------------------------------------------------------------- SeedSequence + PCG64
export class SeedSequence {
  entropy: any; spawn_key: number[]; pool_size = 4; n_children_spawned = 0;
  pool: number[] = [0, 0, 0, 0];
  constructor(entropy?: any, spawn_key: number[] = [], pool_size = 4, n_children_spawned = 0) {
    if (entropy === undefined || entropy === null) {
      entropy = Math.floor(Math.random() * 4294967296) * 4294967296 + Math.floor(Math.random() * 4294967296);
    }
    this.entropy = entropy; this.spawn_key = Array.from(spawn_key); this.pool_size = pool_size;
    this.n_children_spawned = n_children_spawned;
    this.mix_entropy();
  }
  private static int_to_words(v: any): number[] {
    let x = B(v); const out: number[] = [];
    const m = B(0xffffffff);
    if (x === B(0)) return [0];
    while (x > B(0)) { out.push(Number(x & m)); x = x >> B(32); }
    return out;
  }
  private assemble(): number[] {
    let run: number[] = [];
    const e = this.entropy;
    if (typeof e === 'number' || typeof e === 'bigint') run = SeedSequence.int_to_words(e);
    else for (const v of Array.from(arr(e).flat as any)) run = run.concat(SeedSequence.int_to_words(v));
    while (run.length < this.pool_size) run.push(0);
    for (const k of this.spawn_key) run = run.concat(SeedSequence.int_to_words(k));
    return run;
  }
  private mix_entropy() {
    const INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
    const MIX_L = 0xca01f9dd, MIX_R = 0x4973f715;
    let hc = INIT_A;
    const hashmix = (v: number) => {
      v = (v ^ hc) >>> 0; hc = Math.imul(hc, MULT_A) >>> 0;
      v = Math.imul(v, hc) >>> 0;
      return (v ^ (v >>> 16)) >>> 0;
    };
    const mix = (x: number, y: number) => {
      const r = (Math.imul(MIX_L, x) - Math.imul(MIX_R, y)) >>> 0;
      return (r ^ (r >>> 16)) >>> 0;
    };
    const ent = this.assemble(), pool = this.pool = new Array(this.pool_size).fill(0);
    for (let i = 0; i < this.pool_size; i++) pool[i] = hashmix(i < ent.length ? ent[i] : 0);
    for (let s = 0; s < this.pool_size; s++) for (let d = 0; d < this.pool_size; d++)
      if (s !== d) pool[d] = mix(pool[d], hashmix(pool[s]));
    for (let s = this.pool_size; s < ent.length; s++)
      for (let d = 0; d < this.pool_size; d++) pool[d] = mix(pool[d], hashmix(ent[s]));
  }
  generate_state(n_words: number, dtype: DType | 'uint32' | 'uint64' = 'uint32'): NDArray {
    const words = dtype === 'uint64' ? n_words * 2 : n_words;
    const out = new Array(words);
    let hc = 0x8b51f9dd;
    const MULT_B = 0x58f38ded;
    for (let i = 0; i < words; i++) {
      let d = this.pool[i % this.pool_size];
      d = (d ^ hc) >>> 0; hc = Math.imul(hc, MULT_B) >>> 0;
      d = Math.imul(d, hc) >>> 0; d = (d ^ (d >>> 16)) >>> 0;
      out[i] = d;
    }
    if (dtype === 'uint64') {
      const o = new_buffer(n_words, 'uint64');
      for (let i = 0; i < n_words; i++) o[i] = out[2 * i + 1] * 4294967296 + out[2 * i];
      (o as any).__words = out;
      return new NDArray(o, [n_words], 'uint64');
    }
    const o = new_buffer(n_words, 'uint32');
    for (let i = 0; i < n_words; i++) o[i] = out[i];
    return new NDArray(o, [n_words], 'uint32');
  }
  /** words as uint32 pairs, for exact uint64 values */
  generate_state_words(n_words: number): number[] {
    const a = this.generate_state(n_words * 2, 'uint32').flat as any;
    return Array.from(a);
  }
  spawn(n: number): SeedSequence[] {
    const out: SeedSequence[] = [];
    for (let i = 0; i < n; i++)
      out.push(new SeedSequence(this.entropy, this.spawn_key.concat([this.n_children_spawned + i]), this.pool_size));
    this.n_children_spawned += n;
    return out;
  }
}

export class PCG64 {
  state: any; inc: any;
  seed_seq: SeedSequence;
  hi = 0; lo = 0;
  private has32 = false; private stash = 0;
  constructor(seed?: any) {
    this.seed_seq = seed instanceof SeedSequence ? seed : new SeedSequence(seed);
    const w = this.seed_seq.generate_state_words(4);
    const u64 = (i: number) => (B(w[2 * i + 1]) << B(32)) | B(w[2 * i]);
    const initstate = (u64(0) << B(64)) | u64(1), initseq = (u64(2) << B(64)) | u64(3);
    this.state = B(0); this.inc = ((initseq << B(1)) | B(1)) & M128();
    this.step(); this.state = (this.state + initstate) & M128(); this.step();
  }
  private static MULT: any;
  private step() {
    if (!PCG64.MULT) PCG64.MULT = (B("2549297995355413924") << B(64)) + B("4865540595714422341");
    this.state = (this.state * PCG64.MULT + this.inc) & PCG64MASK();
  }
  next64() {
    this.step();
    const s = this.state, m = M64();
    const hi = s >> B(64), lo = s & m;
    const x = hi ^ lo, rot = Number(s >> B(122));
    const r = rot === 0 ? x : (((x >> B(rot)) | (x << B(64 - rot))) & m);
    this.hi = Number(r >> B(32)); this.lo = Number(r & B(0xffffffff));
  }
  next32(): number {
    if (this.has32) { this.has32 = false; return this.stash; }
    this.next64();
    this.has32 = true; this.stash = this.hi;
    return this.lo;
  }
  nextDouble(): number {
    this.next64();
    return (this.hi * 2097152 + (this.lo >>> 11)) / 9007199254740992;
  }
  jumped(): PCG64 { throw new Error("jumped is not implemented"); }
}
function PCG64MASK() { return M128(); }

// ---------------------------------------------------------------- shared math
function loggam(x: number): number {
  if (x === 1 || x === 2) return 0;
  const a = [8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04, -5.952380952380952e-04,
    8.417508417508418e-04, -1.917526917526918e-03, 6.410256410256410e-03, -2.955065359477124e-02,
    1.796443723688307e-01, -1.39243221690590e+00];
  let x0 = x, n = 0;
  if (x <= 7) { n = Math.floor(7 - x); x0 = x + n; }
  const x2 = 1 / (x0 * x0);
  let gl0 = a[9];
  for (let k = 8; k >= 0; k--) gl0 = gl0 * x2 + a[k];
  let gl = gl0 / x0 + 0.5 * Math.log(2 * Math.PI) + (x0 - 0.5) * Math.log(x0) - x0;
  if (x <= 7) for (let k = 1; k <= n; k++) { gl -= Math.log(x0 - 1); x0 -= 1; }
  return gl;
}

/** Interface the distributions need from a bit generator */
interface Core {
  next32(): number;
  nextDouble(): number;
}

// Ziggurat tables, computed as in numpy's create_ziggurat_tables.py
let ZN: { ki: Float64Array, wi: Float64Array, fi: Float64Array } | null = null;
let ZE: { ke: Float64Array, we: Float64Array, fe: Float64Array } | null = null;
const NOR_R = 3.6541528853610088, NOR_INV_R = 0.27366123732975828, NOR_AREA = 0.00492867323399;
const EXP_R = 7.69711747013104972, EXP_AREA = 0.0039496598225815571993;
function zig_normal_tables() {
  if (ZN) return ZN;
  const N = 256, b = Math.pow(2, 52);
  const f = (x: number) => Math.exp(-0.5 * x * x);
  const D = new Float64Array(N + 1); // descending layer edges
  D[0] = NOR_AREA / f(NOR_R);
  D[1] = NOR_R;
  for (let i = 2; i < N; i++) D[i] = Math.sqrt(-2 * Math.log(f(D[i - 1]) + NOR_AREA / D[i - 1]));
  D[N] = 0;
  // numpy stores layer edges in ascending order: X[0] is the base strip, X[k] = D[256 - k]
  const X = new Float64Array(N + 1);
  X[0] = D[0];
  for (let k = 1; k < N; k++) X[k] = D[N - k];
  const ki = new Float64Array(N), wi = new Float64Array(N), fi = new Float64Array(N);
  ki[0] = Math.floor((NOR_R / X[0]) * b);
  ki[1] = 0;
  for (let k = 2; k < N; k++) ki[k] = Math.floor((X[k - 1] / X[k]) * b);
  for (let k = 0; k < N; k++) { wi[k] = X[k] / b; fi[k] = f(X[k]); }
  fi[0] = 1;
  return ZN = { ki, wi, fi };
}
function zig_exp_tables() {
  if (ZE) return ZE;
  const N = 256, b = Math.pow(2, 53);
  const f = (x: number) => Math.exp(-x);
  const D = new Float64Array(N + 1); // descending layer edges
  D[0] = EXP_AREA / f(EXP_R);
  D[1] = EXP_R;
  for (let i = 2; i < N; i++) D[i] = -Math.log(f(D[i - 1]) + EXP_AREA / D[i - 1]);
  D[N] = 0;
  // numpy stores layer edges in ascending order: X[0] is the base strip, X[k] = D[256 - k]
  const X = new Float64Array(N + 1);
  X[0] = D[0];
  for (let k = 1; k < N; k++) X[k] = D[N - k];
  const ke = new Float64Array(N), we = new Float64Array(N), fe = new Float64Array(N);
  ke[0] = Math.floor((EXP_R / X[0]) * b);
  ke[1] = 0;
  for (let k = 2; k < N; k++) ke[k] = Math.floor((X[k - 1] / X[k]) * b);
  for (let k = 0; k < N; k++) { we[k] = X[k] / b; fe[k] = f(X[k]); }
  fe[0] = 1;
  return ZE = { ke, we, fe };
}

// ---------------------------------------------------------------- RandomState (legacy)
export class RandomState {
  _bitgen: MT19937;
  has_gauss = false; gauss = 0;
  constructor(seed?: any) { this._bitgen = seed instanceof MT19937 ? seed : new MT19937(seed); }
  seed(seed?: any) { this._bitgen.seed(seed); this.has_gauss = false; this.gauss = 0; }
  private rd() { return this._bitgen.nextDouble(); }
  /** legacy polar method */
  _gauss(): number {
    if (this.has_gauss) { this.has_gauss = false; return this.gauss; }
    let x1, x2, r2;
    do { x1 = 2 * this.rd() - 1; x2 = 2 * this.rd() - 1; r2 = x1 * x1 + x2 * x2; } while (r2 >= 1 || r2 === 0);
    const f = Math.sqrt(-2 * Math.log(r2) / r2);
    this.gauss = f * x1; this.has_gauss = true;
    return f * x2;
  }
  _std_exp() { return -Math.log(1 - this.rd()); }
  _std_gamma(shape: number): number {
    if (shape === 1) return this._std_exp();
    if (shape === 0) return 0;
    if (shape < 1) {
      for (;;) {
        const U = this.rd(), V = this._std_exp();
        if (U <= 1 - shape) { const X = Math.pow(U, 1 / shape); if (X <= V) return X; }
        else {
          const Y = -Math.log((1 - U) / shape), X = Math.pow(1 - shape + shape * Y, 1 / shape);
          if (X <= Y + V) return X;
        }
      }
    }
    const b = shape - 1 / 3, c = 1 / Math.sqrt(9 * b);
    for (;;) {
      let X, V;
      do { X = this._gauss(); V = 1 + c * X; } while (V <= 0);
      V = V * V * V;
      const U = this.rd();
      if (U < 1 - 0.0331 * (X * X) * (X * X)) return b * V;
      if (Math.log(U) < 0.5 * X * X + b * (1 - V + Math.log(V))) return b * V;
    }
  }
  _beta(a: number, b: number): number {
    if (a <= 1 && b <= 1) {
      for (;;) {
        const U = this.rd(), V = this.rd();
        const X = Math.pow(U, 1 / a), Y = Math.pow(V, 1 / b), XpY = X + Y;
        if (XpY <= 1 && U + V > 0) {
          if (XpY > 0) return X / XpY;
          const lx = Math.log(U) / a, ly = Math.log(V) / b, lm = lx > ly ? lx : ly;
          return Math.exp(lx - lm) / (Math.exp(lx - lm) + Math.exp(ly - lm));
        }
      }
    }
    const Ga = this._std_gamma(a), Gb = this._std_gamma(b);
    return Ga / (Ga + Gb);
  }
  _poisson(lam: number): number {
    if (lam >= 10) {
      const slam = Math.sqrt(lam), loglam = Math.log(lam), b = 0.931 + 2.53 * slam;
      const a = -0.059 + 0.02483 * b, invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
      for (;;) {
        const U = this.rd() - 0.5, V = this.rd(), us = 0.5 - Math.abs(U);
        const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
        if (us >= 0.07 && V <= vr) return k;
        if (k < 0 || (us < 0.013 && V > us)) continue;
        if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - loggam(k + 1)) return k;
      }
    }
    if (lam === 0) return 0;
    const enlam = Math.exp(-lam);
    let X = 0, prod = 1;
    for (;;) { prod *= this.rd(); if (prod > enlam) X++; else return X; }
  }
  _binomial(n: number, p: number): number {
    if (n === 0 || p === 0) return 0;
    if (p === 1) return n;
    const q = p <= 0.5 ? p : 1 - p;
    let x = 0;
    if (n * q <= 30) { // inversion
      const qn = Math.exp(n * Math.log(1 - q)), g = q / (1 - q), a = (n + 1) * g;
      for (;;) {
        x = 0; let px = qn, U = this.rd(), ok = true;
        while (U > px) {
          x++;
          if (x > 110) { ok = false; break; }
          U -= px; px = ((a / x) - g) * px;
        }
        if (ok) break;
      }
    } else { // exact but not numpy's BTPE stream
      for (let i = 0; i < n; i++) if (this.rd() < q) x++;
    }
    return p <= 0.5 ? x : n - x;
  }
  /** unsigned integer in [0, max] by masked rejection (legacy) */
  _interval(max: number): number {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    if (max > 0xffffffff) {
      const hi = Math.floor(max / 4294967296);
      let m = hi; m |= m >>> 1; m |= m >>> 2; m |= m >>> 4; m |= m >>> 8; m |= m >>> 16;
      for (;;) {
        const h = (this._bitgen.next32() & m) >>> 0, l = this._bitgen.next32();
        const v = h * 4294967296 + l;
        if (v <= max) return v;
      }
    }
    let v: number;
    while ((v = (this._bitgen.next32() & mask) >>> 0) > max);
    return v;
  }

  random_sample(size?: any) { return build(size, () => this.rd()); }
  random(size?: any) { return this.random_sample(size); }
  ranf(size?: any) { return this.random_sample(size); }
  sample(size?: any) { return this.random_sample(size); }
  rand(...dims: any[]) { return dims.length === 0 ? this.rd() : this.random_sample(dims.length === 1 && isArrayLike(dims[0]) ? dims[0] : dims); }
  randn(...dims: any[]) { return dims.length === 0 ? this._gauss() : this.standard_normal(dims.length === 1 && isArrayLike(dims[0]) ? dims[0] : dims); }
  standard_normal(size?: any) { return build(size, () => this._gauss()); }
  normal(loc: any = 0, scale: any = 1, size?: any) { return dist(size, [loc, scale], (l, s) => l + s * this._gauss()); }
  uniform(low: any = 0, high: any = 1, size?: any) { return dist(size, [low, high], (l, h) => l + (h - l) * this.rd()); }
  standard_exponential(size?: any) { return build(size, () => this._std_exp()); }
  exponential(scale: any = 1, size?: any) { return dist(size, [scale], (s) => s * this._std_exp()); }
  standard_gamma(shape: any, size?: any) { return dist(size, [shape], (s) => this._std_gamma(s)); }
  gamma(shape: any, scale: any = 1, size?: any) { return dist(size, [shape, scale], (a, s) => s * this._std_gamma(a)); }
  beta(a: any, b: any, size?: any) { return dist(size, [a, b], (x, y) => this._beta(x, y)); }
  chisquare(df: any, size?: any) { return dist(size, [df], (d) => 2 * this._std_gamma(d / 2)); }
  noncentral_chisquare(df: any, nonc: any, size?: any) {
    return dist(size, [df, nonc], (d, nc) => {
      if (nc === 0) return 2 * this._std_gamma(d / 2);
      if (d > 1) { const e = 2 * this._std_gamma((d - 1) / 2), n = this._gauss() + Math.sqrt(nc); return e + n * n; }
      const i = this._poisson(nc / 2); return 2 * this._std_gamma((d + 2 * i) / 2);
    });
  }
  f(dfnum: any, dfden: any, size?: any) {
    return dist(size, [dfnum, dfden], (n, d) => (2 * this._std_gamma(n / 2) * d) / (2 * this._std_gamma(d / 2) * n));
  }
  standard_cauchy(size?: any) { return build(size, () => this._gauss() / this._gauss()); }
  standard_t(df: any, size?: any) {
    return dist(size, [df], (d) => { const num = this._gauss(), den = this._std_gamma(d / 2); return Math.sqrt(d / 2) * num / Math.sqrt(den); });
  }
  laplace(loc: any = 0, scale: any = 1, size?: any) {
    return dist(size, [loc, scale], (l, s) => {
      const U = this.rd();
      return U >= 0.5 ? l - s * Math.log(2 - U - U) : U > 0 ? l + s * Math.log(U + U) : this.rd() >= 0 ? l - s * Math.log(2 - U - U) : 0;
    });
  }
  gumbel(loc: any = 0, scale: any = 1, size?: any) {
    return dist(size, [loc, scale], (l, s) => { const U = 1 - this.rd(); return l - s * Math.log(-Math.log(U)); });
  }
  logistic(loc: any = 0, scale: any = 1, size?: any) {
    return dist(size, [loc, scale], (l, s) => { const U = this.rd(); return l + s * Math.log(U / (1 - U)); });
  }
  lognormal(mean: any = 0, sigma: any = 1, size?: any) { return dist(size, [mean, sigma], (m, s) => Math.exp(m + s * this._gauss())); }
  rayleigh(scale: any = 1, size?: any) { return dist(size, [scale], (s) => s * Math.sqrt(2 * this._std_exp())); }
  wald(mean: any, scale: any, size?: any) {
    return dist(size, [mean, scale], (mu, l) => {
      const mu2l = mu / (2 * l), Y0 = this._gauss(), Y = mu * Y0 * Y0, X = mu + mu2l * (Y - Math.sqrt(4 * l * Y + Y * Y));
      return this.rd() <= mu / (mu + X) ? X : mu * mu / X;
    });
  }
  weibull(a: any, size?: any) { return dist(size, [a], (x) => x === 0 ? 0 : Math.pow(this._std_exp(), 1 / x)); }
  pareto(a: any, size?: any) { return dist(size, [a], (x) => Math.exp(this._std_exp() / x) - 1); }
  power(a: any, size?: any) { return dist(size, [a], (x) => Math.pow(1 - Math.exp(-this._std_exp()), 1 / x)); }
  triangular(left: any, mode: any, right: any, size?: any) {
    return dist(size, [left, mode, right], (l, m, r) => {
      const base = r - l, ml = m - l, leftprop = ml / base, U = this.rd();
      if (U <= leftprop) return l + Math.sqrt(U * base * ml);
      return r - Math.sqrt((1 - U) * base * (r - m));
    });
  }
  poisson(lam: any = 1, size?: any) { return dist(size, [lam], (l) => this._poisson(l), 'int64'); }
  binomial(n: any, p: any, size?: any) { return dist(size, [n, p], (a, b) => this._binomial(a, b), 'int64'); }
  geometric(p: any, size?: any) {
    return dist(size, [p], (q) => q >= 0.333333333333333333 ? this._geom_search(q) : this._geom_inv(q), 'int64');
  }
  private _geom_search(p: number) {
    let X = 1, sum = p, prod = p; const q = 1 - p, U = this.rd();
    while (U > sum) { prod *= q; sum += prod; X++; }
    return X;
  }
  private _geom_inv(p: number) { return Math.ceil(-this._std_exp() / Math.log(1 - p)); }
  negative_binomial(n: any, p: any, size?: any) {
    return dist(size, [n, p], (a, q) => this._poisson(this._std_gamma(a) * ((1 - q) / q)), 'int64');
  }
  zipf(a: any, size?: any) {
    return dist(size, [a], (al) => {
      const am1 = al - 1, b = Math.pow(2, am1);
      for (;;) {
        const U = 1 - this.rd(), V = this.rd(), X = Math.floor(Math.pow(U, -1 / am1));
        if (X > 9223372036854775807 || X < 1) continue;
        const T = Math.pow(1 + 1 / X, am1);
        if (V * X * (T - 1) / (b - 1) <= T / b) return X;
      }
    }, 'int64');
  }
  logseries(p: any, size?: any) {
    return dist(size, [p], (pp) => {
      const r = Math.log1p(-pp);
      for (;;) {
        const V = this.rd();
        if (V >= pp) return 1;
        const U = this.rd(), q = -Math.expm1(r * U);
        if (V <= q * q) { const r2 = Math.floor(1 + Math.log(V) / Math.log(q)); if (r2 < 1 || V === 0) continue; return r2; }
        if (V >= q) return 1;
        return 2;
      }
    }, 'int64');
  }
  vonmises(mu: any, kappa: any, size?: any) {
    return dist(size, [mu, kappa], (m, k) => {
      if (Number.isNaN(k)) return NaN;
      if (k < 1e-8) return Math.PI * (2 * this.rd() - 1);
      let s: number;
      if (k < 1e-5) s = 1 / k + k;
      else if (k <= 1e6) { const r = 1 + Math.sqrt(1 + 4 * k * k), rho = (r - Math.sqrt(2 * r)) / (2 * k); s = (1 + rho * rho) / (2 * rho); }
      else { return m + Math.sqrt(1 / k) * this._gauss(); }
      let W: number;
      for (;;) {
        const U = this.rd(), Z = Math.cos(Math.PI * U); W = (1 + s * Z) / (s + Z);
        const Y = k * (s - W), V = this.rd();
        if (Y * (2 - Y) - V >= 0 || Math.log(Y / V) + 1 - Y >= 0) break;
      }
      const U = this.rd();
      let res = Math.acos(W);
      if (U < 0.5) res = -res;
      res += m;
      const neg = res < 0;
      res = Math.abs(res) % (2 * Math.PI);
      if (res > Math.PI) res -= 2 * Math.PI;
      return neg ? -res : res;
    });
  }
  hypergeometric(ngood: any, nbad: any, nsample: any, size?: any) {
    return dist(size, [ngood, nbad, nsample], (g, b, s) => {
      // simple urn simulation (exact distribution; not numpy's stream)
      let good = g, tot = g + b, x = 0;
      for (let i = 0; i < s; i++) { if (this.rd() * tot < good) { x++; good--; } tot--; }
      return x;
    }, 'int64');
  }
  multinomial(n: number, pvals: any, size?: any) {
    const p = Array.from(arr(pvals).flat as any).map(Number), k = p.length;
    const shape = toShape(size) || [];
    const cnt = prod(shape), out = new_buffer(cnt * k, 'int64');
    for (let t = 0; t < cnt; t++) {
      let remaining = n, rest = 1;
      for (let j = 0; j < k - 1; j++) {
        const pj = rest > 0 ? Math.min(1, p[j] / rest) : 0;
        const v = remaining > 0 ? this._binomial(remaining, pj) : 0;
        out[t * k + j] = v; remaining -= v; rest -= p[j];
      }
      out[t * k + k - 1] = remaining;
    }
    return new NDArray(out, shape.concat([k]), 'int64');
  }
  multivariate_normal(mean: any, cov: any, size?: any) {
    const np = require_np();
    const m = arr(mean), c = arr(cov), k = m.shape[0];
    const [u, s, vt] = np.linalg.svd(c);
    const A = np.matmul(np.multiply(np.reshape(np.sqrt(s), [k, 1]), vt) as any, np.eye(k));
    const shape = toShape(size) || [];
    const z = this.standard_normal(shape.concat([k]));
    return np.add(np.matmul(z, A), m);
  }
  dirichlet(alpha: any, size?: any) {
    const np = require_np(), a = Array.from(arr(alpha).flat as any).map(Number), k = a.length;
    const shape = toShape(size) || [], cnt = prod(shape), out = new_buffer(cnt * k, 'float64');
    for (let t = 0; t < cnt; t++) {
      let sum = 0;
      for (let j = 0; j < k; j++) { const v = this._std_gamma(a[j]); out[t * k + j] = v; sum += v; }
      for (let j = 0; j < k; j++) out[t * k + j] /= sum;
    }
    return new NDArray(out, shape.concat([k]), 'float64');
  }

  randint(low: any, high?: any, size?: any, dtype: any = 'int64') {
    if (high === undefined || high === null) { high = low; low = 0; }
    dtype = canonical_dtype(dtype === Number ? 'int64' : dtype === Boolean ? 'bool' : dtype);
    const lows = isArrayLike(low), highs = isArrayLike(high);
    const t = TYPE_BY_NAME[dtype];
    if (t.kind !== 'i' && t.kind !== 'u' && t.kind !== 'b') throw new TypeError(`Unsupported dtype '${dtype}' for randint`);
    const bits = t.kind === 'b' ? 8 : t.bits;
    const min = t.kind === 'u' || t.kind === 'b' ? 0 : -Math.pow(2, bits - 1);
    const max = t.kind === 'b' ? 1 : t.kind === 'u' ? Math.pow(2, bits) - 1 : Math.pow(2, bits - 1) - 1;
    const rl = param(low), rh = param(high);
    const chk = (l: number, h: number) => {
      if (l < min) throw new Error(`low is out of bounds for ${dtype}`);
      if (h - 1 > max) throw new Error(`high is out of bounds for ${dtype}`);
      if (l >= h) throw new Error(`low >= high`);
    };
    let bcnt = 0, buf = 0;
    const nextBuf = (width: number) => {
      const per = width === 8 ? 4 : 2;
      if (bcnt === 0) { buf = this._bitgen.next32(); bcnt = per - 1; } else { buf = buf >>> width; bcnt--; }
      return buf & (width === 8 ? 0xff : 0xffff);
    };
    let k = 0;
    const draw = () => {
      const i = k++;
      const l = rl(i), h = rh(i);
      chk(l, h);
      const rng = h - 1 - l;
      if (rng === 0) return l;
      let v: number;
      if (bits <= 16 && rng <= 0xffff && (bits === 8 || bits === 16)) {
        const width = bits;
        let mask = rng; mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8;
        if (rng === (width === 8 ? 0xff : 0xffff)) return l + nextBuf(width);
        while ((v = nextBuf(width) & mask) > rng);
        return l + v;
      }
      return l + this._interval(rng);
    };
    const out = build(size === undefined && (lows || highs) ? paramShape(size, low, high) : size, draw, dtype);
    return out;
  }
  random_integers(low: any, high?: any, size?: any) {
    if (high === undefined) { high = low; low = 1; }
    return this.randint(low, high + 1, size);
  }
  bytes(length: number) {
    const out = new Uint8Array(length);
    let i = 0;
    while (i < length) { let v = this._bitgen.next32(); for (let j = 0; j < 4 && i < length; j++, v >>>= 8) out[i++] = v & 255; }
    return out;
  }
  shuffle(x: any) {
    const np = require_np();
    if (!(x instanceof NDArray)) {
      if (Array.isArray(x)) { for (let i = x.length - 1; i > 0; i--) { const j = this._interval(i); const t = x[i]; x[i] = x[j]; x[j] = t; } return; }
      throw new TypeError("shuffle needs an array or list");
    }
    if (x.shape.length === 0) throw new Error("shuffle on a 0-d array");
    const n = x.shape[0];
    const perm: number[] = []; for (let i = 0; i < n; i++) perm.push(i);
    for (let i = n - 1; i > 0; i--) { const j = this._interval(i); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    np.copyto(x, np.take(np.copy(x), perm, 0));
  }
  permutation(x: any) {
    const np = require_np();
    if (typeof x === 'number') { const a = np.arange(x); this.shuffle(a); return a; }
    const a = np.copy(np.array(x));
    this.shuffle(a);
    return a;
  }
  choice(a: any, size?: any, replace: boolean = true, p: any = null) {
    const np = require_np();
    let pop: NDArray | null = null, popN: number;
    if (typeof a === 'number') popN = a;
    else { pop = arr(a); if (pop.shape.length === 0) throw new Error("a must be 1-dimensional or an integer"); popN = pop.shape[0]; }
    if (popN <= 0) throw new Error("a must be a positive integer unless no samples are taken");
    const shape = toShape(size);
    const n = shape === null ? 1 : prod(shape);
    let idx: number[];
    if (p !== null && p !== undefined) {
      const pr = Array.from(arr(p).flat as any).map(Number);
      if (pr.length !== popN) throw new Error("'a' and 'p' must have same size");
      const tot = pr.reduce((x, y) => x + y, 0);
      if (Math.abs(tot - 1) > Math.sqrt(2.220446049250313e-16)) throw new Error("probabilities do not sum to 1");
      if (pr.some(v => v < 0)) throw new Error("probabilities are not non-negative");
      if (!replace) {
        const nnz = pr.filter(v => v > 0).length;
        if (n > nnz) throw new Error("Fewer non-zero entries in p than size");
      }
      const search = (pp: number[], u: number) => {
        const cdf: number[] = []; let s = 0;
        for (const v of pp) { s += v; cdf.push(s); }
        for (let i = 0; i < cdf.length; i++) cdf[i] /= s;
        let lo = 0, hi = cdf.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] <= u) lo = mid + 1; else hi = mid; }
        return lo;
      };
      if (replace) {
        const us: number[] = []; for (let i = 0; i < n; i++) us.push(this.rd());
        idx = us.map(u => search(pr, u));
      } else {
        const pp = pr.slice(), found: number[] = [];
        while (found.length < n) {
          const need = n - found.length, xs: number[] = [];
          for (let i = 0; i < need; i++) xs.push(this.rd());
          for (const f of found) pp[f] = 0;
          const nw = xs.map(u => search(pp, u));
          const seen: { [k: number]: number } = {}, uniq: number[] = [];
          nw.forEach((v, i) => { if (!(v in seen)) { seen[v] = i; uniq.push(i); } });
          uniq.sort((x, y) => x - y);
          for (const i of uniq) found.push(nw[i]);
        }
        idx = found;
      }
    } else if (replace) {
      idx = []; for (let i = 0; i < n; i++) idx.push(this._interval(popN - 1));
    } else {
      if (n > popN) throw new Error("Cannot take a larger sample than population when 'replace=False'");
      const perm = this.permutation(popN).flat as any;
      idx = Array.from(perm as ArrayLike<number>).slice(0, n);
    }
    const sel = (i: number) => i;
    if (pop === null) {
      if (shape === null) return idx[0];
      const buf = new_buffer(n, 'int64'); for (let i = 0; i < n; i++) buf[i] = idx[i];
      return new NDArray(buf, shape, 'int64');
    }
    const tb = new_buffer(n, "int64"); for (let i = 0; i < n; i++) tb[i] = idx[i];
    const taken = np.take(pop, new NDArray(tb, [n], "int64"), 0);
    if (shape === null) return np.take(pop, idx[0], 0);
    return np.reshape(taken, shape.concat(pop.shape.slice(1)));
  }
  get_state(legacy: boolean = true): any {
    const b = this._bitgen;
    const key = new_buffer(624, 'uint32'); for (let i = 0; i < 624; i++) key[i] = b.key[i];
    const k = new NDArray(key, [624], 'uint32');
    if (!legacy) return { bit_generator: 'MT19937', state: { key: k, pos: b.pos }, has_gauss: this.has_gauss ? 1 : 0, gauss: this.gauss };
    return ['MT19937', k, b.pos, this.has_gauss ? 1 : 0, this.gauss];
  }
  set_state(state: any) {
    const b = this._bitgen;
    let key: any, pos: number, hg = 0, g = 0;
    if (Array.isArray(state)) { key = state[1]; pos = state[2]; hg = state[3] || 0; g = state[4] || 0; }
    else { key = state.state.key; pos = state.state.pos; hg = state.has_gauss; g = state.gauss; }
    const kk = Array.from(arr(key).flat as any);
    for (let i = 0; i < 624; i++) b.key[i] = Number(kk[i]);
    b.pos = pos; this.has_gauss = !!hg; this.gauss = g;
  }
  getstate() { return this.get_state(); }
  setstate(s: any) { return this.set_state(s); }
}

// ---------------------------------------------------------------- Generator
export class Generator {
  bit_generator: PCG64 | MT19937;
  constructor(bit_generator?: any) {
    this.bit_generator = bit_generator instanceof Generator ? bit_generator.bit_generator : (bit_generator || new PCG64());
  }
  private get g(): any { return this.bit_generator; }
  private rd() { return this.g.nextDouble(); }
  // --- bounded integers (Lemire) ---
  private lemire32(rng: number): number { // value in [0, rng]
    if (rng === 0xffffffff) return this.g.next32();
    const ex = rng + 1;
    const mul = (a: number, b: number) => { // 32x32 -> [hi, lo]
      const a0 = a & 0xffff, a1 = a >>> 16, b0 = b & 0xffff, b1 = b >>> 16;
      const p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
      const mid = p01 + p10 + Math.floor(p00 / 65536);
      return [p11 + Math.floor(mid / 65536), ((mid % 65536) * 65536 + (p00 % 65536)) >>> 0];
    };
    let [hi, lo] = mul(this.g.next32(), ex);
    if (lo < ex) {
      const thr = (0xffffffff - rng) % ex;
      while (lo < thr) [hi, lo] = mul(this.g.next32(), ex);
    }
    return hi;
  }
  private lemire64(rng: number): number {
    if (rng <= 0xffffffff) return this.lemire32(rng);
    const R = B(rng) , ex = R + B(1), mask = M64();
    const draw = () => { this.g.next64(); return (B(this.g.hi) << B(32)) | B(this.g.lo); };
    let m = draw() * ex, lo = m & mask;
    if (lo < ex) {
      const thr = (mask - R) % ex;
      while (lo < thr) { m = draw() * ex; lo = m & mask; }
    }
    return Number(m >> B(64));
  }
  _interval(max: number): number { return max === 0 ? 0 : this.lemire64(max); }
  /** numpy's random_interval (masked rejection), used by shuffle */
  _masked(max: number): number {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    let v: number;
    while ((v = (this.g.next32() & mask) >>> 0) > max);
    return v;
  }

  random(size?: any, dtype: DType = 'float64') { return build(size, () => this.rd(), dtype); }
  integers(low: any, high?: any, size?: any, dtype: any = 'int64', endpoint: boolean = false) {
    if (high === undefined || high === null) { high = low; low = 0; }
    dtype = canonical_dtype(dtype === Number ? 'int64' : dtype === Boolean ? 'bool' : dtype);
    const t = TYPE_BY_NAME[dtype];
    if (t.kind !== 'i' && t.kind !== 'u' && t.kind !== 'b') throw new TypeError(`Unsupported dtype '${dtype}' for integers`);
    const bits = t.kind === 'b' ? 8 : t.bits;
    const min = t.kind === 'u' || t.kind === 'b' ? 0 : -Math.pow(2, bits - 1);
    const max = t.kind === 'b' ? 1 : t.kind === 'u' ? Math.pow(2, bits) - 1 : Math.pow(2, bits - 1) - 1;
    const rl = param(low), rh = param(high);
    let bcnt = 0, buf = 0;
    const nextBuf = (width: number) => {
      const per = width === 8 ? 4 : 2;
      if (bcnt === 0) { buf = this.g.next32(); bcnt = per - 1; } else { buf = buf >>> width; bcnt--; }
      return buf & (width === 8 ? 0xff : 0xffff);
    };
    let k = 0;
    const draw = () => {
      const i = k++;
      const l = rl(i); let h = rh(i);
      if (!endpoint) h -= 1;
      if (l < min) throw new Error(`low is out of bounds for ${dtype}`);
      if (h > max) throw new Error(`high is out of bounds for ${dtype}`);
      if (l > h) throw new Error(`low >= high${endpoint ? '' : ''}`);
      const rng = h - l;
      if (rng === 0) return l;
      if (bits === 8 || bits === 16) {
        const full = bits === 8 ? 0xff : 0xffff;
        if (rng === full) return l + nextBuf(bits);
        const ex = rng + 1;
        let m = nextBuf(bits) * ex, lo = m & full;
        if (lo < ex) { const thr = (full - rng) % ex; while (lo < thr) { m = nextBuf(bits) * ex; lo = m & full; } }
        return l + Math.floor(m / (full + 1));
      }
      return l + this.lemire64(rng);
    };
    const shape = size === undefined && (isArrayLike(low) || isArrayLike(high)) ? paramShape(size, low, high) : size;
    return build(shape, draw, dtype);
  }
  standard_normal(size?: any) {
    const { ki, wi, fi } = zig_normal_tables();
    return build(size, () => {
      const g = this.g;
      for (;;) {
        g.next64();
        const idx = g.lo & 0xff;
        // r >>= 8 ; sign = r & 1 ; rabs = (r >> 1) & 0x000fffffffffffff
        const hi = g.hi, lo = g.lo;
        const r_lo = ((lo >>> 8) | (hi << 24)) >>> 0, r_hi = hi >>> 8; // r = (hi:lo) >> 8
        const sign = r_lo & 1;
        const rabs_lo = ((r_lo >>> 1) | ((r_hi & 1) << 31)) >>> 0, rabs_hi = (r_hi >>> 1) & 0xfffff;
        const rabs = rabs_hi * 4294967296 + rabs_lo;
        let x = rabs * wi[idx];
        if (sign) x = -x;
        if (rabs < ki[idx]) return x;
        if (idx === 0) {
          for (;;) {
            const xx = -NOR_INV_R * Math.log1p(-this.rd()), yy = -Math.log1p(-this.rd());
            if (yy + yy > xx * xx) return ((rabs_lo >>> 8) & 1) ? -(NOR_R + xx) : NOR_R + xx;
          }
        } else if ((fi[idx - 1] - fi[idx]) * this.rd() + fi[idx] < Math.exp(-0.5 * x * x)) return x;
      }
    });
  }
  private _normal1(): number { return this.standard_normal() as number; }
  private _exp1(): number {
    const { ke, we, fe } = zig_exp_tables();
    const g = this.g;
    for (;;) {
      g.next64();
      // ri >>= 3 ; idx = ri & 0xff ; ri >>= 8
      const hi = g.hi, lo = g.lo;
      const l3 = ((lo >>> 3) | (hi << 29)) >>> 0, h3 = hi >>> 3;
      const idx = l3 & 0xff;
      const l11 = ((l3 >>> 8) | (h3 << 24)) >>> 0, h11 = h3 >>> 8;
      const ri = h11 * 4294967296 + l11;
      const x = ri * we[idx];
      if (ri < ke[idx]) return x;
      if (idx === 0) return EXP_R - Math.log1p(-this.rd());
      if ((fe[idx - 1] - fe[idx]) * this.rd() + fe[idx] < Math.exp(-x)) return x;
    }
  }
  normal(loc: any = 0, scale: any = 1, size?: any) { return dist(size, [loc, scale], (l, s) => l + s * this._normal1()); }
  standard_exponential(size?: any) { return build(size, () => this._exp1()); }
  exponential(scale: any = 1, size?: any) { return dist(size, [scale], (s) => s * this._exp1()); }
  uniform(low: any = 0, high: any = 1, size?: any) { return dist(size, [low, high], (l, h) => l + (h - l) * this.rd()); }
  standard_gamma(shape: any, size?: any) { return dist(size, [shape], (s) => this._gamma1(s)); }
  gamma(shape: any, scale: any = 1, size?: any) { return dist(size, [shape, scale], (a, s) => s * this._gamma1(a)); }
  private _gamma1(shape: number): number {
    if (shape === 1) return this._exp1();
    if (shape === 0) return 0;
    if (shape < 1) {
      for (;;) {
        const U = this.rd(), V = this._exp1();
        if (U <= 1 - shape) { const X = Math.pow(U, 1 / shape); if (X <= V) return X; }
        else { const Y = -Math.log((1 - U) / shape), X = Math.pow(1 - shape + shape * Y, 1 / shape); if (X <= Y + V) return X; }
      }
    }
    const b = shape - 1 / 3, c = 1 / Math.sqrt(9 * b);
    for (;;) {
      let X, V;
      do { X = this._normal1(); V = 1 + c * X; } while (V <= 0);
      V = V * V * V;
      const U = this.rd();
      if (U < 1 - 0.0331 * (X * X) * (X * X)) return b * V;
      if (Math.log(U) < 0.5 * X * X + b * (1 - V + Math.log(V))) return b * V;
    }
  }
  beta(a: any, b: any, size?: any) {
    return dist(size, [a, b], (x, y) => { const Ga = this._gamma1(x), Gb = this._gamma1(y); return Ga / (Ga + Gb); });
  }
  chisquare(df: any, size?: any) { return dist(size, [df], (d) => 2 * this._gamma1(d / 2)); }
  standard_cauchy(size?: any) { return build(size, () => this._normal1() / this._normal1()); }
  standard_t(df: any, size?: any) {
    return dist(size, [df], (d) => { const num = this._normal1(), den = this._gamma1(d / 2); return Math.sqrt(d / 2) * num / Math.sqrt(den); });
  }
  lognormal(mean: any = 0, sigma: any = 1, size?: any) { return dist(size, [mean, sigma], (m, s) => Math.exp(m + s * this._normal1())); }
  laplace(loc: any = 0, scale: any = 1, size?: any) {
    return dist(size, [loc, scale], (l, s) => { const U = this.rd(); return U >= 0.5 ? l - s * Math.log(2 - U - U) : l + s * Math.log(U + U); });
  }
  logistic(loc: any = 0, scale: any = 1, size?: any) {
    return dist(size, [loc, scale], (l, s) => { const U = this.rd(); return l + s * Math.log(U / (1 - U)); });
  }
  gumbel(loc: any = 0, scale: any = 1, size?: any) {
    return dist(size, [loc, scale], (l, s) => l - s * Math.log(-Math.log(1 - this.rd())));
  }
  rayleigh(scale: any = 1, size?: any) { return dist(size, [scale], (s) => s * Math.sqrt(2 * this._exp1())); }
  weibull(a: any, size?: any) { return dist(size, [a], (x) => x === 0 ? 0 : Math.pow(this._exp1(), 1 / x)); }
  pareto(a: any, size?: any) { return dist(size, [a], (x) => Math.expm1(this._exp1() / x)); }
  power(a: any, size?: any) { return dist(size, [a], (x) => Math.pow(-Math.expm1(-this._exp1()), 1 / x)); }
  triangular(left: any, mode: any, right: any, size?: any) {
    return dist(size, [left, mode, right], (l, m, r) => {
      const base = r - l, ml = m - l, lp = ml / base, U = this.rd();
      return U <= lp ? l + Math.sqrt(U * base * ml) : r - Math.sqrt((1 - U) * base * (r - m));
    });
  }
  poisson(lam: any = 1, size?: any) {
    const rs = new RandomState(); (rs as any).rd = () => this.rd(); (rs as any)._gauss = () => this._normal1();
    return dist(size, [lam], (l) => (rs as any)._poisson(l), 'int64');
  }
  binomial(n: any, p: any, size?: any) {
    const rs = new RandomState(); (rs as any).rd = () => this.rd();
    return dist(size, [n, p], (a, b) => (rs as any)._binomial(a, b), 'int64');
  }
  geometric(p: any, size?: any) {
    return dist(size, [p], (q) => {
      if (q >= 0.333333333333333333) { let X = 1, sum = q, prod = q; const qq = 1 - q, U = this.rd(); while (U > sum) { prod *= qq; sum += prod; X++; } return X; }
      return Math.ceil(-this._exp1() / Math.log1p(-q));
    }, 'int64');
  }
  multinomial(n: number, pvals: any, size?: any) {
    const rs = new RandomState(); (rs as any).rd = () => this.rd();
    return rs.multinomial(n, pvals, size);
  }
  dirichlet(alpha: any, size?: any) {
    const a = Array.from(arr(alpha).flat as any).map(Number), k = a.length;
    const shape = toShape(size) || [], cnt = prod(shape), out = new_buffer(cnt * k, 'float64');
    for (let t = 0; t < cnt; t++) {
      let sum = 0;
      for (let j = 0; j < k; j++) { const v = this._gamma1(a[j]); out[t * k + j] = v; sum += v; }
      for (let j = 0; j < k; j++) out[t * k + j] /= sum;
    }
    return new NDArray(out, shape.concat([k]), 'float64');
  }
  bytes(length: number) {
    const out = new Uint8Array(length);
    let i = 0;
    while (i < length) { let v = this.g.next32(); for (let j = 0; j < 4 && i < length; j++, v >>>= 8) out[i++] = v & 255; }
    return out;
  }
  shuffle(x: any, axis: number = 0) {
    if (!(x instanceof NDArray)) {
      if (Array.isArray(x)) { for (let i = x.length - 1; i > 0; i--) { const j = this._masked(i); const t = x[i]; x[i] = x[j]; x[j] = t; } return; }
      throw new TypeError("shuffle needs an array or list");
    }
    const np = require_np();
    const n = x.shape[axis];
    const perm: number[] = []; for (let i = 0; i < n; i++) perm.push(i);
    for (let i = n - 1; i > 0; i--) { const j = this._masked(i); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    const src = np.copy(x);
    np.copyto(x, np.take(src, perm, axis));
  }
  permutation(x: any, axis: number = 0) {
    const np = require_np();
    if (typeof x === 'number') { const a = np.arange(x); this.shuffle(a); return a; }
    const a = np.copy(np.array(x));
    this.shuffle(a, axis);
    return a;
  }
  permuted(x: any, axis?: number) {
    const np = require_np();
    const a = np.copy(np.array(x));
    if (axis === undefined) { const f = np.reshape(a, [-1]); this.shuffle(f); return np.reshape(f, a.shape); }
    const moved = np.copy(np.moveaxis(a, axis, -1));
    const lanes = moved.size / moved.shape[moved.shape.length - 1], n = moved.shape[moved.shape.length - 1];
    const flat = np.reshape(moved, [lanes, n]);
    for (let l = 0; l < lanes; l++) { const row = np.copy(np.take(flat, l, 0)); this.shuffle(row); np.copyto(np.take(flat, [l], 0), np.reshape(row, [1, n])); }
    return np.copy(np.moveaxis(np.reshape(flat, moved.shape), -1, axis));
  }
  choice(a: any, size?: any, replace: boolean = true, p: any = null, axis: number = 0, shuffle: boolean = true) {
    const np = require_np();
    let pop: NDArray | null = null, popN: number;
    if (typeof a === 'number') popN = a;
    else { pop = arr(a); if (pop.shape.length === 0) throw new Error("a must be a sequence or an integer, not a 0-d array"); popN = pop.shape[axis]; }
    const shape = toShape(size);
    const n = shape === null ? 1 : prod(shape);
    let idx: number[];
    if (p !== null && p !== undefined) {
      const pr = Array.from(arr(p).flat as any).map(Number);
      if (pr.length !== popN) throw new Error("a and p must have same size");
      const tot = pr.reduce((x, y) => x + y, 0);
      if (Math.abs(tot - 1) > Math.sqrt(2.220446049250313e-16)) throw new Error("probabilities do not sum to 1");
      const cdf: number[] = []; let s = 0;
      for (const v of pr) { s += v; cdf.push(s); }
      for (let i = 0; i < cdf.length; i++) cdf[i] /= s;
      const search = (u: number) => { let lo = 0, hi = cdf.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] <= u) lo = mid + 1; else hi = mid; } return lo; };
      if (replace) {
        const us: number[] = []; for (let i = 0; i < n; i++) us.push(this.rd());
        idx = us.map(search);
      } else {
        if (n > popN) throw new Error("Cannot take a larger sample than population when replace is False");
        const pp = pr.slice(), found: number[] = [];
        while (found.length < n) {
          const need = n - found.length, xs: number[] = [];
          for (let i = 0; i < need; i++) xs.push(this.rd());
          for (const f of found) pp[f] = 0;
          const cd: number[] = []; let ss = 0; for (const v of pp) { ss += v; cd.push(ss); } for (let i = 0; i < cd.length; i++) cd[i] /= ss;
          const nw = xs.map(u => { let lo = 0, hi = cd.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (cd[mid] <= u) lo = mid + 1; else hi = mid; } return lo; });
          const seen: { [k: number]: number } = {}, uniq: number[] = [];
          nw.forEach((v, i) => { if (!(v in seen)) { seen[v] = i; uniq.push(i); } });
          uniq.sort((x, y) => x - y);
          for (const i of uniq) found.push(nw[i]);
        }
        idx = found;
      }
    } else if (replace) {
      idx = []; for (let i = 0; i < n; i++) idx.push(this._interval(popN - 1));
    } else {
      if (n > popN) throw new Error("Cannot take a larger sample than population when replace is False");
      if (popN > 10000 && n > Math.floor(popN / 50)) { // tail shuffle
        const all: number[] = []; for (let i = 0; i < popN; i++) all.push(i);
        for (let i = popN - 1; i >= Math.max(popN - n, 1); i--) { const j = this._interval(i); const t = all[i]; all[i] = all[j]; all[j] = t; }
        idx = all.slice(popN - n);
      } else { // Floyd's algorithm
        const sel: { [k: number]: boolean } = {}; idx = [];
        for (let j = popN - n; j < popN; j++) {
          let v = this._interval(j);
          if (sel[v]) v = j;
          sel[v] = true; idx.push(v);
        }
        if (shuffle) for (let i = n - 1; i > 0; i--) { const j = this._interval(i); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      }
    }
    if (pop === null) {
      if (shape === null) return idx[0];
      const buf = new_buffer(n, 'int64'); for (let i = 0; i < n; i++) buf[i] = idx[i];
      return new NDArray(buf, shape, 'int64');
    }
    const ib = new_buffer(n, 'int64'); for (let i = 0; i < n; i++) ib[i] = idx[i];
    const taken = np.take(pop, new NDArray(ib, [n], 'int64'), axis);
    if (shape === null) return np.take(pop, idx[0], axis);
    const sh = pop.shape.slice(); sh.splice(axis, 1, ...shape);
    return np.reshape(taken, sh);
  }
  spawn(n: number) { return (this.g as PCG64).seed_seq.spawn(n).map(s => new Generator(new PCG64(s))); }
}

export class BitGenerator { }
/** Distributions that Generator shares with RandomState, run on the Generator's own bit stream (ziggurat normal/exponential). */
function adapter(g: Generator): RandomState {
  const rs = new RandomState() as any;
  rs.rd = () => g.bit_generator.nextDouble();
  rs._gauss = () => (g as any)._normal1();
  rs._std_exp = () => (g as any)._exp1();
  rs._std_gamma = (shape: number) => (g as any)._gamma1(shape);
  return rs;
}
for (const name of ['f', 'noncentral_chisquare', 'noncentral_f', 'wald', 'vonmises', 'zipf', 'logseries', 'negative_binomial', 'hypergeometric', 'multivariate_normal']) {
  if (!(Generator.prototype as any)[name]) (Generator.prototype as any)[name] = function (...a: any[]) { return (adapter(this) as any)[name](...a); };
}
(Generator.prototype as any).multivariate_hypergeometric = function (colors: any, nsample: number, size?: any, method: string = 'marginals') {
  const c = Array.from(arr(colors).flat as any).map(Number), k = c.length, rs = adapter(this) as any;
  const shape = toShape(size) || [], cnt = prod(shape), out = new_buffer(cnt * k, 'int64');
  for (let t = 0; t < cnt; t++) {
    let total = c.reduce((x, y) => x + y, 0), left = nsample;
    for (let j = 0; j < k - 1; j++) {
      total -= c[j];
      const v = left > 0 ? rs.hypergeometric(c[j], total, left) : 0;
      out[t * k + j] = v; left -= v;
    }
    out[t * k + k - 1] = left;
  }
  return new NDArray(out, shape.concat([k]), 'int64');
};
(RandomState.prototype as any).noncentral_f = function (dfnum: any, dfden: any, nonc: any, size?: any) {
  return dist(size, [dfnum, dfden, nonc], (n, d, nc) => {
    const t = this.noncentral_chisquare(n, nc) as number;
    return (t * d) / (this.chisquare(d) as number * n);
  });
};
(Generator.prototype as any).noncentral_f = (RandomState.prototype as any).noncentral_f;
export function default_rng(seed?: any): Generator {
  if (seed instanceof Generator) return seed;
  if (seed instanceof PCG64 || seed instanceof MT19937) return new Generator(seed);
  return new Generator(new PCG64(seed));
}

// ---------------------------------------------------------------- module-level (global RandomState)
let _global = new RandomState();
export function get_state(legacy: boolean = true) { return _global.get_state(legacy); }
export function set_state(s: any) { return _global.set_state(s); }
export function seed(s?: any) { _global.seed(s); }
export const random_sample = (size?: any) => _global.random_sample(size);
export const random = random_sample;
export const ranf = random_sample;
export const sample = random_sample;
export const rand = (...d: any[]) => _global.rand(...d);
export const randn = (...d: any[]) => _global.randn(...d);
export const randint = (low: any, high?: any, size?: any, dtype?: any) => _global.randint(low, high, size, dtype);
export const random_integers = (low: any, high?: any, size?: any) => _global.random_integers(low, high, size);
export const standard_normal = (size?: any) => _global.standard_normal(size);
export const normal = (loc?: any, scale?: any, size?: any) => _global.normal(loc, scale, size);
export const uniform = (low?: any, high?: any, size?: any) => _global.uniform(low, high, size);
export const standard_exponential = (size?: any) => _global.standard_exponential(size);
export const exponential = (scale?: any, size?: any) => _global.exponential(scale, size);
export const standard_gamma = (shape: any, size?: any) => _global.standard_gamma(shape, size);
export const gamma = (shape: any, scale?: any, size?: any) => _global.gamma(shape, scale, size);
export const beta = (a: any, b: any, size?: any) => _global.beta(a, b, size);
export const chisquare = (df: any, size?: any) => _global.chisquare(df, size);
export const noncentral_chisquare = (df: any, nonc: any, size?: any) => _global.noncentral_chisquare(df, nonc, size);
export const noncentral_f = (a: any, b: any, c: any, size?: any) => (_global as any).noncentral_f(a, b, c, size);
export const f = (a: any, b: any, size?: any) => _global.f(a, b, size);
export const standard_cauchy = (size?: any) => _global.standard_cauchy(size);
export const standard_t = (df: any, size?: any) => _global.standard_t(df, size);
export const laplace = (loc?: any, scale?: any, size?: any) => _global.laplace(loc, scale, size);
export const gumbel = (loc?: any, scale?: any, size?: any) => _global.gumbel(loc, scale, size);
export const logistic = (loc?: any, scale?: any, size?: any) => _global.logistic(loc, scale, size);
export const lognormal = (mean?: any, sigma?: any, size?: any) => _global.lognormal(mean, sigma, size);
export const rayleigh = (scale?: any, size?: any) => _global.rayleigh(scale, size);
export const wald = (mean: any, scale: any, size?: any) => _global.wald(mean, scale, size);
export const weibull = (a: any, size?: any) => _global.weibull(a, size);
export const pareto = (a: any, size?: any) => _global.pareto(a, size);
export const power = (a: any, size?: any) => _global.power(a, size);
export const triangular = (l: any, m: any, r: any, size?: any) => _global.triangular(l, m, r, size);
export const poisson = (lam?: any, size?: any) => _global.poisson(lam, size);
export const binomial = (n: any, p: any, size?: any) => _global.binomial(n, p, size);
export const geometric = (p: any, size?: any) => _global.geometric(p, size);
export const negative_binomial = (n: any, p: any, size?: any) => _global.negative_binomial(n, p, size);
export const zipf = (a: any, size?: any) => _global.zipf(a, size);
export const logseries = (p: any, size?: any) => _global.logseries(p, size);
export const vonmises = (mu: any, k: any, size?: any) => _global.vonmises(mu, k, size);
export const hypergeometric = (g: any, b: any, s: any, size?: any) => _global.hypergeometric(g, b, s, size);
export const multinomial = (n: number, p: any, size?: any) => _global.multinomial(n, p, size);
export const multivariate_normal = (m: any, c: any, size?: any) => _global.multivariate_normal(m, c, size);
export const dirichlet = (a: any, size?: any) => _global.dirichlet(a, size);
export const bytes = (n: number) => _global.bytes(n);
export const shuffle = (x: any) => _global.shuffle(x);
export const permutation = (x: any) => _global.permutation(x);
export const choice = (a: any, size?: any, replace?: boolean, p?: any) => _global.choice(a, size, replace, p);
export function shuffled(x: NDArray) { const c = np.copy(x); _global.shuffle(c); return c; }
export const _normal_buffer = (n: number) => { const o = new_buffer(n); for (let i = 0; i < n; i++) o[i] = _global._gauss(); return o; };
