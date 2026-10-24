//@ts-check
/**
 * Public API of numpy.polynomial (functions with the numpy names and keyword arguments, and the series classes)
 * and of the legacy numpy.poly1d / polyval / polyfit... functions.
 */
import { NDArray, asarray, isarray } from "../NDArray";
import { float64, complex128, new_buffer } from "../dtypes";
import { Complex } from "../complex";
import { kwargs_decorator } from "../kwargs/kwargs";
import * as P from "./polynomial";
import { Basis, PolyError, arr, as_series, trimseq } from "./polynomial";
import * as la from "./linalg";

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const nums = (v: any): number[] => Array.from(asarray(v).flat as any, (x: any) => +x);
const asC = (c: any): number[] => {
  const a = asarray(c);
  if (a.dtype === complex128) throw new PolyError("complex coefficients are not supported");
  return nums(a);
};
const coefs = (c: any): number[] => as_series([c], false)[0];
const trimC = (c: any): number[] => as_series([c], true)[0];

function fn(defaults: [string, any][], implementation: (...a: any[]) => any) {
  // 0-d array arguments (e.g. numbers coming from the template parser) are passed as plain JS values
  const impl = (...args: any[]) => implementation(...args.map(a => (isarray(a) && a.shape.length == 0) ? a.item() : a));
  return kwargs_decorator<any, any, any>({ defaults, implementation: impl, parsers: [] }) as (...args: any[]) => any;
}

// ==============================
//   Function families
// ==============================

export function make_functions(B: Basis): { [name: string]: any } {
  const p = B.prefix;
  const out: { [name: string]: any } = {};
  const dom = arr(B.domain.slice());
  out[`${p}domain`] = dom;
  const maybe_int = (v: number[]) => v.every(Number.isInteger) ? arr(v, 'int64') : arr(v);
  out[`${p}zero`] = maybe_int([0]);
  out[`${p}one`] = maybe_int([1]);
  out[`${p}x`] = maybe_int(P.mulx(B, [1]));
  out[`${p}line`] = fn([["off", undefined], ["scl", undefined]], (off, scl) => arr(P.line(B, +off, +scl)));
  out[`${p}add`] = fn([["c1", undefined], ["c2", undefined]], (c1, c2) => arr(P.add(B, trimC(c1), trimC(c2))));
  out[`${p}sub`] = fn([["c1", undefined], ["c2", undefined]], (c1, c2) => arr(P.sub(B, trimC(c1), trimC(c2))));
  out[`${p}mulx`] = fn([["c", undefined]], (c) => arr(P.mulx(B, trimC(c))));
  out[`${p}mul`] = fn([["c1", undefined], ["c2", undefined]], (c1, c2) => arr(P.mul(B, trimC(c1), trimC(c2))));
  out[`${p}div`] = fn([["c1", undefined], ["c2", undefined]], (c1, c2) => { const [q, r] = P.div(B, trimC(c1), trimC(c2)); return [arr(q), arr(r)]; });
  out[`${p}pow`] = fn([["c", undefined], ["pow", undefined], ["maxpower", p === 'poly' ? null : 16]], (c, pw, max) => arr(P.pow(B, trimC(c), pw, max)));
  out[`${p}val`] = fn([["x", undefined], ["c", undefined], ["tensor", true]], (x, c, tensor) => {
    const C = asarray(c);
    if (C.shape.length <= 1) return P.val(B, x, C.shape.length == 0 ? [+C.item()] : asC(C));
    // c has shape (n, ...): coefficients along the first axis. tensor: result shape c.shape[1:] + x.shape
    const n = C.shape[0], rest = C.shape.slice(1), restSize = rest.reduce((a, b) => a * b, 1);
    const cf = C.flat as any;
    const X = asarray(x);
    const parts = range(restSize).map(j => P.val(B, X, range(n).map(k => +cf[k * restSize + j])));
    const xs = X.size;
    const bufferOf = (a: NDArray) => Array.from(a.flat as any) as any[];
    if (tensor) {
      const all = parts.flatMap(bufferOf);
      return new NDArray(new_buffer(all, float64), [...rest, ...X.shape], float64);
    }
    if (X.shape.length !== rest.length || X.shape.some((d, i) => d !== rest[i])) throw new PolyError("x.shape must match c.shape[1:] when tensor is False");
    return new NDArray(new_buffer(parts.map((a, j) => bufferOf(a)[j]), float64), rest, float64);
  });
  out[`${p}val2d`] = fn([["x", undefined], ["y", undefined], ["c", undefined]], (x, y, c) => P.val2d(B, x, y, asarray(c)));
  out[`${p}val3d`] = fn([["x", undefined], ["y", undefined], ["z", undefined], ["c", undefined]], (x, y, z, c) => P.val3d(B, x, y, z, asarray(c)));
  out[`${p}grid2d`] = fn([["x", undefined], ["y", undefined], ["c", undefined]], (x, y, c) => P.grid2d(B, x, y, asarray(c)));
  out[`${p}grid3d`] = fn([["x", undefined], ["y", undefined], ["z", undefined], ["c", undefined]], (x, y, z, c) => P.grid3d(B, x, y, z, asarray(c)));
  out[`${p}der`] = fn([["c", undefined], ["m", 1], ["scl", 1], ["axis", 0]], (c, m, scl) => arr(P.der(B, coefs(c), m, scl)));
  out[`${p}int`] = fn([["c", undefined], ["m", 1], ["k", []], ["lbnd", 0], ["scl", 1], ["axis", 0]], (c, m, k, lbnd, scl) => arr(P.integ(B, coefs(c), m, nums(k), lbnd, scl)));
  out[`${p}fromroots`] = fn([["roots", undefined]], (r) => arr(P.fromroots(B, asC(r))));
  out[`${p}roots`] = fn([["c", undefined]], (c) => P.roots(B, trimC(c)));
  out[`${p}companion`] = fn([["c", undefined]], (c) => P.compan(B, trimC(c)));
  out[`${p}vander`] = fn([["x", undefined], ["deg", undefined]], (x, deg) => P.vander(B, x, deg));
  out[`${p}vander2d`] = fn([["x", undefined], ["y", undefined], ["deg", undefined]], (x, y, deg) => P.vander2d(B, x, y, nums(deg)));
  out[`${p}vander3d`] = fn([["x", undefined], ["y", undefined], ["z", undefined], ["deg", undefined]], (x, y, z, deg) => P.vander3d(B, x, y, z, nums(deg)));
  out[`${p}fit`] = fn([["x", undefined], ["y", undefined], ["deg", undefined], ["rcond", null], ["full", false], ["w", null]], (x, y, deg, rcond, full, w) => P.fit(B, x, y, Array.isArray(deg) || isarray(deg) ? nums(deg) : deg, rcond, !!full, w));
  out[`${p}trim`] = fn([["c", undefined], ["tol", 0]], (c, tol) => arr(P.trimcoef(asC(c), tol)));
  if (p !== 'poly') out[`${p}weight`] = fn([["x", undefined]], (x) => { const X = asarray(x); return new NDArray(new_buffer(nums(X).map(B.weight), float64), X.shape, float64); });
  if (p !== 'poly') {
    out[`${p}gauss`] = fn([["deg", undefined]], (deg) => P.gauss(B, deg));
    out[`${p}2poly`] = fn([["c", undefined]], (c) => arr(P.toPower(B, trimC(c))));
    out[`poly2${p}`] = fn([["pol", undefined]], (pol) => arr(P.fromPower(B, trimC(pol))));
  } else {
    out[`polyvalfromroots`] = fn([["x", undefined], ["r", undefined], ["tensor", true]], (x, r, tensor) => P.valfromroots(x, r, tensor));
  }
  if (p === 'cheb') {
    out['chebpts1'] = fn([["npts", undefined]], (n) => P.pts1(n));
    out['chebpts2'] = fn([["npts", undefined]], (n) => P.pts2(n));
    out['chebinterpolate'] = fn([["func", undefined], ["deg", undefined], ["args", []]], (f, deg, args) => P.chebinterpolate(f, deg, args));
  }
  return out;
}

// ==============================
//   Series classes
// ==============================

export function make_class(B: Basis, F: { [name: string]: any }) {
  const p = B.prefix;
  const defaultDomain = B.domain;
  const toNum = (v: any) => asC(v);
  class Series {
    static readonly basisName = B.name;
    static domain = arr(defaultDomain.slice());
    static window = arr(defaultDomain.slice());
    static basis_name = p;
    coef: NDArray; domain: NDArray; window: NDArray; symbol: string;

    constructor(coef: any, domain: any = null, window: any = null, symbol: string = 'x') {
      // Instances are callable (series(x) evaluates the series), like numpy's series objects
      const self: any = (x: any) => self.call(x);
      Object.setPrototypeOf(self, Series.prototype);
      const c = as_series([coef], false)[0];
      self.coef = arr(c);
      self.domain = domain === null ? arr(defaultDomain.slice()) : arr(toNum(domain));
      self.window = window === null ? arr(defaultDomain.slice()) : arr(toNum(window));
      self.symbol = symbol;
      if (self.domain.size !== 2 || self.window.size !== 2) throw new PolyError("domain and window must have two elements");
      return self;
    }
    make(coef: number[]) { return new Series(coef, nums(this.domain), nums(this.window), this.symbol); }
    c() { return nums(this.coef); }
    get length() { return this.coef.size; }
    degree() { return this.coef.size - 1; }
    copy() { return this.make(this.c()); }
    mapparms(): [number, number] { return P.mapparms(nums(this.domain), nums(this.window)); }
    coerce(other: any): number[] {
      if (other instanceof Series) {
        if (nums(other.domain).some((v, i) => v !== nums(this.domain)[i])) throw new PolyError("Domains differ");
        if (nums(other.window).some((v, i) => v !== nums(this.window)[i])) throw new PolyError("Windows differ");
        return other.c();
      }
      if (other instanceof Object && (other as any).coef) throw new PolyError("Polynomial types differ");
      return trimC(asarray(other).shape.length == 0 ? arr([+asarray(other).item()]) : other);
    }
    /** Evaluate the series at x (mapped from domain to window) */
    call(x: any): NDArray {
      if (x instanceof Series) {
        // composition: self(other(x)) evaluated in the power basis of the argument's variable is not supported
        throw new PolyError("Series composition is not supported");
      }
      const [off, scl] = this.mapparms();
      const X = asarray(x);
      const mapped = X.dtype === complex128
        ? new NDArray((X.flat as any).map((z: Complex) => new Complex(off + scl * z.re, scl * z.im)), X.shape, complex128)
        : P.mapdomain(X, nums(this.domain), nums(this.window));
      return P.val(B, mapped, this.c());
    }
    add(o: any) { return this.make(P.add(B, trimseq(this.c()), this.coerce(o))); }
    sub(o: any) { return this.make(P.sub(B, trimseq(this.c()), this.coerce(o))); }
    mul(o: any) { return this.make(P.mul(B, trimseq(this.c()), this.coerce(o))); }
    neg() { return this.make(this.c().map(v => -v)); }
    divmod(o: any): [Series, Series] { const [q, r] = P.div(B, trimseq(this.c()), this.coerce(o)); return [this.make(q), this.make(r)]; }
    floordiv(o: any) { return this.divmod(o)[0]; }
    mod(o: any) { return this.divmod(o)[1]; }
    div(o: any) {
      const other = this.coerce(o);
      if (other.length === 1) return this.make(this.c().map(v => v / other[0]));
      throw new PolyError("true division by a series is not supported, use floordiv/mod/divmod");
    }
    pow(n: number, maxpower: number = 16) { return this.make(P.pow(B, trimseq(this.c()), n, maxpower)); }
    equals(o: any): boolean {
      if (!(o instanceof Series)) return false;
      const eq = (a: NDArray, b: NDArray) => a.size === b.size && nums(a).every((v, i) => v === nums(b)[i]);
      return eq(this.domain, o.domain) && eq(this.window, o.window) && eq(this.coef, o.coef);
    }
    has_samecoef(o: Series) { return this.coef.size === o.coef.size && nums(this.coef).every((v, i) => v === nums(o.coef)[i]); }
    has_samedomain(o: Series) { return nums(this.domain).every((v, i) => v === nums(o.domain)[i]); }
    has_samewindow(o: Series) { return nums(this.window).every((v, i) => v === nums(o.window)[i]); }
    has_sametype(o: any) { return o instanceof Series; }
    trim(tol: number = 0) { return this.make(P.trimcoef(this.c(), tol)); }
    truncate(size: number) {
      if (!Number.isInteger(size) || size < 1) throw new PolyError("size must be a positive integer");
      const c = this.c();
      return this.make(size >= c.length ? c : c.slice(0, size));
    }
    cutdeg(deg: number) { if (!Number.isInteger(deg) || deg < 0) throw new PolyError("deg must be non-negative integer"); return this.truncate(deg + 1); }
    deriv(m: number = 1) { const [, scl] = this.mapparms(); return this.make(P.der(B, this.c(), m, scl)); }
    integ(m: number = 1, k: number | number[] = [], lbnd: number | null = null) {
      const [off, scl] = this.mapparms();
      const lb = lbnd === null ? 0 : off + scl * lbnd;
      return this.make(P.integ(B, this.c(), m, Array.isArray(k) ? k : [k], lb, 1 / scl));
    }
    roots(): NDArray {
      const r = P.roots(B, this.c());
      const [off, scl] = P.mapparms(nums(this.window), nums(this.domain));
      if (r.dtype === complex128) return new NDArray((r.flat as any).map((z: Complex) => new Complex(off + scl * z.re, scl * z.im)), r.shape, complex128);
      return arr(nums(r).map(v => off + scl * v));
    }
    linspace(n: number = 100, domain: any = null): NDArray[] {
      const d = domain === null ? nums(this.domain) : toNum(domain);
      const x = arr(range(n).map(i => d[0] + (n > 1 ? i * (d[1] - d[0]) / (n - 1) : 0)));
      return [x, this.call(x)];
    }
    /** Converts to another series class (kind) and/or domain/window */
    convert(domain: any = null, kind: any = null, window: any = null) {
      const target: any = kind === null ? Series : kind;
      const d = domain === null ? target.domain : domain;
      const w = window === null ? target.window : window;
      return target.cast(this, d, w);
    }
    static cast(series: any, domain: any = null, window: any = null) {
      const d = domain === null ? Series.domain : domain;
      const w = window === null ? Series.window : window;
      // value-preserving conversion through the power basis of the variable x
      const [off, scl] = P.mapparms(nums(series.domain), nums(series.window));
      const srcBasis: Basis = (series.constructor as any)._basis;
      let pw = P.toPower(srcBasis, nums(series.coef));
      // x_window = off + scl * x_domain  (power series composition)
      pw = compose_linear(pw, off, scl);
      const [off2, scl2] = P.mapparms(nums(asarray(d)), nums(asarray(w)));
      // target variable t = off2 + scl2 * x  =>  x = (t - off2) / scl2
      pw = compose_linear(pw, -off2 / scl2, 1 / scl2);
      return new Series(P.fromPower(B, pw), nums(asarray(d)), nums(asarray(w)), series.symbol);
    }
    static fit(x: any, y: any, deg: number | number[], domain: any = null, rcond: number | null = null, full: boolean = false, w: any = null, window: any = null) {
      const dm = domain === null ? P.getdomain(x) : toNum(domain);
      const dom = (dm[0] === dm[1]) ? [dm[0] - 1, dm[1] + 1] : dm;
      const win = window === null ? defaultDomain.slice() : toNum(window);
      const xnew = P.mapdomain(x, dom, win);
      const res = F[`${p}fit`](xnew, y, deg, rcond, full, w) as any;
      const coef = full ? res[0] : res;
      const series = new Series(coef, dom, win);
      return full ? [series, res[1]] : series;
    }
    static fromroots(roots: any, domain: any = [], window: any = null) {
      const r = asC(roots);
      // domain=[] (the default) means the default domain of the class; null means the smallest interval containing the roots
      const dm = domain === null ? P.getdomain(r) : ((Array.isArray(domain) && domain.length === 0) ? defaultDomain.slice() : toNum(domain));
      const win = window === null ? defaultDomain.slice() : toNum(window);
      const [off, scl] = P.mapparms(dm, win);
      const rnew = r.map(v => off + scl * v);
      const coef = P.fromroots(B, rnew).map(v => v / Math.pow(scl, r.length));
      return new Series(coef, dm, win);
    }
    static identity(domain: any = null, window: any = null) {
      const dm = domain === null ? Series.domain : domain, win = window === null ? Series.window : window;
      const [off, scl] = P.mapparms(nums(asarray(win)), nums(asarray(dm)));
      return new Series(P.line(B, off, scl), nums(asarray(dm)), nums(asarray(win)));
    }
    static basis(deg: number, domain: any = null, window: any = null) {
      if (!Number.isInteger(deg) || deg < 0) throw new PolyError("deg must be non-negative integer");
      const c = new Array(deg + 1).fill(0); c[deg] = 1;
      return new Series(c, domain, window);
    }
    toString() { return `${B.name}([${this.c().join(', ')}], domain=[${nums(this.domain)}], window=[${nums(this.window)}])`; }
  }
  // The exported class can be called with or without `new`
  const Cls: any = function (...args: any[]) { return new (Series as any)(...args); };
  for (const key of ['fit', 'fromroots', 'identity', 'basis', 'cast', 'domain', 'window', 'basis_name', 'basisName']) Cls[key] = (Series as any)[key];
  Cls._basis = B;
  Cls.prototype = Series.prototype;
  Object.defineProperty(Cls, 'name', { value: B.name });
  (Series as any)._basis = B;
  (Series.prototype as any).constructor = Cls;
  return Cls;
}

/** Coefficients (power basis) of p(off + scl x) */
function compose_linear(pw: number[], off: number, scl: number): number[] {
  let res: number[] = [0];
  for (let i = pw.length - 1; i >= 0; i--) {
    // res = res * (off + scl x) + pw[i]
    const next = new Array(res.length + 1).fill(0);
    res.forEach((v, k) => { next[k] += v * off; next[k + 1] += v * scl; });
    next[0] += pw[i];
    res = next;
  }
  return trimseq(res);
}

// ==============================
//   Module objects
// ==============================

export function build_polynomial_package() {
  const bases: [string, Basis][] = [['polynomial', P.POWER], ['chebyshev', P.CHEBYSHEV], ['hermite', P.HERMITE], ['hermite_e', P.HERMITE_E], ['laguerre', P.LAGUERRE], ['legendre', P.LEGENDRE]];
  const pkg: { [name: string]: any } = {};
  for (const [name, B] of bases) {
    const F = make_functions(B);
    const Cls = make_class(B, F);
    pkg[name] = { ...F, [B.name]: Cls };
    pkg[B.name] = Cls;
  }
  pkg.polyutils = {
    as_series: fn([["alist", undefined], ["trim", true]], (alist, trim) => as_series(isarray(alist) && alist.shape.length == 2 ? range(alist.shape[0]).map(i => alist.index(i)) : alist, trim).map(v => arr(v))),
    trimseq: fn([["seq", undefined]], (seq) => arr(trimseq(nums(seq)))),
    trimcoef: fn([["c", undefined], ["tol", 0]], (c, tol) => arr(P.trimcoef(nums(c), tol))),
    getdomain: fn([["x", undefined]], (x) => arr(P.getdomain(x))),
    mapdomain: fn([["x", undefined], ["old", undefined], ["new", undefined]], (x, old, nw) => P.mapdomain(x, nums(old), nums(nw))),
    mapparms: fn([["old", undefined], ["new", undefined]], (old, nw) => P.mapparms(nums(old), nums(nw))),
  };
  return pkg;
}

// ==============================
//   Legacy: numpy.poly1d, polyval, polyfit, roots...
// ==============================

const real_or_complex = (vals: Complex[]): NDArray => vals.every(z => z.im === 0) ? arr(vals.map(z => z.re)) : new NDArray(vals, [vals.length], complex128);

/** Coefficients with the highest degree first */
export class poly1d {
  coeffs: NDArray;
  variable: string;
  constructor(c_or_r: any, r: boolean = false, variable: string = 'x') {
    this.variable = variable;
    let c = asarray(c_or_r);
    if (c.shape.length === 0) c = arr([+c.item()]);
    if (c.shape.length > 1) throw new Error("Polynomial must be 1d only.");
    if (r) c = legacy_poly(c);
    const v = c.dtype === complex128 ? Array.from(c.flat as any) as Complex[] : nums(c);
    // strip leading zeros
    let k = 0;
    while (k < v.length - 1 && (v[k] instanceof Complex ? (v[k] as Complex).isZero() : v[k] === 0)) k++;
    const trimmed = v.slice(k);
    this.coeffs = c.dtype === complex128 ? new NDArray(trimmed as Complex[], [trimmed.length], complex128) : arr(trimmed as number[]);
    if (this.coeffs.size === 0) this.coeffs = arr([0]);
  }
  get order() { return this.coeffs.size - 1; }
  get c() { return this.coeffs; }
  get coef() { return this.coeffs; }
  get coefficients() { return this.coeffs; }
  get o() { return this.order; }
  get roots() { return legacy_roots(this.coeffs); }
  get r() { return this.roots; }
  get length() { return this.order; }
  call(x: any) { return legacy_polyval(this.coeffs, x); }
  add(o: any) { return new poly1d(legacy_polyadd(this.coeffs, o instanceof poly1d ? o.coeffs : o), false, this.variable); }
  sub(o: any) { return new poly1d(legacy_polysub(this.coeffs, o instanceof poly1d ? o.coeffs : o), false, this.variable); }
  mul(o: any) { return new poly1d(legacy_polymul(this.coeffs, o instanceof poly1d ? o.coeffs : o), false, this.variable); }
  neg() { return new poly1d(la_neg(this.coeffs), false, this.variable); }
  pow(n: number) {
    if (!Number.isInteger(n) || n < 0) throw new Error("Power to non-negative integers only.");
    let res = new poly1d([1], false, this.variable);
    for (let i = 0; i < n; i++) res = res.mul(this);
    return res;
  }
  divmod(o: any): [poly1d, poly1d] {
    const [q, r] = legacy_polydiv(this.coeffs, o instanceof poly1d ? o.coeffs : o);
    return [new poly1d(q, false, this.variable), new poly1d(r, false, this.variable)];
  }
  deriv(m: number = 1) { return new poly1d(legacy_polyder(this.coeffs, m), false, this.variable); }
  integ(m: number = 1, k: number | number[] = 0) { return new poly1d(legacy_polyint(this.coeffs, m, k), false, this.variable); }
  get(i: number) { const n = this.coeffs.size; const idx = n - 1 - i; return idx < 0 || i < 0 ? 0 : (this.coeffs.flat as any)[idx]; }
  equals(o: any) { return o instanceof poly1d && o.coeffs.size === this.coeffs.size && nums(o.coeffs).every((v, i) => v === nums(this.coeffs)[i]); }
  toString() {
    const c = nums(this.coeffs), n = c.length - 1;
    const terms: string[] = [];
    c.forEach((v, i) => {
      const deg = n - i;
      if (v === 0 && n > 0) return;
      const mag = Math.abs(v);
      const body = deg === 0 ? `${mag}` : `${mag === 1 ? '' : mag + ' '}${this.variable}${deg > 1 ? '^' + deg : ''}`;
      terms.push(terms.length === 0 ? (v < 0 ? '-' : '') + body : (v < 0 ? '- ' : '+ ') + body);
    });
    return terms.join(' ') || '0';
  }
}
function la_neg(c: NDArray): NDArray {
  return c.dtype === complex128 ? new NDArray((c.flat as any).map((z: Complex) => z.neg()), c.shape, complex128) : arr(nums(c).map(v => -v));
}

function cx(c: any): Complex[] { return Array.from(asarray(c).flat as any, (v: any) => Complex.from(v)); }
const isC = (c: any) => asarray(c).dtype === complex128;
function mkArr(vals: (number | Complex)[], complexOut: boolean, intOut = false): NDArray {
  return complexOut ? new NDArray(vals.map(v => Complex.from(v)), [vals.length], complex128) : arr(vals as number[], intOut ? 'int64' : float64);
}
const is_int_input = (...xs: any[]) => xs.every(x => { const d = asarray(x instanceof poly1d ? x.coeffs : x).dtype; return d === 'int32' || d === 'int64' || d === 'bool'; });

export function legacy_polyval(p: any, x: any): NDArray {
  const P_ = asarray(p instanceof poly1d ? p.coeffs : p);
  const X = asarray(x instanceof poly1d ? x.coeffs : x);
  if (P_.dtype === complex128 || X.dtype === complex128) {
    const pc = cx(P_);
    return new NDArray((X.flat as any ? Array.from(X.flat as any, (v: any) => Complex.from(v)) : []).map((z: Complex) => pc.reduce((acc, c) => acc.mul(z).add(c), new Complex(0, 0))), X.shape, complex128);
  }
  const pc = nums(P_);
  const isInt = P_.dtype !== float64 && X.dtype !== float64 && P_.dtype !== 'float32' && X.dtype !== 'float32';
  const dt = isInt ? (P_.dtype === 'int64' || X.dtype === 'int64' ? 'int64' : 'int32') : float64;
  return new NDArray(new_buffer(Array.from(X.flat as any, (v: any) => pc.reduce((acc, c) => acc * +v + c, 0)), dt as any), X.shape, dt as any);
}

export function legacy_poly(seq_of_zeros: any): NDArray {
  let z = asarray(seq_of_zeros);
  if (z.shape.length === 2) {
    if (z.shape[0] !== z.shape[1]) throw new Error("input must be a square array");
    z = la.eigvals(z);
  } else if (z.shape.length !== 1) throw new Error("input must be 1d or non-empty square 2d array.");
  if (z.size === 0) return new NDArray(new_buffer([1], float64), [], float64);
  const roots = cx(z);
  let a: Complex[] = [new Complex(1, 0)];
  for (const r of roots) {
    const next: Complex[] = new Array(a.length + 1).fill(new Complex(0, 0));
    a.forEach((c, i) => { next[i] = next[i].add(c); next[i + 1] = next[i + 1].sub(c.mul(r)); });
    a = next;
  }
  // like numpy: the result is real if the roots are real or come in conjugate pairs
  if (a.every(c => Math.abs(c.im) < 1e-12 * Math.max(1, Math.abs(c.re)))) {
    return arr(a.map(c => c.re));
  }
  return new NDArray(a, [a.length], complex128);
}

export function legacy_roots(p: any): NDArray {
  const P_ = asarray(p instanceof poly1d ? p.coeffs : p);
  if (P_.shape.length !== 1) throw new Error("Input must be a rank-1 array.");
  let c = cx(P_);
  const nz = c.map((z, i) => z.isZero() ? -1 : i).filter(i => i >= 0);
  if (nz.length === 0) return arr([]);
  const trailing = c.length - 1 - nz[nz.length - 1];
  c = c.slice(nz[0], nz[nz.length - 1] + 1);
  const n = c.length - 1;
  let vals: Complex[] = [];
  if (n >= 1) {
    if (n === 1) vals = [c[1].neg().div(c[0]).neg().neg()];
    if (n === 1) vals = [c[1].div(c[0]).neg()];
    else {
      const A = new Array(n * n).fill(0);
      for (let j = 0; j < n; j++) A[j] = -(c[j + 1].div(c[0]).re);
      for (let i = 1; i < n; i++) A[i * n + i - 1] = 1;
      if (c.some(z => z.im !== 0)) throw new Error("roots of polynomials with complex coefficients are not supported");
      vals = Array.from(la.eigvals(new NDArray(new_buffer(A, float64), [n, n], float64)).flat as any) as Complex[];
    }
  }
  for (let i = 0; i < trailing; i++) vals.push(new Complex(0, 0));
  if (vals.every(z => z.im === 0)) return arr(vals.map(z => z.re));
  return new NDArray(vals, [vals.length], complex128);
}

const pc_nums = (p: any): { v: (number | Complex)[], complex: boolean } => {
  const a = asarray(p instanceof poly1d ? p.coeffs : p);
  return a.dtype === complex128 ? { v: cx(a), complex: true } : { v: nums(a), complex: false };
};

export function legacy_polyadd(a1: any, a2: any): NDArray {
  const x = pc_nums(a1), y = pc_nums(a2);
  const n = Math.max(x.v.length, y.v.length);
  const cplx = x.complex || y.complex;
  const get = (v: (number | Complex)[], i: number) => { const k = i - (n - v.length); return k < 0 ? 0 : v[k]; };
  const vals = range(n).map(i => cplx ? Complex.from(get(x.v, i)).add(Complex.from(get(y.v, i))) : (get(x.v, i) as number) + (get(y.v, i) as number));
  return mkArr(vals, cplx, is_int_input(a1, a2));
}
export function legacy_polysub(a1: any, a2: any): NDArray {
  const y = pc_nums(a2);
  return legacy_polyadd(a1, mkArr(y.v.map(v => y.complex ? (v as Complex).neg() : -(v as number)), y.complex, is_int_input(a2)));
}
export function legacy_polymul(a1: any, a2: any): NDArray {
  const x = pc_nums(a1), y = pc_nums(a2);
  const cplx = x.complex || y.complex;
  const out: (number | Complex)[] = new Array(x.v.length + y.v.length - 1).fill(cplx ? new Complex(0, 0) : 0);
  for (let i = 0; i < x.v.length; i++) for (let j = 0; j < y.v.length; j++)
    out[i + j] = cplx ? (out[i + j] as Complex).add(Complex.from(x.v[i]).mul(Complex.from(y.v[j]))) : (out[i + j] as number) + (x.v[i] as number) * (y.v[j] as number);
  return mkArr(out, cplx, is_int_input(a1, a2));
}
export function legacy_polydiv(u: any, v: any): NDArray[] {
  const U = pc_nums(u), V = pc_nums(v);
  if (U.complex || V.complex) throw new Error("complex polynomial division is not supported");
  const a = U.v as number[], b = V.v as number[];
  const m = a.length - 1, n = b.length - 1;
  const scale = 1 / b[0];
  const q = new Array(Math.max(m - n + 1, 1)).fill(0);
  let r = a.slice();
  for (let k = 0; k < m - n + 1; k++) {
    const d = scale * r[k];
    q[k] = d;
    for (let j = 0; j <= n; j++) r[k + j] -= d * b[j];
  }
  // leading (near) zeros of the remainder are removed, like numpy.allclose(r[0], 0, rtol=1e-14)
  while (r.length > 1 && Math.abs(r[0]) <= 1e-8 + 1e-14 * 0) r = r.slice(1);
  return [arr(q), arr(r)];
}
export function legacy_polyder(p: any, m: number = 1): NDArray {
  if (!Number.isInteger(m) || m < 0) throw new Error("Order of derivative must be positive (see polyint)");
  let { v, complex } = pc_nums(p);
  for (let i = 0; i < m; i++) {
    const n = v.length - 1;
    v = range(Math.max(n, 0)).map(j => complex ? (v[j] as Complex).mul(new Complex(n - j, 0)) : (v[j] as number) * (n - j));
  }
  return mkArr(v, complex, is_int_input(p));
}
export function legacy_polyint(p: any, m: number = 1, k: any = null): NDArray {
  if (!Number.isInteger(m) || m < 0) throw new Error("Order of integral must be positive (see polyder)");
  const ks = k === null ? [] : nums(k);
  if (ks.length > m) throw new Error("k must be a scalar or a rank-1 array of length 1 or >m.");
  const kv = ks.length === 1 && m > 1 ? new Array(m).fill(ks[0]) : ks.concat(new Array(Math.max(0, m - ks.length)).fill(0));
  let { v, complex } = pc_nums(p);
  if (complex) throw new Error("polyint of complex polynomials is not supported");
  let cur = v as number[];
  for (let i = 0; i < m; i++) {
    const n = cur.length;
    cur = [...cur.map((c, j) => c / (n - j)), kv[i]];
  }
  return arr(cur);
}

export function legacy_polyfit(x: any, y: any, deg: number, rcond: number | null = null, full: boolean = false, w: any = null, cov: boolean | string = false): any {
  const X = nums(x);
  const Y = asarray(y);
  const order = deg + 1;
  if (!Number.isInteger(deg) || deg < 0) throw new Error("expected deg >= 0");
  if (asarray(x).shape.length !== 1) throw new Error("expected 1D vector for x");
  if (X.length === 0) throw new Error("expected non-empty vector for x");
  if (Y.shape.length < 1 || Y.shape.length > 2) throw new Error("expected 1D or 2D array for y");
  if (X.length !== Y.shape[0]) throw new Error("expected x and y to have same length");
  const k = Y.shape.length === 2 ? Y.shape[1] : 1;
  const n = X.length;
  let lhs: number[][] = X.map(xv => range(order).map(j => Math.pow(xv, deg - j)));
  let rhs: number[][] = range(n).map(i => range(k).map(j => +(Y.flat as any)[i * k + j]));
  if (w !== null) {
    const wv = nums(w);
    if (asarray(w).shape.length !== 1) throw new Error("expected a 1-d array for weights");
    if (wv.length !== n) throw new Error("expected w and y to have the same length");
    lhs = lhs.map((r, i) => r.map(v => v * wv[i]));
    rhs = rhs.map((r, i) => r.map(v => v * wv[i]));
  }
  const rc = rcond === null ? n * 2.220446049250313e-16 : rcond;
  const scale = range(order).map(j => Math.sqrt(lhs.reduce((acc, r) => acc + r[j] * r[j], 0)) || 1);
  const A = new NDArray(new_buffer(lhs.flatMap(r => r.map((v, j) => v / scale[j])), float64), [n, order], float64);
  const Bm = new NDArray(new_buffer(rhs.flat(), float64), [n, k], float64);
  const [sol, resids, rank, s] = la.lstsq(A, Bm, rc);
  const sf = sol.flat as any;
  const c = new NDArray(new_buffer(range(order * k).map(t => sf[t] / scale[Math.floor(t / k)]), float64), Y.shape.length === 2 ? [order, k] : [order], float64);
  if (full) return [c, resids, rank, s, rc];
  if (cov) {
    // Vbase = inv(A^T A) / outer(scale, scale)
    const AtA = la.dot((A as any).T, A);
    const Vb = la.inv(AtA);
    const vf = Vb.flat as any;
    let fac = 1;
    if (cov !== 'unscaled') {
      if (n <= order) throw new Error("the number of data points must exceed order to scale the covariance matrix");
      const res = resids.size ? nums(resids)[0] : (() => { const Ax = la.matmul(A, sol); const axf = Ax.flat as any; return range(n).reduce((acc, i) => acc + Math.pow(axf[i] - rhs[i][0], 2), 0); })();
      fac = res / (n - order);
    }
    const V = new NDArray(new_buffer(range(order * order).map(t => vf[t] / (scale[Math.floor(t / order)] * scale[t % order]) * fac), float64), [order, order], float64);
    return [c, V];
  }
  return c;
}
