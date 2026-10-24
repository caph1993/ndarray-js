//@ts-check
/**
 * Immutable complex number, the element type of arrays with dtype "complex128".
 * The functions follow the principal branches of the C99 standard (as numpy does).
 */
export class Complex {
  readonly re: number;
  readonly im: number;
  constructor(re: number, im: number = 0) { this.re = re; this.im = im; }

  static from(x: any): Complex {
    if (x instanceof Complex) return x;
    if (typeof x === "boolean") return new Complex(x ? 1 : 0, 0);
    if (typeof x === "number") return new Complex(x, 0);
    if (x === null || x === undefined) return new Complex(NaN, 0);
    if (typeof x === "object" && "re" in x) return new Complex(+x.re, +(x.im ?? 0));
    return new Complex(+x, 0);
  }
  static polar(r: number, theta: number) { return new Complex(r * Math.cos(theta), r * Math.sin(theta)); }

  get abs() { return Math.hypot(this.re, this.im); }
  get arg() { return Math.atan2(this.im, this.re); }
  get isNaN() { return Number.isNaN(this.re) || Number.isNaN(this.im); }
  get isFinite() { return Number.isFinite(this.re) && Number.isFinite(this.im); }
  get isInf() { return !this.isNaN && (Math.abs(this.re) === Infinity || Math.abs(this.im) === Infinity); }

  add(o: Complex) { return new Complex(this.re + o.re, this.im + o.im); }
  sub(o: Complex) { return new Complex(this.re - o.re, this.im - o.im); }
  mul(o: Complex) { return new Complex(this.re * o.re - this.im * o.im, this.re * o.im + this.im * o.re); }
  /** Smith's algorithm */
  div(o: Complex): Complex {
    const { re: a, im: b } = this, { re: c, im: d } = o;
    if (d === 0) return new Complex(a / c, b / c);
    if (Math.abs(c) >= Math.abs(d)) { const r = d / c, den = c + d * r; return new Complex((a + b * r) / den, (b - a * r) / den); }
    const r = c / d, den = c * r + d;
    return new Complex((a * r + b) / den, (b * r - a) / den);
  }
  neg() { return new Complex(-this.re, -this.im); }
  conj() { return new Complex(this.re, -this.im); }
  recip() { return ONE.div(this); }
  square() { return this.mul(this); }
  eq(o: Complex) { return this.re === o.re && this.im === o.im; }
  isZero() { return this.re === 0 && this.im === 0; }

  exp(): Complex {
    if (this.im === 0) return new Complex(Math.exp(this.re), this.im);
    const e = Math.exp(this.re);
    return new Complex(e * Math.cos(this.im), e * Math.sin(this.im));
  }
  log(): Complex { return new Complex(Math.log(this.abs), this.arg); }
  sqrt(): Complex {
    const { re, im } = this;
    if (re === 0 && im === 0) return new Complex(0, im);
    const m = this.abs;
    const t = Math.sqrt((m + Math.abs(re)) / 2);
    if (re >= 0) return new Complex(t, im / (2 * t));
    return new Complex(Math.abs(im) / (2 * t), (im < 0 || Object.is(im, -0)) ? -t : t);
  }
  pow(w: Complex): Complex {
    if (w.im === 0 && Number.isInteger(w.re) && Math.abs(w.re) <= 100) {
      let n = Math.abs(w.re), base: Complex = this, result: Complex = ONE;
      while (n > 0) { if (n & 1) result = result.mul(base); base = base.mul(base); n >>= 1; }
      return w.re < 0 ? ONE.div(result) : result;
    }
    if (this.isZero()) return w.re > 0 ? ZERO : (w.isZero() ? ONE : new Complex(NaN, NaN));
    return this.log().mul(w).exp();
  }
  sin() { return new Complex(Math.sin(this.re) * Math.cosh(this.im), Math.cos(this.re) * Math.sinh(this.im)); }
  cos() { return new Complex(Math.cos(this.re) * Math.cosh(this.im), -Math.sin(this.re) * Math.sinh(this.im)); }
  tan() {
    const a = 2 * this.re, b = 2 * this.im;
    if (Math.abs(b) > 40) return new Complex(0, Math.sign(b));
    const d = Math.cos(a) + Math.cosh(b);
    return new Complex(Math.sin(a) / d, Math.sinh(b) / d);
  }
  sinh() { return new Complex(Math.sinh(this.re) * Math.cos(this.im), Math.cosh(this.re) * Math.sin(this.im)); }
  cosh() { return new Complex(Math.cosh(this.re) * Math.cos(this.im), Math.sinh(this.re) * Math.sin(this.im)); }
  tanh() {
    const a = 2 * this.re, b = 2 * this.im;
    if (Math.abs(a) > 40) return new Complex(Math.sign(a), 0);
    const d = Math.cosh(a) + Math.cos(b);
    return new Complex(Math.sinh(a) / d, Math.sin(b) / d);
  }
  asinh(): Complex { return this.add(this.square().add(ONE).sqrt()).log(); }
  acosh(): Complex { return this.add(this.add(ONE).sqrt().mul(this.sub(ONE).sqrt())).log(); }
  atanh(): Complex {
    if (this.im === 0 && Math.abs(this.re) > 1) {
      const up = !(Object.is(this.im, -0));
      return new Complex(0.5 * Math.log(Math.abs((1 + this.re) / (1 - this.re))), (up ? 1 : -1) * Math.PI / 2);
    }
    return this.atanh_();
  }
  private atanh_(): Complex { return ONE.add(this).div(ONE.sub(this)).log().mul(new Complex(0.5, 0)); }
  asin(): Complex {
    if (this.im === 0 && Math.abs(this.re) > 1) {
      const up = !(Object.is(this.im, -0));
      return new Complex(Math.sign(this.re) * Math.PI / 2, (up ? 1 : -1) * Math.acosh(Math.abs(this.re)));
    }
    return this.asin_();
  }
  private asin_(): Complex { const iz = new Complex(-this.im, this.re); const r = iz.add(ONE.sub(this.square()).sqrt()).log(); return new Complex(r.im, -r.re); }
  acos(): Complex { const a = this.asin(); return new Complex(Math.PI / 2 - a.re, -a.im); }
  atan(): Complex { const iz = new Complex(-this.im, this.re); const r = ONE.sub(iz).log().sub(ONE.add(iz).log()); return new Complex(-r.im / 2, r.re / 2); }
  sign(): Complex { const m = this.abs; return m === 0 ? ZERO : new Complex(this.re / m, this.im / m); }

  toString(): string {
    const f = (x: number) => Number.isNaN(x) ? "nan" : (x === Infinity ? "inf" : (x === -Infinity ? "-inf" : String(Number(x.toPrecision(8)))));
    const neg = this.im < 0 || Object.is(this.im, -0);
    return `${f(this.re)}${neg ? "-" : "+"}${f(Math.abs(this.im))}j`;
  }
  valueOf(): number { return NaN; }
  toJSON() { return { re: this.re, im: this.im }; }
}

export const ZERO = new Complex(0, 0);
export const ONE = new Complex(1, 0);
export const I = new Complex(0, 1);

export function is_complex(x: any): x is Complex { return x instanceof Complex; }

/** Lexicographic order used by numpy to compare/sort complex numbers (nans at the end) */
export function cmp_complex(a: Complex, b: Complex): number {
  const an = a.isNaN, bn = b.isNaN;
  if (an || bn) return an ? (bn ? 0 : 1) : -1;
  if (a.re < b.re) return -1;
  if (a.re > b.re) return 1;
  if (a.im < b.im) return -1;
  if (a.im > b.im) return 1;
  return 0;
}
