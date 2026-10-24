//@ts-check
/**
 * Exact decimal formatting of doubles with round-half-even (what C's printf and numpy's dragon4 produce).
 * JavaScript's toFixed/toExponential round ties away from zero (0.25.toFixed(1) === "0.3").
 */
const BI = (v: string | number): any => (globalThis as any).BigInt(v);

/** x = mantissa * 2**exponent exactly */
function decompose(x: number): { m: any, e: number } {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, x);
  const hi = dv.getUint32(0), lo = dv.getUint32(4);
  const expBits = (hi >>> 20) & 0x7ff;
  let m = (BI(hi & 0xfffff) * BI(4294967296)) + BI(lo);
  let e: number;
  if (expBits === 0) e = -1074;
  else { m = m + BI(1) * BI(4503599627370496); e = expBits - 1075; }
  return { m, e };
}

/** Rounds m * 2**e * 10**d to an integer, half to even (d may be negative) */
function scaled_round(m: any, e: number, d: number): any {
  let num = m, den = BI(1);
  if (e >= 0) num = num * (BI(2) ** BI(e)); else den = den * (BI(2) ** BI(-e));
  if (d >= 0) num = num * (BI(10) ** BI(d)); else den = den * (BI(10) ** BI(-d));
  let q = num / den;
  const r = num - q * den;
  const twice = r * BI(2);
  if (twice > den || (twice === den && (q % BI(2)) === BI(1))) q = q + BI(1);
  return q;
}

/** |x| with `d` digits after the decimal point */
export function fixed_exact(x: number, d: number): string {
  const a = Math.abs(x);
  if (a === 0) return d > 0 ? '0.' + '0'.repeat(d) : '0';
  const { m, e } = decompose(a);
  const q = scaled_round(m, e, d).toString();
  if (d === 0) return q;
  const padded = q.padStart(d + 1, '0');
  return padded.slice(0, padded.length - d) + '.' + padded.slice(padded.length - d);
}

/** |x| as d.ddd e+XX with `d` digits after the point in the mantissa */
export function exp_exact(x: number, d: number): { mantissa: string, exponent: number } {
  const a = Math.abs(x);
  if (a === 0) return { mantissa: d > 0 ? '0.' + '0'.repeat(d) : '0', exponent: 0 };
  const { m, e } = decompose(a);
  let exp10 = Math.floor(Math.log10(a));
  for (let attempt = 0; attempt < 3; attempt++) {
    const q = scaled_round(m, e, d - exp10);        // integer with d+1 digits (or d+2 if rounding carried)
    const s = q.toString();
    if (s.length === d + 1) {
      return { mantissa: d > 0 ? s[0] + '.' + s.slice(1) : s, exponent: exp10 };
    }
    exp10 += s.length > d + 1 ? 1 : -1;
  }
  const s = scaled_round(m, e, d - exp10).toString();
  return { mantissa: d > 0 ? s[0] + '.' + s.slice(1, d + 1) : s[0], exponent: exp10 };
}
