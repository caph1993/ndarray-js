//@ts-check
/**
 * Business day functions. ndarray-js has no datetime64 dtype: dates are given as
 * 'YYYY-MM-DD' strings (or JS Date objects, or arrays of them) and returned as 'YYYY-MM-DD' strings.
 */
import { NDArray } from "../NDArray";
import { np } from "../np_circular_import";
import { new_buffer } from "../dtypes";

const DAY = 86400000;
const NAMES = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];

/** days since 1970-01-01 (NaN for NaT) */
function to_days(d: any): number {
  if (d === null || d === undefined) return NaN;
  if (d instanceof Date) return Number.isNaN(d.getTime()) ? NaN : Math.floor(d.getTime() / DAY);
  if (typeof d === 'number') return d;
  const s = String(d).trim();
  if (s === 'NaT' || s === 'nat') return NaN;
  const m = /^(-?\d{4,6})-(\d{2})-(\d{2})(?:[T ].*)?$/.exec(s);
  if (!m) {
    const m2 = /^(\d{4})-(\d{2})$/.exec(s);
    if (m2) return Date.UTC(+m2[1], +m2[2] - 1, 1) / DAY;
    throw new Error(`Error parsing datetime string "${s}"`);
  }
  const t = Date.UTC(+m[1], +m[2] - 1, +m[3]);
  const back = new Date(t);
  if (back.getUTCMonth() !== +m[2] - 1) throw new Error(`Day out of range in datetime string "${s}"`);
  return Math.round(t / DAY);
}
function from_days(n: number): string {
  if (Number.isNaN(n)) return 'NaT';
  const d = new Date(n * DAY);
  const y = d.getUTCFullYear(), p = (v: number, w: number) => String(v).padStart(w, '0');
  return `${y < 0 ? '-' : ''}${p(Math.abs(y), 4)}-${p(d.getUTCMonth() + 1, 2)}-${p(d.getUTCDate(), 2)}`;
}
/** weekday 0 = Monday */
function weekday(n: number): number { return (((n + 3) % 7) + 7) % 7; }

function parse_weekmask(w: any): boolean[] {
  if (typeof w === 'string') {
    const s = w.trim();
    if (/^[01]{7}$/.test(s)) return s.split('').map(c => c === '1');
    const out = [false, false, false, false, false, false, false];
    for (const tok of s.split(/\s+/).filter(Boolean)) {
      const i = NAMES.indexOf(tok.toLowerCase().slice(0, 3));
      if (i < 0 || tok.length !== 3) throw new Error("Invalid weekmask string");
      out[i] = true;
    }
    return out;
  }
  const a = Array.from((w instanceof NDArray ? w.flat : w) as any).map(Boolean);
  if (a.length !== 7) throw new Error("A business day weekmask array must have length 7");
  return a as boolean[];
}

export class busdaycalendar {
  weekmask: NDArray; holidays: NDArray;
  constructor(weekmask: any = '1111100', holidays: any = []) {
    const wm = parse_weekmask(weekmask);
    if (!wm.some(Boolean)) throw new Error("Cannot construct a numpy.busdaycalendar with a weekmask of all zeros");
    this.weekmask = np.array(wm);
    const hs = Array.from(new Set(flat(holidays).map(to_days).filter(v => !Number.isNaN(v)))).sort((a, b) => a - b);
    this.holidays = np.array(hs.map(from_days));
  }
}
function flat(x: any): any[] {
  if (x instanceof NDArray) return Array.from(x.flat as any);
  if (Array.isArray(x)) return ([] as any[]).concat(...x.map(flat));
  return [x];
}
interface Cal { mask: boolean[]; hol: Set<number>; nbus: number }
function calendar(weekmask: any, holidays: any, cal: any): Cal {
  let mask: boolean[], hol: number[];
  if (cal) { mask = Array.from(cal.weekmask.flat as any).map(Boolean) as boolean[]; hol = flat(cal.holidays).map(to_days); }
  else { mask = parse_weekmask(weekmask); hol = flat(holidays).map(to_days).filter(v => !Number.isNaN(v)); if (!mask.some(Boolean)) throw new Error("Cannot construct a numpy.busdaycalendar with a weekmask of all zeros"); }
  return { mask, hol: new Set(hol), nbus: mask.filter(Boolean).length };
}
const is_bd = (c: Cal, d: number) => c.mask[weekday(d)] && !c.hol.has(d);

function broadcast_map(args: any[], f: (...v: any[]) => any, dtype?: any): any {
  const parts = args.map(a => {
    if (a instanceof NDArray) return { shape: a.shape as number[], flat: Array.from(a.flat as any) };
    if (Array.isArray(a)) { const x = np.array(a); return { shape: x.shape as number[], flat: Array.from(x.flat as any) }; }
    return { shape: [] as number[], flat: [a] };
  });
  const shape = np.broadcast_shapes(...parts.map(p => p.shape));
  const n = shape.reduce((x: number, y: number) => x * y, 1);
  const res: any[] = [];
  const exp = parts.map(p => p.shape.length === 0 || p.flat.length === n ? null : np.ravel(np.broadcast_to(np.reshape(np.array(p.flat), p.shape), shape)));
  for (let i = 0; i < n; i++) res.push(f(...parts.map((p, k) => exp[k] ? (exp[k] as any).flat[i] : p.flat[p.flat.length === 1 ? 0 : i])));
  if (shape.length === 0) return res[0];
  return np.reshape(dtype ? np.array(res, dtype) : np.array(res), shape);
}

export function is_busday(dates: any, weekmask: any = '1111100', holidays: any = [], busdaycal: any = null, out: any = null) {
  const c = calendar(weekmask, holidays, busdaycal);
  const r = broadcast_map([dates], (d) => { const n = to_days(d); return !Number.isNaN(n) && is_bd(c, n); }, 'bool');
  if (out) { np.copyto(out, r); return out; }
  return r;
}
function roll_to(c: Cal, d: number, roll: string): number {
  if (is_bd(c, d)) return d;
  switch (roll) {
    case 'raise': throw new Error("Non-business day date in busday_offset");
    case 'nat': return NaN;
    case 'forward': case 'following': while (!is_bd(c, d)) d++; return d;
    case 'backward': case 'preceding': while (!is_bd(c, d)) d--; return d;
    case 'modifiedfollowing': case 'modifiedpreceding': {
      const orig = d, month = new Date(d * DAY).getUTCMonth();
      const fwd = roll === 'modifiedfollowing';
      let e = d; while (!is_bd(c, e)) e += fwd ? 1 : -1;
      if (new Date(e * DAY).getUTCMonth() !== month) { e = orig; while (!is_bd(c, e)) e += fwd ? -1 : 1; }
      return e;
    }
  }
  throw new Error(`Invalid business day roll parameter "${roll}"`);
}
export function busday_offset(dates: any, offsets: any, roll: string = 'raise', weekmask: any = '1111100', holidays: any = [], busdaycal: any = null, out: any = null) {
  const c = calendar(weekmask, holidays, busdaycal);
  const r = broadcast_map([dates, offsets], (d, o) => {
    let n = to_days(d);
    if (Number.isNaN(n)) return 'NaT';
    n = roll_to(c, n, roll);
    if (Number.isNaN(n)) return 'NaT';
    o = Number(o);
    const step = o > 0 ? 1 : -1;
    // jump whole weeks when there are no holidays
    if (c.hol.size === 0) {
      const weeks = Math.trunc(Math.abs(o) / c.nbus) * (o < 0 ? -1 : 1);
      n += 7 * weeks; o -= weeks * c.nbus;
    }
    for (let left = Math.abs(o); left > 0;) { n += step; if (is_bd(c, n)) left--; }
    return from_days(n);
  });
  if (out) { np.copyto(out, r); return out; }
  return r;
}
export function busday_count(begindates: any, enddates: any, weekmask: any = '1111100', holidays: any = [], busdaycal: any = null, out: any = null) {
  const c = calendar(weekmask, holidays, busdaycal);
  const count = (a: number, b: number) => { let k = 0; for (let d = a; d < b; d++) if (is_bd(c, d)) k++; return k; };
  const r = broadcast_map([begindates, enddates], (a, b) => {
    const x = to_days(a), y = to_days(b);
    if (Number.isNaN(x) || Number.isNaN(y)) throw new Error("busday_count with NaT is not supported");
    if (x <= y) {
      if (c.hol.size === 0) { // whole weeks
        const w = Math.floor((y - x) / 7); let k = w * c.nbus;
        for (let d = x + w * 7; d < y; d++) if (is_bd(c, d)) k++;
        return k;
      }
      return count(x, y);
    }
    return -count(y + 1, x + 1);
  }, 'int64');
  if (out) { np.copyto(out, r); return out; }
  return r;
}
export function datetime_as_string(arr: any, unit: string = 'auto', timezone: string = 'naive', casting: string = 'same_kind') {
  const f = (d: any) => {
    const n = to_days(d); if (Number.isNaN(n)) return 'NaT';
    const s = from_days(n);
    switch (unit) {
      case 'Y': return s.slice(0, s.length - 6);
      case 'M': return s.slice(0, s.length - 3);
      case 'h': return s + 'T00';
      case 'm': return s + 'T00:00';
      case 's': return s + 'T00:00:00';
      case 'ms': return s + 'T00:00:00.000';
      case 'us': return s + 'T00:00:00.000000';
      case 'ns': return s + 'T00:00:00.000000000';
      default: return s;
    }
  };
  return broadcast_map([arr], f);
}
export function datetime_data(dtype: any): [string, number] {
  const s = String(dtype && dtype.name !== undefined ? dtype.name : dtype);
  const m = /^(?:datetime64|timedelta64|M8|m8)(?:\[(\d*)([a-zA-Z]+)\])?$/.exec(s);
  if (!m) throw new TypeError("Cannot get datetime metadata from non-datetime type");
  return [m[2] || 'generic', m[1] ? +m[1] : 1];
}
export function isnat(x: any) {
  return broadcast_map([x], (d) => {
    if (d === null || d === undefined) return true;
    if (d instanceof Date) return Number.isNaN(d.getTime());
    if (typeof d === 'number') return Number.isNaN(d);
    return /^nat$/i.test(String(d));
  }, 'bool');
}
