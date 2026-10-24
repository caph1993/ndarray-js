//@ts-check

import NDArray from "../NDArray";
import { bool, dtype_is_float, object } from '../dtypes';

const EDGE_ITEMS = 3;
const SUMMARIZE_THRESHOLD = 1000;

function format_number(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (Number.isInteger(x) && Math.abs(x) < 1e16) return String(x);
  // 8 significant digits, as numpy
  const s = String(Number(x.toPrecision(8)));
  return s;
}

/**
 * Human readable representation, e.g.
 * np([[1, 2],
 *     [3, 4]], shape=(2, 2), dtype=int32)
 * Large arrays are summarized with "...".
 */
export function humanReadable(arr: NDArray): string {
  const shape = arr.shape;
  const dtype = arr.dtype;
  const suffix = `shape=(${shape.join(', ')}${shape.length == 1 ? ',' : ''}), dtype=${dtype}`;
  const fmt = (v: any) => {
    if (dtype === bool) return v ? "true" : "false";
    if (typeof v === "number") return format_number(v);
    return typeof v === "string" ? JSON.stringify(v) : String(v);
  };
  if (shape.length == 0) return `np(${fmt(arr.flat[0])}, ${suffix})`;
  const summarize = arr.size > SUMMARIZE_THRESHOLD;
  const flat = arr.flat as any;
  const strides: number[] = [];
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { strides[i] = acc; acc *= shape[i]; }
  // Positions to show along each axis
  const shown = shape.map(n => (summarize && n > 2 * EDGE_ITEMS)
    ? [...Array.from({ length: EDGE_ITEMS }, (_, i) => i), -1, ...Array.from({ length: EDGE_ITEMS }, (_, i) => n - EDGE_ITEMS + i)]
    : Array.from({ length: n }, (_, i) => i));
  // Common width for aligned columns
  let width = 0;
  const collect = (axis: number, offset: number): void => {
    for (const i of shown[axis]) {
      if (i < 0) continue;
      if (axis == shape.length - 1) width = Math.max(width, fmt(flat[offset + i]).length);
      else collect(axis + 1, offset + i * strides[axis]);
    }
  };
  collect(0, 0);
  const prefix = "np(";
  const render = (axis: number, offset: number, indent: number): string => {
    const last = axis == shape.length - 1;
    const parts: string[] = shown[axis].map(i => {
      if (i < 0) return "...";
      return last ? fmt(flat[offset + i]).padStart(width) : render(axis + 1, offset + i * strides[axis], indent + 1);
    });
    if (last) return `[${parts.join(', ')}]`;
    const sep = ",\n".repeat(1) + " ".repeat(indent + 1) + "";
    const blank = shape.length - axis > 2 ? "\n" : "";
    return `[${parts.join("," + "\n" + blank + " ".repeat(indent + 1))}]`;
  };
  return `${prefix}${render(0, 0, prefix.length)}, ${suffix})`;
}

NDArray.prototype.toString = function () {
  return humanReadable(this);
}
