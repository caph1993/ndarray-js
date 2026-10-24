//@ts-check
import { NDArray, asarray, broadcast_n_shapes } from "../NDArray";

/** True if the arrays share memory (the exact set of elements is compared) */
export function shares_memory(a: NDArray, b: NDArray, max_work: number | null = null): boolean {
  a = asarray(a); b = asarray(b);
  if (a._flat !== b._flat || a.size == 0 || b.size == 0) return false;
  const ia = a._simpleIndexes ? a._simpleIndexes.indices : null;
  const ib = b._simpleIndexes ? b._simpleIndexes.indices : null;
  if (!ia || !ib) return true;
  const set = new Set<number>(ia);
  for (const i of ib) if (set.has(i)) return true;
  return false;
}

/** Cheap check: True if the arrays are views of the same buffer (they may still not overlap) */
export function may_share_memory(a: NDArray, b: NDArray, max_work: number | null = null): boolean {
  a = asarray(a); b = asarray(b);
  return a._flat === b._flat && a.size > 0 && b.size > 0;
}

/** Broadcast any number of shapes against each other */
export function broadcast_shapes(...shapes: (number | number[])[]): number[] {
  const list = shapes.map(s => typeof s === "number" ? [s] : Array.from(s));
  if (list.length == 0) return [];
  const [, shape] = broadcast_n_shapes(...list);
  return shape as number[];
}

// Numpy's buffer size and error handling settings have no equivalent in JS; these keep the API.
let bufsize = 8192;
export function getbufsize() { return bufsize; }
export function setbufsize(size: number) { const old = bufsize; bufsize = size; return old; }

/** Path to the numpy C headers. Not applicable in JS: always returns an empty string. */
export function get_include() { return ""; }

export const config = {
  name: "ndarray-js",
  language: "TypeScript",
  dtypes: ["bool", "int32", "float32", "float64", "object"],
  backend: "plain JavaScript typed arrays",
};
export function show_config() { console.log(JSON.stringify(config, null, 2)); return config; }
export function show_runtime() {
  const runtime = {
    ...config,
    runtime: typeof process !== "undefined" && (process as any).version ? `node ${(process as any).version}` : "browser",
  };
  console.log(JSON.stringify(runtime, null, 2));
  return runtime;
}

/** Prints the signature/docs of an np function (best effort) */
export function info(obj: any = null) {
  const text = obj === null ? "ndarray-js: see https://caph1993.github.io/ndarray-js/" :
    typeof obj === "function" ? `${obj.name || "function"}\n${obj.toString().slice(0, 300)}` : String(obj);
  console.log(text);
  return text;
}
