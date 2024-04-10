//@ts-check
import { np, nd_modules } from "./_globals";
import { TypedArray, new_buffer } from "../dtypes";
import type NDArray from "../NDArray";


export function random(shape) {
  return nd_modules.basic.new_from(shape, (_) => Math.random(), Float64Array)
};
export function uniform(a, b, shape) {
  return random(shape).multiply(b - a).add(a);
};
export function exponential(mean, shape) {
  return np.multiply(mean, np.subtract(0, np.log(random(shape))));
};

export function _normal_buffer(n: number) {
  const out = new_buffer(n);
  let i = 0;
  while (i < n) {
    let u = Math.random() * 2 - 1;
    let v = Math.random() * 2 - 1;
    let s = u * u + v * v;
    if (s > 1) continue;
    let x = Math.sqrt(-2 * Math.log(s) / s) * u;
    let y = Math.sqrt(-2 * Math.log(s) / s) * v;
    out[i++] = x;
    if (i < n) out[i++] = y;
  }
  return out;
}
export function randn(shape: number[]) {
  const flat = _normal_buffer(shape.reduce((a, b) => a * b, 1));
  return new np.NDArray(flat, shape);
};
export function normal(mean, std, shape) {
  return randn(shape).multiply(std).add(mean);
};


/** @param {any[]} list */
export function _shuffle(list) {
  // Fisher-Yates (aka Knuth) shuffle.
  // https://stackoverflow.com/a/2450976
  for (let i = list.length - 1; i >= 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
}

export function _shuffled<T>(list: T) {
  const out = (list as any).slice();
  _shuffle(out);
  return out;
}

export function shuffled(arr: NDArray) {
  if (arr.shape.length == 0) return arr;
  if (arr.shape.length == 1) {
    const flat = _shuffled(arr.flat)
    return new np.NDArray(flat, arr.shape);
  }
  const perm = _shuffled(Array.from({ length: arr.length }, (_, i) => i));
  const out = np.empty(arr.shape);
  for (let i = 0; i < arr.length; i++) out.assign(arr.index(perm[i]), i);
  return out;
}

export function shuffle(arr: NDArray) {
  arr.assign(shuffled(arr));
}
