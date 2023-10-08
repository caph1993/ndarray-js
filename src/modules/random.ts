//@ts-check
import { np, nd_modules } from "./_globals";


export function random(shape) {
  return nd_modules.basic.new_from(shape, (_) => Math.random(), Number)
};
export function uniform(a, b, shape) {
  return random(shape).multiply(b - a).add(a);
};
export function exponential(mean, shape) {
  return np.multiply(mean, np.subtract(0, np.log(random(shape))));
};

/** @param {number} n */
export function __normal(n) {
  const out: number[] = [];
  while (out.length < n) {
    let u = Math.random() * 2 - 1;
    let v = Math.random() * 2 - 1;
    let s = u * u + v * v;
    if (s >= 1) continue;
    let x = Math.sqrt(-2 * Math.log(s) / s) * u;
    let y = Math.sqrt(-2 * Math.log(s) / s) * v;
    out.push(x);
    out.push(y);
  }
  if (out.length > n) out.pop();
  return out;
}
export function randn(shape) {
  const flat = __normal(np.prod(shape));
  return new np.NDArray(flat, shape, Number);
};
export function normal(mean, std, shape) {
  return np.add(mean, np.multiply(std, shape));
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

/** @param {any[]} list */
export function _shuffled(list) {
  const out = [...list];
  _shuffle(out);
  return out;
}

/** @param {NDArray} arr  @returns {NDArray} */
export function shuffled(arr) {
  if (arr.shape.length == 0) return arr;
  if (arr.shape.length == 1) {
    const flat = _shuffled(arr.flat)
    return new np.NDArray(flat, arr.shape, arr.dtype);
  }
  const perm = _shuffled(Array.from({ length: arr.length }, (_, i) => i));
  const out = np.empty(arr.shape, arr.dtype);
  for (let i = 0; i < arr.length; i++) out.assign([i], arr.index(perm[i]));
  return out;
}

/**
 * @param {NDArray} arr
 */
export function shuffle(arr) {
  arr.assign(shuffled(arr));
}
