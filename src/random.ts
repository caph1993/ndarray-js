//@ts-check
const { np } = require("./globals").GLOBALS;

/** @typedef {typeof import("./core").prototype} NDArray*/


function random(shape) {
  return np.NDArray.prototype._new(shape, (_) => Math.random(), Number)
};
function uniform(a, b, shape) {
  return np.add(a, np.multiply(random(shape), (b - a)));
};
function exponential(mean, shape) {
  return np.multiply(mean, np.subtract(0, np.log(random(shape))));
};

/** @param {number} n */
function __normal(n) {
  const out = [];
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
function randn(shape) {
  const flat = __normal(np.prod(shape));
  return new np.NDArray(flat, shape, Number);
};
function normal(mean, std, shape) {
  return np.add(mean, np.multiply(std, shape));
};


/** @param {any[]} list */
function _shuffle(list) {
  // Fisher-Yates (aka Knuth) shuffle.
  // https://stackoverflow.com/a/2450976
  for (let i = list.length - 1; i >= 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
}

/** @param {any[]} list */
function _shuffled(list) {
  const out = [...list];
  _shuffle(out);
  return out;
}

/** @param {NDArray} arr  @returns {NDArray} */
function shuffled(arr) {
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
function shuffle(arr) {
  arr.assign(shuffled(arr));
}

export default {
  random, uniform, exponential,
  randn, normal, shuffle,
}