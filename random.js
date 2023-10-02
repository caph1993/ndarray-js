//@ts-check
const { np } = require("./globals").GLOBALS;

/** @typedef {typeof np.NDArray} NDArray*/


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

// shuffle,
// shuffled,
// permutation,

module.exports = {
  random, uniform, exponential,
  randn, normal,
}