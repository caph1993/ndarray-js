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


// function __normal(n) {
//   let u = random(n);
//   let v = random(n);
//   let s = np.add(np.square(u), np.square(v));
//   let mask = np.less(s, 1);
//   let x = np.multiply(u, np.sqrt(np.divide(np.subtract(1, s), s)));
//   return np.where(mask, x, __normal(n));
// }
// normal,
// shuffle,
// permutation,

module.exports = {
  random, uniform, exponential
}