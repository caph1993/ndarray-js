//@ts-check
const { np } = require("./globals").GLOBALS;

/** @typedef {typeof np.NDArray} NDArray*/

const thisModule = {};

thisModule.random = function (shape) {
  return np.NDArray.prototype._new(shape, (_) => Math.random(), Number)
};
thisModule.uniform = function (a, b, shape) {
  return np.add(a, np.multiply(thisModule.random(shape), (b - a)));
};
thisModule.exponential = function (mean, shape) {
  return np.multiply(mean, np.subtract(0, np.log(thisModule.random(shape))));
};

// normal,
// shuffle,
// permutation,

module.exports = thisModule;