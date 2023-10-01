//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./globals").GLOBALS;


// Here, we declare only the core functions (those that are methods)

function elementwise(A, func, dtype) {
  if (this instanceof NDArray) return elementwise.bind(NDArray.prototype)(this, ...arguments);
  A = NDArray.prototype.modules.basic.asarray(A);
  return new NDArray(A.flat.map(func), A.shape, dtype);
}


function round(A, decimals = 0) {
  if (this instanceof NDArray) return round.bind(NDArray.prototype)(this, ...arguments);
  if (decimals == 0) elementwise(A, Math.round, Number);
  return elementwise(A, x => parseFloat(x.toFixed(decimals)), Number);
};

function __make_elementwise(func, dtype = Number) {
  return function (A) {
    return elementwise(A, func, dtype);
  }
}

module.exports = {
  elementwise,
  round,
  __make_elementwise,
} 