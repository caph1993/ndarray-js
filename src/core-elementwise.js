//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./globals").GLOBALS;


// Here, we declare only the core functions (those that are methods)

function elementwise(A, func, dtype) {
  A = NDArray.prototype.modules.basic.asarray(A);
  return new NDArray(A.flat.map(func), A.shape, dtype);
}


function round(A, decimals = 0) {
  if (decimals == 0) elementwise(A, Math.round, Number);
  return elementwise(A, x => parseFloat(x.toFixed(decimals)), Number);
};

function __make_elementwise(func, dtype = Number) {
  return function (A) {
    return elementwise(A, func, dtype);
  }
}

function bitwise_not(A) {
  return elementwise(A, x => ~x, Number);
};
function logical_not(A) {
  return elementwise(A, x => !x, Boolean);
};

module.exports = {
  elementwise,
  bitwise_not,
  logical_not,
  round,
  __make_elementwise,
} 