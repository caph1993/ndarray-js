//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./globals").GLOBALS;


/** @typedef {null|number} AxisArg */


// ==============================
//       Reducing functions
// ==============================


function reduce(arr, axis, keepdims, reducer, dtype = Number) {
  const { asarray, __shape_shifts, __as_boolean, __number_collapse } = NDArray.prototype;
  keepdims = __as_boolean(keepdims);
  arr = asarray(arr);
  if (axis == null) return reducer(arr.flat);
  if (axis < 0) axis = arr.shape.length - 1;
  let m = arr.shape[axis];
  let shift = __shape_shifts(arr.shape)[axis];
  const groups = Array.from({ length: m }, (_) =>/**@type {number[]}*/([]));
  arr.flat.forEach((value, i) => groups[(Math.floor(i / shift)) % m].push(value));
  // Transpose it:
  let nCols = arr.size / m;
  const groupsT = [];
  for (let j = 0; j < nCols; j++) {
    const newRow = [];
    for (let i = 0; i < m; i++) newRow.push(groups[i][j]);
    groupsT.push(newRow);
  }
  const flat = groupsT.map(reducer);
  let shape = [...arr.shape];
  if (keepdims) shape[axis] = 1;
  else shape = shape.filter((_, i) => i != axis);
  const out = new NDArray(flat, shape, dtype)
  return __number_collapse(out);
};

function __make_reducer(dtype, reducer) {
  /**
   * @param {Array} arr
  * @param {AxisArg} axis
   * @param {boolean} keepdims
   */
  return function (arr, axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    return reduce(arr, axis, keepdims, reducer, dtype);
  };
}

const reducers = {
  sum: __make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0)),
  product: __make_reducer(Number, (arr) => arr.reduce((a, b) => a * b, 1)),
  any: __make_reducer(Boolean, (arr) => {
    for (let x of arr) if (x) return true;
    return false;
  }),
  all: __make_reducer(Boolean, (arr) => {
    for (let x of arr) if (!x) return false;
    return true;
  }),
  max: __make_reducer(Number, (arr) => Math.max(...arr)),
  min: __make_reducer(Number, (arr) => Math.min(...arr)),
  argmax: __make_reducer(Number, (arr) => arr.indexOf(Math.max(...arr))),
  argmin: __make_reducer(Number, (arr) => arr.indexOf(Math.min(...arr))),
  mean: __make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0) / arr.length),
  var: function (arr, axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    const { mean, subtract, multiply } = NDArray.prototype;
    const arrMean = mean.bind({ axis, keepdims: true })(arr);
    arr = subtract(arr, arrMean);
    arr = multiply(arr, arr);
    return mean.bind({ axis, keepdims })(arr);
  },
  std: function (arr, axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    const { pow, var: _var } = NDArray.prototype;
    const variance = _var.bind({ axis, keepdims })(arr);
    return pow(variance, 0.5);
  },
}

module.exports = {
  reduce, reducers,
} 