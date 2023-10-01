//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./globals").GLOBALS;


/**
 * @param {NDArray} arr
 * @param {number} axis
 */
function apply_along_axis(arr, axis, transform, dtype = Number) {
  arr = NDArray.prototype.modules.basic.asarray(arr)
  const { __shape_shifts, __number_collapse } = NDArray.prototype;
  if (axis == null) return transform(arr.flat);
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
  const data = groupsT.map(transform);
  const tmp = NDArray.prototype.fromJS(data);
  const shape = [...arr.shape.slice(0, axis), ...tmp.shape.slice(1), ...arr.shape.slice(axis + 1),];
  const out = new NDArray(tmp.flat, shape, dtype)
  return __number_collapse(out);
}

/**
 * @param {NDArray} A
 */
function sort(A, axis = -1) {
  return apply_along_axis(A, axis, (arr) => {
    const cpy = [...arr];
    cpy.sort((a, b) => a - b);
    return cpy;
  }, A.dtype);
}


module.exports = {
  apply_along_axis, sort
} 