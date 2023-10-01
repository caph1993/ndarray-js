//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./globals").GLOBALS;


/** @returns {NDArray} */
function asarray(A) {
  if (A instanceof NDArray) return A;
  else return NDArray.prototype.fromJS(A);
}

/** @returns {NDArray} */
function array(A) {
  // @ts-ignore
  if (A instanceof NDArray) {
    let flat = A._simpleIndexes == null ? [...A.flat] : A.flat;
    return new NDArray(flat, A.shape, A.dtype);
  }
  else return NDArray.prototype.fromJS(A);
}

function reshape(A, shape, ...more_shape) {
  A = asarray(A);
  const { __parse_shape, __as_number } = NDArray.prototype;
  if (!more_shape.length) shape = __parse_shape(shape);
  else shape = [shape, ...more_shape].map(__as_number)
  const n = A.size;
  const inferredIndex = shape.indexOf(-1);
  if (inferredIndex !== -1) {
    const productOfKnownDims = shape.filter(dim => dim !== -1).reduce((acc, val) => acc * val, 1);
    if (n % productOfKnownDims !== 0) {
      throw new Error("Invalid shape. The total number of elements must match the product of the known dimensions.");
    }
    shape[inferredIndex] = n / productOfKnownDims;
  }
  return new NDArray(A.flat, shape, A.dtype);
};
function ravel(A) {
  A = asarray(A);
  return new NDArray(A.flat, [A.size], A.dtype);
};
function copy(A) {
  A = asarray(A);
  return new NDArray([...A.flat], A.shape, A.dtype);
};


module.exports = {
  asarray, array, reshape, ravel, copy
} 