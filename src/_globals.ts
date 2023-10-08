//@ts-check
/**
 * This file exists for the sole purpose splitting the class methods across multiple files
 * while preserving all features of intellisense or JSDoc without errors.
 * The main issue is that the implementation of the methods require NDArray very often.
 * It resolves circular dependencies by using a global variable imported in each module.
 * The main file must define `require('./core-globals').GLOBALS.NDArray = NDArray;` before
 * importing any of the files that use it.
 * 
 * A template header for files importing NDArray from this file is given below.
 *    DO NOT use `const NDArray = require("./core-globals").GLOBALS.NDArray;`.
 *    Use const {NDArray} = ... instead as indicated. (Intellisense stops working otherwise)
 */


type NDArrayPrototype = typeof import("./NDArray-class").NDArray;
type npPrototype = typeof import(".").np;

//@ts-ignore
const _np: npPrototype = null, _array: NDArrayPrototype = null;

export const GLOBALS = {
  NDArray: _array,
  np: _np,
};
