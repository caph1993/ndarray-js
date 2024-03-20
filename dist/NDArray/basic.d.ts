/** @typedef {import('../NDArray-class')} NDArray */
import type NDArray from "../NDArray-class";
import { DType } from "../NDArray-class";
import { isarray, asarray, array, new_NDArray, _NDArray } from "./_globals";
export { isarray, asarray, array, new_NDArray, _NDArray };
/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 * @param {NDArray} arr
 * @param {boolean} [expect=false]
 * @returns {NDArray | number}
 */
export declare function number_collapse(arr: NDArray, expect?: boolean): NDArray | number;
/**
 * @param {any} obj
 * @returns {boolean}
 */
export declare function as_boolean(obj: any): boolean;
/**
 * @param {any} obj
 * @returns {number}
 */
export declare function as_number(obj: any): number;
/**
 * @param {any} shape
 * @returns {number[]}
 */
export declare function shape_shifts(shape: any): number[];
/**
 * @param {number | number[] | NDArray} list
 * @returns {number[]}
 */
export declare function parse_shape(list: number | number[] | NDArray): number[];
/**
 * @param {NDArray} A
 * @param {number | number[]} shape_or_first
 * @param {...number} [more_shape]
 * @returns {NDArray}
 */
export declare function reshape(A: NDArray, shape_or_first: number | number[], ...more_shape: number[]): NDArray;
/**
 * @param {NDArray} A
 * @returns {NDArray}
 */
export declare function ravel(A: NDArray): NDArray;
/**
 * @param {any} shape
 * @param {any} [f=undefined]
 * @param {DType} [dtype=Number]
 * @returns {NDArray}
 */
export declare function new_from(shape: any, f?: any, dtype?: DType): NDArray;
/**
 * @param {any} shape
 * @param {DType} [dtype=Number]
 * @returns {NDArray}
 */
export declare function empty(shape: any, dtype?: DType): NDArray;
/**
 * @param {NDArray} A
 * @returns {NDArray}
 */
export declare function copy(A: NDArray): NDArray;
//# sourceMappingURL=basic.d.ts.map