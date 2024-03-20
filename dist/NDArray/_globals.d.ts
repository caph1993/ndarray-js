/** @typedef {import('../NDArray-class')} NDArray */
import type NDArray from "../NDArray-class";
import { DType } from "../NDArray-class";
export declare const _NDArray: typeof NDArray;
/**
 * @param {any} A
 * @returns {A is NDArray}
 */
export declare function isarray(A: any): A is NDArray;
/**
 * @param {number[]} flat
 * @param {number[]} shape
 * @param {DType} dtype
 * @returns {NDArray}
 */
export declare const new_NDArray: (flat: number[], shape: number[], dtype: DType) => NDArray;
/**
 * @param {any} A
 * @returns {NDArray}
 */
export declare function asarray(A: any): NDArray;
/**
 * @param {any} A
 * @returns {NDArray}
 */
export declare function array(A: any): NDArray;
//# sourceMappingURL=_globals.d.ts.map