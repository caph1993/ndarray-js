/** @typedef {import('../NDArray-class')} NDArray */
import { DType } from '../NDArray-class';
import type NDArray from "../NDArray-class";
import { ArrayOrConstant } from './operators';
/**
 * @param {NDArray} arr
 * @param {number} axis
 * @param {any} transform
 * @param {DType} [dtype=Number]
 * @returns {ArrayOrConstant}
 */
export declare function apply_along_axis(arr: NDArray, axis: number, transform: any, dtype?: DType): ArrayOrConstant;
/**
 * @param {NDArray} A
 * @param {number} [axis=-1]
 * @returns {ArrayOrConstant}
 */
export declare function sort(A: NDArray, axis?: number): ArrayOrConstant;
/**
 * @param {NDArray} arr
 * @param {null | number[]} [axes=null]
 * @returns {NDArray}
 */
export declare function transpose(arr: NDArray, axes?: null | number[]): NDArray;
/**
 * @param {NDArray} arr
 * @param {number} axisA
 * @param {number} axisB
 * @returns {NDArray}
 */
export declare function swapAxes(arr: NDArray, axisA: number, axisB: number): NDArray;
/**
 * @param {NDArray[]} arrays
 * @param {number | null} [axis=null]
 * @returns {NDArray}
 */
export declare function concatenate(arrays: NDArray[], axis?: number | null): NDArray;
/**
 * @param {NDArray[]} arrays
 * @param {number} [axis=0]
 * @returns {NDArray}
 */
export declare function stack(arrays: NDArray[], axis?: number): NDArray;
//# sourceMappingURL=transform.d.ts.map