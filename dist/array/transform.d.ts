import { DType } from '../NDArray';
import type NDArray from "../NDArray";
import { ArrayOrConstant } from './operators';
export declare function apply_along_axis(arr: NDArray, axis: number, transform: any, dtype?: DType): ArrayOrConstant;
export declare function sort(A: NDArray, axis?: number): ArrayOrConstant;
export declare function transpose(arr: NDArray, axes?: null | number[]): NDArray;
export declare function swapAxes(arr: NDArray, axisA: number, axisB: number): NDArray;
export declare function concatenate(arrays: NDArray[], axis?: number | null): NDArray;
export declare function stack(arrays: NDArray[], axis?: number): NDArray;
//# sourceMappingURL=transform.d.ts.map