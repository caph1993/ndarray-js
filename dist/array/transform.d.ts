import type NDArray from "../NDArray";
import { ArrayOrConstant } from './operators';
import { Func_a_lastAxis } from './kwargs';
import { TypedArrayConstructor } from '../dtypes';
/**
 * This function can reduce, sort, operate pointwise, or increase the dimensionality.
 */
export declare function apply_along_axis(arr: NDArray<any>, axis: number, transform: any, dtype?: TypedArrayConstructor): ArrayOrConstant;
export declare const cmp_nan_at_the_end: (a: number, b: number) => number;
export declare function sort(a: NDArray<any>, axis: number): NDArray<Float64ArrayConstructor>;
export declare function transpose(arr: NDArray<any>, axes?: null | number[]): NDArray<TypedArrayConstructor>;
export declare function swapAxes(arr: NDArray<any>, axisA: number, axisB: number): NDArray<TypedArrayConstructor>;
export declare function concatenate(arrays: NDArray<any>[], axis?: number | null): NDArray<TypedArrayConstructor>;
export declare function stack(arrays: NDArray[], axis?: number): NDArray<TypedArrayConstructor>;
export declare const kw_exported: {
    sort: Func_a_lastAxis.Wrapper;
};
//# sourceMappingURL=transform.d.ts.map