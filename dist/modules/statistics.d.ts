import NDArray from "../NDArray";
import { AxisArg, Func_a_q_axis, Func_a_axis_keepdims } from "../array/kwargs";
/**
 * Compute the q-th percentile of the data along the specified axis.
 */
export declare function percentile(a: NDArray<any>, q: NDArray<any> | number, axis: AxisArg | null): NDArray<import("../dtypes").TypedArrayConstructor>;
/**
 * Compute the q-th quantile of the data along the specified axis.
 */
export declare function quantile(a: NDArray<any>, q: NDArray<any> | number, axis: number, _nan_handling?: boolean): NDArray<import("../dtypes").TypedArrayConstructor>;
/**
 * Compute the median along the specified axis.
 */
export declare function median(a: NDArray, axis: AxisArg | null, keepdims: boolean): NDArray<import("../dtypes").TypedArrayConstructor>;
/**
 * Compute the weighted average along the specified axis.
 */
export declare function average(a: NDArray, axis: AxisArg | null, weights: NDArray | null, keepdims: boolean): NDArray<any>;
/**
 * Compute the q-th percentile of the data along the specified axis, while ignoring nan values.
*/
export declare function nanpercentile(a: NDArray, q: NDArray | number, axis: AxisArg | null): NDArray<import("../dtypes").TypedArrayConstructor>;
/**
 * Compute the q-th quantile of the data along the specified axis, while ignoring nan values.
 */
export declare function nanquantile(a: NDArray, q: NDArray | number, axis: AxisArg | null): NDArray<import("../dtypes").TypedArrayConstructor>;
/**
 * Compute the median along the specified axis, while ignoring NaNs.
 */
export declare function nanmedian(a: NDArray, axis: number, keepdims: boolean): NDArray<import("../dtypes").TypedArrayConstructor>;
export declare const kw_exported: {
    quantile: Func_a_q_axis.Wrapper;
    nanquantile: Func_a_q_axis.Wrapper;
    percentile: Func_a_q_axis.Wrapper;
    nanpercentile: Func_a_q_axis.Wrapper;
    median: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    nanmedian: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
};
//# sourceMappingURL=statistics.d.ts.map