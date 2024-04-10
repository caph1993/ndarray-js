import NDArray from "../NDArray";
import { AxisArg, Func_a_axis_ddof_keepdims, Func_a_axis_keepdims, Func_a_ord_axis_keepdims } from './kwargs';
export declare const reducers: {
    sum: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    product: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    mean: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    max: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    min: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    argmax: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Int32ArrayConstructor>;
    argmin: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Int32ArrayConstructor>;
    len: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Int32ArrayConstructor>;
    any: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Uint8ArrayConstructor>;
    all: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => NDArray<Uint8ArrayConstructor>;
    norm: (arr: NDArray, ord: number, axis: AxisArg, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    var: (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
    std: (arr: NDArray, axis: AxisArg, ddof: number, keepdims: boolean) => NDArray<Float64ArrayConstructor>;
};
export declare const kw_reducers: {
    sum: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    product: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    mean: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    max: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    min: Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    argmax: Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    argmin: Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    len: Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    any: Func_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    all: Func_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    norm: Func_a_ord_axis_keepdims.Wrapper;
    var: Func_a_axis_ddof_keepdims.Wrapper;
    std: Func_a_axis_ddof_keepdims.Wrapper;
};
//# sourceMappingURL=reduce.d.ts.map