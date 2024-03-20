import NDArray from "../NDArray";
import { AxisArg } from './kwargs';
export declare const reducers: {
    sum: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => any;
    product: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => any;
    mean: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => number | NDArray;
    max: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => number | NDArray;
    min: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => number | NDArray;
    argmax: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => number | NDArray;
    argmin: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => number | NDArray;
    len: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => number | NDArray;
    any: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => boolean | NDArray;
    all: (arr: NDArray, axis: AxisArg | null, keepdims: boolean) => boolean | NDArray;
    norm: (arr: NDArray, axis: AxisArg, keepdims: boolean, ord: number) => any;
    var: (arr: NDArray, axis: AxisArg, keepdims: boolean) => NDArray;
    std: (arr: NDArray, axis: AxisArg, keepdims: boolean, ddof: number) => NDArray;
};
export declare const kw_reducers: {
    sum: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    product: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    mean: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    max: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    min: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    argmax: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    argmin: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    len: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    any: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    all: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    norm: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceNormKwargs, keepdims?: boolean | import("./kwargs").ReduceNormKwargs, ord?: number | import("./kwargs").ReduceNormKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceNormKwargs, keepdims?: boolean | import("./kwargs").ReduceNormKwargs, ord?: number | import("./kwargs").ReduceNormKwargs) => NDArray;
    };
    var: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceKwargs, keepdims?: boolean | import("./kwargs").ReduceKwargs) => NDArray;
    };
    std: {
        as_function: (arr: number | boolean | NDArray, axis?: number | import("./kwargs").ReduceStdKwargs, keepdims?: boolean | import("./kwargs").ReduceStdKwargs, ddof?: number | import("./kwargs").ReduceStdKwargs) => NDArray;
        as_method: (axis?: number | import("./kwargs").ReduceStdKwargs, keepdims?: boolean | import("./kwargs").ReduceStdKwargs, ddof?: number | import("./kwargs").ReduceStdKwargs) => NDArray;
    };
};
//# sourceMappingURL=reduce.d.ts.map