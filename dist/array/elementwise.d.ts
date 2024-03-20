import type NDArray from "../NDArray";
export declare function elementwise(A: NDArray, func: any, dtype: any, out?: NDArray): NDArray;
export declare const funcs: {
    sign: (A: NDArray, out?: NDArray) => NDArray;
    sqrt: (A: NDArray, out?: NDArray) => NDArray;
    square: (A: NDArray, out?: NDArray) => NDArray;
    exp: (A: NDArray, out?: NDArray) => NDArray;
    log: (A: NDArray, out?: NDArray) => NDArray;
    log2: (A: NDArray, out?: NDArray) => NDArray;
    log10: (A: NDArray, out?: NDArray) => NDArray;
    log1p: (A: NDArray, out?: NDArray) => NDArray;
    sin: (A: NDArray, out?: NDArray) => NDArray;
    cos: (A: NDArray, out?: NDArray) => NDArray;
    tan: (A: NDArray, out?: NDArray) => NDArray;
    asin: (A: NDArray, out?: NDArray) => NDArray;
    acos: (A: NDArray, out?: NDArray) => NDArray;
    atan: (A: NDArray, out?: NDArray) => NDArray;
    cosh: (A: NDArray, out?: NDArray) => NDArray;
    sinh: (A: NDArray, out?: NDArray) => NDArray;
    tanh: (A: NDArray, out?: NDArray) => NDArray;
    acosh: (A: NDArray, out?: NDArray) => NDArray;
    asinh: (A: NDArray, out?: NDArray) => NDArray;
    atanh: (A: NDArray, out?: NDArray) => NDArray;
};
export declare const ops: {
    "~": (A: NDArray, out?: NDArray) => NDArray;
    not: (A: NDArray, out?: NDArray) => NDArray;
    "+": (A: NDArray, out?: NDArray) => NDArray;
    "-": (A: NDArray, out?: NDArray) => NDArray;
    round: (arr: NDArray, decimals: number, out?: NDArray) => NDArray;
    negative: (A: NDArray, out?: NDArray) => NDArray;
    bitwise_not: (A: NDArray, out?: NDArray) => NDArray;
    logical_not: (A: NDArray, out?: NDArray) => NDArray;
    valueOf: (A: NDArray, out?: NDArray) => NDArray;
    abs: (A: NDArray, out?: NDArray) => NDArray;
};
export declare const kw_ops: {
    bitwise_not: {
        as_function: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (out?: NDArray) => NDArray;
    };
    logical_not: {
        as_function: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (out?: NDArray) => NDArray;
    };
    negative: {
        as_function: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (out?: NDArray) => NDArray;
    };
    abs: {
        as_function: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (out?: NDArray) => NDArray;
    };
    round: {
        as_function: (arr: number | boolean | NDArray, decimals?: number) => NDArray;
        as_method: (decimals?: number) => NDArray;
    };
};
//# sourceMappingURL=elementwise.d.ts.map