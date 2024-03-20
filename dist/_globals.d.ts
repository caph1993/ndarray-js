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
/** @ignore */
export declare const GLOBALS: {
    NDArray: typeof import("./NDArray").NDArray;
    np: {
        (template: number | boolean | any[] | TemplateStringsArray, ...variables: any[]): any;
        NDArray: typeof import("./NDArray").NDArray;
        tolist(template: number | boolean | any[] | TemplateStringsArray, ...variables: any[]): any;
        fromlist: typeof import("./array/js-interface").fromlist;
        ravel: typeof import("./array/basic").ravel;
        reshape: typeof import("./array/basic").reshape;
        array: typeof import("./array/_globals").array;
        asarray: typeof import("./array/_globals").asarray;
        sum: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        product: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        prod: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        any: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => boolean | import("./NDArray").NDArray;
        all: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => boolean | import("./NDArray").NDArray;
        max: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        min: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        argmax: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        argmin: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        mean: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        norm: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceNormKwargs, keepdims?: boolean | import("./array/kwargs").ReduceNormKwargs, ord?: number | import("./array/kwargs").ReduceNormKwargs) => number | import("./NDArray").NDArray;
        var: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | import("./NDArray").NDArray;
        std: (arr: number | boolean | import("./NDArray").NDArray, axis?: number | import("./array/kwargs").ReduceStdKwargs, keepdims?: boolean | import("./array/kwargs").ReduceStdKwargs, ddof?: number | import("./array/kwargs").ReduceStdKwargs) => number | import("./NDArray").NDArray;
        transpose: typeof import("./array/transform").transpose;
        apply_along_axis: typeof import("./array/transform").apply_along_axis;
        sort: typeof import("./array/transform").sort;
        concatenate: typeof import("./array/transform").concatenate;
        stack: typeof import("./array/transform").stack;
        add: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        subtract: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        multiply: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        divide: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        mod: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        divide_int: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        pow: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        bitwise_or: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        bitwise_and: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        bitwise_xor: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        bitwise_shift_left: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        bitwise_shift_right: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        greater: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        less: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        greater_equal: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        less_equal: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        equal: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        not_equal: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        maximum: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        minimum: (arr: number | boolean | import("./NDArray").NDArray, other: number | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        logical_or: (arr: number | boolean | import("./NDArray").NDArray, other: boolean | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        logical_and: (arr: number | boolean | import("./NDArray").NDArray, other: boolean | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        atan2: (y: import("./NDArray").NDArray, x: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        allclose: typeof import("./array/operators").allclose;
        isclose: typeof import("./array/operators").isclose;
        sign: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        sqrt: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        square: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        exp: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        log: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        log2: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        log10: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        log1p: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        sin: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        cos: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        tan: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        asin: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        acos: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        atan: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        cosh: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        sinh: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        tanh: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        acosh: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        asinh: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        atanh: (A: import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        abs: (arr: number | boolean | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        bitwise_not: (arr: number | boolean | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        logical_not: (arr: number | boolean | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        negative: (arr: number | boolean | import("./NDArray").NDArray, out?: import("./NDArray").NDArray) => import("./NDArray").NDArray;
        round: (arr: number | boolean | import("./NDArray").NDArray, decimals?: number) => import("./NDArray").NDArray;
        modules: {
            constructors: typeof import("./modules/constructors");
            grammar: typeof import("./modules/grammar");
            random: typeof import("./modules/random");
        };
        random: typeof import("./modules/random");
        empty: typeof import("./modules/constructors").empty;
        zeros: typeof import("./modules/constructors").zeros;
        ones: typeof import("./modules/constructors").ones;
        arange: typeof import("./modules/constructors").arange;
        linspace: typeof import("./modules/constructors").linspace;
        geomspace: typeof import("./modules/constructors").geomspace;
        pi: number;
        e: number;
    };
};
//# sourceMappingURL=_globals.d.ts.map