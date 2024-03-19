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
export declare const GLOBALS: {
    NDArray: typeof import("./NDArray-class").NDArray;
    np: {
        (template: number | boolean | any[] | TemplateStringsArray, ...variables: any[]): any;
        NDArray: typeof import("./NDArray-class").NDArray;
        tolist(template: number | boolean | any[] | TemplateStringsArray, ...variables: any[]): any;
        fromlist: typeof import("./NDArray/js-interface").fromlist;
        ravel: typeof import("./NDArray/basic").ravel;
        reshape: typeof import("./NDArray/basic").reshape;
        array: typeof import("./NDArray/_globals").array;
        asarray: typeof import("./NDArray/_globals").asarray;
        sum: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        product: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        prod: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        any: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => boolean | import("./NDArray-class").NDArray;
        all: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => boolean | import("./NDArray-class").NDArray;
        max: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        min: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        argmax: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        argmin: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        mean: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        norm: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceNormKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceNormKwargs, ord?: number | import("./NDArray/kwargs").ReduceNormKwargs) => number | import("./NDArray-class").NDArray;
        var: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceKwargs) => number | import("./NDArray-class").NDArray;
        std: (arr: number | boolean | import("./NDArray-class").NDArray, axis?: number | import("./NDArray/kwargs").ReduceStdKwargs, keepdims?: boolean | import("./NDArray/kwargs").ReduceStdKwargs, ddof?: number | import("./NDArray/kwargs").ReduceStdKwargs) => number | import("./NDArray-class").NDArray;
        transpose: typeof import("./NDArray/transform").transpose;
        apply_along_axis: typeof import("./NDArray/transform").apply_along_axis;
        sort: typeof import("./NDArray/transform").sort;
        concatenate: typeof import("./NDArray/transform").concatenate;
        stack: typeof import("./NDArray/transform").stack;
        add: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        subtract: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        multiply: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        divide: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        mod: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        divide_int: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        pow: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        bitwise_or: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        bitwise_and: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        bitwise_xor: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        bitwise_shift_left: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        bitwise_shift_right: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        greater: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        less: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        greater_equal: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        less_equal: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        equal: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        not_equal: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        maximum: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        minimum: (arr: number | boolean | import("./NDArray-class").NDArray, other: number | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        logical_or: (arr: number | boolean | import("./NDArray-class").NDArray, other: boolean | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        logical_and: (arr: number | boolean | import("./NDArray-class").NDArray, other: boolean | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        atan2: (y: import("./NDArray-class").NDArray, x: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        allclose: typeof import("./NDArray/operators").allclose;
        isclose: typeof import("./NDArray/operators").isclose;
        sign: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        sqrt: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        square: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        exp: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        log: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        log2: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        log10: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        log1p: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        sin: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        cos: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        tan: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        asin: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        acos: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        atan: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        cosh: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        sinh: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        tanh: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        acosh: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        asinh: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        atanh: (A: import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        abs: (arr: number | boolean | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        bitwise_not: (arr: number | boolean | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        logical_not: (arr: number | boolean | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        negative: (arr: number | boolean | import("./NDArray-class").NDArray, out?: import("./NDArray-class").NDArray) => import("./NDArray-class").NDArray;
        round: (arr: number | boolean | import("./NDArray-class").NDArray, decimals?: number) => import("./NDArray-class").NDArray;
        modules: {
            constructors: typeof import("./modules/constructors");
            grammar: typeof import("./modules/grammar");
            random: typeof import("./modules/random");
        };
        empty: typeof import("./modules/constructors").empty;
        zeros: typeof import("./modules/constructors").zeros;
        ones: typeof import("./modules/constructors").ones;
        arange: typeof import("./modules/constructors").arange;
        linspace: typeof import("./modules/constructors").linspace;
        geomspace: typeof import("./modules/constructors").geomspace;
        random: typeof import("./modules/random");
        pi: number;
        e: number;
    };
};
//# sourceMappingURL=_globals.d.ts.map