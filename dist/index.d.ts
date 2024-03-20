/**
 * Namespace for the ndarray-js package.
 * <script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>
 * @remarks
 * `np` is both the main namespace and a numpy parser: ``np`...` `` is equivalent to ``np.numpy`...` ``.
 */
declare const np: {
    (template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]): any;
    /** @category Main */
    NDArray: typeof NDArray;
    /** @category Casting and reshaping */
    tolist(template: TemplateStringsArray | any[] | number | boolean, ...variables: any[]): any;
    /** @category Casting and reshaping */
    fromlist: typeof import("./array/js-interface").fromlist;
    /** @category Casting and reshaping */
    ravel: typeof import("./array/basic").ravel;
    /** @category Casting and reshaping */
    reshape: typeof import("./array/basic").reshape;
    /** @category Casting and reshaping */
    array: typeof import("./array/_globals").array;
    /** @category Casting and reshaping */
    asarray: typeof import("./array/_globals").asarray;
    /** @category Reducers */
    sum: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    product: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    prod: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    any: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => boolean | NDArray;
    /** @category Reducers */
    all: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => boolean | NDArray;
    /** @category Reducers */
    max: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    min: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    argmax: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    argmin: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    mean: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    norm: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceNormKwargs, keepdims?: boolean | import("./array/kwargs").ReduceNormKwargs, ord?: number | import("./array/kwargs").ReduceNormKwargs) => number | NDArray;
    /** @category Reducers */
    var: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceKwargs, keepdims?: boolean | import("./array/kwargs").ReduceKwargs) => number | NDArray;
    /** @category Reducers */
    std: (arr: number | boolean | NDArray, axis?: number | import("./array/kwargs").ReduceStdKwargs, keepdims?: boolean | import("./array/kwargs").ReduceStdKwargs, ddof?: number | import("./array/kwargs").ReduceStdKwargs) => number | NDArray;
    /** @category Transformations */
    transpose: typeof import("./array/transform").transpose;
    /** @category Transformations */
    apply_along_axis: typeof import("./array/transform").apply_along_axis;
    /** @category Transformations */
    sort: typeof import("./array/transform").sort;
    /** @category Transformations */
    concatenate: typeof import("./array/transform").concatenate;
    /** @category Transformations */
    stack: typeof import("./array/transform").stack;
    /** @category Binary operators */
    add: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    subtract: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    multiply: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    divide: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    mod: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    divide_int: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    pow: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    bitwise_or: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    bitwise_and: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    bitwise_xor: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    bitwise_shift_left: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    bitwise_shift_right: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    greater: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    less: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    greater_equal: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    less_equal: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    equal: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    not_equal: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    maximum: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    minimum: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    logical_or: (arr: number | boolean | NDArray, other: boolean | NDArray, out?: NDArray) => NDArray;
    /** @category Binary operators */
    logical_and: (arr: number | boolean | NDArray, other: boolean | NDArray, out?: NDArray) => NDArray;
    atan2: (y: NDArray, x: NDArray, out?: NDArray) => NDArray;
    allclose: typeof import("./array/operators").allclose;
    isclose: typeof import("./array/operators").isclose;
    /** @category Elementwise operators */
    sign: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    sqrt: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    square: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    exp: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    log: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    log2: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    log10: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    log1p: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    sin: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    cos: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    tan: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    asin: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    acos: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    atan: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    cosh: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    sinh: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    tanh: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    acosh: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    asinh: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    atanh: (A: NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    abs: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    bitwise_not: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    logical_not: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    negative: (arr: number | boolean | NDArray, out?: NDArray) => NDArray;
    /** @category Elementwise operators */
    round: (arr: number | boolean | NDArray, decimals?: number) => NDArray;
    /** @category Main */
    modules: {
        constructors: typeof import("./modules/constructors");
        grammar: typeof import("./modules/grammar");
        random: typeof import("./modules/random");
    };
    /** @category Modules */
    random: typeof import("./modules/random");
    /** @category Constructors */
    empty: typeof import("./modules/constructors").empty;
    /** @category Constructors */
    zeros: typeof import("./modules/constructors").zeros;
    /** @category Constructors */
    ones: typeof import("./modules/constructors").ones;
    /** @category Constructors */
    arange: typeof import("./modules/constructors").arange;
    /** @category Constructors */
    linspace: typeof import("./modules/constructors").linspace;
    /** @category Constructors */
    geomspace: typeof import("./modules/constructors").geomspace;
    /** @category Math constants */
    pi: number;
    /** @category Math constants */
    e: number;
};
import NDArray from './NDArray';
export { np };
//# sourceMappingURL=index.d.ts.map