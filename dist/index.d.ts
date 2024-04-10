/**
 * Parser and main namespace for the ndarray-js package.
 * <script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>
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
    sum: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    /** @category Reducers */
    product: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    /** @category Reducers */
    prod: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    /** @category Reducers */
    any: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    /** @category Reducers */
    all: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    /** @category Reducers */
    max: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    /** @category Reducers */
    min: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    /** @category Reducers */
    argmax: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    /** @category Reducers */
    argmin: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    /** @category Reducers */
    mean: import("./array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    /** @category Reducers */
    norm: import("./array/kwargs").Func_a_ord_axis_keepdims.Wrapper;
    /** @category Reducers */
    var: import("./array/kwargs").Func_a_axis_ddof_keepdims.Wrapper;
    /** @category Reducers */
    std: import("./array/kwargs").Func_a_axis_ddof_keepdims.Wrapper;
    /** @category Transformations */
    transpose: typeof import("./array/transform").transpose;
    /** @category Transformations */
    apply_along_axis: typeof import("./array/transform").apply_along_axis;
    /** @category Transformations */
    sort: import("./array/kwargs").Func_a_lastAxis.Wrapper;
    /** @category Transformations */
    concatenate: typeof import("./array/transform").concatenate;
    /** @category Transformations */
    stack: typeof import("./array/transform").stack;
    /** @category Binary operators */
    add: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    subtract: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    multiply: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    divide: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    mod: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    divide_int: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    pow: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_or: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_and: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_xor: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_shift_left: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_shift_right: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    greater: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    less: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    greater_equal: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    less_equal: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    equal: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    not_equal: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    maximum: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    minimum: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    logical_or: Func_a_other_out.Wrapper;
    /** @category Binary operators */
    logical_and: Func_a_other_out.Wrapper;
    atan2: import("./array/kwargs").Func_y_x_out.Wrapper;
    assign: <T extends import("./dtypes").TypedArrayConstructor>(a: NDArray<T>, values: NDArray<Float64ArrayConstructor>, where: import("./array/indexes").GeneralIndexSpec[]) => NDArray<T>;
    allclose: typeof import("./array/operators").allclose;
    isclose: typeof import("./array/operators").isclose;
    /** @category Elementwise operators */
    sign: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    sqrt: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    square: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    exp: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    log: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    log2: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    log10: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    log1p: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    sin: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    cos: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    tan: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    asin: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    acos: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    atan: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    cosh: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    sinh: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    tanh: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    acosh: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    asinh: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    atanh: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    floor: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    ceil: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    isnan: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    isfinite: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    abs: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    bitwise_not: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    logical_not: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    negative: import("./array/kwargs").Func_a_out.Wrapper;
    /** @category Elementwise operators */
    round: import("./array/kwargs").Func_a_decimals_out.Wrapper;
    /** @category Main @namespace*/
    modules: {
        constructors: typeof import("./modules/constructors");
        grammar: typeof import("./modules/grammar");
        random: typeof import("./modules/random");
        indexing: typeof import("./modules/indexing");
        statistics: typeof import("./modules/statistics");
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
    take: typeof import("./modules/indexing").take;
    where: typeof import("./modules/indexing").where;
    nonzero: typeof import("./modules/indexing").nonzero;
    quantile: import("./array/kwargs").Func_a_q_axis.Wrapper;
    nanquantile: import("./array/kwargs").Func_a_q_axis.Wrapper;
    /** @category Math constants */
    pi: number;
    /** @category Math constants */
    e: number;
};
import NDArray from './NDArray';
import { Func_a_other_out } from './array/kwargs';
export { np };
//# sourceMappingURL=index.d.ts.map