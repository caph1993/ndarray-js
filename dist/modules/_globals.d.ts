declare const np: {
    (template: number | boolean | any[] | TemplateStringsArray, ...variables: any[]): any;
    NDArray: typeof import("../NDArray").NDArray;
    tolist(template: number | boolean | any[] | TemplateStringsArray, ...variables: any[]): any;
    fromlist: typeof import("../array/js-interface").fromlist;
    ravel: typeof import("../array/basic").ravel;
    reshape: typeof import("../array/basic").reshape;
    array: typeof import("../array/_globals").array;
    asarray: typeof import("../array/_globals").asarray;
    sum: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    product: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    prod: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    any: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    all: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    max: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    min: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    argmax: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    argmin: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Int32ArrayConstructor>;
    mean: import("../array/kwargs").Func_a_axis_keepdims.Wrapper<Float64ArrayConstructor>;
    norm: import("../array/kwargs").Func_a_ord_axis_keepdims.Wrapper;
    var: import("../array/kwargs").Func_a_axis_ddof_keepdims.Wrapper;
    std: import("../array/kwargs").Func_a_axis_ddof_keepdims.Wrapper;
    transpose: typeof import("../array/transform").transpose;
    apply_along_axis: typeof import("../array/transform").apply_along_axis;
    sort: import("../array/kwargs").Func_a_lastAxis.Wrapper;
    concatenate: typeof import("../array/transform").concatenate;
    stack: typeof import("../array/transform").stack;
    add: import("../array/kwargs").Func_a_other_out.Wrapper;
    subtract: import("../array/kwargs").Func_a_other_out.Wrapper;
    multiply: import("../array/kwargs").Func_a_other_out.Wrapper;
    divide: import("../array/kwargs").Func_a_other_out.Wrapper;
    mod: import("../array/kwargs").Func_a_other_out.Wrapper;
    divide_int: import("../array/kwargs").Func_a_other_out.Wrapper;
    pow: import("../array/kwargs").Func_a_other_out.Wrapper;
    bitwise_or: import("../array/kwargs").Func_a_other_out.Wrapper;
    bitwise_and: import("../array/kwargs").Func_a_other_out.Wrapper;
    bitwise_xor: import("../array/kwargs").Func_a_other_out.Wrapper;
    bitwise_shift_left: import("../array/kwargs").Func_a_other_out.Wrapper;
    bitwise_shift_right: import("../array/kwargs").Func_a_other_out.Wrapper;
    greater: import("../array/kwargs").Func_a_other_out.Wrapper;
    less: import("../array/kwargs").Func_a_other_out.Wrapper;
    greater_equal: import("../array/kwargs").Func_a_other_out.Wrapper;
    less_equal: import("../array/kwargs").Func_a_other_out.Wrapper;
    equal: import("../array/kwargs").Func_a_other_out.Wrapper;
    not_equal: import("../array/kwargs").Func_a_other_out.Wrapper;
    maximum: import("../array/kwargs").Func_a_other_out.Wrapper;
    minimum: import("../array/kwargs").Func_a_other_out.Wrapper;
    logical_or: import("../array/kwargs").Func_a_other_out.Wrapper;
    logical_and: import("../array/kwargs").Func_a_other_out.Wrapper;
    atan2: import("../array/kwargs").Func_y_x_out.Wrapper;
    assign: <T extends import("../dtypes").TypedArrayConstructor>(a: import("../NDArray").NDArray<T>, values: import("../NDArray").NDArray<Float64ArrayConstructor>, where: import("../array/indexes").GeneralIndexSpec[]) => import("../NDArray").NDArray<T>;
    allclose: typeof import("../array/operators").allclose;
    isclose: typeof import("../array/operators").isclose;
    sign: import("../array/kwargs").Func_a_out.Wrapper;
    sqrt: import("../array/kwargs").Func_a_out.Wrapper;
    square: import("../array/kwargs").Func_a_out.Wrapper;
    exp: import("../array/kwargs").Func_a_out.Wrapper;
    log: import("../array/kwargs").Func_a_out.Wrapper;
    log2: import("../array/kwargs").Func_a_out.Wrapper;
    log10: import("../array/kwargs").Func_a_out.Wrapper;
    log1p: import("../array/kwargs").Func_a_out.Wrapper;
    sin: import("../array/kwargs").Func_a_out.Wrapper;
    cos: import("../array/kwargs").Func_a_out.Wrapper;
    tan: import("../array/kwargs").Func_a_out.Wrapper;
    asin: import("../array/kwargs").Func_a_out.Wrapper;
    acos: import("../array/kwargs").Func_a_out.Wrapper;
    atan: import("../array/kwargs").Func_a_out.Wrapper;
    cosh: import("../array/kwargs").Func_a_out.Wrapper;
    sinh: import("../array/kwargs").Func_a_out.Wrapper;
    tanh: import("../array/kwargs").Func_a_out.Wrapper;
    acosh: import("../array/kwargs").Func_a_out.Wrapper;
    asinh: import("../array/kwargs").Func_a_out.Wrapper;
    atanh: import("../array/kwargs").Func_a_out.Wrapper;
    floor: import("../array/kwargs").Func_a_out.Wrapper;
    ceil: import("../array/kwargs").Func_a_out.Wrapper;
    isnan: import("../array/kwargs").Func_a_out.Wrapper;
    isfinite: import("../array/kwargs").Func_a_out.Wrapper;
    abs: import("../array/kwargs").Func_a_out.Wrapper;
    bitwise_not: import("../array/kwargs").Func_a_out.Wrapper;
    logical_not: import("../array/kwargs").Func_a_out.Wrapper;
    negative: import("../array/kwargs").Func_a_out.Wrapper;
    round: import("../array/kwargs").Func_a_decimals_out.Wrapper;
    modules: {
        constructors: typeof import("./constructors");
        grammar: typeof import("./grammar");
        random: typeof import("./random");
        indexing: typeof import("./indexing");
        statistics: typeof import("./statistics");
    };
    random: typeof import("./random");
    empty: typeof import("./constructors").empty;
    zeros: typeof import("./constructors").zeros;
    ones: typeof import("./constructors").ones;
    arange: typeof import("./constructors").arange;
    linspace: typeof import("./constructors").linspace;
    geomspace: typeof import("./constructors").geomspace;
    take: typeof import("./indexing").take;
    where: typeof import("./indexing").where;
    nonzero: typeof import("./indexing").nonzero;
    quantile: import("../array/kwargs").Func_a_q_axis.Wrapper;
    nanquantile: import("../array/kwargs").Func_a_q_axis.Wrapper;
    pi: number;
    e: number;
};
export { np };
export declare const nd_modules: {
    basic: typeof import("../array/basic");
    jsInterface: typeof import("../array/js-interface");
    indexes: typeof import("../array/indexes");
    elementwise: typeof import("../array/elementwise");
    print: typeof import("../array/print");
    reduce: typeof import("../array/reduce");
    operators: typeof import("../array/operators");
    transform: typeof import("../array/transform");
};
//# sourceMappingURL=_globals.d.ts.map