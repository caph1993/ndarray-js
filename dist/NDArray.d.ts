import { TypedArrayConstructor } from './dtypes';
/** @ignore */
export type ArrayOrConstant = NDArray | number | boolean;
/**
 * Multi dimensional array.
 */
declare class NDArray<T extends TypedArrayConstructor = Float64ArrayConstructor> {
    /** @ignore */
    _flat: InstanceType<T>;
    /** @category Attributes @readonly */
    shape: number[];
    /** @category Attributes @readonly */
    get dtype(): T;
    /** @category Indexing / slicing */
    index: (...where: Where) => NDArray<T>;
    /** @ignore */
    modules: typeof import("./array").modules;
    /** @category Reducers */
    any: Method_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    /** @category Reducers */
    all: Method_a_axis_keepdims.Wrapper<Uint8ArrayConstructor>;
    /** @category Reducers */
    sum: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    product: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    max: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    min: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    argmax: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    argmin: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    mean: Method_a_axis_keepdims.Wrapper;
    /** @category Reducers */
    var: Method_a_axis_ddof_keepdims.Wrapper;
    /** @category Reducers */
    std: Method_a_axis_ddof_keepdims.Wrapper;
    /** @category Reducers */
    norm: Method_a_ord_axis_keepdims.Wrapper;
    /** @category Binary operators */
    add: Method_other_out.Wrapper;
    /** @category Binary operators */
    subtract: Method_other_out.Wrapper;
    /** @category Binary operators */
    multiply: Method_other_out.Wrapper;
    /** @category Binary operators */
    divide: Method_other_out.Wrapper;
    /** @category Binary operators */
    mod: Method_other_out.Wrapper;
    /** @category Binary operators */
    divide_int: Method_other_out.Wrapper;
    /** @category Binary operators */
    pow: Method_other_out.Wrapper;
    /** @category Binary operators */
    maximum: Method_other_out.Wrapper;
    /** @category Binary operators */
    minimum: Method_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_or: Method_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_and: Method_other_out.Wrapper;
    /** @category Binary operators */
    bitwise_shift_right: Method_other_out.Wrapper;
    /** @category Binary logical operators */
    logical_xor: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Binary logical operators */
    logical_or: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Binary logical operators */
    logical_and: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    greater: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    less: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    greater_equal: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    less_equal: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    equal: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    not_equal: Method_other_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Comparison operators */
    isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray<T>;
    /** @category Comparison operators */
    allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;
    /** @category Unary operators */
    round: Method_a_decimals_out.Wrapper;
    /** @category Unary operators */
    abs: Method_out.Wrapper;
    /** @category Unary operators */
    negative: Method_out.Wrapper;
    /** @category Unary operators */
    bitwise_not: Method_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Unary logical operators */
    logical_not: Method_out.Wrapper<Uint8ArrayConstructor>;
    /** @category Operators with assignment */
    assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    add_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    subtract_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    multiply_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    divide_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    mod_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    pow_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    divide_int_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    maximum_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    minimum_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    bitwise_and_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    bitwise_or_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    logical_or_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    bitwise_shift_right_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    bitwise_shift_left_assign: Method_values_where.Wrapper;
    /** @category Operators with assignment */
    logical_and_assign: Method_values_where.Wrapper;
    /** @category Transformations */
    ravel: () => NDArray<T>;
    /** @category Transformations */
    reshape: (shape: any, ...more_shape: any[]) => NDArray<T>;
    /** @category Transformations */
    sort: (axis?: number) => NDArray<T>;
    /** @category Transformations */
    transpose: (axes?: number[]) => NDArray<T>;
    /** @category Casting */
    tolist: () => any;
    /**
     * Generic operator function. See {@link GenericOperatorFunction} for details.
     */
    op: GenericOperatorFunction;
    constructor(flat: InstanceType<T>, shape?: number[]);
    /** @ignore */
    _simpleIndexes: import("./array/indexes").AxesIndex | null;
    /** @category Attributes @readonly */
    get size(): number;
    /** @category Attributes @readonly */
    get flat(): InstanceType<T>;
    /** @internal */
    set flat(list: InstanceType<T>);
    /**
     * @category Transformations
     * Transpose.
    */
    get T(): NDArray<T>;
    /**
     * Iterator over the first axis.
     * For 1D arrays, yields numbers.
     * For multidimensional arrays, yields array views.
     */
    [Symbol.iterator](): Generator<NDArray<T>, void, unknown>;
    /** @category Attributes @readonly */
    get length(): number;
    /** @category Transformations */
    copy: () => NDArray<T>;
    /** @category Casting */
    item(): number | boolean;
}
import { Method_other_out, Method_a_axis_keepdims, Method_values_where, Method_out, Method_a_decimals_out, Method_a_ord_axis_keepdims, Method_a_axis_ddof_keepdims } from './array/kwargs';
type BinaryOpSymbol = "+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min";
type AssignmentOpSymbol = "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and=";
type Where = import("./array/indexes").Where;
/**
 * Generic function to apply an operation to an array.
 * It can be used to apply unary, binary or assignment operations.
 *
 * @Example
 * ```javascript
 * const A = np.arange(10).op("+", 1);
 * A.index('::2').op("=", 0);
 * console.log(A.tolist());
 * ```
 */
export type GenericOperatorFunction = {
    (): NDArray;
    (where: Where): ArrayOrConstant;
    (where: Where, op: AssignmentOpSymbol, B: ArrayOrConstant): NDArray;
    (op: AssignmentOpSymbol, B: ArrayOrConstant): NDArray;
    (op: BinaryOpSymbol, B: ArrayOrConstant): NDArray;
    (UnaryOpSymbol: any): NDArray;
};
export { NDArray };
export default NDArray;
//# sourceMappingURL=NDArray.d.ts.map