/** @ignore */
export type DType = NumberConstructor | BooleanConstructor;
/** @ignore */
export type ArrayOrConstant = NDArray | number | boolean;
/**
 * Multi dimensional array.
 */
declare class NDArray {
    /** @ignore */
    _flat: number[];
    /** @category Attributes @readonly */
    shape: number[];
    /** @category Attributes @readonly */
    dtype: DType;
    /** @category Indexing / slicing */
    index: (...where: Where) => NDArray;
    /** @ignore */
    modules: typeof import("./array").modules;
    /** @category Reducers */
    any: ReduceSignatureBool;
    /** @category Reducers */
    all: ReduceSignatureBool;
    /** @category Reducers */
    sum: ReduceSignature;
    /** @category Reducers */
    product: ReduceSignature;
    /** @category Reducers */
    max: ReduceSignature;
    /** @category Reducers */
    min: ReduceSignature;
    /** @category Reducers */
    argmax: ReduceSignature;
    /** @category Reducers */
    argmin: ReduceSignature;
    /** @category Reducers */
    mean: ReduceSignature;
    /** @category Reducers */
    var: ReduceSignature;
    /** @category Reducers */
    std: ReduceStdSignature;
    /** @category Reducers */
    norm: ReduceNormSignature;
    /** @category Binary operators */
    add: BinaryOperatorMethod;
    /** @category Binary operators */
    subtract: BinaryOperatorMethod;
    /** @category Binary operators */
    multiply: BinaryOperatorMethod;
    /** @category Binary operators */
    divide: BinaryOperatorMethod;
    /** @category Binary operators */
    mod: BinaryOperatorMethod;
    /** @category Binary operators */
    divide_int: BinaryOperatorMethod;
    /** @category Binary operators */
    pow: BinaryOperatorMethod;
    /** @category Binary operators */
    maximum: BinaryOperatorMethod;
    /** @category Binary operators */
    minimum: BinaryOperatorMethod;
    /** @category Binary operators */
    bitwise_or: BinaryOperatorMethod;
    /** @category Binary operators */
    bitwise_and: BinaryOperatorMethod;
    /** @category Binary operators */
    bitwise_shift_right: BinaryOperatorMethod;
    /** @category Binary logical operators */
    logical_xor: BinaryOperatorMethod<boolean>;
    /** @category Binary logical operators */
    logical_or: BinaryOperatorMethod<boolean>;
    /** @category Binary logical operators */
    logical_and: BinaryOperatorMethod<boolean>;
    /** @category Comparison operators */
    greater: BinaryOperatorMethod;
    /** @category Comparison operators */
    less: BinaryOperatorMethod;
    /** @category Comparison operators */
    greater_equal: BinaryOperatorMethod;
    /** @category Comparison operators */
    less_equal: BinaryOperatorMethod;
    /** @category Comparison operators */
    equal: BinaryOperatorMethod;
    /** @category Comparison operators */
    not_equal: BinaryOperatorMethod;
    /** @category Comparison operators */
    isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray;
    /** @category Comparison operators */
    allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;
    /** @category Unary operators */
    round: RoundSignature;
    /** @category Unary operators */
    abs: UnaryOperatorMethod;
    /** @category Unary operators */
    negative: UnaryOperatorMethod;
    /** @category Unary operators */
    bitwise_not: UnaryOperatorMethod;
    /** @category Unary logical operators */
    logical_not: UnaryOperatorMethod;
    /** @category Operators with assignment */
    assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    add_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    subtract_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    multiply_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    divide_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    mod_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    pow_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    divide_int_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    maximum_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    minimum_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    bitwise_and_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    bitwise_or_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    logical_or_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    bitwise_shift_right_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    bitwise_shift_left_assign: SelfAssignmentOperator;
    /** @category Operators with assignment */
    logical_and_assign: SelfAssignmentOperator;
    /** @category Transformations */
    ravel: () => NDArray;
    /** @category Transformations */
    reshape: (shape: any, ...more_shape: any[]) => any;
    /** @category Transformations */
    sort: (axis?: number) => NDArray;
    /** @category Transformations */
    transpose: (axes?: number[]) => NDArray;
    /** @category Casting */
    tolist: () => any;
    /**
     * Generic operator function. See {@link GenericOperatorFunction} for details.
     */
    op: GenericOperatorFunction;
    constructor(flat: number[], shape: number[], dtype?: any);
    /** @ignore */
    _simpleIndexes: import("./array/indexes").AxesIndex | null;
    /** @category Attributes @readonly */
    get size(): number;
    /** @category Attributes @readonly */
    get flat(): number[];
    /** @internal */
    set flat(list: number[]);
    /**
     * @category Transformations
     * Transpose.
    */
    get T(): NDArray;
    /**
     * Iterator over the first axis.
     * For 1D arrays, yields numbers.
     * For multidimensional arrays, yields array views.
     */
    [Symbol.iterator](): Generator<NDArray, void, unknown>;
    /** @category Attributes @readonly */
    get length(): number;
    /** @category Transformations */
    copy: () => NDArray;
    /** @category Casting */
    item(): number;
}
import { SelfAssignmentOperator } from './array/operators';
import { BinaryOperatorMethod, ReduceNormSignature, ReduceSignature, ReduceSignatureBool, ReduceStdSignature, RoundSignature, UnaryOperatorMethod } from './array/kwargs';
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