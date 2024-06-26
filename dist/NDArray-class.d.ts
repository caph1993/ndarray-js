/** @typedef {NumberConstructor | BooleanConstructor} DType */
export type DType = NumberConstructor | BooleanConstructor;
/** @typedef {NDArray | number | boolean} ArrayOrConstant */
export type ArrayOrConstant = NDArray | number | boolean;
declare class NDArray {
    _flat: number[];
    shape: number[];
    dtype: DType;
    index: (...where: any[]) => any;
    reshape: (shape: any, ...more_shape: any[]) => any;
    modules: typeof import("./NDArray").modules;
    any: ReduceSignatureBool;
    all: ReduceSignatureBool;
    sum: ReduceSignature;
    product: ReduceSignature;
    max: ReduceSignature;
    min: ReduceSignature;
    argmax: ReduceSignature;
    argmin: ReduceSignature;
    mean: ReduceSignature;
    var: ReduceSignature;
    std: ReduceStdSignature;
    norm: ReduceNormSignature;
    add: BinaryOperatorSignature;
    subtract: BinaryOperatorSignature;
    multiply: BinaryOperatorSignature;
    divide: BinaryOperatorSignature;
    mod: BinaryOperatorSignature;
    divide_int: BinaryOperatorSignature;
    pow: BinaryOperatorSignature;
    maximum: BinaryOperatorSignature;
    minimum: BinaryOperatorSignature;
    bitwise_or: BinaryOperatorSignature;
    bitwise_and: BinaryOperatorSignature;
    bitwise_shift_right: BinaryOperatorSignature;
    logical_xor: BinaryOperatorSignature<boolean>;
    logical_or: BinaryOperatorSignature<boolean>;
    logical_and: BinaryOperatorSignature<boolean>;
    greater: BinaryOperatorSignature;
    less: BinaryOperatorSignature;
    greater_equal: BinaryOperatorSignature;
    less_equal: BinaryOperatorSignature;
    equal: BinaryOperatorSignature;
    not_equal: BinaryOperatorSignature;
    isclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => number | boolean | NDArray;
    allclose: (A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean) => boolean;
    abs: UnaryOperatorSignature;
    negative: UnaryOperatorSignature;
    logical_not: UnaryOperatorSignature;
    bitwise_not: UnaryOperatorSignature;
    assign: SelfAssignmentOperator;
    add_assign: SelfAssignmentOperator;
    subtract_assign: SelfAssignmentOperator;
    multiply_assign: SelfAssignmentOperator;
    divide_assign: SelfAssignmentOperator;
    mod_assign: SelfAssignmentOperator;
    pow_assign: SelfAssignmentOperator;
    divide_int_assign: SelfAssignmentOperator;
    maximum_assign: SelfAssignmentOperator;
    minimum_assign: SelfAssignmentOperator;
    bitwise_and_assign: SelfAssignmentOperator;
    bitwise_or_assign: SelfAssignmentOperator;
    logical_or_assign: SelfAssignmentOperator;
    bitwise_shift_right_assign: SelfAssignmentOperator;
    bitwise_shift_left_assign: SelfAssignmentOperator;
    logical_and_assign: SelfAssignmentOperator;
    tolist: () => any;
    round: RoundSignature;
    sort: (axis?: number) => NDArray;
    transpose: (axes?: number[]) => NDArray;
    op: (...args: any[]) => NDArray;
    ravel: () => NDArray;
    constructor(flat: number[], shape: number[], dtype?: any);
    _simpleIndexes: import("./NDArray/indexes").AxesIndex | null;
    get size(): number;
    get flat(): number[];
    set flat(list: number[]);
    get T(): NDArray;
    /**
       * @returns {Generator<any, void, unknown>}
       */
    [Symbol.iterator](): Generator<any, void, unknown>;
    get length(): number;
    copy: () => NDArray;
    /**
       * @returns {number}
       */
    item(): number;
}
import { SelfAssignmentOperator } from './NDArray/operators';
import { BinaryOperatorSignature, ReduceNormSignature, ReduceSignature, ReduceSignatureBool, ReduceStdSignature, RoundSignature, UnaryOperatorSignature } from './NDArray/kwargs';
export { NDArray };
export default NDArray;
/** @typedef {NumberConstructor | BooleanConstructor} DType */
/** @typedef {NDArray | number | boolean} ArrayOrConstant */
/** @typedef {"+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min"} BinaryOpSymbol */
/** @typedef {"=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and="} AssignmentOpSymbol */
/** @typedef {"~" | "not" | "-"} UnaryOpSymbol */
/** @typedef {import("./NDArray/indexes").Where} Where */
/** @typedef {Object} Op */
//# sourceMappingURL=NDArray-class.d.ts.map