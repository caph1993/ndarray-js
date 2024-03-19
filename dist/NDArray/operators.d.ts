import * as indexes from './indexes';
import type NDArray from "../NDArray-class";
export type ArrayOrConstant = NDArray | number | boolean;
type Index = indexes.Where;
export declare function binary_operation(A: ArrayOrConstant, B: ArrayOrConstant, func: any, dtype: any, out?: NDArray | null): ArrayOrConstant;
export declare function _broadcast_shapes(shapeA: any, shapeB: any): any[];
export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray;
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray;
export declare function __make_operator(dtype: any, func: any): BinaryOperator;
export declare function __make_operator_special(funcNum: any, funcBool: any): BinaryOperator;
export declare const op_binary: {
    "+": BinaryOperator;
    "-": BinaryOperator;
    "*": BinaryOperator;
    "/": BinaryOperator;
    "%": BinaryOperator;
    "//": BinaryOperator;
    "**": BinaryOperator;
    "<": BinaryOperator;
    ">": BinaryOperator;
    ">=": BinaryOperator;
    "<=": BinaryOperator;
    "==": BinaryOperator;
    "!=": BinaryOperator;
    "|": BinaryOperator;
    "&": BinaryOperator;
    "^": BinaryOperator;
    "<<": BinaryOperator;
    ">>": BinaryOperator;
    or: BinaryOperator;
    and: BinaryOperator;
    xor: BinaryOperator;
    max: BinaryOperator;
    min: BinaryOperator;
};
export declare function assign_operation(tgt: NDArray, src: ArrayOrConstant, where: Index, func: any, dtype: any): void;
export declare function _assign_operation_toJS(tgtJS: any[], src: any, where: Index, func: any, dtype: any): void;
export type AssignmentOperator = {
    (tgt: NDArray, src: ArrayOrConstant): NDArray;
    (tgt: NDArray, where: Index, src: ArrayOrConstant): NDArray;
};
export type SelfAssignmentOperator = {
    (other: ArrayOrConstant): NDArray;
    (where: Index, other: ArrayOrConstant): NDArray;
};
export declare function __make_assignment_operator(dtype: any, func: any): AssignmentOperator;
export declare const op_assign: {
    "=": AssignmentOperator;
    "+=": AssignmentOperator;
    "-=": AssignmentOperator;
    "*=": AssignmentOperator;
    "/=": AssignmentOperator;
    "%=": AssignmentOperator;
    "//=": AssignmentOperator;
    "**=": AssignmentOperator;
    "|=": AssignmentOperator;
    "&=": AssignmentOperator;
    "^=": AssignmentOperator;
    "<<=": AssignmentOperator;
    ">>=": AssignmentOperator;
    "max=": AssignmentOperator;
    "min=": AssignmentOperator;
    "or=": AssignmentOperator;
    "and=": AssignmentOperator;
};
export declare function isclose(A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean): ArrayOrConstant;
export declare function allclose(A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean): boolean;
export declare const kw_op_binary: {
    "+": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "-": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "*": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "/": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "%": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "//": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "**": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "<": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    ">": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    ">=": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "<=": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "==": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "!=": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "|": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "&": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "^": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    "<<": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    ">>": {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    max: {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    min: {
        as_function: (arr: number | boolean | NDArray, other: number | NDArray, out?: NDArray) => NDArray;
        as_method: (other: number | NDArray, out?: NDArray) => NDArray;
    };
    or: {
        as_function: (arr: number | boolean | NDArray, other: boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (other: boolean | NDArray, out?: NDArray) => NDArray;
    };
    and: {
        as_function: (arr: number | boolean | NDArray, other: boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (other: boolean | NDArray, out?: NDArray) => NDArray;
    };
    xor: {
        as_function: (arr: number | boolean | NDArray, other: boolean | NDArray, out?: NDArray) => NDArray;
        as_method: (other: boolean | NDArray, out?: NDArray) => NDArray;
    };
};
export declare const atan2: (y: NDArray, x: NDArray, out?: NDArray) => NDArray;
export {};
//# sourceMappingURL=operators.d.ts.map