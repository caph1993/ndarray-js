/** @typedef {import('../NDArray-class')} NDArray */
import * as indexes from './indexes';
import type NDArray from "../NDArray-class";
/** @typedef {NDArray | number | boolean} ArrayOrConstant */
export type ArrayOrConstant = NDArray | number | boolean;
/** @typedef {indexes.Where} Index */
type Index = indexes.Where;
/**
 * @param {ArrayOrConstant} A
 * @param {ArrayOrConstant} B
 * @param {any} func
 * @param {any} dtype
 * @param {NDArray | null} [out=null]
 * @returns {ArrayOrConstant}
 */
export declare function binary_operation(A: ArrayOrConstant, B: ArrayOrConstant, func: any, dtype: any, out?: NDArray | null): ArrayOrConstant;
/**
 * @param {any} shapeA
 * @param {any} shapeB
 * @returns {any[]}
 */
export declare function _broadcast_shapes(shapeA: any, shapeB: any): any[];
/** @typedef {(A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray} BinaryOperator */
export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray;
/** @typedef {(other: ArrayOrConstant, out?: NDArray | null) => NDArray} SelfBinaryOperator */
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray;
/**
 * @param {any} dtype
 * @param {any} func
 * @returns {BinaryOperator}
 */
export declare function __make_operator(dtype: any, func: any): BinaryOperator;
/**
 * @param {any} funcNum
 * @param {any} funcBool
 * @returns {BinaryOperator}
 */
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
/**
 * @param {NDArray} tgt
 * @param {ArrayOrConstant} src
 * @param {Index} where
 * @param {any} func
 * @param {any} dtype
 * @returns {void}
 */
export declare function assign_operation(tgt: NDArray, src: ArrayOrConstant, where: Index, func: any, dtype: any): void;
/**
 * @param {any[]} tgtJS
 * @param {any} src
 * @param {Index} where
 * @param {any} func
 * @param {any} dtype
 * @returns {void}
 */
export declare function _assign_operation_toJS(tgtJS: any[], src: any, where: Index, func: any, dtype: any): void;
/** @typedef {Object} AssignmentOperator */
export type AssignmentOperator = {
    (tgt: NDArray, src: ArrayOrConstant): NDArray;
    (tgt: NDArray, where: Index, src: ArrayOrConstant): NDArray;
};
/** @typedef {Object} SelfAssignmentOperator */
export type SelfAssignmentOperator = {
    (other: ArrayOrConstant): NDArray;
    (where: Index, other: ArrayOrConstant): NDArray;
};
/**
 * @param {any} dtype
 * @param {any} func
 * @returns {AssignmentOperator}
 */
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
/**
 * @param {any} A
 * @param {any} B
 * @param {number} [rtol=1.e-5]
 * @param {number} [atol=1.e-8]
 * @param {boolean} [equal_nan=false]
 * @returns {ArrayOrConstant}
 */
export declare function isclose(A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean): ArrayOrConstant;
/**
 * @param {any} A
 * @param {any} B
 * @param {number} [rtol=1.e-5]
 * @param {number} [atol=1.e-8]
 * @param {boolean} [equal_nan=false]
 * @returns {boolean}
 */
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
/** @typedef {NDArray | number | boolean} ArrayOrConstant */
/** @typedef {indexes.Where} Index */
/** @typedef {(A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray} BinaryOperator */
/** @typedef {(other: ArrayOrConstant, out?: NDArray | null) => NDArray} SelfBinaryOperator */
/** @typedef {Object} AssignmentOperator */
/** @typedef {Object} SelfAssignmentOperator */
//# sourceMappingURL=operators.d.ts.map