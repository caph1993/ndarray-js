import * as indexes from './indexes';
import NDArray from "../NDArray";
import { Func_y_x_out } from './kwargs';
import { Where } from './indexes';
import { TypedArrayConstructor } from '../dtypes';
export type ArrayOrConstant = NDArray<any> | number | boolean;
export declare function binary_operation<T extends TypedArrayConstructor = Float64ArrayConstructor>(A: ArrayOrConstant, B: ArrayOrConstant, func: any, dtype: T, out?: NDArray<T> | null): NDArray<T>;
export declare function __make_operator<T extends TypedArrayConstructor = Float64ArrayConstructor>(dtype: T, func: any): BinaryOperator;
export type BinaryOperator = (A: ArrayOrConstant, B: ArrayOrConstant, out?: NDArray | null) => NDArray;
export type SelfBinaryOperator = (other: ArrayOrConstant, out?: NDArray | null) => NDArray;
export declare const op_binary: {
    "+": BinaryOperator;
    "-": BinaryOperator;
    "*": BinaryOperator;
    "/": BinaryOperator;
    "%": BinaryOperator;
    "|": BinaryOperator;
    "&": BinaryOperator;
    "^": BinaryOperator;
    "<<": BinaryOperator;
    ">>": BinaryOperator;
    "**": BinaryOperator;
    "//": BinaryOperator;
    "<": BinaryOperator;
    ">": BinaryOperator;
    ">=": BinaryOperator;
    "<=": BinaryOperator;
    "==": BinaryOperator;
    "!=": BinaryOperator;
    or: BinaryOperator;
    and: BinaryOperator;
    xor: BinaryOperator;
    max: BinaryOperator;
    min: BinaryOperator;
};
export declare function assign_operation<T extends TypedArrayConstructor = Float64ArrayConstructor>(tgt: NDArray<T>, src: ArrayOrConstant, where: Where, func: any): NDArray<T>;
export declare function _assign_operation_toJS(tgtJS: any[], src: any, where: Where, func: any): any[];
export declare function __make_assignment_operator(func: any): <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
export declare const op_assign: {
    "=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "+=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "-=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "*=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "/=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "%=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "//=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "**=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "|=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "&=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "^=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "<<=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    ">>=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "max=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "min=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "or=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
    "and=": <T extends TypedArrayConstructor>(a: NDArray<T>, values: NDArray, where: indexes.GeneralIndexSpec[]) => NDArray<T>;
};
export declare function isclose(A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean): NDArray<Uint8ArrayConstructor>;
export declare function allclose(A: any, B: any, rtol?: number, atol?: number, equal_nan?: boolean): boolean;
export declare const atan2: Func_y_x_out.Wrapper;
export declare function n_ary_operation<F extends (...args: any) => NDArray | number | number[] | boolean | boolean[]>(arrs: NDArray[], elem_shape: number[], func: F): NDArray;
//# sourceMappingURL=operators.d.ts.map