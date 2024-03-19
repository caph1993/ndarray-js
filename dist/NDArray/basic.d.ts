import type NDArray from "../NDArray-class";
import { DType } from "../NDArray-class";
import { isarray, asarray, array, new_NDArray, _NDArray } from "./_globals";
export { isarray, asarray, array, new_NDArray, _NDArray };
/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 */
export declare function number_collapse(arr: NDArray, expect?: boolean): NDArray | number;
export declare function as_boolean(obj: any): boolean;
export declare function as_number(obj: any): number;
export declare function shape_shifts(shape: any): number[];
export declare function parse_shape(list: number | number[] | NDArray): number[];
export declare function reshape(A: NDArray, shape_or_first: number | number[], ...more_shape: number[]): NDArray;
export declare function ravel(A: NDArray): NDArray;
export declare function new_from(shape: any, f?: any, dtype?: DType): NDArray;
export declare function empty(shape: any, dtype?: DType): NDArray;
export declare function copy(A: NDArray): NDArray;
//# sourceMappingURL=basic.d.ts.map