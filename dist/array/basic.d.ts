import type NDArray from "../NDArray";
import type { TypedArrayConstructor } from "../dtypes";
import { isarray, asarray, array, new_NDArray, _NDArray } from "./_globals";
export { isarray, asarray, array, new_NDArray, _NDArray };
/**
 * If the array is 0D, it returns it's unique element (number or boolean).
 * The signature is kept as NDArray for type consistency, even though the
 * output is a number or a boolean. This is consistent with the facts that
 * (1) all functions requiring arrays work with numbers as well because they call asarray,
 * and (2) semantically, a constant is an array.
 */
export declare function number_collapse(arr: NDArray<any>, expect?: boolean): NDArray | number;
export declare function as_boolean(obj: any): boolean;
export declare function as_number(obj: any): number;
export declare function shape_shifts(shape: any): number[];
export type Shape = number | number[] | NDArray;
export declare function parse_shape(list: Shape): number[];
export declare function reshape(A: NDArray, shape_or_first: Shape, ...more_shape: number[]): NDArray<TypedArrayConstructor>;
export declare function ravel(A: NDArray): NDArray<TypedArrayConstructor>;
export declare function new_from(shape: Shape, f?: any, dtype?: TypedArrayConstructor): NDArray<TypedArrayConstructor>;
export declare function empty(shape: Shape, dtype?: TypedArrayConstructor): NDArray<TypedArrayConstructor>;
export declare function copy(A: NDArray): NDArray<TypedArrayConstructor>;
//# sourceMappingURL=basic.d.ts.map