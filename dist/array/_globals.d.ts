import type NDArray from "../NDArray";
import { TypedArrayConstructor } from "../dtypes";
export declare const _NDArray: typeof NDArray;
export declare function isarray(A: any): A is NDArray;
export declare function new_NDArray<T extends TypedArrayConstructor>(flat: InstanceType<T>, shape: number[]): NDArray<T>;
export declare function asarray<T extends TypedArrayConstructor = any>(A: NDArray<T> | any): NDArray<T>;
export declare function array(A: NDArray<any> | any): NDArray<any>;
export declare function broadcast_shapes(shapeA: number[], shapeB: number[]): any[][];
export declare function broadcast_n_shapes(...shapes: number[][]): any[][];
//# sourceMappingURL=_globals.d.ts.map