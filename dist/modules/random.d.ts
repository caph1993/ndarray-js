import type NDArray from "../NDArray";
export declare function random(shape: any): NDArray<import("../dtypes").TypedArrayConstructor>;
export declare function uniform(a: any, b: any, shape: any): NDArray<Float64ArrayConstructor>;
export declare function exponential(mean: any, shape: any): NDArray<any>;
export declare function _normal_buffer(n: number): Float64Array;
export declare function randn(shape: number[]): NDArray<Float64ArrayConstructor>;
export declare function normal(mean: any, std: any, shape: any): NDArray<Float64ArrayConstructor>;
/** @param {any[]} list */
export declare function _shuffle(list: any): void;
export declare function _shuffled<T>(list: T): any;
export declare function shuffled(arr: NDArray): NDArray<import("../dtypes").TypedArrayConstructor>;
export declare function shuffle(arr: NDArray): void;
//# sourceMappingURL=random.d.ts.map