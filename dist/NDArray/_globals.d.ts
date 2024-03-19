import type NDArray from "../NDArray-class";
import { DType } from "../NDArray-class";
export declare const _NDArray: typeof NDArray;
export declare function isarray(A: any): A is NDArray;
export declare const new_NDArray: (flat: number[], shape: number[], dtype: DType) => NDArray;
export declare function asarray(A: any): NDArray;
export declare function array(A: any): NDArray;
//# sourceMappingURL=_globals.d.ts.map