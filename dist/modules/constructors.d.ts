import NDArray from '../NDArray';
/**
 * Creates a new array with the specified shape and type, without initializing entries.
 */
export declare function empty(shape: number[], dtype?: import('../NDArray').DType): NDArray;
/**
 * Creates a new array of zeros with the specified shape and dtype.
 */
export declare function zeros(shape: number[], dtype?: import('../NDArray').DType): NDArray;
export declare function ones(shape: number[], dtype?: import('../NDArray').DType): NDArray;
export declare function arange(arg0: any, arg1?: any): NDArray;
export declare function linspace(start: any, stop: any, num?: number, endpoint?: boolean): NDArray;
export declare function geomspace(start: any, stop: any, num?: number, endpoint?: boolean): NDArray;
//# sourceMappingURL=constructors.d.ts.map