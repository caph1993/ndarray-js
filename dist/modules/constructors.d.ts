import NDArray from '../NDArray';
import { Shape } from '../array/basic';
import { TypedArrayConstructor } from '../dtypes';
/**
 * Creates a new array with the specified shape and type, without initializing entries.
 */
export declare function empty(shape: Shape, dtype?: TypedArrayConstructor): NDArray<TypedArrayConstructor>;
/**
 * Creates a new array of zeros with the specified shape and dtype.
 */
export declare function zeros(shape: Shape, dtype?: TypedArrayConstructor): NDArray<TypedArrayConstructor>;
export declare function ones(shape: Shape, dtype?: TypedArrayConstructor): NDArray<TypedArrayConstructor>;
export declare function arange(arg0: any, arg1?: any): NDArray<TypedArrayConstructor>;
export declare function linspace(start: number, stop: number, steps?: number, endpoint?: boolean): NDArray<Float64ArrayConstructor>;
export declare function geomspace(start: number, stop: number, steps?: number, endpoint?: boolean): NDArray<Float64ArrayConstructor> | NDArray<TypedArrayConstructor>;
//# sourceMappingURL=constructors.d.ts.map