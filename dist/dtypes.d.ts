export declare const dtypes: (Int8ArrayConstructor | Uint8ArrayConstructor | Uint8ClampedArrayConstructor | Int16ArrayConstructor | Uint16ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor | Float32ArrayConstructor | Float64ArrayConstructor)[];
export declare function new_buffer<T extends TypedArrayConstructor = Float64ArrayConstructor>(size_or_array: number | ArrayLike<number> | ArrayLike<boolean> | TypedArray, dtype?: T): InstanceType<T>;
export declare function dtype_eq<T extends TypedArrayConstructor>(A: T, B: T): boolean;
export declare function dtype_leq(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean;
export declare function dtype_lt(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean;
export declare function dtype_is_integer(A: TypedArrayConstructor): boolean;
export declare function dtype_is_boolean(A: TypedArrayConstructor): boolean;
export declare function dtype_is_float(A: TypedArrayConstructor): boolean;
type SupposedlyAnyTypedArrayConstructor = Float64ArrayConstructor;
export declare function dtype_least_ancestor(...dtypes: TypedArrayConstructor[]): SupposedlyAnyTypedArrayConstructor;
export type TypedArrayConstructor = Int8ArrayConstructor | Uint8ArrayConstructor | Uint8ClampedArrayConstructor | Int16ArrayConstructor | Uint16ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor | Float32ArrayConstructor | Float64ArrayConstructor;
export type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;
export {};
//# sourceMappingURL=dtypes.d.ts.map