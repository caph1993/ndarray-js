//@ts-check

// Mozilla docs:
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypedArray

// This is a mess:
// https://stackoverflow.com/questions/65129070/
// https://stackoverflow.com/questions/49242232/
// https://stackoverflow.com/questions/69783310/

type BufferConstructor<B> = {
  new(size: number | ArrayLike<any> | any[]): B
}

interface DType_<Name extends string, Buffer> {
  readonly name: Name
  readonly BufferType: BufferConstructor<Buffer>

  // phantom (type only)
  readonly __buffer?: Buffer
}

export const bool = {
  name: "bool",
  BufferType: Uint8Array
} as const satisfies DType_<"bool", Uint8Array>

export const int32 = {
  name: "int32",
  BufferType: Int32Array
} as const satisfies DType_<"int32", Int32Array>

export const float32 = {
  name: "float32",
  BufferType: Float32Array
} as const satisfies DType_<"float32", Float32Array>

export const float64 = {
  name: "float64",
  BufferType: Float64Array
} as const satisfies DType_<"float64", Float64Array>

export const object = {
  name: "object",
  BufferType: Array
} as const satisfies DType_<"object", any[]>




// type F32 = Float32Array & DType<"float32">
// type I32 = Int32Array & DType<"int32">
// type Bool = Uint8Array & DType<"bool">

export type DType =
  | typeof bool
  | typeof int32
  | typeof float32
  | typeof float64
  | typeof object;

export type NameOf<D extends DType_<any, any>> = D["name"]

export type PromoteName<A, B> =
  A extends "object" ? "object" :
  B extends "object" ? "object" :
  A extends "float64" ? "float64" :
  B extends "float64" ? "float64" :
  A extends "float32" ? "float32" :
  B extends "float32" ? "float32" :
  A extends "int32" ? "int32" :
  B extends "int32" ? "int32" :
  "bool";

export type DTypeByName<N> =
  N extends "bool" ? typeof bool :
  N extends "int32" ? typeof int32 :
  N extends "float32" ? typeof float32 :
  N extends "float64" ? typeof float64 :
  N extends "object" ? typeof object :
  never

export interface HasDType {
  readonly dtype: DType
}

export type DTypeNameOf<T extends HasDType> =
  NameOf<T["dtype"]>

export type DTypeOf<T extends HasDType> =
  T["dtype"]

export type Promote<
  A extends HasDType,
  B extends HasDType
> = DTypeByName<PromoteName<DTypeNameOf<A>, DTypeNameOf<B>>>


const _infer_dtype = {
  Float32Array: "float32" as const,
  Int32Array: "int32" as const,
  Uint8Array: "bool" as const,
  Array: "object" as const,
}

export function infer_dtype<T extends HasDType>(arr: T): DTypeOf<T> {
  const key = arr.dtype.constructor.name;
  return _infer_dtype[key] || object;
}

export function new_buffer<T extends DType>(
  size_or_array: number | InstanceType<T["BufferType"]>,
  dtype?: T
): InstanceType<T["BufferType"]> {
  //@ts-ignore
  return new (dtype || float64).BufferType(size_or_array) as InstanceType<T["BufferType"]>;
}

export function promote(a: DType, b: DType): DType {
  if (a.name === "object" || b.name === "object") return object
  if (a.name === "float64" || b.name === "float64") return float64
  if (a.name === "float32" || b.name === "float32") return float32
  if (a.name === "int32" || b.name === "int32") return int32
  return bool
}

export function dtype_cmp(t1: DType, t2: DType): number {
  if (t1.name === t2.name) return 0;
  if (t1.name === "object") return 1;
  if (t2.name === "object") return -1;
  if (t1.name === "float64") return 1;
  if (t2.name === "float64") return -1;
  if (t1.name === "float32") return 1;
  if (t2.name === "float32") return -1;
  if (t1.name === "int32") return 1;
  if (t2.name === "int32") return -1;
  return 0; // Should not happen
}
export function dtype_max(...dtypes: DType[]): DType {
  if (dtypes.length === 0) return float64;
  return dtypes.reduce((a, b) => promote(a, b));
}



export type DtypeResolver = {
  (dtypes: DType[], out?: DType | HasDType): DType;
};

function get_out_dtype(out?: DType | HasDType) {
  if (!out) return null;
  if ("dtype" in out) return out.dtype;
  return out as DType;
}

export function float_out(dtypes: DType[], out?: DType | HasDType): DType {
  let dtype = get_out_dtype(out);
  let max = dtype_max(...dtypes);
  // Output is float64. If you want float32, provide out.
  const inferred = (dtype_cmp(max, float32) >= 0) ? max : float64;
  if (dtype && dtype_cmp(dtype, inferred) < 0)
    throw new Error(`Output array has dtype ${dtype.name}, which cannot hold all values of type ${inferred.name}.`);
  return dtype || inferred;
}

export function bool_out<T extends DType>(dtypes: DType[], out?: DType | HasDType): T {
  let dtype = get_out_dtype(out);
  if (dtype && dtype_cmp(dtype, bool) != 0)
    throw new Error(`Output array has dtype ${dtype.name} but expected bool.`);
  return bool as T;
}


// export const dtypes = [Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array];//, BigInt64Array, BigUint64Array]


// export function new_buffer<
//   T extends TypedArrayConstructor = Float64ArrayConstructor
// >(
//   size_or_array: number | ArrayLike<number> | ArrayLike<boolean> | TypedArray,
//   dtype?: T
// ): InstanceType<T> {
//   //@ts-ignore
//   return dtype ? new dtype(size_or_array) : new Float64Array(size_or_array);
// }


// dtype comparison (hierarchical)
// export function dtype_eq<T extends TypedArrayConstructor>(A: T, B: T): boolean {
//   return A === B;
// }
// export function dtype_leq(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean {
//   return hierarchy_leq[A.name + B.name] || false;
// }
// export function dtype_lt(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean {
//   return hierarchy_lt[A.name + B.name] || false;
// }
// export function dtype_is_integer(A: TypedArrayConstructor): boolean {
//   return A === Uint32Array || A === Uint8Array || A === Int8Array || A === Uint8ClampedArray || A === Int16Array || A === Uint16Array || A === Int32Array;// || A === BigInt64Array || A === BigUint64Array;
// }
// export function dtype_is_boolean(A: TypedArrayConstructor): boolean {
//   return A === Uint8Array;
// }
// export function dtype_is_float(A: TypedArrayConstructor): boolean {
//   return A === Float32Array || A === Float64Array;
// }

// type SupposedlyAnyTypedArrayConstructor = Float64ArrayConstructor; // Otherwise I can't use dtype.from because TypedArrayConstructor doesn't have a from method (even though all the literal constructors do have it)
// export function dtype_least_ancestor(...dtypes: TypedArrayConstructor[]): SupposedlyAnyTypedArrayConstructor {
//   return dtypes.reduce((a, b) => hierarchy_max[a.name + b.name], Float64Array);
// }

// const { hierarchy_leq, hierarchy_lt, hierarchy_max } = (() => {
//   const descendants: { [key: string]: string[] } = {
//     Uint8Array: [],
//     Int8Array: [],
//     Uint8ClampedArray: [],
//     Int16Array: ['Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
//     Uint16Array: ['Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
//     Int32Array: ['Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
//     Uint32Array: ['Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
//     BigInt64Array: ['Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
//     BigUint64Array: ['Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
//     Float32Array: ['Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray', 'BigUint32Array', 'BigInt32Array'],
//     Float64Array: ['Float32Array', 'Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray', 'BigUint32Array', 'BigInt32Array', 'BigUint64Array', 'BigInt64Array'],
//   }
//   const ltPairs: string[] = [];
//   const leqPairs: string[] = [];
//   for (const [key, value] of Object.entries(descendants)) {
//     leqPairs.push(key + key);
//     leqPairs.push(...value.map(v => key + v));
//     ltPairs.push(...value.map(v => key + v));
//   }
//   const hierarchy_leq = Object.fromEntries(leqPairs.map(pair => [pair, true]));
//   const hierarchy_lt = Object.fromEntries(ltPairs.map(pair => [pair, false]));
//   const hierarchy_max = {};
//   for (let A of dtypes) {
//     for (let B of dtypes) {
//       let lca: any = Float64Array;
//       for (let C of dtypes) {
//         if (leqPairs[A.name + C.name] && leqPairs[B.name + C.name] && leqPairs[C.name + lca.name]) {
//           lca = C;
//         }
//       }
//       hierarchy_max[A.name + B.name] = lca;
//     }
//   }
//   return { hierarchy_leq, hierarchy_lt, hierarchy_max };
// })();


// export type TypedArrayConstructor =
//   Int8ArrayConstructor |
//   Uint8ArrayConstructor |
//   Uint8ClampedArrayConstructor |
//   Int16ArrayConstructor |
//   Uint16ArrayConstructor |
//   Int32ArrayConstructor |
//   Uint32ArrayConstructor |
//   Float32ArrayConstructor |
//   Float64ArrayConstructor;
// // BigInt64ArrayConstructor |
// // BigUint64ArrayConstructor;


// export type TypedArray =
//   Int8Array |
//   Uint8Array |
//   Uint8ClampedArray |
//   Int16Array |
//   Uint16Array |
//   Int32Array |
//   Uint32Array |
//   Float32Array |
//   Float64Array;
// // BigInt64Array |
// // BigUint64Array;



