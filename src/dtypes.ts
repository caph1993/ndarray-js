//@ts-check
import { Complex, ZERO, is_complex } from './complex';

// Mozilla docs:
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypedArray

// This is a mess:
// https://stackoverflow.com/questions/65129070/
// https://stackoverflow.com/questions/49242232/
// https://stackoverflow.com/questions/69783310/

export type DType = "bool" | "int32" | "int64" | "float32" | "float64" | "complex128" | "object";

export const bool = "bool";
export const int32 = "int32";
/** Integers stored in a Float64Array: exact up to 2**53. Values are truncated to integers when stored. */
export const int64 = "int64";
export const float32 = "float32";
export const float64 = "float64";
/** Complex numbers: the buffer is an Array of Complex objects */
export const complex128 = "complex128";
export const object = "object";

// export const bool = {
//   name: "bool",
//   BufferType: Uint8Array
// } as const satisfies DType_<"bool", Uint8Array>

// export const int32 = {
//   name: "int32",
//   BufferType: Int32Array
// } as const satisfies DType_<"int32", Int32Array>

// export const float32 = {
//   name: "float32",
//   BufferType: Float32Array
// } as const satisfies DType_<"float32", Float32Array>

// export const float64 = {
//   name: "float64",
//   BufferType: Float64Array
// } as const satisfies DType_<"float64", Float64Array>

// export const object = {
//   name: "object",
//   BufferType: Array
// } as const satisfies DType_<"object", any[]>




// type F32 = Float32Array & DType<"float32">
// type I32 = Int32Array & DType<"int32">
// type Bool = Uint8Array & DType<"bool">

// export type DType =
//   | typeof bool
//   | typeof int32
//   | typeof float32
//   | typeof float64
//   | typeof object;

// export type NameOf<D extends DType_<any, any>> = D["name"]

export type PromoteDType<A, B> =
  A extends "object" ? "object" :
  B extends "object" ? "object" :
  A extends "float64" ? "float64" :
  B extends "float64" ? "float64" :
  A extends "float32" ? "float32" :
  B extends "float32" ? "float32" :
  A extends "int32" ? "int32" :
  B extends "int32" ? "int32" :
  "bool";

// export type DTypeByName<N> =
//   N extends "bool" ? typeof bool :
//   N extends "int32" ? typeof int32 :
//   N extends "float32" ? typeof float32 :
//   N extends "float64" ? typeof float64 :
//   N extends "object" ? typeof object :
//   never

export interface HasDType {
  readonly dtype: DType
}

export type DTypeOf<T extends HasDType> = T["dtype"]

export type Promote<
  A extends HasDType,
  B extends HasDType
> = PromoteDType<A["dtype"], B["dtype"]>


export const infer_dtype = {
  Float32Array: float32,
  Float64Array: float64,
  Int32Array: int32,
  Uint8Array: bool,
  Array: object,
}

export const dtype_constructors = {
  bool: Uint8Array,
  int32: Int32Array,
  int64: Float64Array,
  float32: Float32Array,
  float64: Float64Array,
  complex128: Array,
  object: Array,
}

export type Buffer = {
  length: number
  [i: number]: any
  [Symbol.iterator](): IterableIterator<any>
  forEach: any;
  map: any;
  slice: (start?: number, end?: number) => Buffer;
}

export function new_buffer(
  size_or_array: number | Buffer,
  dtype?: DType
): InstanceType<any> {

  dtype = dtype || float64;
  if (dtype === "complex128") {
    if (typeof size_or_array === "number") return new Array(size_or_array).fill(ZERO) as any;
    return Array.from(size_or_array as any, (x: any) => Complex.from(x)) as any;
  }
  if (dtype === "object") {
    // Handled separately because Array is not a typed array and doesn't have the same constructor signature.
    if (Array.isArray(size_or_array)) {
      return size_or_array as any;
    } else if (typeof size_or_array === "number") {
      return Array.from({ length: size_or_array }) as any;
    } else {
      return Array.from(size_or_array as any) as any;
    }
    // else {
    // throw new Error(`Invalid input for object dtype: ${size_or_array}`);
    // }
  }
  return new dtype_constructors[dtype](size_or_array as any) as InstanceType<any>;
}

/** Copies (and casts) a buffer or array into a new buffer of the given dtype */
export function convert_buffer(src: ArrayLike<any>, dtype: DType): any {
  const n = src.length;
  const out = new_buffer(n, dtype);
  if (dtype === "complex128") for (let i = 0; i < n; i++) out[i] = Complex.from(src[i]);
  else if (n > 0 && is_complex(src[0]) && dtype !== "object") {
    // complex -> real: the imaginary part is discarded (as numpy does)
    const re = (v: any) => is_complex(v) ? v.re : +v;
    if (dtype === "bool") for (let i = 0; i < n; i++) out[i] = truthy(src[i]) ? 1 : 0;
    else for (let i = 0; i < n; i++) out[i] = re(src[i]);
  }
  else if (dtype === "bool") for (let i = 0; i < n; i++) out[i] = truthy(src[i]) ? 1 : 0;
  else if (dtype === "int64") for (let i = 0; i < n; i++) { const v = Math.trunc(+src[i]); out[i] = Number.isFinite(v) ? v : 0; }
  else if (dtype === "object") for (let i = 0; i < n; i++) out[i] = src[i];
  else for (let i = 0; i < n; i++) out[i] = +src[i];
  return out;
}

/** Python-like truthiness, except that nan is true (as in numpy) */
export function truthy(x: any): boolean { return x !== x ? true : (x instanceof Complex ? (x.re !== 0 || x.im !== 0 || x.isNaN) : !!x); }

/** Makes sure that the values of a buffer are valid for the dtype (int64 values are stored as truncated floats) */
export function fix_buffer(buffer: any, dtype: DType): any {
  if (dtype === "int64") for (let i = 0; i < buffer.length; i++) buffer[i] = Number.isFinite(buffer[i]) ? Math.trunc(buffer[i]) : 0;
  return buffer;
}

export function promote(a: DType, b: DType): DType {
  if (a === "object" || b === "object") return object
  if (a === "complex128" || b === "complex128") return complex128
  if (a === "float64" || b === "float64") return float64
  // As in numpy, int32 does not fit in float32
  if ((a === "float32" && (b === "int32" || b === "int64")) || (b === "float32" && (a === "int32" || a === "int64"))) return float64
  if (a === "float32" || b === "float32") return float32
  if (a === "int64" || b === "int64") return int64
  if (a === "int32" || b === "int32") return int32
  return bool
}

export function dtype_cmp(t1: DType, t2: DType): number {
  if (t1 === t2) return 0;
  if (t1 === "object") return 1;
  if (t2 === "object") return -1;
  if (t1 === "complex128") return 1;
  if (t2 === "complex128") return -1;
  if (t1 === "float64") return 1;
  if (t2 === "float64") return -1;
  if (t1 === "float32") return 1;
  if (t2 === "float32") return -1;
  if (t1 === "int64") return 1;
  if (t2 === "int64") return -1;
  if (t1 === "int32") return 1;
  if (t2 === "int32") return -1;
  return 0; // Should not happen
}
export function dtype_max(...dtypes: DType[]): DType {
  if (dtypes.length === 0) return float64;
  return dtypes.reduce((a, b) => promote(a, b));
}

/**
 * dtype of a plain JS scalar (weak, like python scalars) combined with another operand:
 * it never upcasts within the same kind (int < float), as in numpy 2 (NEP 50).
 */
export function weak_dtype(weak: DType, other: DType): DType {
  if (other === "object" || weak === "object") return object;
  if (other === "complex128" || weak === "complex128") return complex128;
  if (weak === "bool") return other;
  if (other === "bool") return weak === "float32" ? float64 : weak;
  if (dtype_is_float(weak) && dtype_is_integer(other)) return float64;
  return other;
}

export function dtype_is_integer(dtype: DType): boolean {
  return dtype === "int32" || dtype === "int64";
}
export function dtype_is_boolean(dtype: DType): boolean {
  return dtype === "bool";
}
export function dtype_is_complex(dtype: DType): boolean {
  return dtype === "complex128";
}
export function dtype_is_float(dtype: DType): boolean {
  return dtype === "float32" || dtype === "float64";
}



export type DtypeResolver = {
  (dtypes: DType[], out?: DType | HasDType): DType;
};

function get_out_dtype(out?: DType | HasDType) {
  if (!out) return null;
  if (typeof out === "string") return out;
  if ("dtype" in out) return out.dtype;
  return out as DType;
}

export function addition_out(dtypes: DType[], out?: DType | HasDType): DType {
  out = get_out_dtype(out);
  let inferred = dtype_max(int32, ...dtypes);
  if (out && dtype_cmp(out, inferred) < 0)
    throw new Error(`Output array has dtype ${out}, which cannot hold all values of type ${inferred}.`);
  return out || inferred;
}
export function bitwise_out(dtypes: DType[], out?: DType | HasDType): DType {
  out = get_out_dtype(out);
  let inferred = dtype_max(...dtypes);
  if (out && dtype_cmp(out, inferred) < 0)
    throw new Error(`Output array has dtype ${out}, which cannot hold all values of type ${inferred}.`);
  return out || inferred;
}

function check_out(inferred: DType, out: DType | null): DType {
  if (out && dtype_cmp(out, inferred) < 0)
    throw new Error(`Output array has dtype ${out}, which cannot hold all values of type ${inferred}.`);
  return out || inferred;
}

/** Same dtype as the inputs (bool stays bool). Used by add, multiply, maximum, ... */
export function same_out(dtypes: DType[], out?: DType | HasDType): DType {
  return check_out(dtype_max(...dtypes), get_out_dtype(out));
}

/** Like same_out but bool is not allowed (subtract, negative). */
export function numeric_out(dtypes: DType[], out?: DType | HasDType): DType {
  const inferred = dtype_max(...dtypes);
  if (inferred === bool) throw new Error(`Numpy boolean subtract, the \`-\` operator, is not supported, use the bitwise_xor, the \`^\` operator, or the logical_xor function instead.`);
  return check_out(inferred, get_out_dtype(out));
}

/** Integer or bool only (bitwise operators and shifts). */
export function integer_out(dtypes: DType[], out?: DType | HasDType): DType {
  const inferred = dtype_max(...dtypes);
  if (dtype_is_float(inferred) || inferred === object || inferred === complex128) throw new Error(`Bitwise operations are not supported for dtype ${inferred}. Only bool and integer arrays are supported.`);
  return check_out(inferred, get_out_dtype(out));
}

/** Integers or bool promoted to at least int32, floats unchanged (sum, cumsum, prod). */
export function sum_out(dtypes: DType[], out?: DType | HasDType): DType {
  const inferred = dtype_max(...dtypes);
  return check_out((inferred === bool || inferred === int32) ? int64 : inferred, get_out_dtype(out));
}

/** Result of functions that return the real-valued magnitude of complex numbers (abs, real, imag, angle) */
export function realpart_out(dtypes: DType[], out?: DType | HasDType): DType {
  const m = dtype_max(...dtypes);
  return check_out(m === complex128 ? float64 : (dtype_cmp(m, float32) >= 0 ? m : float64), get_out_dtype(out));
}

export function float_out(dtypes: DType[], out?: DType | HasDType): DType {
  out = get_out_dtype(out);
  let max = dtype_max(...dtypes);
  // Output is float64. If you want float32, provide out.
  const inferred = (dtype_cmp(max, float32) >= 0) ? max : float64;
  if (out && dtype_cmp(out, inferred) < 0)
    throw new Error(`Output array has dtype ${out}, which cannot hold all values of type ${inferred}.`);
  return out || inferred;
}

export function bool_out<T extends DType>(dtypes: DType[], out?: DType | HasDType): T {
  out = get_out_dtype(out);
  if (out && dtype_cmp(out, bool) != 0)
    throw new Error(`Output array has dtype ${out} but expected bool.`);
  return bool as T;
}

export function argmax_out(dtypes: DType[], out?: DType | HasDType): DType {
  out = get_out_dtype(out);
  const inferred = int32;
  if (out && !dtype_is_integer(out)) {
    throw new Error(`Output array has dtype ${out}, which is not an integer type.`);
  }
  return out || inferred;
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
//   return hierarchy_leq[A + B] || false;
// }
// export function dtype_lt(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean {
//   return hierarchy_lt[A + B] || false;
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
//   return dtypes.reduce((a, b) => hierarchy_max[a + b], Float64Array);
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
//         if (leqPairs[A + C] && leqPairs[B + C] && leqPairs[C + lca]) {
//           lca = C;
//         }
//       }
//       hierarchy_max[A + B] = lca;
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



