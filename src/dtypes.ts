//@ts-check

// Mozilla docs:
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypedArray

// This is a mess:
// https://stackoverflow.com/questions/65129070/
// https://stackoverflow.com/questions/49242232/
// https://stackoverflow.com/questions/69783310/


export const dtypes = [Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array];//, BigInt64Array, BigUint64Array]


export function new_buffer<
  T extends TypedArrayConstructor = Float64ArrayConstructor
>(
  size_or_array: number | ArrayLike<number> | ArrayLike<boolean> | TypedArray,
  dtype?: T
): InstanceType<T> {
  //@ts-ignore
  return dtype ? new dtype(size_or_array) : new Float64Array(size_or_array);
}


// dtype comparison (hierarchical)
export function dtype_eq<T extends TypedArrayConstructor>(A: T, B: T): boolean {
  return A === B;
}
export function dtype_leq(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean {
  return hierarchy_leq[A.name + B.name] || false;
}
export function dtype_lt(A: TypedArrayConstructor, B: TypedArrayConstructor): boolean {
  return hierarchy_lt[A.name + B.name] || false;
}
export function dtype_is_integer(A: TypedArrayConstructor): boolean {
  return A === Uint32Array || A === Uint8Array || A === Int8Array || A === Uint8ClampedArray || A === Int16Array || A === Uint16Array || A === Int32Array;// || A === BigInt64Array || A === BigUint64Array;
}
export function dtype_is_boolean(A: TypedArrayConstructor): boolean {
  return A === Uint8Array;
}
export function dtype_is_float(A: TypedArrayConstructor): boolean {
  return A === Float32Array || A === Float64Array;
}

type SupposedlyAnyTypedArrayConstructor = Float64ArrayConstructor; // Otherwise I can't use dtype.from because TypedArrayConstructor doesn't have a from method (even though all the literal constructors do have it)
export function dtype_least_ancestor(...dtypes: TypedArrayConstructor[]): SupposedlyAnyTypedArrayConstructor {
  return dtypes.reduce((a, b) => hierarchy_max[a.name + b.name], Float64Array);
}

const { hierarchy_leq, hierarchy_lt, hierarchy_max } = (() => {
  const descendants: { [key: string]: string[] } = {
    Uint8Array: [],
    Int8Array: [],
    Uint8ClampedArray: [],
    Int16Array: ['Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
    Uint16Array: ['Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
    Int32Array: ['Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
    Uint32Array: ['Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
    BigInt64Array: ['Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
    BigUint64Array: ['Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray'],
    Float32Array: ['Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray', 'BigUint32Array', 'BigInt32Array'],
    Float64Array: ['Float32Array', 'Int32Array', 'Uint32Array', 'Int16Array', 'Uint16Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray', 'BigUint32Array', 'BigInt32Array', 'BigUint64Array', 'BigInt64Array'],
  }
  const ltPairs: string[] = [];
  const leqPairs: string[] = [];
  for (const [key, value] of Object.entries(descendants)) {
    leqPairs.push(key + key);
    leqPairs.push(...value.map(v => key + v));
    ltPairs.push(...value.map(v => key + v));
  }
  const hierarchy_leq = Object.fromEntries(leqPairs.map(pair => [pair, true]));
  const hierarchy_lt = Object.fromEntries(ltPairs.map(pair => [pair, false]));
  const hierarchy_max = {};
  for (let A of dtypes) {
    for (let B of dtypes) {
      let lca: any = Float64Array;
      for (let C of dtypes) {
        if (leqPairs[A.name + C.name] && leqPairs[B.name + C.name] && leqPairs[C.name + lca.name]) {
          lca = C;
        }
      }
      hierarchy_max[A.name + B.name] = lca;
    }
  }
  return { hierarchy_leq, hierarchy_lt, hierarchy_max };
})();


export type TypedArrayConstructor =
  Int8ArrayConstructor |
  Uint8ArrayConstructor |
  Uint8ClampedArrayConstructor |
  Int16ArrayConstructor |
  Uint16ArrayConstructor |
  Int32ArrayConstructor |
  Uint32ArrayConstructor |
  Float32ArrayConstructor |
  Float64ArrayConstructor;
// BigInt64ArrayConstructor |
// BigUint64ArrayConstructor;


export type TypedArray =
  Int8Array |
  Uint8Array |
  Uint8ClampedArray |
  Int16Array |
  Uint16Array |
  Int32Array |
  Uint32Array |
  Float32Array |
  Float64Array;
// BigInt64Array |
// BigUint64Array;



