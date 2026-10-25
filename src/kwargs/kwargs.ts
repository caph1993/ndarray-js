import { NDArray, broadcast_n_shapes } from '../NDArray';
import { Arr, ArrOrAny, ArrOrConst } from "../array/js-interface";
import { isarray, asarray } from '../NDArray';
/** Methods that every binary ufunc has (np.add.reduce, np.add.accumulate, ...) */
export type UfuncMethods = {
  reduce: (array: ArrOrAny, axis?: AxisArg | any, dtype?: any, out?: any, keepdims?: boolean, initial?: any, where?: any) => any;
  accumulate: (array: ArrOrAny, axis?: number | any, dtype?: any, out?: any) => NDArray;
  outer: (A: ArrOrAny, B: ArrOrAny, out?: any) => NDArray;
  reduceat: (array: ArrOrAny, indices: ArrOrAny, axis?: number | any, dtype?: any, out?: any) => NDArray;
  at: (a: NDArray, indices: any, b?: any) => void;
  identity: any; nin: number; nout: number; nargs: number;
};
export type AxisArg = number | number[] | null;

export type ArrOrNull = NDArray | null;

import { empty } from "../NDArray";
import { Where } from '../array/indexes';
import { DType, canonical_dtype } from '../dtypes';

export { Arr, ArrOrAny, ArrOrConst };




// We need 2 types of wrapper on top of the implementation:
// The implementation expects NDArray on all arguments and returns NDArray
// The "library wrapper" expects NDArray | constant and returns NDArray
// The "user wrapper" returns NDArray | constant



export type Parser = (kwargs: any) => void;

/** True for `{...}` literals (and Object.create(null)), false for arrays, NDArrays, typed arrays, functions, class instances */
export function is_plain_object(value: any): boolean {
  if (value === null || typeof value !== "object") return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * Set while a numpy-syntax template is evaluated: 0-d results stay arrays (like numpy scalars), so that they keep their dtype and have methods:
 * as JS numbers an integer-valued np.mean(x) == 2.0 would become a python int in the next operation: `np.arange(3) - np.arange(3).mean()` must be float.
 * The final value of the template is unwrapped.
 */
export const template_mode = { depth: 0 };
const keeps_float = (out: any) => template_mode.depth > 0 && isarray(out) && out.shape.length == 0;

export function kwargs_decorator<
  Wrapper extends (...args: any[]) => T,
  Implementation extends (...args: any[]) => T,
  T = NDArray
>({ defaults, implementation, parsers, this_as_first_arg, keep_weak }: {
  defaults: readonly (readonly [string, any])[],
  implementation: Implementation,
  parsers?: Parser[],
  this_as_first_arg?: boolean,
  /** names of parameters that keep python-like float scalars (weak 0-d arrays) instead of unwrapping them to JS numbers */
  keep_weak?: string[],
}) {
  const kwParser = kwargs_parser<Wrapper, Implementation>(defaults, keep_weak || [], ...(parsers || []));
  return (function (...args: Parameters<Wrapper>) {
    //@ts-ignore
    if (this_as_first_arg) args = [this, ...args];
    const parsedArgs = kwParser(args);
    const out = implementation(...parsedArgs);
    if (isarray(out) && out.shape.length == 0) return keeps_float(out) ? out : out.item();
    if (Array.isArray(out) && out.length && out.every(o => isarray(o)) && out.some(o => o.shape.length == 0)) return Object.assign(out.map(o => o.shape.length == 0 ? o.item() : o), out);
    return out;
  }) as Wrapper;
}

function kwargs_parser<
  Wrapper extends (...args: any[]) => any,
  Implementation extends (...args: any[]) => any,
>(
  defaults_: readonly (readonly [string, any])[],
  keep_weak: string[],
  ...parsers: Parser[]
) {
  // The name starting with "..." collects all remaining positional arguments into an array.
  // Names after it are keyword-only.
  const names = defaults_.map(([key]) => key.startsWith('...') ? key.slice(3) : key);
  const spreadIdx = defaults_.findIndex(([key]) => key.startsWith('...'));
  const spreadKey: string | null = spreadIdx >= 0 ? names[spreadIdx] : null;
  const nPositional = spreadIdx >= 0 ? spreadIdx : names.length;
  const defaultsDict: { [key: string]: any } = {};
  defaults_.forEach(([_, value], i) => { defaultsDict[names[i]] = value; });
  const validKwargsNames: { [key: string]: boolean } = {};
  names.forEach(name => { validKwargsNames[name] = true; });
  const shorten = (s: any) => {
    try { s = JSON.stringify(s); } catch (e) { s = String(s); }
    if (typeof s !== "string") s = String(s);
    return s.length > 100 ? s.slice(0, 100) + '...' : s;
  }
  return (args: Parameters<Wrapper>) => {
    let kwargs = {};
    if (spreadKey) kwargs[spreadKey] = [...defaultsDict[spreadKey]]; // copy
    for (let i = 0; i < args.length; i++) {
      let value = args[i];
      if (is_plain_object(value)) {
        for (let key in value) {
          if (key == spreadKey) kwargs[key].push(value[key]);
          else {
            if (key in kwargs) throw new Error(`Duplicate argument ${key} in ${shorten(args)}`);
            if (!validKwargsNames[key]) throw new Error(`Invalid argument ${key} in ${shorten(args)}. Valid arguments: ${names.join(', ')}`);
            kwargs[key] = value[key];
          }
        }
      } else if (value !== undefined) {
        if (spreadKey && i >= nPositional) kwargs[spreadKey].push(value);
        else if (i < nPositional) {
          const key = names[i];
          if (key in kwargs) throw new Error(`Duplicate argument ${key} in ${shorten(args)}`);
          kwargs[key] = value;
        }
        else throw new Error(`Too many arguments ${shorten(args)}`);
      }
    }
    for (let key in defaultsDict) {
      if (!(key in kwargs)) kwargs[key] = defaultsDict[key];
      if (kwargs[key] === undefined) {
        throw new Error(`Missing argument ${key} in ${shorten(args)}.`);
      }
    }
    // Parameters whose default is a plain number/boolean take JS scalars: unwrap 0-d arrays (e.g. from the template parser).
    // Optional parameters (default null) get python-like scalars (weak 0-d arrays, as produced by the template parser) unwrapped too.
    for (const key in defaultsDict) {
      const d = defaultsDict[key], v = kwargs[key];
      if (!isarray(v) || v.shape.length != 0 || keep_weak.includes(key)) continue;
      if (typeof d === "number" || typeof d === "boolean" || (d === null && v._weak)) kwargs[key] = v.item();
    }
    for (let parser of parsers) parser(kwargs);
    return names.map(key => kwargs[key]) as Parameters<Implementation>;
  }
}


export function as_boolean(obj: boolean | number | NDArray) {
  if (isarray(obj)) obj = obj.item();
  if (typeof obj === 'boolean') return obj;
  if (typeof obj === 'number' && (obj == 0 || obj == 1)) return !!obj;
  throw new Error(`'${typeof obj}' object can not be interpreted as boolean: ${obj}`);
}
export function as_number(obj: number | boolean | NDArray) {
  if (isarray(obj)) obj = obj.item();
  if (typeof obj === 'number') return obj;
  if (typeof obj === 'boolean') return obj ? 1 : 0;
  throw new Error(`'${typeof obj}' object can not be interpreted as number: ${obj}`);
}

export const frequently_used_parsers = {
  'a': (kwargs: any) => { kwargs.a = asarray(kwargs.a); },
  'a_axis': (kwargs: any) => {
    kwargs.a = asarray(kwargs.a);
    if (kwargs.axis == null) {
      kwargs.axis = -1;
    }
    const ndims = kwargs.a.shape.length;
    if (kwargs.axis + ndims < 0 || kwargs.axis >= ndims) {
      throw new Error(`Invalid axis ${kwargs.axis} for array of shape ${JSON.stringify(kwargs.a.shape)}`);
    }
    if (kwargs.axis < 0) {
      kwargs.axis += ndims;
    }
  },
  'a_axis_flatten': (kwargs: any) => {
    kwargs.a = asarray(kwargs.a);
    if (kwargs.axis == null) {
      // flatten
      kwargs.a = kwargs.a.reshape(-1);
      kwargs.axis = -1;
    }
    const ndims = kwargs.a.shape.length;
    if (kwargs.axis + ndims < 0 || kwargs.axis >= ndims) {
      throw new Error(`Invalid axis ${kwargs.axis} for array of shape ${JSON.stringify(kwargs.a.shape)}`);
    }
    if (kwargs.axis < 0) {
      kwargs.axis += ndims;
    }
  },
  'keepdims': (kwargs: any) => { kwargs.keepdims = !!kwargs.keepdims; },
  'axis_or_axes': (kwargs: any) => {
    const ax = kwargs.axis;
    const ok = (v: any) => typeof v === "number" && Number.isInteger(v);
    if (ax != null && !(ok(ax) || (Array.isArray(ax) && ax.every(ok)))) {
      throw new Error(`Invalid "axis" argument. Expected integer, list of integers or null. Found ${JSON.stringify(ax)}`);
    }
  },
  'is_integer': (key: string) => (kwargs: any) => {
    if (typeof kwargs[key] !== 'number' || kwargs[key] % 1 !== 0) {
      throw new Error(`Invalid "${key}" argument. Expected integer. Found ${kwargs[key]}`);
    }
  },
  'dtype_or_null': (kwargs: any) => {
    const d = kwargs.dtype;
    if (d != null) kwargs.dtype = canonical_dtype(typeof d === "string" ? d : (d && d.name) || String(d));
  },
  'isarray_or_null': (key: string) => (kwargs: any) => {
    if (kwargs[key] != null && !isarray(kwargs[key])) {
      throw new Error(`Invalid "${key}" argument. Expected NDArray. Found type ${typeof kwargs[key]} (${kwargs[key].prototype})`);
    }
  },
  'out_broadcast': (keys: string[], out = 'out', reshape = true) => (kwargs: any) => {
    const [shapes, shape] = broadcast_n_shapes(keys.map(key => kwargs[key].shape));
    const dtype = kwargs[out] ? kwargs[out].dtype : Float64Array;
    let nd_out = kwargs[out];
    if (!nd_out) nd_out = empty(shape, dtype);
    kwargs[out] = nd_out;
    if (reshape) for (let i in keys) {
      const key = keys[i];
      kwargs[key] = kwargs[key].reshape(shapes[i]);
    }
  },
  'asarray': (key: string) => (kwargs: any) => {
    kwargs[key] = asarray(kwargs[key]);
  },
  'asarray_or_null': (key: string) => (kwargs: any) => {
    if (kwargs[key] === null) return;
    else kwargs[key] = asarray(kwargs[key]);
  },
  'isarray': (key: string) => (kwargs: any) => {
    if (!isarray(kwargs[key])) {
      throw new Error(`Invalid "${key}" argument. Expected NDArray. Found type ${typeof kwargs[key]} (${kwargs[key].prototype})`);
    }
  },
  'is_integer_non_neg': (key: string) => (kwargs: any) => {
    if (typeof kwargs[key] !== 'number' || kwargs[key] < 0 || kwargs[key] % 1 !== 0) {
      throw new Error(`Invalid "${key}" argument. Expected non-negative integer. Found ${kwargs[key]}`);
    }
  },
  'is_pnorm': (key: string) => (kwargs: any) => {
    if (typeof kwargs[key] !== 'number' || kwargs[key] <= 0) {
      throw new Error(`Invalid "${key}" argument. Expected non-negative integer. Found ${kwargs[key]}`);
    }
  },
  'as_boolean': (key: string) => (kwargs: any) => {
    kwargs[key] = as_boolean(kwargs[key]);
  },
  'as_number': (key: string) => (kwargs: any) => {
    kwargs[key] = as_number(kwargs[key]);
  },
}

// ==============================
//  Compact signature definitions
// ==============================

/** Arguments that accept NDArrays also accept anything that can be converted to one */
export type Loose<T> = [T] extends [NDArray] ? ArrOrAny : [T] extends [NDArray | null] ? ArrOrAny | null : T;
type StripDots<N extends string> = N extends `...${infer R}` ? R : N;
/** Object with the named (keyword) arguments of an implementation: {a?: ..., axis?: ...} */
export type KwargsOf<P extends readonly any[], D extends readonly (readonly [string, any])[]> =
  D extends readonly [readonly [infer N extends string, any], ...infer DR extends readonly (readonly [string, any])[]]
  ? P extends readonly [infer P0, ...infer PR]
  ? { [K in StripDots<N>]?: Loose<P0> } & KwargsOf<PR, DR>
  : {}
  : {};
type Rest<P extends readonly any[]> = P extends readonly [any, ...infer R] ? R : [];
type OrKw<R extends readonly any[], K> = { [Q in keyof R]?: Loose<R[Q]> | K };
type AsArray<T> = T extends any[] ? T : never;
/** Function: first argument required, the others optional. Any argument can be replaced by an object of keyword arguments */
export type FuncWrapper<I extends (...args: any) => any, K> =
  Parameters<I> extends [infer A, ...any[]]
  ? (a: Loose<A> | K, ...rest: AsArray<OrKw<Rest<Required<Parameters<I>>>, K>>) => ReturnType<I>
  : (...args: any[]) => ReturnType<I>;
/** Method: same as the function but `a` is the array itself */
export type MethodWrapper<I extends (...args: any) => any, K> = (...args: AsArray<OrKw<Rest<Required<Parameters<I>>>, K>>) => ReturnType<I>;

/**
 * Defines a signature family compactly:
 *   const sig = signature<Impl>()([["a", undefined], ["axis", null]] as const, [parsers...]);
 *   np.f = sig.func(impl);          // np.f(a, axis) | np.f({a, axis}) | np.f(a, {axis}) ...
 *   NDArray.prototype.f = sig.method(impl);  // x.f(axis)
 * Defaults are in numpy's positional order; `undefined` means required.
 */
export function signature<I extends (...args: any[]) => any>() {
  return <D extends readonly (readonly [string, any])[]>(defaults: D, parsers: Parser[] = [], options: { keep_weak?: string[] } = {}) => {
    type K = KwargsOf<Required<Parameters<I>>, D>;
    // variadic signatures ("...shape") also accept the values spread: x.reshape(2, 3)
    type Spread = [Extract<D[number][0], `...${string}`>] extends [never] ? unknown : ((...args: any[]) => ReturnType<I>);
    return {
      defaults, parsers,
      func: (implementation: I) => kwargs_decorator<any, any, any>({ defaults, implementation, parsers, ...options }) as FuncWrapper<I, K> & Spread,
      method: (implementation: I) => kwargs_decorator<any, any, any>({ defaults, implementation, parsers, this_as_first_arg: true, ...options }) as MethodWrapper<I, K> & Spread,
    };
  };
}

// Used by sort:
export namespace Func_a_lastAxis {
  export type Implementation = (a: ArrOrAny, axis: number) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", -1]];
  export const parsers = [
    frequently_used_parsers['a_axis_flatten'],
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}


// Used by elementwise operators and methods:
export namespace Func_a_out {
  export type Implementation = (a: NDArray, out: NDArray | null) => NDArray;
  export type Kwargs = { a?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = (a: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_out {
  export type Implementation = Func_a_out.Implementation;
  export type Kwargs = { out?: NDArray | null };
  export type Wrapper = (out?: NDArray | null | Kwargs) => NDArray;
  export const defaults: [string, any][] = [["a", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}
// Used by round
export namespace Func_a_decimals_out {
  export type Implementation = (a: NDArray, decimals: number, out: NDArray | null) => NDArray;
  export type Kwargs = { a?: ArrOrAny, decimals?: number, out?: NDArray | null };
  export type Wrapper = (a: ArrOrAny | Kwargs, decimals: number | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["decimals", 0], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.is_integer('decimals'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_a_decimals_out {
  export type Implementation = Func_a_decimals_out.Implementation;
  export type Kwargs = { decimals?: number, out?: NDArray | null };
  export type Wrapper = (decimals: number | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["decimals", 0], ["out", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.is_integer('decimals'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers, this_as_first_arg: true,
  });
}



// Used by binary operators and methods:
export namespace Func_a_other_out {
  export type Implementation = (a: NDArray, other: NDArray, out: NDArray | null) => NDArray;
  export type Kwargs = { a: ArrOrAny, other?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = ((a: ArrOrAny | Kwargs, other: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray) & UfuncMethods;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["other", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray('other'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_other_out {
  export type Implementation = Func_a_other_out.Implementation;
  export type Kwargs = { other?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = (other: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const defaults: [string, any][] = [["a", undefined], ["other", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.asarray('other'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}

// Used by assign operators and methods:
export namespace Func_a_values_where {
  export type Implementation = (a: NDArray, values: NDArray, where: Where) => NDArray;
  export type Kwargs = { a: NDArray, values?: ArrOrAny, where: Where };
  export type Wrapper = (a: NDArray | Kwargs, values: ArrOrAny | Kwargs, ...where: Where) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["values", undefined], ["...where", []]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray('values'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_values_where {
  export type Implementation = Func_a_values_where.Implementation;
  export type Kwargs = { values?: ArrOrAny, where: Where };
  export type Wrapper = (values: ArrOrAny | Kwargs, ...where: Where) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["values", undefined], ["...where", []]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.asarray('values'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers, this_as_first_arg: true,
  });
}

// Used by reduce functions and methods (max, min, argmax, any, all, ...):
export namespace Func_a_axis_keepdims {
  export type Implementation = (a: NDArray, axis: AxisArg, keepdims: boolean, out: NDArray | null) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg, keepdims?: boolean, out?: NDArray | null };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["keepdims", false], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.axis_or_axes,
    frequently_used_parsers.keepdims,
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers
    });
  }
}

export namespace Method_a_axis_keepdims {
  export type Implementation = Func_a_axis_keepdims.Implementation;
  export type Kwargs = { axis?: AxisArg, keepdims?: boolean, out?: NDArray | null };
  export type Wrapper = (axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults = Func_a_axis_keepdims.defaults;
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    ...Func_a_axis_keepdims.parsers.slice(1),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}

// Used by sum, product, mean, ...:
export namespace Func_a_axis_keepdims_dtype_out {
  export type Implementation = (a: NDArray, axis: AxisArg, keepdims: boolean, dtype: DType | null, out: NDArray | null) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg, keepdims?: boolean, dtype?: DType | null, out?: NDArray | null };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs, dtype?: DType | null | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["keepdims", false], ["dtype", null], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.axis_or_axes,
    frequently_used_parsers.keepdims,
    frequently_used_parsers.dtype_or_null,
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({ defaults, implementation, parsers });
  }
}
export namespace Method_a_axis_keepdims_dtype_out {
  export type Implementation = Func_a_axis_keepdims_dtype_out.Implementation;
  export type Kwargs = { axis?: AxisArg, keepdims?: boolean, dtype?: DType | null, out?: NDArray | null };
  export type Wrapper = (axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs, dtype?: DType | null | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const defaults = Func_a_axis_keepdims_dtype_out.defaults;
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    ...Func_a_axis_keepdims_dtype_out.parsers.slice(1),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({ defaults, implementation, parsers, this_as_first_arg: true });
  }
}

// For norm:
export namespace Func_a_ord_axis_keepdims {
  export type Implementation = (a: NDArray, ord: number, axis: AxisArg, keepdims: boolean) => NDArray;
  export type Kwargs = { a?: ArrOrAny, ord?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (a: ArrOrAny | Kwargs, ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["ord", 2], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.axis_or_axes,
    frequently_used_parsers.is_pnorm('ord'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers,
    });
  }
}
export namespace Method_a_ord_axis_keepdims {
  export type Kwargs = { ord?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export type Implementation = Func_a_ord_axis_keepdims.Implementation;
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults: Func_a_ord_axis_keepdims.defaults,
      parsers: [frequently_used_parsers.isarray('a'), ...Func_a_ord_axis_keepdims.parsers.slice(1)],
      implementation,
      this_as_first_arg: true,
    });
  }
}
// For std and var:
export namespace Func_a_axis_ddof_keepdims {
  export type Implementation = (a: NDArray, axis: AxisArg, ddof: number, keepdims: boolean, dtype: DType | null, out: NDArray | null) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg, ddof?: number, keepdims?: boolean, dtype?: DType | null, out?: NDArray | null };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs, ddof?: number | Kwargs, keepdims?: boolean | Kwargs, dtype?: DType | null | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["ddof", 0], ["keepdims", false], ["dtype", null], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.axis_or_axes,
    frequently_used_parsers.is_integer_non_neg('ddof'),
    frequently_used_parsers.keepdims,
    frequently_used_parsers.dtype_or_null,
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers,
    });
  }
}
export namespace Method_a_axis_ddof_keepdims {
  export type Implementation = Func_a_axis_ddof_keepdims.Implementation;
  export type Kwargs = { axis?: AxisArg, ddof?: number, keepdims?: boolean, dtype?: DType | null, out?: NDArray | null };
  export type Wrapper = (axis?: AxisArg | Kwargs, ddof?: number | Kwargs, keepdims?: boolean | Kwargs, dtype?: DType | null | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults: Func_a_axis_ddof_keepdims.defaults,
      parsers: [frequently_used_parsers.isarray('a'), ...Func_a_axis_ddof_keepdims.parsers.slice(1)],
      implementation,
      this_as_first_arg: true,
    });
  }
}


// Used by atan2:
export namespace Func_y_x_out {
  export type Implementation = (y: ArrOrAny, x: ArrOrAny, out?: NDArray | null) => NDArray;
  export type Kwargs = { y?: ArrOrAny, x?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = (y: ArrOrAny | Kwargs, x: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["y", undefined], ["x", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('y'),
    frequently_used_parsers.asarray('x'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}


// Used by clip:
export namespace Func_a_a_min_a_max_out {
  export type Implementation = (a: NDArray, a_min?: NDArray, a_max?: NDArray, out?: NDArray | null) => NDArray;
  export type Kwargs = { a?: ArrOrAny, a_min?: ArrOrAny, a_max?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = (a: ArrOrAny | Kwargs, a_min?: ArrOrAny | Kwargs, a_max?: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["a_min", null], ["a_max", null], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray_or_null('a_min'),
    frequently_used_parsers.asarray_or_null('a_max'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

export namespace Method_a_min_a_max_out {
  export type Implementation = Func_a_a_min_a_max_out.Implementation;
  export type Kwargs = { a_min?: ArrOrAny, a_max?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = (a_min?: ArrOrAny | Kwargs, a_max?: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults: Func_a_a_min_a_max_out.defaults,
      parsers: [frequently_used_parsers.isarray('a'), ...Func_a_a_min_a_max_out.parsers.slice(1)],
      implementation, this_as_first_arg: true,
    });
  }
}

// Used by heaviside:
export namespace Func_x1_x2_out {
  export type Implementation = (x1: ArrOrAny, x2: ArrOrAny, out?: NDArray | null) => NDArray;
  export type Kwargs = { x1?: ArrOrAny, x2?: ArrOrAny, out?: NDArray | null };
  export type Wrapper = (x1: ArrOrAny | Kwargs, x2: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["x1", undefined], ["x2", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('x1'),
    frequently_used_parsers.asarray('x2'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}


export namespace Func_isarray {
  export type Implementation = (A: any) => A is NDArray;
  export type Kwargs = { A?: any };
  export type Wrapper = (A: any | Kwargs) => boolean;
  export const decorator = kwargs_decorator<Wrapper, Implementation, boolean>;
  export const defaults: [string, any][] = [
    ["A", undefined],
  ];
  export const parsers = [
  ];
}

export namespace Func_new_NDArray {
  export type Implementation = (flat: Buffer, shape: number[], dtype?: DType) => NDArray;
  export type Kwargs = { flat?: Buffer, shape?: number[], dtype?: DType };
  export type Wrapper = (flat: Buffer | Kwargs, shape?: number[], dtype?: DType) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["flat", undefined],
    ["shape", []],
    ["dtype", null],
  ];
}

export namespace Func_asarray {
  export type Implementation = (A: NDArray | any, dtype: DType) => NDArray;
  export type Kwargs = { A?: NDArray | any, dtype?: DType };
  export type Wrapper = (A: NDArray | any | Kwargs, dtype?: DType) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["A", undefined],
    ["dtype", null],
  ];
}

export namespace Func_array {
  export type Implementation = (A: NDArray | any, dtype: DType) => NDArray;
  export type Kwargs = { A?: NDArray | any, dtype?: DType };
  export type Wrapper = (A: NDArray | any | Kwargs, dtype?: DType) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["A", undefined],
    ["dtype", null],
  ];
}

export namespace Func_broadcast_shapes {
  export type Implementation = (shapeA: number[], shapeB: number[]) => [number[], number[], number[]];
  export type Kwargs = { shapeA?: number[], shapeB?: number[] };
  export type Wrapper = (shapeA: number[] | Kwargs, shapeB?: number[]) => [number[], number[], number[]];
  export const decorator = kwargs_decorator<Wrapper, Implementation, [number[], number[], number[]]>;
  export const defaults: [string, any][] = [
    ["shapeA", []],
    ["shapeB", []],
  ];
}

export namespace Func_broadcast_n_shapes {
  export type Implementation = (...shapes: number[][]) => [number[][], number[]];
  export type Kwargs = { shapes?: number[][] };
  export type Wrapper = (shape: number[] | Kwargs[], ...shapes: number[][]) => [number[][], number[]];
  export const decorator = kwargs_decorator<Wrapper, Implementation, [number[][], number[]]>;
  export const defaults: [string, any][] = [
    ["shapes", []],
  ];
}
