import { NDArray, broadcast_n_shapes } from '../NDArray';
import { Arr, ArrOrAny, ArrOrConst } from "../array/js-interface";
import { isarray, asarray } from '../NDArray';
export type AxisArg = number | null;

export type ArrOrNull = NDArray | null;

import { empty } from "../NDArray";
import { Where } from '../array/indexes';
import { DType } from '../dtypes';

export { Arr, ArrOrAny, ArrOrConst };




// We need 2 types of wrapper on top of the implementation:
// The implementation expects NDArray on all arguments and returns NDArray
// The "library wrapper" expects NDArray | constant and returns NDArray
// The "user wrapper" returns NDArray | constant



export function kwargs_decorator<
  Wrapper extends (...args: any[]) => T,
  Implementation extends (...args: any[]) => T,
  T = NDArray
>({ defaults, implementation, parsers, this_as_first_arg }: {
  defaults: [string, any][],
  implementation: Implementation,
  parsers?: ((kwargs: any) => void)[],
  this_as_first_arg?: boolean,
}) {
  const kwParser = kwargs_parser<Wrapper, Implementation>(defaults, ...parsers);
  return (function (...args: Parameters<Wrapper>) {
    //@ts-ignore
    if (this_as_first_arg) args = [this, ...args];
    const parsedArgs = kwParser(args);
    const out = implementation(...parsedArgs);
    if (isarray(out) && out.shape.length == 0) return out.flat[0];
    return out;
  }) as Wrapper;
}

function kwargs_parser<
  Wrapper extends (...args: any[]) => any,
  Implementation extends (...args: any[]) => any,
>(
  defaults: [string, any][],
  ...parsers: ((kwargs: any) => void)[]
) {
  let spreadKey: string | null = null;
  const last = defaults.length && defaults.slice(-1)[0];
  if (last && last[0].startsWith('...')) {
    spreadKey = last[0] = last[0].slice(3);
  }
  const defaultsDict = Object.assign(Object.fromEntries(defaults));
  const validKwargsNames = Object.fromEntries(defaults.map(([key]) => [key, true]));
  const shorten = (s: any) => {
    s = JSON.stringify(s);
    // s = s.toString();
    return s.length > 100 ? s.slice(0, 100) + '...' : s;
  }
  return (args: Parameters<Wrapper>) => {
    let kwargs = {};
    if (spreadKey) kwargs[spreadKey] = [...defaultsDict[spreadKey]]; // copy
    for (let i = 0; i < args.length; i++) {
      let value = args[i];
      if (value instanceof Object && !isarray(value) && !Array.isArray(value)) {
        for (let key in value) {
          if (key == spreadKey) kwargs[key].push(value[key]);
          else {
            if (key in kwargs) throw new Error(`Duplicate argument ${key} in ${shorten(args)}`);
            if (!validKwargsNames[key]) throw new Error(`Invalid argument ${key} in ${shorten(args)}`);
            kwargs[key] = value[key];
          }
        }
      } else if (value !== undefined) {
        if (spreadKey && i >= defaults.length - 1) kwargs[spreadKey].push(value);
        else if (i < defaults.length) {
          const key = defaults[i][0];
          if (key in kwargs) throw new Error(`Duplicate argument ${key} in ${shorten(args)}`);
          if (!validKwargsNames[key]) throw new Error(`Invalid argument ${key} in ${shorten(args)}`);
          kwargs[key] = value;
        }
        else throw new Error(`Too many arguments ${shorten(args)}`);
      }
    }
    for (let key in defaultsDict) {
      if (!(key in kwargs)) kwargs[key] = defaultsDict[key];
      if (kwargs[key] === undefined) {
        throw new Error(`Missing argument ${key} in ${shorten(args)}. Found only ${JSON.stringify(Object.keys(args))}`);
      }
    }
    for (let parser of parsers) parser(kwargs);
    let sortedArgs = defaults.map(([key]) => kwargs[key]);
    return sortedArgs as Parameters<Implementation>;
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

// Used by sort:
export namespace Func_a_lastAxis {
  export type Implementation = (a: ArrOrAny, axis: number) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", -1]];
  export const parsers = [
    frequently_used_parsers['a_axis'],
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
    frequently_used_parsers.is_integer_non_neg('decimals'),
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
    frequently_used_parsers.is_integer_non_neg('decimals'),
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
  export type Wrapper = (a: ArrOrAny | Kwargs, other: ArrOrAny | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
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
  export const defaults: [string, any][] = [["a", undefined], ["values", undefined], ["...where", null]];
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
  export const defaults: [string, any][] = [["a", undefined], ["values", undefined], ["...where", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.asarray('values'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers, this_as_first_arg: true,
  });
}

// Used by reduce functions and methods:
export namespace Func_a_axis_keepdims {
  export type Implementation = (a: NDArray, axis: number, keepdims: boolean) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers
    });
  }
}

export namespace Method_a_axis_keepdims {
  export type Implementation = Func_a_axis_keepdims.Implementation;
  export type Kwargs = { axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.a_axis_flatten,
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}

// For norm:
export namespace Func_a_ord_axis_keepdims {
  export type Implementation = (a: NDArray, ord: number, axis: number, keepdims: boolean) => NDArray;
  export type Kwargs = { a?: ArrOrAny, ord?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (a: ArrOrAny | Kwargs, ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["ord", 2], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
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
      parsers: Func_a_ord_axis_keepdims.parsers,
      implementation,
      this_as_first_arg: true,
    });
  }
}
// For std:
export namespace Func_a_axis_ddof_keepdims {
  export type Implementation = (a: NDArray, axis: number, ddof: number, keepdims: boolean) => NDArray;
  export type Kwargs = { a?: ArrOrAny, axis?: AxisArg, ddof?: number, keepdims?: boolean };
  export type Wrapper = (a: ArrOrAny | Kwargs, axis?: AxisArg | Kwargs, ddof?: number | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["ddof", 0], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
    frequently_used_parsers.is_integer_non_neg('ddof'),
  ];
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers,
    });
  }
}
export namespace Method_a_axis_ddof_keepdims {
  export type Implementation = Func_a_axis_ddof_keepdims.Implementation;
  export type Kwargs = { ddof?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (ddof?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export function decorate(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults: Func_a_axis_ddof_keepdims.defaults,
      parsers: Func_a_axis_ddof_keepdims.parsers,
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
  export const defaults: [string, any][] = [["x", undefined], ["y", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('x'),
    frequently_used_parsers.asarray('y'),
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
