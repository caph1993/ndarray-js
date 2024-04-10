import { NDArray } from '../NDArray';
import { TypedArrayConstructor } from '../dtypes';
import { asarray, isarray } from "./_globals";
import { Where } from './indexes';

export type AxisArg = number | null;

// We need 2 types of wrapper on top of the implementation:
// The implementation expects NDArray on all arguments and returns NDArray
// The "library wrapper" expects NDArray | constant and returns NDArray
// The "user wrapper" returns NDArray | constant



// The last one can be [argumentName, defaultValue, transformFunction]
// The last one can be [`...${argumentName}`, defaultValue, transformFunction]
type KwTuple = [string, any] | [string, any, ((x: any) => any) | ((x: any, kwargs: any) => any)];


export function kwargs_decorator<
  Wrapper extends (...args: any[]) => NDArray_,
  Implementation extends (...args: any[]) => NDArray,
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


const frequently_used_parsers = {
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
  'asarray': (key: string) => (kwargs: any) => {
    kwargs[key] = asarray(kwargs[key]);
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
  }
}

type NDArray_non_0D = NDArray<any> | number[];

// Used in statistics.ts:
export namespace Func_a_q_axis {
  export type Implementation = (a: NDArray<any>, q: number, axis: number) => NDArray<any>;
  export type Kwargs = { a?: NDArray_non_0D, q?: number, axis?: AxisArg };
  export type Wrapper = (a: NDArray_non_0D | Kwargs, q: number | Kwargs, axis?: AxisArg | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["q", undefined], ["axis", null]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
    (kwargs) => { kwargs.q = asarray(kwargs.q); },
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  })
}

// Used by sort:
export namespace Func_a_lastAxis {
  export type Implementation = (a: NDArray_, axis: number) => NDArray<any>;
  export type Kwargs = { a?: NDArray_, axis?: AxisArg };
  export type Wrapper = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", -1]];
  export const parsers = [
    frequently_used_parsers['a_axis'],
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}


// Used by elementwise operators and methods:
export namespace Func_a_out {
  export type Implementation = (a: NDArray<any>, out: NDArray<any> | null) => NDArray<any>;
  export type Kwargs = { a?: NDArray_, out?: NDArray<any> | null };
  export type Wrapper = (a: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_out {
  export type Implementation = Func_a_out.Implementation;
  export type Kwargs = { out?: NDArray<any> | null };
  export type Wrapper<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  > = (out?: NDArray<any> | null | Kwargs) => NDArray<T>;
  export const defaults: [string, any][] = [["a", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function defaultDecorator<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  >(implementation: Implementation) {
    return kwargs_decorator<Wrapper<T>, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}
// Used by round
export namespace Func_a_decimals_out {
  export type Implementation = (a: NDArray<any>, decimals: number, out: NDArray<any> | null) => NDArray<any>;
  export type Kwargs = { a?: NDArray_, decimals?: number, out?: NDArray<any> | null };
  export type Wrapper = (a: NDArray_ | Kwargs, decimals: number | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["decimals", 0], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.is_integer_non_neg('decimals'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_a_decimals_out {
  export type Implementation = Func_a_decimals_out.Implementation;
  export type Kwargs = { decimals?: number, out?: NDArray<any> | null };
  export type Wrapper = (decimals: number | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["decimals", 0], ["out", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.is_integer_non_neg('decimals'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers, this_as_first_arg: true,
  });
}


type NDArray_ = NDArray<any> | number | boolean;

// Used by binary operators and methods:
export namespace Func_a_other_out {
  export type Implementation = (a: NDArray<any>, other: NDArray<any>, out: NDArray<any> | null) => NDArray<any>;
  export type Kwargs = { a: NDArray_, other?: NDArray_, out?: NDArray<any> | null };
  export type Wrapper = (a: NDArray_ | Kwargs, other: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["other", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray('other'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_other_out {
  export type Implementation = Func_a_other_out.Implementation;
  export type Kwargs = { other?: NDArray_, out?: NDArray<any> | null };
  export type Wrapper<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  > = (other: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<T>;
  export const defaults: [string, any][] = [["a", undefined], ["other", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.asarray('other'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export function defaultDecorator<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  >(implementation: Implementation) {
    return kwargs_decorator<Wrapper<T>, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}

// Used by assign operators and methods:
export namespace Func_a_values_where {
  export type Implementation = (a: NDArray<any>, values: NDArray<any>, where: Where) => NDArray<any>;
  export type Kwargs = { a: NDArray<any>, values?: NDArray_, where: Where };
  export type Wrapper = (a: NDArray<any> | Kwargs, values: NDArray_ | Kwargs, ...where: Where) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["values", undefined], ["...where", null]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray('values'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
export namespace Method_values_where {
  export type Implementation = Func_a_values_where.Implementation;
  export type Kwargs = { values?: NDArray_, where: Where };
  export type Wrapper = (values: NDArray_ | Kwargs, ...where: Where) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["values", undefined], ["...where", null]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.asarray('values'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers, this_as_first_arg: true,
  });
}

// Used by reduce functions and methods:
export namespace Func_a_axis_keepdims {
  export type Implementation = (a: NDArray<any>, axis: number, keepdims: boolean) => NDArray<any>;
  export type Kwargs = { a?: NDArray_, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  > = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<T>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
  ];
  export function defaultDecorator<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  >(implementation: Implementation) {
    return kwargs_decorator<Wrapper<T>, Implementation>({
      defaults, implementation, parsers
    });
  }
}
export namespace Method_a_axis_keepdims {
  export type Implementation = Func_a_axis_keepdims.Implementation;
  export type Kwargs = { axis?: AxisArg, keepdims?: boolean };
  export type Wrapper<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  > = (axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<T>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.isarray('a'),
    frequently_used_parsers.a_axis_flatten,
  ];
  export function defaultDecorator<
    T extends TypedArrayConstructor = Float64ArrayConstructor,
  >(implementation: Implementation) {
    return kwargs_decorator<Wrapper<T>, Implementation>({
      defaults, implementation, parsers, this_as_first_arg: true,
    });
  }
}

// For norm:
export namespace Func_a_ord_axis_keepdims {
  export type Implementation = (a: NDArray<any>, ord: number, axis: number, keepdims: boolean) => NDArray<any>;
  export type Kwargs = { a?: NDArray_, ord?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (a: NDArray_ | Kwargs, ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["ord", 2], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
    frequently_used_parsers.is_pnorm('ord'),
  ];
  export function defaultDecorator(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers,
    });
  }
}
export namespace Method_a_ord_axis_keepdims {
  export type Kwargs = { ord?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (ord?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
  export type Implementation = Func_a_ord_axis_keepdims.Implementation;
  export function defaultDecorator(implementation: Implementation) {
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
  export type Implementation = (a: NDArray<any>, axis: number, ddof: number, keepdims: boolean) => NDArray<any>;
  export type Kwargs = { a?: NDArray_, axis?: AxisArg, ddof?: number, keepdims?: boolean };
  export type Wrapper = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs, ddof?: number | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["ddof", 0], ["axis", null], ["keepdims", false]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
    frequently_used_parsers.is_integer_non_neg('ddof'),
  ];
  export function defaultDecorator(implementation: Implementation) {
    return kwargs_decorator<Wrapper, Implementation>({
      defaults, implementation, parsers,
    });
  }
}
export namespace Method_a_axis_ddof_keepdims {
  export type Implementation = Func_a_axis_ddof_keepdims.Implementation;
  export type Kwargs = { ddof?: number, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (ddof?: number | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray<any>;
  export function defaultDecorator(implementation: Implementation) {
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
  export type Implementation = (y: NDArray_, x: NDArray_, out?: NDArray | null) => NDArray<any>;
  export type Kwargs = { y?: NDArray_, x?: NDArray_, out?: NDArray<any> | null };
  export type Wrapper = (y: NDArray_ | Kwargs, x: NDArray_ | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["x", undefined], ["y", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers.asarray('x'),
    frequently_used_parsers.asarray('y'),
    frequently_used_parsers.isarray_or_null('out'),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}