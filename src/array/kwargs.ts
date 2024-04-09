import { NDArray } from '../NDArray';
import { asarray, isarray } from "./_globals";
import { Where } from './indexes';

export type AxisArg = number | null;

export type ReduceKwargs = { axis?: AxisArg, keepdims?: boolean };
export type ReduceSignature<T = number> = (axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray;
export type ReduceSignatureBool = ReduceSignature<boolean>;
export type ReduceParsedKwargs = [AxisArg, boolean];

export type ReduceStdKwargs = { axis?: number, keepdims?: boolean, ddof?: number };
export type ReduceStdSignature = (axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray;
export type ReduceStdParsedKwargs = [AxisArg, boolean, number];

export type ReduceNormKwargs = { axis?: number, keepdims?: boolean, ord?: number };
export type ReduceNormSignature = (axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray;
export type ReduceNormParsedKwargs = [AxisArg, boolean, number];


export type RoundKwargs = { decimals?: number };
export type RoundSignature = (decimals?: number) => NDArray;
export type RoundParsedKwargs = [number];


export type BinaryOperatorMethod<T = number> = (other: NDArray | T, out?: NDArray | null) => NDArray;
export type BinaryOperatorKwargs = { other: NDArray, out?: NDArray | null };
export type BinaryOperatorParsedKwargs = [NDArray, NDArray | null];


export type UnaryOperatorMethod = (out?: NDArray | null) => NDArray;
export type UnaryOperatorKwargs = { out?: NDArray | null };
export type UnaryOperatorParsedKwargs = [NDArray | null];




export type AssignmentOperatorMethod<T = number> = (other: NDArray | T, ...where: Where) => NDArray;
export type AssignmentOperatorKwargs = { other: NDArray };
export type AssignmentOperatorParsedKwargs = [NDArray];


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
>({ defaults, implementation, parsers }: {
  defaults: [string, any][],
  implementation: Implementation,
  parsers?: ((kwargs: any) => void)[],
}) {
  const kwParser = kwargs_parser<Wrapper, Implementation>(defaults, ...parsers);
  return ((...args: Parameters<Wrapper>) => {
    const parsedArgs = kwParser(...args);
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
  return (...args: Parameters<Wrapper>) => {
    let kwargs = {};
    if (spreadKey) kwargs[spreadKey] = [...defaultsDict[spreadKey]]; // copy
    for (let i = 0; i < args.length; i++) {
      let value = args[i];
      if (value instanceof Object && !isarray(value)) {
        for (let key in value) {
          if (key == spreadKey) kwargs[key].push(value[key]);
          else {
            if (key in kwargs) throw new Error(`Duplicate argument ${key} in ${args}`);
            if (!validKwargsNames[key]) throw new Error(`Invalid argument ${key} in ${args}`);
            kwargs[key] = value[key];
          }
        }
      }
      else if (value !== undefined) {
        if (spreadKey && i >= defaults.length - 1) kwargs[spreadKey].push(value);
        else if (i < defaults.length) {
          const key = defaults[i][0];
          if (key in kwargs) throw new Error(`Duplicate argument ${key} in ${args}`);
          if (!validKwargsNames[key]) throw new Error(`Invalid argument ${key} in ${args}`);
          kwargs[key] = value;
        }
        else throw new Error(`Too many arguments ${args}`);
      }
    }
    for (let key in defaultsDict) {
      if (!(key in kwargs)) kwargs[key] = defaultsDict[key];
      if (kwargs[key] === undefined) {
        throw new Error(`Missing argument ${key} in ${args}. Found only ${JSON.stringify(Object.keys(args))}`);
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
  'out': (kwargs: any) => {
    if (kwargs.out != null && !isarray(kwargs)) {
      throw new Error(`Invalid out argument. Expected NDArray. Found type ${typeof kwargs.out} (${kwargs.out.prototype})`);
    }
  },
}

type NDArray_ = NDArray | number | boolean;

// Used in statistics.ts:
export namespace Func_a_q_axis {
  export type Implementation = (a: NDArray, q: number, axis: number) => NDArray;
  export type Kwargs = { a?: NDArray, q?: number, axis?: AxisArg };
  export type Wrapper = (a: NDArray | Kwargs, q: number | Kwargs, axis?: AxisArg | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["q", undefined], ["axis", null]];
  export const parsers = [
    frequently_used_parsers['a_axis_flatten'],
    (kwargs) => { kwargs.q = asarray(kwargs.q); },
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  })
}


export namespace Func_a_axis_keepdims {
  export type Implementation = (a: NDArray, axis: number, keepdims: boolean) => NDArray;
  export type Kwargs = { a?: NDArray, axis?: AxisArg, keepdims?: boolean };
  export type Wrapper = (a: NDArray | Kwargs, axis?: AxisArg | Kwargs, keepdims?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["keepdims", false], ["axis", null]];
  export const parsers = [
    frequently_used_parsers['a_axis_flatten'],
    (kwargs) => { kwargs.q = asarray(kwargs.q); },
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

// Used by sort:
export namespace Func_a_lastAxis {
  export type Implementation = (a: NDArray, axis: number) => NDArray;
  export type Kwargs = { a?: NDArray_, axis?: AxisArg };
  export type Wrapper = (a: NDArray_ | Kwargs, axis?: AxisArg | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["axis", -1]];
  export const parsers = [
    frequently_used_parsers['a_axis'],
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

//Used by elementwise operators:
export namespace Func_a_out {
  export type Implementation = (a: NDArray, out: NDArray | null) => NDArray;
  export type Kwargs = { a?: NDArray, out?: NDArray | null };
  export type Wrapper = (a: NDArray | Kwargs, out?: NDArray | null | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["out", null]];
  export const parsers = [
    frequently_used_parsers['a'],
    frequently_used_parsers['out'],
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}




// OLD VERSION:


export function kwDecorator<Signature extends (...args: any[]) => any, Parsed extends any[]>(
  { defaults, func }: { defaults: KwTuple[], func: (...args: Parsed) => any }) {
  return new KwParser<Signature, Parsed>(defaults).as_function(func);
}
export function kwDecorators<Signature extends (...args: any[]) => any, Parsed extends any[]>(
  { defaults, func }: { defaults: KwTuple[], func: (arr: NDArray, ...args: Parsed) => any }) {
  return new KwParser<Signature, Parsed>(defaults).decorators(func);
}


export class KwParser<Signature extends (...args: any[]) => any, Parsed extends any[]> {
  defaults: KwTuple[];
  spreadKey: string | null;

  constructor(defaults: KwTuple[]) {
    this.defaults = defaults;
    const last = defaults.length && defaults.slice(-1)[0];
    if (last && last[0].startsWith('...')) {
      this.spreadKey = last[0] = last[0].slice(3);
    }
  }

  parse(...args: Parameters<Signature>): Parsed {
    let defaults = this.defaults;
    let kwargs = Object.assign(Object.fromEntries(defaults));
    if (this.spreadKey) kwargs[this.spreadKey] = [...kwargs[this.spreadKey]]; // copy
    for (let i = 0; i < args.length; i++) {
      let value = args[i];
      if (value instanceof Object && !isarray(value)) Object.assign(kwargs, value);
      else if (value !== undefined) {
        if (this.spreadKey && i >= defaults.length - 1) kwargs[this.spreadKey].push(value);
        else if (i < defaults.length) kwargs[defaults[i][0]] = value;
        else throw new Error(`Too many arguments ${args}`);
      }
    }
    let sortedArgs = defaults.map((_, i) => {
      let key = defaults[i][0];
      let value = kwargs[key];
      if (value === undefined) {
        throw new Error(`Missing argument ${key} in ${args}`);
      }
      if (defaults[i].length === 3) {
        value = defaults[i][2](value, kwargs);
      }
      return value
    });
    return sortedArgs as Parsed;
  }

  as_arr_function<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): (arr: NDArray | number | boolean, ...args: Parameters<Signature>) => ReturnType<Signature> {
    let self = this;
    return function (arr: NDArray, ...args: Parameters<Signature>) {
      const parsed = self.parse(...args);
      return func(asarray(arr), ...parsed);
    };
  }

  as_arr_method<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): (...args: Parameters<Signature>) => ReturnType<Signature> {
    let self = this;
    return function (...args: Parameters<Signature>) {
      const parsed = self.parse(...args);
      return func(this, ...parsed);
    } as any;
  }

  decorators<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F) {
    return { as_function: this.as_arr_function(func), as_method: this.as_arr_method(func) };
  }

  as_function<F extends (...args: Parsed) => ReturnType<Signature>>(func: F): (...args: Parameters<Signature>) => ReturnType<Signature> {
    let self = this;
    return function (...args: Parameters<Signature>) {
      const parsed = self.parse(...args);
      return func(...parsed);
    };
  }

}
