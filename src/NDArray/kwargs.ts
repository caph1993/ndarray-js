import { NDArray } from "../NDArray-class";

export type AxisArg = null | number;

export type ReduceKwargs = { axis?: number, keepdims?: boolean };
export type ReduceSignature<T = number> = (axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray | T;
export type ReduceSignatureBool = ReduceSignature<boolean>;
export type ReduceParsedKwargs = [number, boolean];

export type ReduceStdKwargs = { axis?: number, keepdims?: boolean, ddof?: number };
export type ReduceStdSignature = (axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray | number;
export type ReduceStdParsedKwargs = [number, boolean, number];

export type ReduceNormKwargs = { axis?: number, keepdims?: boolean, ord?: number };
export type ReduceNormSignature = (axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray | number;
export type ReduceNormParsedKwargs = [number, boolean, number];


export type RoundKwargs = { decimals?: number };
export type RoundSignature = (decimals?: number) => NDArray;
export type RoundParsedKwargs = [number];

type WithArray<Signature extends (...args: any[]) => any> = (arr: NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;

export class KwParser<Signature extends (...args: any[]) => any, Parsed extends any[]> {
  defaults: [string, any][];

  constructor(defaults: [string, any][]) {
    this.defaults = defaults;
  }

  parse(...args: Parameters<Signature>): Parsed {
    let defaults = this.defaults;
    let kwargs = Object.assign(Object.fromEntries(defaults));
    for (let i = 0; i < args.length; i++) {
      let value = args[i];
      if (value instanceof Object) Object.assign(kwargs, value);
      else if (value !== undefined) kwargs[defaults[i][0]] = value;
    }
    let sortedArgs = defaults.map(([key, _]) => kwargs[key]);
    return sortedArgs as Parsed;
  }

  decorator_func<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): WithArray<Signature> {
    return function (arr: NDArray, ...args: Parameters<Signature>) {
      const parsed = this.parse(...args);
      return func(arr, ...parsed);
    };
  }

  decorator_method<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): Signature {
    let self = this;
    return function (...args: Parameters<Signature>) {
      const parsed = self.parse(...args);
      return func(this, ...parsed);
    } as any;
  }
  decorators<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F) {
    return { as_function: this.decorator_func(func), as_method: this.decorator_method(func) };
  }
}
