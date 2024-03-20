import { NDArray } from '../NDArray';
import { asarray, isarray } from "./_globals";

export type AxisArg = null | number;

export type ReduceKwargs = { axis?: AxisArg, keepdims?: boolean };
export type ReduceSignature<T = number> = (axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray | T;
export type ReduceSignatureBool = ReduceSignature<boolean>;
export type ReduceParsedKwargs = [AxisArg, boolean];

export type ReduceStdKwargs = { axis?: number, keepdims?: boolean, ddof?: number };
export type ReduceStdSignature = (axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray | number;
export type ReduceStdParsedKwargs = [AxisArg, boolean, number];

export type ReduceNormKwargs = { axis?: number, keepdims?: boolean, ord?: number };
export type ReduceNormSignature = (axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray | number;
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


type KwTuple = [string, any] | [string, any, (x: any) => any];

export function kwDecorators<Signature extends (...args: any[]) => any, Parsed extends any[]>(
  { defaults, func }: { defaults: KwTuple[], func: (arr: NDArray, ...args: Parsed) => any }) {
  return new KwParser<Signature, Parsed>(defaults).decorators(func);
}

export function kwDecorator<Signature extends (...args: any[]) => any, Parsed extends any[]>(
  { defaults, func }: { defaults: KwTuple[], func: (...args: Parsed) => any }) {
  return new KwParser<Signature, Parsed>(defaults).as_function(func);
}

export class KwParser<Signature extends (...args: any[]) => any, Parsed extends any[]> {
  defaults: KwTuple[];

  constructor(defaults: KwTuple[]) {
    this.defaults = defaults;
  }

  parse(...args: Parameters<Signature>): Parsed {
    let defaults = this.defaults;
    let kwargs = Object.assign(Object.fromEntries(defaults));
    for (let i = 0; i < args.length; i++) {
      let value = args[i];
      if (value instanceof Object && !isarray(value)) Object.assign(kwargs, value);
      else if (value !== undefined) kwargs[defaults[i][0]] = value;
    }
    let sortedArgs = defaults.map((_, i) => {
      let key = defaults[i][0];
      let value = kwargs[key];
      if (value === undefined) {
        throw new Error(`Missing argument ${key} in ${args}`);
      }
      if (defaults[i].length === 3) {
        value = defaults[i][2](value);
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
