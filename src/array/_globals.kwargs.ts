//@ts-check
import NDArray from "../NDArray";
import { Arr, ArrOrAny, kwargs_decorator, frequently_used_parsers } from "../array/kwargs";


export namespace Func_clip {
  export type Implementation = (a: Arr, a_min: Arr, a_max: Arr, out: Arr) => Arr;
  export type Kwargs = { a?: ArrOrAny, a_min?: ArrOrAny, a_max?: ArrOrAny, out?: NDArray<any> | null };
  export type Wrapper = (a: ArrOrAny, a_min: ArrOrAny, a_max: ArrOrAny, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["a", undefined],
    ["a_min", undefined],
    ["a_max", undefined],
    ["out", null],
  ];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray('a_min'),
    frequently_used_parsers.asarray('a_max'),
    frequently_used_parsers.out_broadcast(['a', 'a_min', 'a_max'], 'out', true),
  ];
  export const defaultDecorator = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}