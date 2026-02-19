//@ts-check
import { AxisArg, kwargs_decorator, frequently_used_parsers, Arr } from "./kwargs";
import NDArray, { asarray } from "../NDArray";


type NDArray_non_0D = Arr | number[];

// Used in statistics.ts:
export namespace Func_a_q_axis {
  export type Implementation = (a: Arr, q: number, axis: number) => Arr;
  export type Kwargs = { a?: NDArray_non_0D, q?: number, axis?: AxisArg };
  export type Wrapper = (a: NDArray_non_0D | Kwargs, q: number | Kwargs, axis?: AxisArg | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["q", undefined], ["axis", null]];
  export const parsers = [
    frequently_used_parsers.a_axis_flatten,
    (kwargs: any) => { kwargs.q = asarray(kwargs.q); },
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  })
}
