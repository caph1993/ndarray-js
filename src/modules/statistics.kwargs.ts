//@ts-check
import { AxisArg, kwargs_decorator, frequently_used_parsers, Func_a_axis_keepdims, ArrOrConst, Arr } from "../array/kwargs";
import NDArray, { asarray } from "../NDArray";
import * as stats from "./statistics";


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

export const quantile = Func_a_q_axis.decorate(stats.quantile);
export const nanquantile = Func_a_q_axis.decorate(stats.nanquantile);
export const percentile = Func_a_q_axis.decorate(stats.percentile);
export const nanpercentile = Func_a_q_axis.decorate(stats.nanpercentile);
export const median = Func_a_axis_keepdims.decorate(stats.median);
export const nanmedian = Func_a_axis_keepdims.decorate(stats.nanmedian);