//@ts-check
import { Arr, AxisArg, kwargs_decorator, frequently_used_parsers, asarray, ArrOrConst, Func_a_axis_keepdims } from "../array/kwargs";
import * as stats from "./statistics";


type NDArray_non_0D = Arr | number[];

// Used in statistics.ts:
export namespace Func_a_q_axis {
  export type Implementation = (a: Arr, q: number, axis: number) => Arr;
  export type Kwargs = { a?: NDArray_non_0D, q?: number, axis?: AxisArg };
  export type Wrapper = (a: NDArray_non_0D | Kwargs, q: number | Kwargs, axis?: AxisArg | Kwargs) => ArrOrConst;
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

export const quantile = Func_a_q_axis.defaultDecorator(stats.quantile);
export const nanquantile = Func_a_q_axis.defaultDecorator(stats.nanquantile);
export const percentile = Func_a_q_axis.defaultDecorator(stats.percentile);
export const nanpercentile = Func_a_q_axis.defaultDecorator(stats.nanpercentile);
export const median = Func_a_axis_keepdims.defaultDecorator(stats.median);
export const nanmedian = Func_a_axis_keepdims.defaultDecorator(stats.nanmedian);