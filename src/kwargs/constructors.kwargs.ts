import { isarray, asarray } from "../NDArray";
//@ts-check
import { kwargs_decorator, frequently_used_parsers as P, signature } from "./kwargs";
import type { Shape } from "../NDArray";
import NDArray from "../NDArray";
import { DType, float64, int32 } from "../dtypes";
import * as constructors from "../modules/constructors";


// Used in constructors.ts:
export namespace Func_shape_dtype {
  export type Implementation = (shape: Shape, dtype: DType) => NDArray;
  export type Kwargs = { shape?: Shape, dtype?: DType };
  export type Wrapper = (shape: Shape | Kwargs, dtype?: DType | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["shape", undefined], ["dtype", float64]];
  export const parsers = [P.dtype_or_null];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

/** arange(stop) | arange(start, stop) | arange(start, stop, step), with optional dtype (also accepted as 3rd positional argument) */
export const arange = signature<typeof constructors.arange>()(
  [["start", undefined], ["stop", null], ["step", 1], ["dtype", null]] as const,
  [(kwargs: any) => {
    // Python-like scalars: keep the weak float-ness of the first arguments for dtype inference

    // Backwards compatibility: arange(n, null, 'float64')
    if (typeof kwargs.step === "string") { kwargs.dtype = kwargs.step; kwargs.step = 1; }
    P.dtype_or_null(kwargs);
  }], { keep_weak: ["start", "stop", "step"] });

const num_or_array = (...keys: string[]) => (kwargs: any) => {
  for (const key of keys) {
    const v = kwargs[key];
    if (isarray(v) && v.shape.length === 0) kwargs[key] = v.item();
    else if (Array.isArray(v) || isarray(v)) kwargs[key] = asarray(v);
  }
};
export const linspace = signature<typeof constructors.linspace>()(
  [["start", undefined], ["stop", undefined], ["num", 50], ["endpoint", true], ["retstep", false], ["dtype", null], ["axis", 0]] as const,
  [num_or_array('start', 'stop'), P.dtype_or_null, (kwargs: any) => { kwargs.endpoint = !!kwargs.endpoint; kwargs.retstep = !!kwargs.retstep; }]);
export const geomspace = signature<typeof constructors.geomspace>()(
  [["start", undefined], ["stop", undefined], ["num", 50], ["endpoint", true], ["dtype", null], ["axis", 0]] as const,
  [num_or_array('start', 'stop'), P.dtype_or_null, (kwargs: any) => { kwargs.endpoint = !!kwargs.endpoint; }]);
