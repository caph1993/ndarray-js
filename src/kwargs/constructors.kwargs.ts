//@ts-check
import { kwargs_decorator } from "./kwargs";
import type { Shape } from "../NDArray";
import NDArray from "../NDArray";
import { DType, float64, int32 } from "../dtypes";


// Used in constructors.ts:
export namespace Func_shape_dtype {
  export type Implementation = (shape: Shape, dtype: DType) => NDArray;
  export type Kwargs = { shape?: Shape, dtype?: DType };
  export type Wrapper = (shape: Shape | Kwargs, dtype?: DType | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["shape", undefined], ["dtype", float64]];
  export const parsers = [];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

export namespace Func_arg0_arg1_dtype {
  export type Implementation = (arg0: number, arg1: number | null, dtype: DType) => NDArray;
  export type Kwargs = { arg0?: number, arg1?: number | null, dtype?: DType };
  export type Wrapper = (arg0: number | Kwargs, arg1?: number | null | Kwargs, dtype?: DType | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["arg0", undefined], ["arg1", null], ["dtype", int32]];
  export const parsers = [];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

export namespace Func_start_stop_steps_endpoint {
  export type Implementation = (start: number, stop: number, steps: number, endpoint: boolean) => NDArray;
  export type Kwargs = { start?: number, stop?: number, steps?: number, endpoint?: boolean };
  export type Wrapper = (start: number | Kwargs, stop: number | Kwargs, steps?: number | Kwargs, endpoint?: boolean | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["start", undefined], ["stop", undefined], ["steps", 50], ["endpoint", true]];
  export const parsers = [];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
