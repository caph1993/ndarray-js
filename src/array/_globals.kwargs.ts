//@ts-check
import { Arr, ArrOrAny, ArrOrNull, kwargs_decorator, frequently_used_parsers, ArrOrConst } from "../array/kwargs";
import * as _globals from "./_globals";
import type NDArray from "../NDArray";
import type { DType, Buffer } from "../dtypes";

export namespace Func_isarray {
  export type Implementation = (A: any) => A is NDArray;
  export type Kwargs = { A?: any };
  export type Wrapper = (A: any | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["A", undefined],
  ];
  export const parsers = [
  ];
}

export namespace Func_new_NDArray {
  export type Implementation = (flat: Buffer, shape: number[], dtype?: DType) => NDArray;
  export type Kwargs = { flat?: Buffer, shape?: number[], dtype?: DType };
  export type Wrapper = (flat: Buffer | Kwargs, shape?: number[], dtype?: DType) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["flat", undefined],
    ["shape", []],
    ["dtype", null],
  ];
}

export namespace Func_asarray {
  export type Implementation = (A: NDArray | any, dtype: DType) => NDArray;
  export type Kwargs = { A?: NDArray | any, dtype?: DType };
  export type Wrapper = (A: NDArray | any | Kwargs, dtype?: DType) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["A", undefined],
    ["dtype", null],
  ];
}

export namespace Func_array {
  export type Implementation = (A: NDArray | any, dtype: DType) => NDArray;
  export type Kwargs = { A?: NDArray | any, dtype?: DType };
  export type Wrapper = (A: NDArray | any | Kwargs, dtype?: DType) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["A", undefined],
    ["dtype", null],
  ];
}

export namespace Func_broadcast_shapes {
  export type Implementation = (shapeA: number[], shapeB: number[]) => [number[], number[], number[]];
  export type Kwargs = { shapeA?: number[], shapeB?: number[] };
  export type Wrapper = (shapeA: number[] | Kwargs, shapeB?: number[]) => [number[], number[], number[]];
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["shapeA", []],
    ["shapeB", []],
  ];
}

export namespace Func_broadcast_n_shapes {
  export type Implementation = (...shapes: number[][]) => [number[][], number[]];
  export type Kwargs = { shapes?: number[][] };
  export type Wrapper = (...shapes: number[][] | Kwargs) => [number[][], number[]];
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [
    ["shapes", []],
  ];
}