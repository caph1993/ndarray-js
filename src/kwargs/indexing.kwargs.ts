//@ts-check
import { kwargs_decorator, frequently_used_parsers, ArrOrAny, Arr } from "./kwargs";
import NDArray, { asarray } from "../NDArray";


// Used in indexing.ts:
export namespace Func_a_indices_axis {
  export type Implementation = (a: Arr, indices: Arr, axis: number) => NDArray;
  export type Kwargs = { a?: ArrOrAny, indices?: ArrOrAny, axis?: number };
  export type Wrapper = (a: ArrOrAny | Kwargs, indices: ArrOrAny | Kwargs, axis: number | Kwargs) => NDArray;
  export const decorator = kwargs_decorator<Wrapper, Implementation>;
  export const defaults: [string, any][] = [["a", undefined], ["indices", undefined], ["axis", undefined]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
    frequently_used_parsers.asarray('indices'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

export namespace Func_condition_x_y_out {
  export type ReturnType = NDArray | number | boolean | any[];
  export type Implementation = (condition: ArrOrAny, x: ArrOrAny, y: ArrOrAny, out: NDArray | null) => ReturnType;
  export type Kwargs = { condition?: ArrOrAny, x?: ArrOrAny, y?: ArrOrAny, out?: ArrOrAny | null };
  export type Wrapper = (condition: ArrOrAny | Kwargs, x: ArrOrAny | Kwargs, y: ArrOrAny | Kwargs, out?: ArrOrAny | null | Kwargs) => ReturnType;
  export const decorator = kwargs_decorator<Wrapper, Implementation, ReturnType>;
  export const defaults: [string, any][] = [["condition", undefined], ["x", undefined], ["y", undefined], ["out", null]];
  export const parsers = [
    (kwargs: any) => {
      if (typeof kwargs.condition !== 'boolean') {
        kwargs.condition = asarray(kwargs.condition);
        kwargs.x = asarray(kwargs.x);
        kwargs.y = asarray(kwargs.y);
        if (kwargs.out !== null) kwargs.out = asarray(kwargs.out);
      }
    }
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}

export namespace Func_a {
  export type Implementation = (a: Arr) => NDArray[];
  export type Kwargs = { a?: ArrOrAny };
  export type Wrapper = (a: ArrOrAny | Kwargs) => NDArray[];
  export const decorator = kwargs_decorator<Wrapper, Implementation, NDArray[]>;
  export const defaults: [string, any][] = [["a", undefined]];
  export const parsers = [
    frequently_used_parsers.asarray('a'),
  ];
  export const decorate = (implementation: Implementation) => decorator({
    defaults, implementation, parsers
  });
}
