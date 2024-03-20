import { NDArray } from '../NDArray';
export type AxisArg = null | number;
export type ReduceKwargs = {
    axis?: AxisArg;
    keepdims?: boolean;
};
export type ReduceSignature<T = number> = (axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray | T;
export type ReduceSignatureBool = ReduceSignature<boolean>;
export type ReduceParsedKwargs = [AxisArg, boolean];
export type ReduceStdKwargs = {
    axis?: number;
    keepdims?: boolean;
    ddof?: number;
};
export type ReduceStdSignature = (axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray | number;
export type ReduceStdParsedKwargs = [AxisArg, boolean, number];
export type ReduceNormKwargs = {
    axis?: number;
    keepdims?: boolean;
    ord?: number;
};
export type ReduceNormSignature = (axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray | number;
export type ReduceNormParsedKwargs = [AxisArg, boolean, number];
export type RoundKwargs = {
    decimals?: number;
};
export type RoundSignature = (decimals?: number) => NDArray;
export type RoundParsedKwargs = [number];
export type BinaryOperatorMethod<T = number> = (other: NDArray | T, out?: NDArray | null) => NDArray;
export type BinaryOperatorKwargs = {
    other: NDArray;
    out?: NDArray | null;
};
export type BinaryOperatorParsedKwargs = [NDArray, NDArray | null];
export type UnaryOperatorMethod = (out?: NDArray | null) => NDArray;
export type UnaryOperatorKwargs = {
    out?: NDArray | null;
};
export type UnaryOperatorParsedKwargs = [NDArray | null];
type KwTuple = [string, any] | [string, any, (x: any) => any];
export declare function kwDecorators<Signature extends (...args: any[]) => any, Parsed extends any[]>({ defaults, func }: {
    defaults: KwTuple[];
    func: (arr: NDArray, ...args: Parsed) => any;
}): {
    as_function: (arr: number | boolean | NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;
    as_method: (...args: Parameters<Signature>) => ReturnType<Signature>;
};
export declare function kwDecorator<Signature extends (...args: any[]) => any, Parsed extends any[]>({ defaults, func }: {
    defaults: KwTuple[];
    func: (...args: Parsed) => any;
}): (...args: Parameters<Signature>) => ReturnType<Signature>;
export declare class KwParser<Signature extends (...args: any[]) => any, Parsed extends any[]> {
    defaults: KwTuple[];
    constructor(defaults: KwTuple[]);
    parse(...args: Parameters<Signature>): Parsed;
    as_arr_function<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): (arr: NDArray | number | boolean, ...args: Parameters<Signature>) => ReturnType<Signature>;
    as_arr_method<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): (...args: Parameters<Signature>) => ReturnType<Signature>;
    decorators<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): {
        as_function: (arr: number | boolean | NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;
        as_method: (...args: Parameters<Signature>) => ReturnType<Signature>;
    };
    as_function<F extends (...args: Parsed) => ReturnType<Signature>>(func: F): (...args: Parameters<Signature>) => ReturnType<Signature>;
}
export {};
//# sourceMappingURL=kwargs.d.ts.map