import { NDArray } from "../NDArray-class";
/** @typedef {null | number} AxisArg */
export type AxisArg = null | number;
/**
 * @typedef {Object} ReduceKwargs
 * @property {AxisArg} [axis]
 * @property {boolean} [keepdims]
 */
export type ReduceKwargs = {
    axis?: AxisArg;
    keepdims?: boolean;
};
/**
 * @typedef {(axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray | T} ReduceSignature
 * @template [T=number]
 */
export type ReduceSignature<T = number> = (axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray | T;
/** @typedef {ReduceSignature<boolean>} ReduceSignatureBool */
export type ReduceSignatureBool = ReduceSignature<boolean>;
/** @typedef {[AxisArg, boolean]} ReduceParsedKwargs */
export type ReduceParsedKwargs = [AxisArg, boolean];
/**
 * @typedef {Object} ReduceStdKwargs
 * @property {number} [axis]
 * @property {boolean} [keepdims]
 * @property {number} [ddof]
 */
export type ReduceStdKwargs = {
    axis?: number;
    keepdims?: boolean;
    ddof?: number;
};
/** @typedef {(axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray | number} ReduceStdSignature */
export type ReduceStdSignature = (axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray | number;
/** @typedef {[AxisArg, boolean, number]} ReduceStdParsedKwargs */
export type ReduceStdParsedKwargs = [AxisArg, boolean, number];
/**
 * @typedef {Object} ReduceNormKwargs
 * @property {number} [axis]
 * @property {boolean} [keepdims]
 * @property {number} [ord]
 */
export type ReduceNormKwargs = {
    axis?: number;
    keepdims?: boolean;
    ord?: number;
};
/** @typedef {(axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray | number} ReduceNormSignature */
export type ReduceNormSignature = (axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray | number;
/** @typedef {[AxisArg, boolean, number]} ReduceNormParsedKwargs */
export type ReduceNormParsedKwargs = [AxisArg, boolean, number];
/**
 * @typedef {Object} RoundKwargs
 * @property {number} [decimals]
 */
export type RoundKwargs = {
    decimals?: number;
};
/** @typedef {(decimals?: number) => NDArray} RoundSignature */
export type RoundSignature = (decimals?: number) => NDArray;
/** @typedef {[number]} RoundParsedKwargs */
export type RoundParsedKwargs = [number];
/**
 * @typedef {(other: NDArray | T, out?: NDArray | null) => NDArray} BinaryOperatorSignature
 * @template [T=number]
 */
export type BinaryOperatorSignature<T = number> = (other: NDArray | T, out?: NDArray | null) => NDArray;
/**
 * @typedef {Object} BinaryOperatorKwargs
 * @property {NDArray} other
 * @property {NDArray | null} [out]
 */
export type BinaryOperatorKwargs = {
    other: NDArray;
    out?: NDArray | null;
};
/** @typedef {[NDArray, NDArray | null]} BinaryOperatorParsedKwargs */
export type BinaryOperatorParsedKwargs = [NDArray, NDArray | null];
/** @typedef {(out?: NDArray | null) => NDArray} UnaryOperatorSignature */
export type UnaryOperatorSignature = (out?: NDArray | null) => NDArray;
/**
 * @typedef {Object} UnaryOperatorKwargs
 * @property {NDArray | null} [out]
 */
export type UnaryOperatorKwargs = {
    out?: NDArray | null;
};
/** @typedef {[NDArray | null]} UnaryOperatorParsedKwargs */
export type UnaryOperatorParsedKwargs = [NDArray | null];
/** @typedef {[string, any] | [string, any, (x: any) => any]} KwTuple */
type KwTuple = [string, any] | [string, any, (x: any) => any];
/**
 * @param {{ defaults: KwTuple[], func: (arr: NDArray, ...args: Parsed) => any }}
 * @returns {{ as_function: (arr: number | boolean | NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>; as_method: (...args: Parameters<Signature>) => ReturnType<Signature>; }}
 */
export declare function kwDecorators<Signature extends (...args: any[]) => any, Parsed extends any[]>({ defaults, func }: {
    defaults: KwTuple[];
    func: (arr: NDArray, ...args: Parsed) => any;
}): {
    as_function: (arr: number | boolean | NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;
    as_method: (...args: Parameters<Signature>) => ReturnType<Signature>;
};
/**
 * @param {{ defaults: KwTuple[], func: (...args: Parsed) => any }}
 * @returns {(...args: Parameters<Signature>) => ReturnType<Signature>}
 */
export declare function kwDecorator<Signature extends (...args: any[]) => any, Parsed extends any[]>({ defaults, func }: {
    defaults: KwTuple[];
    func: (...args: Parsed) => any;
}): (...args: Parameters<Signature>) => ReturnType<Signature>;
export declare class KwParser<Signature extends (...args: any[]) => any, Parsed extends any[]> {
    defaults: KwTuple[];
    constructor(defaults: KwTuple[]);
    /**
       * @param {...Parameters<Signature>} [args]
       * @returns {Parsed}
       */
    parse(...args: Parameters<Signature>): Parsed;
    /**
       * @param {F} func
       * @returns {(arr: NDArray | number | boolean, ...args: Parameters<Signature>) => ReturnType<Signature>}
       */
    as_arr_function<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): (arr: NDArray | number | boolean, ...args: Parameters<Signature>) => ReturnType<Signature>;
    /**
       * @param {F} func
       * @returns {(...args: Parameters<Signature>) => ReturnType<Signature>}
       */
    as_arr_method<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): (...args: Parameters<Signature>) => ReturnType<Signature>;
    /**
       * @param {F} func
       * @returns {{ as_function: (arr: number | boolean | NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>; as_method: (...args: Parameters<Signature>) => ReturnType<Signature>; }}
       */
    decorators<F extends (arr: NDArray, ...args: Parsed) => ReturnType<Signature>>(func: F): {
        as_function: (arr: number | boolean | NDArray, ...args: Parameters<Signature>) => ReturnType<Signature>;
        as_method: (...args: Parameters<Signature>) => ReturnType<Signature>;
    };
    /**
       * @param {F} func
       * @returns {(...args: Parameters<Signature>) => ReturnType<Signature>}
       */
    as_function<F extends (...args: Parsed) => ReturnType<Signature>>(func: F): (...args: Parameters<Signature>) => ReturnType<Signature>;
}
export {};
/** @typedef {null | number} AxisArg */
/**
 * @typedef {Object} ReduceKwargs
 * @property {AxisArg} [axis]
 * @property {boolean} [keepdims]
 */
/**
 * @typedef {(axis?: AxisArg | ReduceKwargs, keepdims?: boolean | ReduceKwargs) => NDArray | T} ReduceSignature
 * @template [T=number]
 */
/** @typedef {ReduceSignature<boolean>} ReduceSignatureBool */
/** @typedef {[AxisArg, boolean]} ReduceParsedKwargs */
/**
 * @typedef {Object} ReduceStdKwargs
 * @property {number} [axis]
 * @property {boolean} [keepdims]
 * @property {number} [ddof]
 */
/** @typedef {(axis?: AxisArg | ReduceStdKwargs, keepdims?: boolean | ReduceStdKwargs, ddof?: number | ReduceStdKwargs) => NDArray | number} ReduceStdSignature */
/** @typedef {[AxisArg, boolean, number]} ReduceStdParsedKwargs */
/**
 * @typedef {Object} ReduceNormKwargs
 * @property {number} [axis]
 * @property {boolean} [keepdims]
 * @property {number} [ord]
 */
/** @typedef {(axis?: AxisArg | ReduceNormKwargs, keepdims?: boolean | ReduceNormKwargs, ord?: number | ReduceNormKwargs) => NDArray | number} ReduceNormSignature */
/** @typedef {[AxisArg, boolean, number]} ReduceNormParsedKwargs */
/**
 * @typedef {Object} RoundKwargs
 * @property {number} [decimals]
 */
/** @typedef {(decimals?: number) => NDArray} RoundSignature */
/** @typedef {[number]} RoundParsedKwargs */
/**
 * @typedef {(other: NDArray | T, out?: NDArray | null) => NDArray} BinaryOperatorSignature
 * @template [T=number]
 */
/**
 * @typedef {Object} BinaryOperatorKwargs
 * @property {NDArray} other
 * @property {NDArray | null} [out]
 */
/** @typedef {[NDArray, NDArray | null]} BinaryOperatorParsedKwargs */
/** @typedef {(out?: NDArray | null) => NDArray} UnaryOperatorSignature */
/**
 * @typedef {Object} UnaryOperatorKwargs
 * @property {NDArray | null} [out]
 */
/** @typedef {[NDArray | null]} UnaryOperatorParsedKwargs */
/** @typedef {[string, any] | [string, any, (x: any) => any]} KwTuple */
//# sourceMappingURL=kwargs.d.ts.map