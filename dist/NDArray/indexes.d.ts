/** @typedef {import('../NDArray-class')} NDArray */
import type NDArray from "../NDArray-class";
/** @typedef {string} RangeSpec */
export type RangeSpec = string;
/** @typedef {':' | number | RangeSpec | NDArray | number[]} indexSpec */
export type indexSpec = ':' | number | RangeSpec | NDArray | number[];
/** @typedef {':' | '...' | 'None' | null | indexSpec} GeneralIndexSpec */
export type GeneralIndexSpec = ':' | '...' | 'None' | null | indexSpec;
/** @typedef {null | GeneralIndexSpec[]} Where */
export type Where = null | GeneralIndexSpec[];
/**
 * @param {NDArray} arr
 * @param {Where} where
 * @returns {number | NDArray}
 */
export declare function index(arr: NDArray, where: Where): number | NDArray;
export declare class AxesIndex {
    shape: any;
    internalShape: any;
    axisIndexes: AxisIndex[];
    /**
       * @private
       */
    private _indices;
    /**
       * @private
       */
    private _size;
    isSimple: boolean;
    isConstant: boolean;
    parse: (shape: number[], where: Where) => AxesIndex;
    /**
     * @param {AxisIndex[]} axisIndexes
     */
    constructor(apparentShape: any, internalShape: any, axisIndexes: AxisIndex[]);
    get indices(): number[];
    get __slices(): any[];
    get size(): number;
}
/**
 * @param {any} axis_size
 * @param {{ start: any; stop: any; step: any; }}
 * @returns {any[]}
 */
export declare function __parse_sliceRange(axis_size: any, { start, stop, step }: {
    start: any;
    stop: any;
    step: any;
}): any[];
/** @typedef {{ type: ':', size: number } | { type: 'number', index: number } | { type: 'range', range: { start: number, step: number, nSteps: number } } | { type: 'array', indices: number[] }} AxisIndexSpec */
export type AxisIndexSpec = {
    type: ':';
    size: number;
} | {
    type: 'number';
    index: number;
} | {
    type: 'range';
    range: {
        start: number;
        step: number;
        nSteps: number;
    };
} | {
    type: 'array';
    indices: number[];
};
export declare class AxisIndex {
    spec: AxisIndexSpec;
    /**
       * @private
       */
    private _indices;
    isSimple: boolean;
    isConstant: boolean;
    parse: (indexSpec: indexSpec | undefined, size: number) => {
        axisIndex: AxisIndex;
        span: number;
    };
    parse_range: (size: number, start?: number, stop?: number, step?: number) => {
        start: number;
        step: number;
        nSteps: number;
    };
    parse_range_spec: (rangeString: string) => {
        start: number;
        stop: number;
        step: number;
    };
    /**
     * Invariant: Immutable
     * @param {AxisIndexSpec} spec
     */
    constructor(spec: AxisIndexSpec);
    get indices(): any;
    get size(): number;
}
/** @typedef {string} RangeSpec */
/** @typedef {':' | number | RangeSpec | NDArray | number[]} indexSpec */
/** @typedef {':' | '...' | 'None' | null | indexSpec} GeneralIndexSpec */
/** @typedef {null | GeneralIndexSpec[]} Where */
/** @typedef {{ type: ':', size: number } | { type: 'number', index: number } | { type: 'range', range: { start: number, step: number, nSteps: number } } | { type: 'array', indices: number[] }} AxisIndexSpec */
//# sourceMappingURL=indexes.d.ts.map