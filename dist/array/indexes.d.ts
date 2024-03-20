import type NDArray from "../NDArray";
export type RangeSpec = string;
export type indexSpec = ':' | number | RangeSpec | NDArray | number[];
export type GeneralIndexSpec = ':' | '...' | 'None' | null | indexSpec;
export type Where = null | GeneralIndexSpec[];
export declare function index(arr: NDArray, where: Where): NDArray;
export declare class AxesIndex {
    shape: any;
    internalShape: any;
    axisIndexes: AxisIndex[];
    private _indices;
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
export declare function __parse_sliceRange(axis_size: any, { start, stop, step }: {
    start: any;
    stop: any;
    step: any;
}): any[];
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
//# sourceMappingURL=indexes.d.ts.map