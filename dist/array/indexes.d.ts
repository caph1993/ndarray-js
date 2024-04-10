import type NDArray from "../NDArray";
export type RangeSpec = string;
export type IndexSpec = ':' | number | RangeSpec | number[] | NDArray<Float64ArrayConstructor> | NDArray<Int32ArrayConstructor> | NDArray<Uint8ArrayConstructor>;
export type GeneralIndexSpec = '...' | 'None' | null | boolean | IndexSpec;
export type Where = null | GeneralIndexSpec[];
export declare function index(arr: NDArray, where: Where): NDArray<import("../dtypes").TypedArrayConstructor>;
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
    get __slices(): number[][];
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
} | {
    type: false;
};
export declare class AxisIndex {
    spec: AxisIndexSpec;
    private _indices;
    isSimple: boolean;
    isConstant: boolean;
    parse: (indexSpec: IndexSpec | undefined, size: number) => {
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
     */
    constructor(spec: AxisIndexSpec);
    get indices(): number[];
    get size(): number;
}
//# sourceMappingURL=indexes.d.ts.map