/** @typedef {NumberConstructor|BooleanConstructor} DType */
/** @typedef {NDArray|number|boolean} ArrayOrConstant */
declare class NDArray {
    shape: number[];
    private _flat;
    dtype: any;
    transpose: (axes?: null) => any;
    /**
     * @param {number[]} flat actually number|boolean
     * @param {number[]} shape
     * @param {*} dtype
     */
    constructor(flat: number[], shape: number[], dtype?: any);
    /** @type {import("./core-indexes").AxesIndex|null} */ _simpleIndexes: any;
    get size(): any;
    get flat(): any;
    set flat(list: any);
    get T(): any;
    __popKwArgs(): any;
    /** @param {Object<string, any>} kwArgs */
    withKwArgs(kwArgs: any): this;
    get length(): number;
}
export default NDArray;
