import NDArray from '../NDArray-class';
/**
 * @param {any} shape
 * @param {import('../NDArray-class').DType} [dtype=Number]
 * @returns {NDArray}
 */
export declare function empty(shape: any, dtype?: import('../NDArray-class').DType): NDArray;
/**
 * @param {any} shape
 * @param {import('../NDArray-class').DType} [dtype=Number]
 * @returns {NDArray}
 */
export declare function zeros(shape: any, dtype?: import('../NDArray-class').DType): NDArray;
/**
 * @param {any} shape
 * @param {import('../NDArray-class').DType} [dtype=Number]
 * @returns {NDArray}
 */
export declare function ones(shape: any, dtype?: import('../NDArray-class').DType): NDArray;
/**
 * @param {any} arg0
 * @param {any} [arg1=null]
 * @returns {NDArray}
 */
export declare function arange(arg0: any, arg1?: any): NDArray;
/**
 * @param {any} start
 * @param {any} stop
 * @param {number} [num=50]
 * @param {boolean} [endpoint=true]
 * @returns {NDArray}
 */
export declare function linspace(start: any, stop: any, num?: number, endpoint?: boolean): NDArray;
/**
 * @param {any} start
 * @param {any} stop
 * @param {number} [num=50]
 * @param {boolean} [endpoint=true]
 * @returns {NDArray}
 */
export declare function geomspace(start: any, stop: any, num?: number, endpoint?: boolean): NDArray;
//# sourceMappingURL=constructors.d.ts.map