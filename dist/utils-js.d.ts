/**
 * @param {any} A
 * @param {any} B
 * @param {any} func
 * @returns {boolean}
 */
export declare function binary_operation(A: any, B: any, func: any): boolean;
/**
 * @param {any} A
 * @returns {any[]}
 */
export declare function ravel(A: any): any[];
/**
 * @param {any} A
 * @param {any} B
 * @param {boolean} [nan_equal=false]
 * @returns {boolean}
 */
export declare function allEq(A: any, B: any, nan_equal?: boolean): boolean;
/**
 * @param {any} A
 * @param {any} B
 * @param {number} [rtol=1.e-5]
 * @param {number} [atol=1.e-8]
 * @param {boolean} [nan_equal=false]
 * @returns {boolean}
 */
export declare function allClose(A: any, B: any, rtol?: number, atol?: number, nan_equal?: boolean): boolean;
//# sourceMappingURL=utils-js.d.ts.map