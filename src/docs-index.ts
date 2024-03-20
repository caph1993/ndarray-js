/**
 * @module ndarray-js
 * @description
 * # Welcome to the documentation of ndarray-js
 * ndarray-js is a reimplementation of numpy for javascript that aims to make the coding experience as similar to numpy as possible.
 * ## API Index
 *  - Global namespace: {@link np}
 *  - NDArray class: np.{@link NDArray}.
 * [[include: index.md]]
 * 
 * ## Browser usage
 * @example
 * ```html
 * <script src="https://cdn.jsdelivr.net/npm/ndarray-js@1.0.0/dist/index.js"></script>
 * <script>
 * const a = np.arange(18).reshape(3, 2, 3);
 * console.log(a.tolist());
 * </script>
 * ```
*/

//@ts-check
import { np } from './index';
import NDArray from './NDArray';

export { np, NDArray };
