/**
 * @module ndarray-js
 * @description
 * <style>h1 a code{ font-size: inherit; }</style>
 * # Welcome to the documentation of `ndarray-js`
 * ndarray-js is a reimplementation of numpy for javascript that aims to make the coding experience as similar to numpy as possible.
 * ## Interactive demo:</h3>
 * [[include: docs-index.md]]
 * 
 * ## API Index
 *  - Global namespace: {@link np}
 *  - NDArray class: np.{@link NDArray}.
 * 
 * ## Browser usage
 * In html:
 * ```html
 * <script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>
 * <script>
 * a = np.arange(18).reshape(3, 2, 3);
 * console.log(a.tolist());
 * </script>
 * ```
 * Or press F12 and play with the browser console directly.
 * 
 * ## Node usage
 * Installation:
 * ```bash
 * npm install ndarray-js
 * ```
 * Use in nodejs with require:
 * ```javascript
 * const { np } = require('ndarray-js');
 * a = np.arange(18).reshape(3, 2, 3);
 * console.log(a.tolist());
 * ```
 * 
 * Use in nodejs with typescript (run with ts-node or compile with tsc):
 * ```typescript
 * import { np } from 'ndarray-js';
 * const a = np.arange(18).reshape(3, 2, 3);
 * console.log(a.tolist());
 * ```
 * 
 * ## Use in Jupyter notebooks:
 * You need to install the ipykernel (the setup is a bit more complex).
 * 
 * Example notebook: https://github.com/caph1993/numpy-js/blob/main/notebooks/normal-scatter.ipynb 
 * 
 * [[include: clean-index.md]]
*/

//@ts-check
import { np } from './index';
import NDArray from './NDArray';

export { np, NDArray };

/** @namespace */
export const modules = np.modules;

