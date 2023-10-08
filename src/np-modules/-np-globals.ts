//@ts-check
import { GLOBALS } from '../globals';
const { np, NDArray: _NDArray } = GLOBALS;
if (!np) throw new Error(`Programming error: np not defined`);

export { np };
export const nd_modules = _NDArray.prototype.modules;