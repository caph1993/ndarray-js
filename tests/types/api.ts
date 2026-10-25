// Type-level checks of the public API. Compiled (not run) with `npx tsc -p tests/types/tsconfig.json`.
import { np } from "../../src/index";
import type { NDArray } from "../../src/NDArray";

const x: NDArray = np.arange(12).reshape(3, 4);

// kwargs friendly: positional, options object, and mixed
const a1: NDArray = x.mean(-1, true);
const a2: NDArray = x.mean({ axis: -1, keepdims: true });
const a3: NDArray = x.mean(-1, { keepdims: true });
const a4: NDArray = np.mean(x, { axis: -1 });
const a5: NDArray = np.sum(x, 0);
const a6: NDArray = np.cumsum(x, { axis: 1 });
const a7: NDArray = np.sort(x, { axis: 0 });
const a8: NDArray = np.concatenate([x, x], 1);
const a9: NDArray = np.reshape(x, [4, 3]);
const a10: NDArray = np.where(np.greater(x, 3), x, 0);
const a11: NDArray = np.transpose(x);
const a12: NDArray = x.T;
const a13: NDArray = np.clip(x, 2, 7);
const a14: NDArray = np.linspace(0, 1, 5);
const a15: NDArray = np.zeros([2, 3]);
const a16: NDArray = np.einsum('ij,jk->ik', x, x.T);

const v1: NDArray = x.reshape(4, 3);
const v2: NDArray = x.reshape([4, 3]);
const v3: NDArray = x.reshape(-1);
const v4: NDArray = x.transpose(1, 0);
const v5: NDArray[] = np.meshgrid(np.arange(3), np.arange(2));
const v6: NDArray = np.sum(x, [0, 1]);
const v7: NDArray = x.sum({ axis: 0, keepdims: true });
const v8: NDArray = np.add.reduce(x);
const v9: NDArray = np.argmax(x, 1);

// misspelled or unknown options are compile errors
// @ts-expect-error
np.sum(x, { axiss: 0 });
// @ts-expect-error
x.mean({ keepdim: true });
// @ts-expect-error
np.sort(x, { axis: 'a' });

// attributes and methods
const shape: number[] = x.shape;
const size: number = x.size;
const nd: number = x.ndim;
const t: NDArray = x.take([0, 1], 1);
const d: NDArray = x.diagonal();
const l: any = x.tolist();
const dt: string = x.dtype;

// binary operators by name
const b1: NDArray = np.add(x, 1);
const b2: NDArray = x.add(1).multiply(2);
const b3: NDArray = np.power(x, 2);
const b4: NDArray = np.maximum(x, x.T.T);

// random
const r1: any = np.random.rand(3);
const g = np.random.default_rng(0);
const r2: any = g.integers(0, 10, 5);

export { v1, v2, v3, v4, v5, v6, v7, v8, v9, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, shape, size, nd, t, d, l, dt, b1, b2, b3, b4, r1, r2 };
