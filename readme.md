# ndarray-js

`ndarray-js` is a reimplementation of `numpy` for javascript that aims to make the coding experience as similar to `numpy` as possible.


Documentation and interactive demo: [click here](https://caph1993.github.io/ndarray-js/).

# Features

The library implements a large subset of numpy operations (see the [documentation website](https://caph1993.github.io/ndarray-js/))

```js
// Feature 1: A parsing system that can interpret numpy code as ndarray-js instructions
x = np`np.exp(0.5 * np.linspace(0, 1, 100).reshape(5, 4, 5)).mean(axis=0)`
y = np`(${x} * ${x}) / 2`

// Equivalent code:
x = np.exp(0.5 * np.linspace(0, 1, 100).reshape(5, 4, 5)).mean({axis:0})
y = np.divide(np.multiply(x * x), 2)


// Feature 2. Handling of args and kwargs following numpy order
x = np.geomspace(1, 10, 100).reshape([10,10])
x.mean({axis: -1, keepdims: true})
x.mean(-1, true)
x.mean(-1, {keepdims: true})
np.mean(x, {axis: -1})


// Feature 3. All types of indexing: (number, start:end:step, ':', '...', list of indices, NDArray boolean mask, NDArray boolean mask, NDArray indices)
y = x.index(2, ':', mask, '...', indices, '-1:-1:-2', -1);
y = np`${x}[2, :, ${mask}, ..., ${indices}, -1:-1:-2, -1]`;

// Feature 4. Broadcasting:
mu = np.array([1.5, 0]);
x = np.random.randn([100, 2]).add(x);
norm = x.pow(2).sum({axis: -1}).pow(0.5);
x.divide_assign(norm.index(':', 'None'));
for(let point2d of x){
  console.log(point2d.tolist(), point2d.norm(), point2d.max());
  if(np.all(point2d.equal(x.index(5)))) break;
}

// Feature 5. Most of numpy: all numeric dtypes (int8..uint64, float16..64, complex64/128), linalg, fft,
// numpy.polynomial, strings/char, npy/npz/text I/O, format_float_*, numpy-exact printing, and seeded random
// streams that reproduce numpy's (np.random.seed, RandomState, default_rng/PCG64 + SeedSequence).
np.random.seed(0);
np.random.rand(3);                                   // [0.5488135, 0.71518937, 0.60276338], like numpy
np.random.default_rng(0).integers(0, 10, 5);         // [0, 0, 1, 8, 6], like numpy
np.linalg.eigvals([[0, -1], [1, 0]]);                // complex128, like numpy
np.fft.fft(np.arange(8));
```

# Installation

Numpy-js can be used in:

- web applications: `<script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>`
- node applications: `npm install ndarray-js`
- jupyter notebooks with ijavascript kernel.
[Example](https://github.com/caph1993/numpy-js/blob/main/notebooks/normal-scatter.ipynb).

Check the [documentation website](https://caph1993.github.io/ndarray-js/) for details.

## Differences with numpy

 - Dtypes: bool, int8-int64, uint8-uint64 (64-bit integers are stored as JS doubles, exact up to 2**53), float16/32/64, complex64/128 and object (plus string arrays). There is no datetime64 (the business-day functions take and return `'YYYY-MM-DD'` strings).
   `sum`, `prod` and `cumsum` of ints/bools are `int64`, like in numpy.
 - JS numbers do not distinguish `2` from `2.0`: `np.array([1, 2])` is `int32`. Inside the template parser, float literals are kept as floats: ``np`np.array([1., 2])` `` is `float64`.
 - 0-d results are returned as plain JS numbers/booleans.
 - Reducers take `keepdims` as the third argument: `x.sum(axis, keepdims, dtype, out)`.
 - Indexing with several lists/masks is orthogonal (one per axis) instead of numpy's "zipped" fancy indexing.
 - Python scalars are "weak" as in numpy 2: `np.array([1, 2], 'float32') + 1.5` stays `float32`.

## Development

```bash
npm install --legacy-peer-deps
npx vitest run                         # unit tests, and differential tests against real numpy
NDARRAYJS_PYTHON=/path/to/venv/bin/python npx vitest run   # python with numpy installed (default: python3)
```

The tests in `tests/numpy/` evaluate the same numpy expression with python/numpy and with ndarray-js
(through its numpy-syntax parser) and compare values, shapes and dtype kinds.
They are skipped when numpy is not available.

## ToDo

Not provided (yet):
 - `numpy.ma` (masked arrays), datetime64/timedelta64 and structured dtypes (the business-day functions work on `'YYYY-MM-DD'` strings).
 - Philox and SFC64 bit generators; numpy's exact streams for `binomial` (large `n*p`) and `hypergeometric` (the distributions are right, the seeded values differ).
 - True O(1) strided reads: views (slices, transposes) share memory with their base like in numpy, but reading a non-contiguous view first gathers its elements (a few times slower than a plain array).

Beyond numpy, `np.group_by(keys, values?, axis?, func?)` splits `values` by the distinct entries of `keys`:
```js
const g = np.group_by(['b', 'a', 'b'], [10, 20, 30]);
g.keys.tolist();                        // ['a', 'b']
g.groups.map(x => x.tolist());          // [[20], [10, 30]]
np.group_by([1, 0, 1], [1, 2, 3], {func: np.sum}).values.tolist();  // [2, 4]
```
