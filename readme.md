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
```

# Installation

Numpy-js can be used in:

- web applications: `<script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>`
- node applications: `npm install ndarray-js`
- jupyter notebooks with ijavascript kernel.
[Example](https://github.com/caph1993/numpy-js/blob/main/notebooks/normal-scatter.ipynb).

Check the [documentation website](https://caph1993.github.io/ndarray-js/) for details.

## ToDo

It does not provide yet:
 - trapz, binning.
 - a simple group_by operator.
 - the linear algebra module
 - all the random module. Only uniform, normal, exponential, gamma, beta and some others are provided.

<script src="https://cdn.jsdelivr.net/npm/ndarray-js@latest/dist/index.js"></script>