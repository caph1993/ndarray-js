# ndarray-js

`ndarray-js` is a reimplementation of `numpy` for javascript that aims to make the coding experience as similar to `numpy` as possible.

It features:

 - A parser that can execute numpy code with numpy (python) syntax like ```x = np`np.arange(120).reshape(20,3,2).mean(axis=0)` ```.
 - Handling of args/kwargs, e.g. you can use any of `x.mean(-1, true)` or `x.mean({axis: -1, keepdims: true})` or `x.mean(-1, {keepdims: true})`.

 - All types of advanced indexing, for example ```y = np`${x}[2, :, mask, ..., indices, -1:-1:-2, -1]` ```, or without parsing, `y = x.index(2, ':', mask, '...', indices, '-1:-1:-2', -1)`.
 - broadcasting, both manual and automatic for all basic operations.

 - a simple group_by operator.

It does not provide yet:
 - the linear algebra module
 - all the random module. Only uniform, normal, exponential, gamma, beta and some others are provided.


# Examples using the parser

```js
console.log(np.JS`np.linspace(-5, 5, 5) ** 2`)
// Prints: [ 25, 6.25, 0, 6.25, 25 ]
console.log(np.JS`np.arange(120).reshape(20,3,2).mean(axis=0)`)
// Prints: [ [ 57, 58 ], [ 59, 60 ], [ 61, 62 ] ]

var x = np`np.linspace(-10, 10, 51)`;
console.log(x.shape, x.min()); // [ 51 ] -10
var x_abs = x.copy();
np`${x_abs}[${x_abs} < 0] *= -1` // Notice the numpy notation: x_abs[x_abs < 0] *= -1
console.log(x_abs.shape, x_abs.min()); // [ 51 ] 0
console.log(np.allclose(x_abs, np.abs(x))); // true
console.log(np`${x_abs}[1:-20:15].round(3)`.JS()); // [ 9.6, 3.6, 2.4 ]
```

# Examples without the parser:

```js
var x = np.arange(120).reshape(20, 3, 2).mean({axis: 0});
x = x.pow(2);
x.index(0, ':', ':').op('=', -1);
x.index('...', ':2').op('+=', 5);

console.log(x.min());
console.log(x.argmin());

// Invalid operation (prints a warning)

console.log('First row:', x.index(0));
console.log('Second row:', x.index(1));


var x = np`np.linspace(-10, 10, 51)`;
console.log(x.shape, x.min()); // [ 51 ] -10
var x_abs = x.copy();
np`${x_abs}[${x_abs} < 0] *= -1` // Notice the numpy notation: x_abs[x_abs < 0] *= -1
console.log(x_abs.shape, x_abs.min()); // [ 51 ] 0
console.log(np.allclose(x_abs, np.abs(x))); // true
console.log(np`${x_abs}[1:-20:15].round(3)`.JS()); // [ 9.6, 3.6, 2.4 ]
```


# Usage

Find more details and usage patterns in the [documentation](https://caph1993.github.io/numpy-js/).

Numpy-js can be used in:

- web applications: `<script src="https://cdn.jsdelivr.net/npm/ndarray-js@1.0.0/dist/index.js"></script>`
- node applications: `npm install ndarray-js`
- jupyter notebooks with ijavascript kernel. [Example](https://github.com/caph1993/numpy-js/blob/main/notebooks/normal-scatter.ipynb).


<!-- As of October 2023, the library is under development and testing. -->

