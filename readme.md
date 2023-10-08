# Numpy-js

Numpy-js is a reimplementation of numpy for javascript that aims to make the coding experience as similar to numpy as possible.

```js
console.log(np.JS`np.linspace(-5, 5, 5) ** 2`)
// Prints: [ 25, 6.25, 0, 6.25, 25 ]
console.log(np.JS`np.arange(120).reshape(20,3,2).mean(axis=0)`)
// Prints: [ [ 57, 58 ], [ 59, 60 ], [ 61, 62 ] ]

var x = np`np.linspace(-10, 10, 51)`;
console.log(x.shape, x.min()); // [ 51 ] -10
var x_abs = x.copy();
np`${x_abs}[${x_abs} < 0] *= -1`
console.log(x_abs.shape, x_abs.min()); // [ 51 ] 0
console.log(np.allclose(x_abs, np.abs(x))); // true
console.log(np`${x_abs}[1:-20:15].round(3)`.JS()); // [ 9.6, 3.6, 2.4 ]
```

Find more details and usage patterns in the [documentation](https://caph1993.github.io/numpy-js/).


Numpy-js can be used in:

- web applications: `<script src="(TODO: put a cdn url here)">`
- node applications: `npm install @caph1993/numpy-js`
- jupyter notebooks with ijavascript kernel. [Example](https://github.com/caph1993/numpy-js/blob/main/notebooks/normal-scatter.ipynb).


