//@ts-check
const { np } = require("./globals").GLOBALS;

/** @typedef {typeof np.NDArray} NDArray*/


function empty(shape, /**@type {import('./core').DType} */dtype = Number) {
  return np.NDArray.prototype._new(shape, (_) => undefined, dtype)
};
function zeros(shape, /**@type {import('./core').DType} */dtype = Number) {
  const c = dtype == Boolean ? false : 0;
  return np.NDArray.prototype._new(shape, (_) => c, dtype)
};
function ones(shape, /**@type {import('./core').DType} */dtype = Number) {
  const c = dtype == Boolean ? true : 1;
  return np.NDArray.prototype._new(shape, (_) => c, dtype)
};

function arange(arg0, arg1 = null) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return np.NDArray.prototype._new(end - start, (_, i) => start + i, Number)
};

function linspace(start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  const { __as_number } = np.NDArray.prototype;
  start = __as_number(start);
  stop = __as_number(stop);
  let n = (num - (endpoint ? 1 : 0))
  let arr = np.add(np.multiply(arange(num), (stop - start) / n), start);
  return arr;
}

function geomspace(start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  start = np.log(start);
  stop = np.log(stop);
  return np.exp(linspace(start, stop, num, endpoint));
}

module.exports = {
  empty,
  zeros,
  ones,
  arange,
  linspace,
  geomspace,
};