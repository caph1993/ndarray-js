//@ts-check
/** @typedef {NumberConstructor|BooleanConstructor} DType */
/** @typedef {NDArray|number|boolean} ArrayOrConstant */

class NDArray {
  /**
   * @param {number[]} flat actually number|boolean
   * @param {number[]} shape
   * @param {*} dtype
   */
  constructor(flat, shape, dtype = Number) {
    this.shape = shape; // invariant: immutable
    this._flat = flat;
    this.dtype = dtype;
    this._simpleIndexes = null;
  }
  /** @type {indexes.AxesIndex|null} */ _simpleIndexes;

  get size() {
    return this._simpleIndexes == null ? this._flat.length : this._simpleIndexes.size;
  }
  get flat() {
    if (this._simpleIndexes == null) return this._flat;
    const indices = this._simpleIndexes.indices;
    return indices.map(i => this._flat[i]);
  }
  set flat(list) {
    if (list.length != this.size)
      throw new Error(`Length mismatch. Can't write ${list.length} values into ${this.size} available positions.`);
    const n = this.size;
    if (this._simpleIndexes == null) {
      for (let i = 0; i < n; i++) this._flat[i] = list[i];
    } else {
      const indices = this._simpleIndexes.indices;
      for (let i = 0; i < n; i++) this._flat[indices[i]] = list[i];
    }
  }
}

require('./core-globals').GLOBALS.NDArray = NDArray;


// ==============================
//    Basic methods
// ==============================


NDArray.prototype.reshape = function (A, shape, ...more_shape) {
  const { __parse_shape, __as_number } = NDArray.prototype;
  if (!more_shape.length) shape = __parse_shape(shape);
  else shape = [shape, ...more_shape].map(__as_number)
  const n = A.size;
  const inferredIndex = shape.indexOf(-1);
  if (inferredIndex !== -1) {
    const productOfKnownDims = shape.filter(dim => dim !== -1).reduce((acc, val) => acc * val, 1);
    if (n % productOfKnownDims !== 0) {
      throw new Error("Invalid shape. The total number of elements must match the product of the known dimensions.");
    }
    shape[inferredIndex] = n / productOfKnownDims;
  }
  return new NDArray(A.flat, shape, A.dtype);
};

NDArray.prototype.ravel = function (A) {
  A = NDArray.prototype.asarray(A);
  return new NDArray(A.flat, [A.size], A.dtype);
};


NDArray.prototype.copy = function (A) {
  return new NDArray([...A.flat], A.shape, A.dtype);
};

NDArray.prototype.isarray = function (arr) {
  return arr instanceof NDArray;
};

/** @returns {NDArray} */
NDArray.prototype.asarray = function (A) {
  if (A instanceof NDArray) return A;
  else return NDArray.prototype.fromJS(A);
}
/** @returns {NDArray} */
NDArray.prototype.array = function (A) {
  // @ts-ignore
  if (A instanceof NDArray) {
    let flat = A._simpleIndexes == null ? [...A.flat] : A.flat;
    return new NDArray(flat, A.shape, A.dtype);
  }
  else return NDArray.prototype.fromJS(A);
}

NDArray.prototype.__shape_shifts = function (shape) {
  // increasing one by one on a given axis is increasing by shifts[axis] in flat representation
  const shifts = Array.from({ length: shape.length }, (_) => 0);
  shifts[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) shifts[i] = shifts[i + 1] * shape[i + 1];
  return shifts;
}

/**
 * 
 * @param {NDArray} arr 
 * @returns {NDArray|number|boolean}
 */
NDArray.prototype.__number_collapse = function (arr, expect = false) {
  if (!arr.shape.length) return arr.flat[0];
  if (expect) throw new Error(`Expected constant. Got array with shape ${arr.shape}`);
  return arr;
}

NDArray.prototype.__parse_shape = function (list) {
  if (Array.isArray(list)) return list;
  if (list instanceof NDArray) {
    if (list.shape.length > 1) {
      throw new Error(`Expected flat list. Got array with shape ${list.shape}`);
    }
    return list.flat;
  }
  if (typeof list == "number") return [list];
  throw new Error(`Expected list. Got ${list}`);
}

NDArray.prototype.__as_boolean = function (obj) {
  if (obj instanceof NDArray) obj = NDArray.prototype.__number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return !!(0 + obj);
}
NDArray.prototype.__as_number = function (obj) {
  if (obj instanceof NDArray) obj = NDArray.prototype.__number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return parseFloat(obj);
}


NDArray.prototype._new = function (shape, f, dtype) {
  shape = NDArray.prototype.__parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  const flat = Array.from({ length: size }, f);
  return new NDArray(flat, shape, dtype);
};

NDArray.prototype.zeros = function (shape, /**@type {DType} */dtype = Number) {
  const c = dtype == Boolean ? false : 0;
  return NDArray.prototype._new(shape, (_) => c, dtype)
};
NDArray.prototype.empty = function (shape, /**@type {DType} */dtype = Number) {
  return NDArray.prototype._new(shape, (_) => undefined, dtype)
};
NDArray.prototype.ones = function (shape, /**@type {DType} */dtype = Number) {
  const c = dtype == Boolean ? true : 1;
  return NDArray.prototype._new(shape, (_) => c, dtype)
};
NDArray.prototype.arange = function (arg0, arg1 = null) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return NDArray.prototype._new(end - start, (_, i) => start + i, Number)
};
NDArray.prototype.__random = function (shape) {
  return NDArray.prototype._new(shape, (_) => Math.random(), Number)
};





/**
 * @template {Function} T
 * @param {T} func
 * @returns {T}
 */
function classMethodDecorator(func) {
  // @ts-ignore
  return function () {
    if (this instanceof NDArray) return func.bind(NDArray.prototype)(this, ...arguments);
    return func(...arguments);
  }
}

// ==============================
//    Slicing
// ==============================

var indexes = require('./core-indexes');
NDArray.prototype.slice = classMethodDecorator(indexes.slice);


// ==============================
//    Printing
// ==============================

var print = require('./core-print');
NDArray.prototype.toString = classMethodDecorator(print.humanReadable);


// ==============================
//    Grammar parser
// ==============================

var grammar = require('./core-grammar');
NDArray.prototype.grammar = grammar;
NDArray.prototype.parse = NDArray.prototype.grammar.parse;
NDArray.prototype.parseJS = NDArray.prototype.grammar.parseJS;




var reduce = require('./core-reduce');

NDArray.prototype.sum = classMethodDecorator(reduce.reducers.sum);
NDArray.prototype.product = classMethodDecorator(reduce.reducers.product);
NDArray.prototype.any = classMethodDecorator(reduce.reducers.any);
NDArray.prototype.all = classMethodDecorator(reduce.reducers.all);
NDArray.prototype.max = classMethodDecorator(reduce.reducers.max);
NDArray.prototype.min = classMethodDecorator(reduce.reducers.min);
NDArray.prototype.argmax = classMethodDecorator(reduce.reducers.argmax);
NDArray.prototype.argmin = classMethodDecorator(reduce.reducers.argmin);
NDArray.prototype.mean = classMethodDecorator(reduce.reducers.mean);
NDArray.prototype.var = classMethodDecorator(reduce.reducers.var);
NDArray.prototype.std = classMethodDecorator(reduce.reducers.std);



// ==============================
//       Operators: Binary operations, assignment operations and unary boolean_not
// ==============================

var op = require('./core-op');

NDArray.prototype.op = op.op;
NDArray.prototype._binary_operation = op.binary_operation;

NDArray.prototype.op_assign = op.op_assign;
NDArray.prototype.opx = op.opx;


NDArray.prototype.add = NDArray.prototype.op["+"];
NDArray.prototype.subtract = NDArray.prototype.op["-"];
NDArray.prototype.multiply = NDArray.prototype.op["*"];
NDArray.prototype.divide = NDArray.prototype.op["/"];
NDArray.prototype.mod = NDArray.prototype.op["%"];
NDArray.prototype.divide_int = NDArray.prototype.op["//"];
NDArray.prototype.pow = NDArray.prototype.op["**"];
NDArray.prototype.boolean_or = NDArray.prototype.op["|"];
NDArray.prototype.boolean_and = NDArray.prototype.op["&"];
NDArray.prototype.boolean_xor = NDArray.prototype.op["^"];
NDArray.prototype.boolean_shift_left = NDArray.prototype.op["<<"];
NDArray.prototype.boolean_shift_right = NDArray.prototype.op[">>"];
NDArray.prototype.gt = NDArray.prototype.op[">"];
NDArray.prototype.lt = NDArray.prototype.op["<"];
NDArray.prototype.geq = NDArray.prototype.op[">="];
NDArray.prototype.leq = NDArray.prototype.op["<="];
NDArray.prototype.eq = NDArray.prototype.op["=="];
NDArray.prototype.neq = NDArray.prototype.op["!="];
NDArray.prototype.maximum = NDArray.prototype.op["↑"];
NDArray.prototype.minimum = NDArray.prototype.op["↓"];

// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.boolean_not = function (A) { return NDArray.prototype.boolean_xor(A, 1); };


NDArray.prototype.isclose = op.isclose;
NDArray.prototype.allclose = op.allclose;





// ==============================
//    array instantiation and reshaping
// ==============================

var jsInterface = require('./core-js-interface');
NDArray.prototype.fromJS = jsInterface.fromJS;
NDArray.prototype.toJS = jsInterface.toJS;
NDArray.prototype.nested = jsInterface.nested;





// ==============================
//    pointwise math functions
// ==============================

NDArray.prototype._apply = function (A, func, dtype) {
  A = NDArray.prototype.asarray(A);
  return new NDArray(A.flat.map(func), A.shape, dtype);
}

NDArray.prototype.__make_pointwise = function (func, dtype = Number) {
  return function (A) {
    return NDArray.prototype._apply(A, func, dtype);
  }
}

NDArray.prototype.sign = NDArray.prototype.__make_pointwise(Math.sign);
NDArray.prototype.sqrt = NDArray.prototype.__make_pointwise(Math.sqrt);
NDArray.prototype.abs = NDArray.prototype.__make_pointwise(Math.abs);
NDArray.prototype.exp = NDArray.prototype.__make_pointwise(Math.exp);
NDArray.prototype.log = NDArray.prototype.__make_pointwise(Math.log);
NDArray.prototype.log2 = NDArray.prototype.__make_pointwise(Math.log2);
NDArray.prototype.log10 = NDArray.prototype.__make_pointwise(Math.log10);
NDArray.prototype.log1p = NDArray.prototype.__make_pointwise(Math.log1p);
NDArray.prototype.sin = NDArray.prototype.__make_pointwise(Math.sin);
NDArray.prototype.cos = NDArray.prototype.__make_pointwise(Math.cos);
NDArray.prototype.tan = NDArray.prototype.__make_pointwise(Math.tan);
NDArray.prototype.asin = NDArray.prototype.__make_pointwise(Math.asin);
NDArray.prototype.acos = NDArray.prototype.__make_pointwise(Math.acos);
NDArray.prototype.atan = NDArray.prototype.__make_pointwise(Math.atan);
NDArray.prototype.atan2 = NDArray.prototype.__make_pointwise(Math.atan2);
NDArray.prototype.cosh = NDArray.prototype.__make_pointwise(Math.cosh);
NDArray.prototype.sinh = NDArray.prototype.__make_pointwise(Math.sinh);
NDArray.prototype.tanh = NDArray.prototype.__make_pointwise(Math.tanh);
NDArray.prototype.acosh = NDArray.prototype.__make_pointwise(Math.acosh);
NDArray.prototype.asinh = NDArray.prototype.__make_pointwise(Math.asinh);
NDArray.prototype.atanh = NDArray.prototype.__make_pointwise(Math.atanh);
NDArray.prototype._round = NDArray.prototype.__make_pointwise(Math.round);
NDArray.prototype.round = function (A, decimals = 0) {
  if (decimals == 0) NDArray.prototype._round(A);
  return NDArray.prototype._apply(A, x => parseFloat(x.toFixed(decimals)), Number);
}


//=============================



NDArray.prototype.linspace = function (start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  const { multiply, arange, add, __as_number } = NDArray.prototype;
  start = __as_number(start);
  stop = __as_number(stop);
  let n = (num - (endpoint ? 1 : 0))
  let arr = add(multiply(arange(num), (stop - start) / n), start);
  return arr;
}

NDArray.prototype.geomspace = function (start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  const { exp, log, linspace } = NDArray.prototype;
  start = log(start);
  stop = log(stop);
  return exp(linspace(start, stop, num, endpoint));
}







NDArray.prototype.random = {
  random(shape) {
    return NDArray.prototype.__random(shape);
  },
  uniform(a, b, shape) {
    const { random, op } = NDArray.prototype;
    return op['+'](a, op['*'](random.random(shape), b - a));
  },
  exponential(mean, shape) {
    const { random, log, op } = NDArray.prototype;
    return op['*'](mean, op['-'](0, log(random.random(shape))));
  },
  // normal,
  // shuffle,
  // permutation,
};





module.exports = NDArray;