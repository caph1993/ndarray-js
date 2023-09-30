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

require('./globals').GLOBALS.NDArray = NDArray;


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

NDArray.prototype.empty = function (shape, /**@type {DType} */dtype = Number) {
  return NDArray.prototype._new(shape, (_) => undefined, dtype)
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
    return func.bind(this)(...arguments);
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
//    Reduce
// ==============================

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


NDArray.prototype.add = op.op["+"];
NDArray.prototype.subtract = op.op["-"];
NDArray.prototype.multiply = op.op["*"];
NDArray.prototype.divide = op.op["/"];
NDArray.prototype.mod = op.op["%"];
NDArray.prototype.divide_int = op.op["//"];
NDArray.prototype.pow = op.op["**"];
NDArray.prototype.boolean_or = op.op["|"];
NDArray.prototype.boolean_and = op.op["&"];
NDArray.prototype.boolean_xor = op.op["^"];
NDArray.prototype.boolean_shift_left = op.op["<<"];
NDArray.prototype.boolean_shift_right = op.op[">>"];
NDArray.prototype.gt = op.op[">"];
NDArray.prototype.lt = op.op["<"];
NDArray.prototype.geq = op.op[">="];
NDArray.prototype.leq = op.op["<="];
NDArray.prototype.eq = op.op["=="];
NDArray.prototype.neq = op.op["!="];
NDArray.prototype.maximum = op.op["↑"];
NDArray.prototype.minimum = op.op["↓"];

// Unary operations: only boolean_not. Positive is useless and negative is almost useless
NDArray.prototype.boolean_not = function (A) { return NDArray.prototype.boolean_xor(A, 1); };


NDArray.prototype.isclose = op.isclose;
NDArray.prototype.allclose = op.allclose;





// ==============================
//    array instantiation and reshaping
// ==============================

var jsInterface = require('./core-js-interface');
NDArray.prototype.fromJS = classMethodDecorator(jsInterface.fromJS);
NDArray.prototype.toJS = classMethodDecorator(jsInterface.toJS);





// ==============================
//    pointwise math functions
// ==============================

function apply_pointwise(A, func, dtype) {
  if (this instanceof NDArray) return apply_pointwise.bind(NDArray.prototype)(this, ...arguments);
  A = NDArray.prototype.asarray(A);
  return new NDArray(A.flat.map(func), A.shape, dtype);
}

NDArray.prototype.apply_pointwise = apply_pointwise;

function round(A, decimals = 0) {
  if (this instanceof NDArray) return round.bind(NDArray.prototype)(this, ...arguments);
  if (decimals == 0) apply_pointwise(A, Math.round, Number);
  return apply_pointwise(A, x => parseFloat(x.toFixed(decimals)), Number);
};

NDArray.prototype.round = round;


//=============================










module.exports = NDArray;