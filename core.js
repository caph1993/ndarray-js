//@ts-check
/** @typedef {NumberConstructor|BooleanConstructor} DType */
/** @typedef {NDArray|number|boolean} ArrayOrConstant */
/** @typedef {null|number} Axis */

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

var indexes = require('./core-indexes');
var grammar = require('./core-grammar');
var print = require('./core-print');



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

NDArray.prototype.copy = function (A) {
  return new NDArray([...A.flat], A.shape, A.dtype);
};

NDArray.prototype.isarray = function (arr) {
  return arr instanceof NDArray;
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
NDArray.prototype.slice = classMethodDecorator(indexes.slice);
NDArray.prototype.toString = classMethodDecorator(print.humanReadable);

NDArray.prototype.grammar = grammar;
NDArray.prototype.parse = NDArray.prototype.grammar.parse;
NDArray.prototype.parseJS = NDArray.prototype.grammar.parseJS;




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

NDArray.prototype._new = function (shape, f, dtype) {
  shape = NDArray.prototype.__parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  const flat = Array.from({ length: size }, f);
  return new NDArray(flat, shape, dtype);
};

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


NDArray.prototype._new = function (shape, f, dtype) {
  shape = NDArray.prototype.__parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  const flat = Array.from({ length: size }, f);
  return new NDArray(flat, shape, dtype);
};

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

// ==============================
//       Reducing functions
// ==============================


NDArray.prototype._reduce = function (arr, axis, keepdims, reducer, dtype = Number) {
  const { asarray, __shape_shifts, __as_boolean, __number_collapse } = NDArray.prototype;
  keepdims = __as_boolean(keepdims);
  arr = asarray(arr);
  if (axis == null) return reducer(arr.flat);
  if (axis < 0) axis = arr.shape.length - 1;
  let m = arr.shape[axis];
  let shift = __shape_shifts(arr.shape)[axis];
  const groups = Array.from({ length: m }, (_) =>/**@type {number[]}*/([]));
  arr.flat.forEach((value, i) => groups[(Math.floor(i / shift)) % m].push(value));
  // Transpose it:
  let nCols = arr.size / m;
  const groupsT = [];
  for (let j = 0; j < nCols; j++) {
    const newRow = [];
    for (let i = 0; i < m; i++) newRow.push(groups[i][j]);
    groupsT.push(newRow);
  }
  const flat = groupsT.map(reducer);
  let shape = [...arr.shape];
  if (keepdims) shape[axis] = 1;
  else shape = shape.filter((_, i) => i != axis);
  const out = new NDArray(flat, shape, dtype)
  return __number_collapse(out);
};

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

NDArray.prototype.__make_reducer = function (dtype, reducer) {
  const { _reduce } = NDArray.prototype;
  /**
   * @param {ArrayOrConstant} arr
   * @param {Axis} axis
   * @param {boolean} keepdims
   */
  return function (arr, axis = null, keepdims = false) {
    ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
    return _reduce(arr, axis, keepdims, reducer, dtype);
  };
}


NDArray.prototype.sum = NDArray.prototype.__make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0));
NDArray.prototype.product = NDArray.prototype.__make_reducer(Number, (arr) => arr.reduce((a, b) => a * b, 1));
NDArray.prototype.any = NDArray.prototype.__make_reducer(Boolean, (arr) => {
  for (let x of arr) if (x) return true;
  return false;
});
NDArray.prototype.all = NDArray.prototype.__make_reducer(Boolean, (arr) => {
  for (let x of arr) if (!x) return false;
  return true;
});
NDArray.prototype.max = NDArray.prototype.__make_reducer(Number, (arr) => Math.max(...arr));
NDArray.prototype.min = NDArray.prototype.__make_reducer(Number, (arr) => Math.min(...arr));
NDArray.prototype.argmax = NDArray.prototype.__make_reducer(Number, (arr) => arr.indexOf(Math.max(...arr)));
NDArray.prototype.argmin = NDArray.prototype.__make_reducer(Number, (arr) => arr.indexOf(Math.min(...arr)));
NDArray.prototype.mean = NDArray.prototype.__make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0) / arr.length);
NDArray.prototype.var = function (arr, axis = null, keepdims = false) {
  ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
  const { mean, subtract, multiply } = NDArray.prototype;
  const arrMean = mean.bind({ axis, keepdims: true })(arr);
  arr = subtract(arr, arrMean);
  arr = multiply(arr, arr);
  return mean.bind({ axis, keepdims })(arr);
};
NDArray.prototype.std = function (arr, axis = null, keepdims = false) {
  ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
  const { pow, var: _var } = NDArray.prototype;
  const variance = _var.bind({ axis, keepdims })(arr);
  return pow(variance, 0.5);
};



// ==============================
//       Binary operations (and boolean_not)
// ==============================

/**
 * 
 * @param {ArrayOrConstant} A 
 * @param {ArrayOrConstant} B 
 * @param {*} func
 * @param {*} dtype
 * @param {NDArray?} out
 * @returns {ArrayOrConstant}
 */
NDArray.prototype._binary_operation = function (A, B, func, dtype, out = null) {
  // Find output shape and input broadcast shapes
  const { asarray, _broadcast_shapes, __shape_shifts, empty, __number_collapse } = NDArray.prototype;
  A = asarray(A);
  B = asarray(B);
  const [shape, shapeA, shapeB] = _broadcast_shapes(A.shape, B.shape);
  if (out == null) out = empty(shape, dtype);
  else if (!(out instanceof NDArray)) throw new Error(`Out must be of type ${NDArray}. Got ${typeof out}`);
  // Iterate with broadcasted indices
  const flatOut = [];
  const shiftsA = __shape_shifts(shapeA);
  const shiftsB = __shape_shifts(shapeB);
  const flatA = A.flat;
  const flatB = B.flat;
  for (let i = 0; i < out.size; i++) {
    let idxA = 0, idxB = 0, idx = i;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      idxA += shiftsA[axis] * (idx % shapeA[axis]);
      idxB += shiftsB[axis] * (idx % shapeB[axis]);
      idx = Math.floor(idx / shape[axis]);
    }
    flatOut.push(func(flatA[idxA], flatB[idxB]));
  };
  out.flat = flatOut;
  return __number_collapse(out);
}

NDArray.prototype._broadcast_shapes = function (shapeA, shapeB) {
  const shape = [];
  const maxDim = Math.max(shapeA.length, shapeB.length);
  shapeA = [...Array.from({ length: maxDim - shapeA.length }, () => 1), ...shapeA];
  shapeB = [...Array.from({ length: maxDim - shapeB.length }, () => 1), ...shapeB];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dim1 = shapeA[axis];
    const dim2 = shapeB[axis];
    if (dim1 !== 1 && dim2 !== 1 && dim1 !== dim2)
      throw new Error(`Can not broadcast axis ${axis} with sizes ${dim1} and ${dim2}`);
    shape.unshift(Math.max(dim1, dim2));
  }
  return [shape, shapeA, shapeB];
}

/** @typedef {(A:ArrayOrConstant, B:ArrayOrConstant, out?:NDArray)=>ArrayOrConstant} BinaryOperator */

/**@returns {BinaryOperator} */
NDArray.prototype.__make_operator = function (dtype, func) {
  /** @param {NDArray?} out */
  return function (A, B, out = null) {
    return NDArray.prototype._binary_operation(A, B, func, dtype, out);
  };
}

/**@returns {BinaryOperator} */
NDArray.prototype.__make_operator_special = function (funcNum, funcBool) {
  /** @param {NDArray?} out */
  return function (A, B, out = null) {
    A = NDArray.prototype.asarray(A);
    B = NDArray.prototype.asarray(B);
    let dtype = A.dtype, func;
    if (A.dtype != B.dtype) console.warn(`Warning: operating arrays of different dtypes. Using ${dtype}`);
    if (dtype == Boolean) func = funcBool;
    else func = funcNum;
    return NDArray.prototype._binary_operation(A, B, func, dtype, out);
  };
}

/**@type {Object.<string, BinaryOperator>} */
NDArray.prototype.op = {
  "+": NDArray.prototype.__make_operator(Number, (a, b) => a + b),
  "-": NDArray.prototype.__make_operator(Number, (a, b) => a - b),
  "*": NDArray.prototype.__make_operator(Number, (a, b) => a * b),
  "/": NDArray.prototype.__make_operator(Number, (a, b) => a / b),
  "%": NDArray.prototype.__make_operator(Number, (a, b) => (a % b)),
  "//": NDArray.prototype.__make_operator(Number, (a, b) => Math.floor(a / b)),
  "**": NDArray.prototype.__make_operator(Number, (a, b) => Math.pow(a, b)),
  "<": NDArray.prototype.__make_operator(Boolean, (a, b) => a < b),
  ">": NDArray.prototype.__make_operator(Boolean, (a, b) => a > b),
  ">=": NDArray.prototype.__make_operator(Boolean, (a, b) => a >= b),
  "<=": NDArray.prototype.__make_operator(Boolean, (a, b) => a <= b),
  "==": NDArray.prototype.__make_operator(Boolean, (a, b) => a == b),
  "!=": NDArray.prototype.__make_operator(Boolean, (a, b) => a != b),
  "|": NDArray.prototype.__make_operator_special((a, b) => a | b, (a, b) => a || b),
  "&": NDArray.prototype.__make_operator_special((a, b) => a & b, (a, b) => a && b),
  "^": NDArray.prototype.__make_operator(Number, (a, b) => a ^ b),
  "<<": NDArray.prototype.__make_operator(Number, (a, b) => a << b),
  ">>": NDArray.prototype.__make_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "||": NDArray.prototype.__make_operator(Boolean, (a, b) => a || b),
  "&&": NDArray.prototype.__make_operator(Boolean, (a, b) => a && b),
  "max": NDArray.prototype.__make_operator(Number, (a, b) => Math.max(a, b)),
  "min": NDArray.prototype.__make_operator(Number, (a, b) => Math.min(a, b)),
};


/** @typedef {(A:ArrayOrConstant, B:ArrayOrConstant, slicesSpec:any)=>void} AssignmentOperator */

/**@returns {AssignmentOperator} */
NDArray.prototype.__make_assignment_operator = function (dtype, func) {
  const { _binary_operation, asarray, ravel } = NDArray.prototype;
  /** @param {*?} slicesSpec */
  return function (tgt, src, slicesSpec) {
    if (!(tgt instanceof NDArray)) throw new Error(`Can not assign to a non-array. Found ${typeof tgt}: ${tgt}`);
    if (!slicesSpec) {
      _binary_operation(tgt, src, func, dtype, tgt);
    } else {
      src = asarray(src);
      let { indices } = indexes.AxesIndex.prototype.parse(tgt.shape, slicesSpec);
      let tmpTgt = asarray(indices.map(i => tgt._flat[i]));
      _binary_operation(tmpTgt, ravel(src), func, dtype, tmpTgt);
      for (let i of indices) tgt._flat[i] = tmpTgt._flat[i];
    }
  };
}

/**@type {Object.<string, AssignmentOperator>} */
NDArray.prototype.op_assign = {
  "=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => b),
  "+=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a + b),
  "-=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a - b),
  "*=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a * b),
  "/=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a / b),
  "%=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => (a % b)),
  "//=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => Math.floor(a / b)),
  "**=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => Math.pow(a, b)),
  "|=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a | b),
  "&=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a & b),
  "^=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a ^ b),
  "<<=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a << b),
  ">>=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => Math.max(a, b)),
  "min=": NDArray.prototype.__make_assignment_operator(Number, (a, b) => Math.min(a, b)),
  "||=": NDArray.prototype.__make_assignment_operator(Boolean, (a, b) => a || b),
  "&&=": NDArray.prototype.__make_assignment_operator(Boolean, (a, b) => a && b),
};


// Extended, non-ascii operator names for fun
NDArray.prototype.opx = Object.assign({
  // Operators with custom non-ascii identifiers:
  // "≈≈": MyArray.prototype.isclose,
  "↑": NDArray.prototype.op["max"],
  "↓": NDArray.prototype.op["min"],
  "≤": NDArray.prototype.op["leq"],
  "≥": NDArray.prototype.op["geq"],
  "≠": NDArray.prototype.op["neq"],
  "↑=": NDArray.prototype.op["max="],
  "↓=": NDArray.prototype.op["min="],
}, NDArray.prototype.op);


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



NDArray.prototype.isclose = function (A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  const func = (a, b) => {
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  return NDArray.prototype._binary_operation(A, B, func, Boolean)
}

NDArray.prototype.allclose = function (A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  // Equivalent to all(isclose(A, B, rtol, atol, equal_nan)), but shortcutting if false 
  const func = (a, b) => { //copied from isclose
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  const different = new Error('');
  const wrapper = (a, b) => {
    if (!func(a, b)) throw different;
    return 0;
  }
  try { NDArray.prototype._binary_operation(A, B, wrapper, Number) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
}









/** @param {indexes.GeneralSliceSpec[]} slicesSpec */

NDArray.prototype._assign_operation = function (tgt, src, slicesSpec, func, dtype) {
  // @ts-ignore
  if (this !== NDArray.prototype) return NDArray.prototype._assign_operation(this, ...arguments);
  const { _binary_operation, asarray, ravel } = NDArray.prototype;
  if (!(tgt instanceof NDArray)) return _assign_operation_toJS(tgt, src, slicesSpec, func, dtype)
  if (!slicesSpec) {
    _binary_operation(tgt, src, func, dtype, tgt);
  } else {
    src = asarray(src);
    let { indices } = indexes.AxesIndex.prototype.parse(tgt.shape, slicesSpec);
    let tmpTgt;
    if (func == null) {
      // Small optimization: unlike "+=", "*=", etc., for "=", we don't need to reed the target
      func = (a, b) => b;
      tmpTgt = NDArray.prototype._new(indices.length, () => undefined, tgt.dtype);
    } else {
      tmpTgt = asarray(indices.map(i => tgt._flat[i]));
    }
    _binary_operation(tmpTgt, ravel(src), func, dtype, tmpTgt);
    for (let i of indices) tgt._flat[i] = tmpTgt._flat[i];
  }
};


/**
 * @param {any[]} tgtJS
 * @param {any} src
 * @param {any} func
 * @param {any} dtype
 * @param {indexes.GeneralSliceSpec[]} slicesSpec
 */
function _assign_operation_toJS(tgtJS, src, slicesSpec, func, dtype) {
  const { asarray, toJS } = NDArray.prototype;
  if (!Array.isArray(tgtJS)) throw new Error(`Can not assign to a non-array. Found ${typeof tgtJS}: ${tgtJS}`);
  console.warn('Assignment to JS array is experimental and slow.')
  // Parse the whole array
  const cpy = asarray(tgtJS);
  NDArray.prototype._assign_operation(cpy, src, slicesSpec, func, dtype);
  // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
  const outJS = toJS(cpy);
  while (tgtJS.length) tgtJS.pop();
  // @ts-ignore
  tgtJS.push(...outJS);
}

// ==============================
//    array instantiation and reshaping
// ==============================

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

NDArray.prototype.fromJS = function (arr) {
  if (typeof arr === "number") return new NDArray([arr], [], Number);
  if (typeof arr === "boolean") return new NDArray([arr ? 1 : 0], [], Boolean);
  if (!Array.isArray(arr)) throw new Error(`Can't parse as array: ${arr}`);
  const shape = [];
  let root = arr;
  while (Array.isArray(root)) {
    shape.push(root.length);
    root = root[0];
    if (shape.length > 256) throw new Error(`Circular reference or excessive array depth`);
  }
  let dtype = typeof root === "boolean" ? Boolean : Number;
  const flat = [];
  const pushToFlat = (arr, axis) => {
    // Check consistency
    if (axis == shape.length - 1) {
      for (let elem of arr) {
        if (Array.isArray(elem)) throw new Error(`Inconsistent shape`);
        flat.push(elem);
        // Update dtype
      }
    } else {
      if (!Array.isArray(arr)) throw new Error(`Inconsistent shape`);
      for (let sub of arr) {
        if (sub.length != shape[axis + 1])
          throw new Error(`Inconsistent shape: found sibling arrays of lengths ${sub.length} and ${shape[axis + 1]}`);
        pushToFlat(sub, axis + 1);
      }
    }
  }
  pushToFlat(arr, 0);
  return new NDArray(flat, shape, dtype)
}

NDArray.prototype.toJS = function (arr) {
  if (typeof arr == "number" || typeof arr == "boolean") return arr;
  if (Array.isArray(arr)) return arr.map(NDArray.prototype.toJS);
  if (!(arr instanceof NDArray)) throw new Error(`Expected MyArray. Got ${typeof arr}: ${arr}`);
  arr = NDArray.prototype.__number_collapse(arr);
  if (!(arr instanceof NDArray)) return arr;

  // let out = [], top;
  // let q = /**@type {[MyArray, any][]}*/([[arr, out]])
  // while (top = q.pop()) {
  //   let [arr, out] = top;
  //   if (arr.shape.length <= 1) {
  //     out.push(...arr.flat);
  //   } else {
  //     for (let i = 0; i < arr.shape[0]; i++) {
  //       let l = []
  //       out.push(l);
  //       q.push([arr.slice(i), l]);
  //     }
  //   }
  // }
  // return out;
  function recursiveReshape(flatArr, shapeArr) {
    if (shapeArr.length === 0) {
      return flatArr.shift();
    }
    const innerShape = shapeArr.slice(1);
    const outerSize = shapeArr[0];
    const innerArray = [];
    for (let i = 0; i < outerSize; i++) {
      innerArray.push(recursiveReshape(flatArr, innerShape));
    }
    return innerArray;
  }
  return recursiveReshape([...arr.flat], arr.shape);
}

NDArray.prototype.ravel = function (A) {
  A = NDArray.prototype.asarray(A);
  return new NDArray(A.flat, [A.size], A.dtype);
};


// =========================================
//     Slicing
// =========================================

NDArray.prototype.__shape_shifts = function (shape) {
  // increasing one by one on a given axis is increasing by shifts[axis] in flat representation
  const shifts = Array.from({ length: shape.length }, (_) => 0);
  shifts[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) shifts[i] = shifts[i + 1] * shape[i + 1];
  return shifts;
}
NDArray.prototype.__parse_sliceRange = function (axis_size, { start, stop, step }) {
  if (start == null) start = 0;
  else if (start < 0) start = axis_size + start;
  if (stop == null) stop = axis_size;
  else if (stop < 0) stop = axis_size + stop;
  if (step == null) step = 1;
  else if (step == 0) throw new Error(`Slice range with step size of zero`);
  if (!isFinite(start) || !isFinite(stop) || !isFinite(step)) throw new Error(`Invalid slice ${[start, stop, step]}. Axis size ${axis_size}`);
  let indices = [];
  if (step > 0) {
    start = Math.max(start, 0);
    stop = Math.min(stop, axis_size);
    for (let i = start; i < stop; i += step) indices.push(i);
  } else {
    stop = Math.max(stop, 0);
    start = Math.min(start, axis_size);
    for (let i = start; i > stop; i += step) indices.push(i);
  }
  return indices;
}


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














// ==============================
//    utils for js lists
// ==============================

NDArray.prototype.nested = {
  _binary_operation(A, B, func) {
    // Pointwise check for equality for arbitrary nested js arrays (without broadcasting)
    const C = [];
    const q = [[A, B, C, 0]];
    let seen;
    while (true) {
      const _next = q.pop();
      if (!_next) return true;
      const [a, b, c, depth] = _next
      if (Array.isArray(a) && Array.isArray(b) && a.length == b.length) {
        for (let i in a) {
          const c_i = [];
          c.push(c_i);
          q.push([a[i], b[i], c_i, depth + 1]);
        }
      }
      else c.push(func(a, b));
      if (depth > 10000 && Array.isArray(a)) { // Activate circular reference detection
        // Checking only A suffices (the other will exhaust otherwise)
        seen = /**@type {any[]}*/(seen || []);
        if (seen.includes(a)) throw new Error(`Circular reference found. ${a}`)
        seen[depth - 10000] = a;
      }
    }
  },
  ravel(A) {
    // Flatten js array
    const q = [[A, 0]], flat = [];
    let seen;
    while (true) {
      const _next = q.pop();
      if (!_next) break;
      const [a, depth] = _next
      if (depth > 10000 && Array.isArray(a)) { // Activate circular reference detection
        seen = /**@type {any[]}*/(seen || []);
        if (seen.includes(a)) throw new Error(`Circular reference found. ${a}`)
        seen[depth - 10000] = a;
      }
      if (Array.isArray(a)) {
        q.push(...a.map(v => [v, depth + 1]));
        continue;
      }
      flat.push(a);
    }
    return flat;
  },
  allEq(A, B, nan_equal = false) {
    const different = new Error('');
    const func = (a, b) => {
      if (a !== b && !(nan_equal && Number.isNaN(a) && Number.isNaN(b))) throw different;
      return 0;
    }
    try { NDArray.prototype.nested._binary_operation(A, B, func) }
    catch (err) {
      if (err === different) return false;
      else throw err;
    }
    return true;
  },
  allClose(A, B, rtol = 1.e-5, atol = 1.e-8, nan_equal = false,) {
    const func = (a, b) => { //copied from isclose
      if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * Math.abs(b);
      return (a === b) || (nan_equal && Number.isNaN(a) && Number.isNaN(b));
    }
    const different = new Error('');
    const wrapper = (a, b) => {
      if (!func(a, b)) throw different;
      return 0;
    }
    try { NDArray.prototype.nested._binary_operation(A, B, wrapper) }
    catch (err) {
      if (err === different) return false;
      else throw err;
    }
    return true;
  },
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