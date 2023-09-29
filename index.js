//@ts-check
const { easeCircleOut } = require('d3');
var ohm = require('ohm-js');

/** @typedef {NumberConstructor|BooleanConstructor} DType */
/** @typedef {MyArray|number|boolean} ArrayOrConstant */
/** @typedef {null|number} Axis */
/** @typedef {null|{size:Number, ranges:{refSize:number, range:number|[number,number,number]|null}[], indices:null|number[]}} SimpleIndex */

/**@template {boolean|number} T*/
class MyArray {
  /**
   * @param {T[]} flat
   * @param {number[]} shape
   * @param {*} dtype
   */
  constructor(flat, shape, dtype = Number) {
    this._flat = flat;
    this.shape = shape; // invariant: immutable
    this.dtype = dtype;
    this._simpleIndex = null;
  }
  /** @type {SimpleIndex|null} */ _simpleIndex;

  get size() {
    return this._simpleIndex == null ? this._flat.length : this._simpleIndex.size;
  }
  get flat() {
    if (this._simpleIndex == null) return this._flat;
    const indices = MyArray.prototype.__simpleIndex_to_slices(this._simpleIndex);
    return indices.map(i => this._flat[i]);
  }
  set flat(list) {
    if (list.length != this.size)
      throw new Error(`Length mismatch. Can't write ${list.length} values into ${this.size} available positions.`);
    const n = this.size;
    if (this._simpleIndex == null) {
      for (let i = 0; i < n; i++) this._flat[i] = list[i];
    } else {
      const indices = MyArray.prototype.__simpleIndex_to_slices(this._simpleIndex);
      for (let i = 0; i < n; i++) this._flat[indices[i]] = list[i];
    }
  }
}

MyArray.prototype.__simpleIndex_to_slices = function (/** @type {SimpleIndex} */ simpleIndex) {
  if (!simpleIndex) throw new Error(`This function can only be called on views`);
  if (simpleIndex.indices) return simpleIndex.indices;
  const { ranges } = simpleIndex;
  const slices = ranges.map(({ refSize, range }) => {
    let [start, stop, step] = range == null ? [0, refSize, 1] : typeof range == "number" ? [range, range + 1, 1] : range;
    let indices = [];
    for (let i = start; i < stop; i += step) indices.push(i);
    return indices;
  });
  const indices = MyArray.prototype.__slices_to_indices(this.shape, slices);
  return simpleIndex.indices = indices;
}

/**
 * 
 * @param {SimpleIndex} first 
 * @param {SimpleIndex} second 
 * @returns {SimpleIndex}
 */
MyArray.prototype.__compose_simpleIndexes = function (first, second) {
  if (first == null) return second;
  if (second == null) return first;
  const inRange = (x, start, stop) => (start <= x && x < stop) || (start >= x && x > stop);
  let size = 1, j = 0;
  const ranges = [];
  for (let i = 0; i < first.ranges.length; i++) {
    let { refSize: refSizeA, range: rangeA } = first.ranges[i];
    if (typeof rangeA == "number") continue;
    let { refSize: refSizeB, range: rangeB } = second.ranges[j] || { refSize: refSizeA, range: null };
    if (refSizeA != refSizeB) throw new Error(`Mismatch of index reference size: ${refSizeA}!=${refSizeB}`);
    let /**@type {null|number|[number,number,number]} */ range;
    if (rangeA == null) range = rangeB;
    else if (rangeB == null) range = rangeA;
    else {
      let [startA, stopA, stepA] = rangeA;
      if (typeof rangeB == "number") {
        range = startA + rangeB * stepA;
        if (!inRange(range, startA, stopA)) throw new Error(`Index out of bounds`);
      } else {
        let [startB, stopB, stepB] = rangeB;
        let step = stepA * stepB;
        let start = startA + startB * stepA;
        let stop = stopA - stopB * stepA;
        if (!inRange(start, startA, stopA)) start = stop = 0;
        range = [start, stop, step];
        size *= Math.floor(Math.abs(start - stop) / step);
      }
    }
    ranges.push({ refSize: refSizeA, range: range });
    j++;
  }
  if (j < second.ranges.length) throw new Error(`Index too long. Expected ${j} axes. Found ${second.ranges.length}`)
  return { ranges, size, indices: null };
}


/**
 * Computes the indices wr to shape of the cartesian products of the slices.
 * We have shape.length==slices.length, and the elements in slices[axis] are
 * integers between 0 and shape[axis]-1
 * @param {number[]} shape 
 * @param {number[][]} slices 
 * @returns {number[]}
 */
MyArray.prototype.__slices_to_indices = function (shape, slices) {
  const { __shape_shifts } = MyArray.prototype;
  const shifts = __shape_shifts(shape);
  const iShifts = slices.map((indices, axis) => {
    // out[i] = How much does the cursor increase if we change from [...,indices[i],...] to [...,indices[(i+1)%n],...]
    let out = [], n = indices.length;
    for (let i = 0; i < n - 1; i++) out.push(shifts[axis] * (indices[i + 1] - indices[i]));
    out[n - 1] = shifts[axis] * (indices[0] - indices[n - 1]);
    return out;
  });
  const indices = [];
  const lastAxis = shape.length - 1;
  const tuple = new Array(shape.length).fill(0);
  let cursor = slices.map((l, i) => l[tuple[i]] * shifts[i]).reduce((a, b) => a + b, 0);
  while (true) {
    if (!isFinite(cursor)) throw new Error(`Programming error`);
    indices.push(cursor);
    let axis = lastAxis;
    while (axis >= 0) {
      cursor += iShifts[axis][tuple[axis]++];
      if (tuple[axis] < iShifts[axis].length) break;
      tuple[axis--] = 0; // Overflow
    };
    if (axis < 0) break;
  }
  return indices;
}


/**
 * 
 * @param {MyArray} arr 
 * @returns {MyArray|number|boolean}
 */
MyArray.prototype.__number_collapse = function (arr, expect = false) {
  if (!arr.shape.length) return arr.flat[0];
  if (expect) throw new Error(`Expected constant. Got array with shape ${arr.shape}`);
  return arr;
}


MyArray.prototype._new = function (shape, f, dtype) {
  shape = MyArray.prototype.__parse_shape(shape);
  const size = shape.reduce((a, b) => a * b, 1);
  const flat = Array.from({ length: size }, f);
  return new MyArray(flat, shape, dtype);
};

MyArray.prototype.__parse_shape = function (list) {
  if (Array.isArray(list)) return list;
  if (list instanceof MyArray) {
    if (list.shape.length > 1) {
      throw new Error(`Expected flat list. Got array with shape ${list.shape}`);
    }
    return list.flat;
  }
  if (typeof list == "number") return [list];
  throw new Error(`Expected list. Got ${list}`);
}
MyArray.prototype.zeros = function (shape, /**@type {DType} */dtype = Number) {
  const c = dtype == Boolean ? false : 0;
  return MyArray.prototype._new(shape, (_) => c, dtype)
};
MyArray.prototype.empty = function (shape, /**@type {DType} */dtype = Number) {
  return MyArray.prototype._new(shape, (_) => undefined, dtype)
};
MyArray.prototype.ones = function (shape, /**@type {DType} */dtype = Number) {
  const c = dtype == Boolean ? true : 1;
  return MyArray.prototype._new(shape, (_) => c, dtype)
};
MyArray.prototype.arange = function (arg0, arg1 = null) {
  let start, end;
  if (arg1 === null) start = 0, end = arg0;
  else start = arg0, end = arg1;
  return MyArray.prototype._new(end - start, (_, i) => start + i, Number)
};

// ==============================
//       Reducing functions
// ==============================


MyArray.prototype._reduce = function (arr, axis, keepdims, reducer, dtype = Number) {
  const { asarray, __shape_shifts, __as_boolean, __number_collapse } = MyArray.prototype;
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
  const out = new MyArray(flat, shape, dtype)
  return __number_collapse(out);
};

MyArray.prototype.__as_boolean = function (obj) {
  if (obj instanceof MyArray) obj = MyArray.prototype.__number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return !!(0 + obj);
}
MyArray.prototype.__as_number = function (obj) {
  if (obj instanceof MyArray) obj = MyArray.prototype.__number_collapse(obj, true);
  else if (typeof obj == 'string') throw new Error(`'string' object can not be interpreted as boolean: ${obj}`);
  return parseFloat(obj);
}

MyArray.prototype.__make_reducer = function (dtype, reducer) {
  const { _reduce } = MyArray.prototype;
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


MyArray.prototype.sum = MyArray.prototype.__make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0));
MyArray.prototype.product = MyArray.prototype.__make_reducer(Number, (arr) => arr.reduce((a, b) => a * b, 1));
MyArray.prototype.any = MyArray.prototype.__make_reducer(Boolean, (arr) => {
  for (let x of arr) if (x) return true;
  return false;
});
MyArray.prototype.all = MyArray.prototype.__make_reducer(Boolean, (arr) => {
  for (let x of arr) if (!x) return false;
  return true;
});
MyArray.prototype.max = MyArray.prototype.__make_reducer(Number, (arr) => Math.max(...arr));
MyArray.prototype.min = MyArray.prototype.__make_reducer(Number, (arr) => Math.min(...arr));
MyArray.prototype.argmax = MyArray.prototype.__make_reducer(Number, (arr) => arr.indexOf(Math.max(...arr)));
MyArray.prototype.argmin = MyArray.prototype.__make_reducer(Number, (arr) => arr.indexOf(Math.min(...arr)));
MyArray.prototype.mean = MyArray.prototype.__make_reducer(Number, (arr) => arr.reduce((a, b) => a + b, 0) / arr.length);
MyArray.prototype.var = function (arr, axis = null, keepdims = false) {
  ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
  const { mean, subtract, multiply } = MyArray.prototype;
  const arrMean = mean.bind({ axis, keepdims: true })(arr);
  arr = subtract(arr, arrMean);
  arr = multiply(arr, arr);
  return mean.bind({ axis, keepdims })(arr);
};
MyArray.prototype.std = function (arr, axis = null, keepdims = false) {
  ({ axis, keepdims } = Object.assign({ axis, keepdims }, this));
  const { pow, var: _var } = MyArray.prototype;
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
 * @param {MyArray?} out
 * @returns {ArrayOrConstant}
 */
MyArray.prototype._binary_operation = function (A, B, func, dtype, out = null) {
  // Find output shape and input broadcast shapes
  const { asarray, _broadcast_shapes, __shape_shifts, empty, __number_collapse } = MyArray.prototype;
  A = asarray(A);
  B = asarray(B);
  const [shape, shapeA, shapeB] = _broadcast_shapes(A.shape, B.shape);
  if (out == null) out = empty(shape, dtype);
  else if (!(out instanceof MyArray)) throw new Error(`Out must be of type ${MyArray}. Got ${typeof out}`);
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

MyArray.prototype._broadcast_shapes = function (shapeA, shapeB) {
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

/** @typedef {(A:ArrayOrConstant, B:ArrayOrConstant, out?:MyArray)=>ArrayOrConstant} BinaryOperator */

/**@returns {BinaryOperator} */
MyArray.prototype.__make_operator = function (dtype, func) {
  /** @param {MyArray?} out */
  return function (A, B, out = null) {
    return MyArray.prototype._binary_operation(A, B, func, dtype, out);
  };
}

/**@type {Object.<string, BinaryOperator>} */
MyArray.prototype.op = {
  "+": MyArray.prototype.__make_operator(Number, (a, b) => a + b),
  "-": MyArray.prototype.__make_operator(Number, (a, b) => a - b),
  "*": MyArray.prototype.__make_operator(Number, (a, b) => a * b),
  "/": MyArray.prototype.__make_operator(Number, (a, b) => a / b),
  "%": MyArray.prototype.__make_operator(Number, (a, b) => (a % b)),
  "//": MyArray.prototype.__make_operator(Number, (a, b) => Math.floor(a / b)),
  "**": MyArray.prototype.__make_operator(Number, (a, b) => Math.pow(a, b)),
  ">": MyArray.prototype.__make_operator(Boolean, (a, b) => a > b),
  "<": MyArray.prototype.__make_operator(Boolean, (a, b) => a < b),
  ">=": MyArray.prototype.__make_operator(Boolean, (a, b) => a >= b),
  "<=": MyArray.prototype.__make_operator(Boolean, (a, b) => a <= b),
  "==": MyArray.prototype.__make_operator(Boolean, (a, b) => a == b),
  "!=": MyArray.prototype.__make_operator(Boolean, (a, b) => a != b),
  "|": MyArray.prototype.__make_operator(Number, (a, b) => a | b),
  "&": MyArray.prototype.__make_operator(Number, (a, b) => a & b),
  "^": MyArray.prototype.__make_operator(Number, (a, b) => a ^ b),
  "<<": MyArray.prototype.__make_operator(Number, (a, b) => a << b),
  ">>": MyArray.prototype.__make_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "||": MyArray.prototype.__make_operator(Boolean, (a, b) => a || b),
  "&&": MyArray.prototype.__make_operator(Boolean, (a, b) => a && b),
  "max": MyArray.prototype.__make_operator(Number, (a, b) => Math.max(a, b)),
  "min": MyArray.prototype.__make_operator(Number, (a, b) => Math.min(a, b)),
};


/** @typedef {(A:ArrayOrConstant, B:ArrayOrConstant, slicesSpec:any)=>void} AssignmentOperator */
/**@returns {AssignmentOperator} */
MyArray.prototype.__make_assignment_operator = function (dtype, func) {
  const { _binary_operation, _idx_slice, asarray, ravel } = MyArray.prototype;
  /** @param {*?} slicesSpec */
  return function (tgt, src, slicesSpec) {
    if (!(tgt instanceof MyArray)) throw new Error(`Can not assign to a non-array. Found ${typeof tgt}: ${tgt}`);
    if (!slicesSpec) {
      _binary_operation(tgt, src, func, dtype, tgt);
    } else {
      src = asarray(src);
      let [_, indices] = _idx_slice(tgt.shape, slicesSpec);
      let tmpTgt = asarray(indices.map(i => tgt._flat[i]));
      _binary_operation(tmpTgt, ravel(src), func, dtype, tmpTgt);
      for (let i of indices) tgt._flat[i] = tmpTgt._flat[i];
    }
  };
}

/**@type {Object.<string, AssignmentOperator>} */
MyArray.prototype.op_assign = {
  "=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => b),
  "+=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a + b),
  "-=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a - b),
  "*=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a * b),
  "/=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a / b),
  "%=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => (a % b)),
  "//=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => Math.floor(a / b)),
  "**=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => Math.pow(a, b)),
  "|=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a | b),
  "&=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a & b),
  "^=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a ^ b),
  "<<=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a << b),
  ">>=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => Math.max(a, b)),
  "min=": MyArray.prototype.__make_assignment_operator(Number, (a, b) => Math.min(a, b)),
  "||=": MyArray.prototype.__make_assignment_operator(Boolean, (a, b) => a || b),
  "&&=": MyArray.prototype.__make_assignment_operator(Boolean, (a, b) => a && b),
};


// Extended, non-ascii operator names for fun
MyArray.prototype.opx = Object.assign({
  // Operators with custom non-ascii identifiers:
  // "≈≈": MyArray.prototype.isclose,
  "↑": MyArray.prototype.op["max"],
  "↓": MyArray.prototype.op["min"],
  "≤": MyArray.prototype.op["leq"],
  "≥": MyArray.prototype.op["geq"],
  "≠": MyArray.prototype.op["neq"],
  "↑=": MyArray.prototype.op["max="],
  "↓=": MyArray.prototype.op["min="],
}, MyArray.prototype.op);


MyArray.prototype.add = MyArray.prototype.op["+"];
MyArray.prototype.subtract = MyArray.prototype.op["-"];
MyArray.prototype.multiply = MyArray.prototype.op["*"];
MyArray.prototype.divide = MyArray.prototype.op["/"];
MyArray.prototype.mod = MyArray.prototype.op["%"];
MyArray.prototype.divide_int = MyArray.prototype.op["//"];
MyArray.prototype.pow = MyArray.prototype.op["**"];
MyArray.prototype.boolean_or = MyArray.prototype.op["|"];
MyArray.prototype.boolean_and = MyArray.prototype.op["&"];
MyArray.prototype.boolean_xor = MyArray.prototype.op["^"];
MyArray.prototype.boolean_shift_left = MyArray.prototype.op["<<"];
MyArray.prototype.boolean_shift_right = MyArray.prototype.op[">>"];
MyArray.prototype.gt = MyArray.prototype.op[">"];
MyArray.prototype.lt = MyArray.prototype.op["<"];
MyArray.prototype.geq = MyArray.prototype.op[">="];
MyArray.prototype.leq = MyArray.prototype.op["<="];
MyArray.prototype.eq = MyArray.prototype.op["=="];
MyArray.prototype.neq = MyArray.prototype.op["!="];
MyArray.prototype.maximum = MyArray.prototype.op["↑"];
MyArray.prototype.minimum = MyArray.prototype.op["↓"];

// Unary operations: only boolean_not. Positive is useless and negative is almost useless
MyArray.prototype.boolean_not = function (A) { return MyArray.prototype.boolean_xor(A, 1); };



MyArray.prototype.isclose = function (A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  const func = (a, b) => {
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  return MyArray.prototype._binary_operation(A, B, func, Boolean)
}

MyArray.prototype.allclose = function (A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
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
  try { MyArray.prototype._binary_operation(A, B, wrapper, Number) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
}


// ==============================
//    array instantiation and reshaping
// ==============================

MyArray.prototype.asarray = function (A) {
  if (A instanceof MyArray) return A;
  else return MyArray.prototype.from_js_array(A);
}
MyArray.prototype.array = function (A) {
  // @ts-ignore
  if (A instanceof MyArray) {
    let flat = A._simpleIndex == null ? [...A.flat] : A.flat;
    return new MyArray(flat, A.shape, A.dtype);
  }
  else return MyArray.prototype.from_js_array(A);
}
MyArray.prototype.from_js_array = function (arr) {
  if (typeof arr === "number") return new MyArray([arr], [], Number);
  if (typeof arr === "boolean") return new MyArray([arr ? 1 : 0], [], Boolean);
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
  return new MyArray(flat, shape, dtype)
}
MyArray.prototype.to_js_array = function (arr) {
  if (!(arr instanceof MyArray)) throw new Error(`Expected MyArray: ${arr}`);
  arr = MyArray.prototype.__number_collapse(arr);
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
  return recursiveReshape(arr.flat, arr.shape);
}
MyArray.prototype.ravel = function (A) {
  A = MyArray.prototype.asarray(A);
  return new MyArray(A.flat, [A.size], A.dtype);
};


// =========================================
//     Slicing
// =========================================


MyArray.prototype.slice = function (arr, slicesSpec) {
  // This can result either in a view or a copy
  const [shape, indices] = MyArray.prototype._idx_slice(arr.shape, slicesSpec);
  return new MyArray(indices.map(i => arr._flat[i]), shape, arr.dtype);
}

MyArray.prototype.__shape_shifts = function (shape) {
  // increasing one by one on a given axis is increasing by shifts[axis] in flat representation
  const shifts = Array.from({ length: shape.length }, (_) => 0);
  shifts[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) shifts[i] = shifts[i + 1] * shape[i + 1];
  return shifts;
}
MyArray.prototype.__parse_sliceRange = function (axis_size, { start, stop, step }) {
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


/**@typedef {{type:':', size:number}|{type:'number', index:number}|{type:'range', range:{start:number, step:number, nSteps:number}}|{type:'array', indices:number[]}} AxisIndexSpec */

class AxisIndex {
  /**
   * @param {AxisIndexSpec} spec
   */
  constructor(spec) {
    this.spec = spec;
    this._indices = null;
  }
  get indices() {
    if (this._indices) return this._indices;
    let indices;
    if (this.spec.type == ':') indices = Array.from({ length: this.spec.size }, (_, i) => i);
    else if (this.spec.type === "number") indices = [this.spec.index];
    else if (this.spec.type === "array") indices = this.spec.indices;
    else if (this.spec.type == "range") {
      const { nSteps, step, start } = this.spec.range;
      indices = Array.from({ length: nSteps }, (_, i) => start + i * step);
    } else throw new Error(`Unknown spec type ${this.spec['type']}`);
    return this._indices = indices;
  }
  get size() {
    if (this.spec.type == ':') return this.spec.size;
    else if (this.spec.type === "number") return 1;
    else if (this.spec.type === "array") return this.spec.indices.length;
    else if (this.spec.type == "range") return this.spec.range.nSteps;
    else throw new Error(`Unknown spec type ${this.spec['type']}`);
  }
}

/**
 * 
 * @param {number} size
 * @param {number|null} start
 * @param {number|null} stop
 * @param {number|null} step
 * @returns
 */
AxisIndex.prototype.parse_range = function (size, start = null, stop = null, step = null) {
  if (step == null) step = 1;
  else if (step == 0) throw new Error(`Index specification error. Step must be different from zero.`);
  /**
   *  @param {number|null} i @param {number} ifNull @param {number} min @param {number} max */
  const parse = (i, ifNull, min, max) => {
    if (i == null) return ifNull;
    if (i < 0) i = Math.max(0, size - i);
    return Math.min(max, Math.max(min, i));
  }
  let nSteps;
  if (step > 0) start = parse(start, 0, 0, size - 1), stop = parse(stop, size, 0, size);
  else start = parse(start, size - 1, 0, size - 1), stop = parse(stop, size - 1, -1, size - 1);
  stop = Math.max(stop, start);
  nSteps = Math.floor(Math.abs(stop - start) / step);
  return { start, step, nSteps };
}

class AxesIndex {
  /**
   * @param {AxisIndex[]} axisIndexes
   */
  constructor(outShape, vShape, axisIndexes) {
    this.shape = outShape;
    this.vShape = vShape;
    this.axisIndexes = axisIndexes;
    this._indices = null;
  }
  get indices() {
    if (this._indices) return this._indices;
    let indices = MyArray.prototype.__slices_to_indices(this.vShape, this.slices);
    return this._indices = indices;
  }
  get slices() {
    return this.axisIndexes.map(idx => idx.indices);
  }
}


/**@typedef {':'|'...'|'None'|number|{isRange:boolean, start:null|number, stop:null|number, step:null|number}|MyArray|number[]} SliceSpec */
/**@typedef {':'|'number'|'range'|'array'} IndexType */


/**
 * @param {SliceSpec[]} slicesSpec
 * @returns {AxesIndex}
 */
AxesIndex.prototype.parse = function (shape, slicesSpec) {
  /**
   * 
   * vShape (virtual shape) matches shape unless there are boolean masks spanning
   * over several axes/dimensions.
   * For example, in `np.ones((2,3,4,5))[:, np.arange(12).reshape((3,4))>5, 1]`,
   * the boolean mask is spanning over axes 1 and 2. In this case, the output should
   * merge these axes, resulting in an a vShape of (2, 12, 5).
   * The boolean mask is then converted to indices in the flattened merged axis.
   */
  const vShape = [];
  const outShape = [];
  let /**@type {AxisIndex[]}*/ axisIndexes = [];
  let ellipsis = {
    axisDir: 1,
    vShape: NaN,
    outShape: NaN,
    axisIndexes: NaN,
  }
  let axis = 0, sliceI = 0, nonVisitedAxes = shape.length;
  while (nonVisitedAxes > 0) {
    let inputSpec = slicesSpec[sliceI];
    //@ts-ignore
    slicesSpec[sliceI] = undefined; // For ellipsis to avoid reading twice
    sliceI += ellipsis.axisDir;
    if (inputSpec == "None") {
      outShape.push(1);
      continue;
    } else if (inputSpec == "...") {
      if (ellipsis.axisDir == -1) throw new Error(`Index can only have a single ellipsis ('...')`)
      ellipsis.axisDir = -1;
      ellipsis.vShape = vShape.length;
      ellipsis.outShape = outShape.length;
      ellipsis.axisIndexes = axisIndexes.length;
      sliceI = slicesSpec.length - 1;
      axis = shape.length - 1;
      continue;
    }
    /**@type {AxisIndexSpec} */
    let spec;
    let axisSize = shape[axis];
    let visitedAxes = 1;
    if (inputSpec == ':' || inputSpec === undefined) {
      spec = { type: ':', size: shape[axis] };
    }
    else if (typeof inputSpec === "number") {
      spec = { type: 'number', index: inputSpec };
    }
    else if (inputSpec instanceof MyArray || Array.isArray(inputSpec)) {
      let indices;
      inputSpec = MyArray.prototype.asarray(inputSpec)
      if (inputSpec.dtype == Number) {
        // Array of indices
        if (inputSpec.shape.length > 1) throw new Error(
          `Expected 1D array of indices or nD array of booleans. ` +
          `Found shape=${inputSpec.shape} and dtype=${inputSpec.dtype}`
        );
        indices = inputSpec.flat;
      } else {
        // Boolean mask
        indices = [];
        inputSpec.flat.forEach((if_value, i) => if_value && indices.push(i));
        // Next lines: the boolean mask spans over more than 1 axis
        visitedAxes = Math.max(1, inputSpec.shape.length);
        // Multiply the (possibly inverted) interval
        for (let n = visitedAxes - 1, i = axis; n; n--) axisSize *= shape[i += ellipsis.axisDir];
        if (axisSize < 0) throw new Error(`Index with boolean mask spans over more dimensions than available`);
      }
      spec = { type: 'array', indices };
    }
    else if (inputSpec.isRange) {
      let { start, stop, step } = inputSpec;
      const range = AxisIndex.prototype.parse_range(shape[axis], start, stop, step);
      if (range.start == 0 && range.nSteps == shape[axis] && range.step == 1) {
        // Small optimization: all of these are just ":": ["::","0::1", ":axisSize:", etc.]
        spec = { type: ':', size: shape[axis] };
      } else {
        spec = { type: 'range', range };
      }
    }
    else throw new Error(`Unknown index type. Found ${typeof inputSpec}: ${inputSpec}`);

    const axisIndex = new AxisIndex(spec);
    axisIndexes.push(axisIndex);
    vShape.push(axisSize);
    if (axisIndex.spec.type != "number") outShape.push(axisIndex.size);
    axis += ellipsis.axisDir * visitedAxes;
    nonVisitedAxes -= visitedAxes;
  }
  //if (slicesSpec.filter(x => x !== undefined).length) throw new Error(`Index exceeds the number of dimensions`);
  if (ellipsis.axisDir == -1) {
    // reverse the right to left elements
    vShape.splice(0, ellipsis.vShape).concat(vShape.reverse());
    outShape.splice(0, ellipsis.outShape).concat(outShape.reverse());
    axisIndexes.splice(0, ellipsis.axisIndexes).concat(axisIndexes.reverse());
  }
  return new AxesIndex(outShape, vShape, axisIndexes);
}


MyArray.prototype._idx_slice = function (shape, slicesSpec) {
  // Iterative cartesian product of the slices.
  const axesIndex = AxesIndex.prototype.parse(shape, slicesSpec);

  const outShape = axesIndex.shape;
  const slices = axesIndex.slices;
  const vShape = axesIndex.vShape;

  if (slices.map(l => l.length).reduce((a, b) => a * b, 1) == 0) {
    return [outShape, []];
  }
  // const indices = MyArray.prototype.__slices_to_indices(vShape, slices);
  const indices = axesIndex.indices;
  return [outShape, indices];
}


// ==============================
//    pointwise math functions
// ==============================

MyArray.prototype._apply = function (A, func, dtype) {
  A = MyArray.prototype.asarray(A);
  return new MyArray(A.flat.map(func), A.shape, dtype);
}

MyArray.prototype.__make_pointwise = function (func, dtype = Number) {
  return function (A) {
    return MyArray.prototype._apply(A, func, dtype);
  }
}

MyArray.prototype.sign = MyArray.prototype.__make_pointwise(Math.sign);
MyArray.prototype.sqrt = MyArray.prototype.__make_pointwise(Math.sqrt);
MyArray.prototype.abs = MyArray.prototype.__make_pointwise(Math.abs);
MyArray.prototype.exp = MyArray.prototype.__make_pointwise(Math.exp);
MyArray.prototype.log = MyArray.prototype.__make_pointwise(Math.log);
MyArray.prototype.log2 = MyArray.prototype.__make_pointwise(Math.log2);
MyArray.prototype.log10 = MyArray.prototype.__make_pointwise(Math.log10);
MyArray.prototype.log1p = MyArray.prototype.__make_pointwise(Math.log1p);
MyArray.prototype.sin = MyArray.prototype.__make_pointwise(Math.sin);
MyArray.prototype.cos = MyArray.prototype.__make_pointwise(Math.cos);
MyArray.prototype.tan = MyArray.prototype.__make_pointwise(Math.tan);
MyArray.prototype.asin = MyArray.prototype.__make_pointwise(Math.asin);
MyArray.prototype.acos = MyArray.prototype.__make_pointwise(Math.acos);
MyArray.prototype.atan = MyArray.prototype.__make_pointwise(Math.atan);
MyArray.prototype.cosh = MyArray.prototype.__make_pointwise(Math.cosh);
MyArray.prototype.sinh = MyArray.prototype.__make_pointwise(Math.sinh);
MyArray.prototype.tanh = MyArray.prototype.__make_pointwise(Math.tanh);
MyArray.prototype.acosh = MyArray.prototype.__make_pointwise(Math.acosh);
MyArray.prototype.asinh = MyArray.prototype.__make_pointwise(Math.asinh);
MyArray.prototype.atanh = MyArray.prototype.__make_pointwise(Math.atanh);
MyArray.prototype._round = MyArray.prototype.__make_pointwise(Math.round);
MyArray.prototype.round = function (A, decimals = 0) {
  if (decimals == 0) MyArray.prototype._round(A);
  return MyArray.prototype._apply(A, x => parseFloat(x.toFixed(decimals)), Number);
}

// ==============================
//    utils for js lists
// ==============================

MyArray.prototype.nested = {
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
    try { MyArray.prototype.nested._binary_operation(A, B, func) }
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
    try { MyArray.prototype.nested._binary_operation(A, B, wrapper) }
    catch (err) {
      if (err === different) return false;
      else throw err;
    }
    return true;
  },
}



MyArray.prototype.grammar = {}
MyArray.prototype.grammar.grammar = String.raw`
ArrayGrammar {
  Instruction
  = Variable "[" Slice "]" AssignSymbol ArithmeticLogicExp -- sliceAssignment
  | ArithmeticLogicExp                       -- expression
  
  Variable
   = "#" digit+ "#"
  
  AssignSymbol
  ="="|"+="|"-="|"/="|"%="|"&="|"|="|"^="|"@="
  
  /* Declaration in precedence order (weakest first) */
  ArithmeticLogicExp = Precedence11

  /* https://docs.python.org/3/reference/expressions.html */
  Operator11 = "<" | "<=" | ">" | ">=" | "!=" | "=="
  Operator10 = "|"
  Operator09 = "^"
  Operator08 = "&"
  Operator07 = "<<" | ">>"
  Operator06 = "+" | "-"
  Operator05 = "*" | "@" | "/" | "//" | "%"
  Operator04 = "~" /* Unary */
  Operator03 = "+" | "-" /* Unary. Special treatment to prevent "-1.3" to be "-(array of 1.3)" */
  Operator02 = "**"
  /* Operator01 = "x[index]" | "x[index:index]" | "x(arguments...)" | "x.attribute" */
  /* Operator00 = "(expressions...)" */

  Precedence11 = Precedence11 Operator11 Precedence10 | "" "" Precedence10
  Precedence10 = Precedence10 Operator10 Precedence09 | "" "" Precedence09
  Precedence09 = Precedence09 Operator09 Precedence08 | "" "" Precedence08
  Precedence08 = Precedence08 Operator08 Precedence07 | "" "" Precedence07
  Precedence07 = Precedence07 Operator07 Precedence06 | "" "" Precedence06
  Precedence06 = Precedence06 Operator06 Precedence05 | "" "" Precedence05
  Precedence05 = Precedence05 Operator05 Precedence04 | "" "" Precedence04
  Precedence04 = ""           Operator04 Precedence03 | "" "" Precedence03 /* Unary */
  Precedence03 = ""           Operator03 Precedence02 | "" "" Precedence02 /* Special */
  Precedence02 = Precedence02 Operator02 Precedence03 | "" "" Precedence01
  Precedence01 = Arr
  
  Parenthesis = "(" ArithmeticLogicExp ")"
  Arr
    = Arr "." Name CallArgs     -- method
    | Arr "." Name              -- attribute
    | Arr "[" Slice "]"         -- slice
    | Parenthesis
    | Name ("." Name)* CallArgs -- call
    | number
    | Variable

  Name  (an identifier)
    = (letter|"_") (letter|"_"|digit)*

  number  (a number)
    = ("+"|"-")? digit* "." digit+ "E" ("+"|"-")? "digit+"
    | ("+"|"-")? digit* "." digit+ "e" ("+"|"-")? "digit+"
    | ("+"|"-")? digit* "." digit+  ""  ""        ""
    | ("+"|"-")? digit+ ""  ""      ""  ""        ""
  
  int (an integer) = ""  digit+ | "-" digit+ | "+" digit+

  CallArgs // Using empty strings instead of separate rules
   = "(" Args ","  KwArgs ","? ")"
   | "(" Args ","? ""      ""  ")"
   | "(" ""   ","? KwArgs ","? ")"
   | "(" ""    ""  ""      ""  ")"
   
  Args = NonemptyListOf<ArgValue, ",">
  KwArgs = NonemptyListOf<KwArg, ",">
  KwArg = Name "=" ArgValue

  ArgValue = Constant | JsArray | ArithmeticLogicExp
  Constant = "True" | "False" | "None" | "np.nan" | "np.inf" | String
  JsArray
    = "[" ListOf<ArgValue, ","> ","? "]"
    | "(" ListOf<ArgValue, ","> ","? ")"

  String = "\'" any* "\'" | "\"" any* "\""
   
  Slice = NonemptyListOf<SliceTerm, ",">
  SliceTerm
    = SliceRange
    | (":" | "..." | "None") -- constant
    | JsArray
    | ArithmeticLogicExp
  
  SliceRange
    = int ":" int ":" int
    | int ":" int ""  ""
    | int ":" ""  ":" int
    | int ":" ""  ""  ""
    | ""  ":" int ":" int
    | ""  ":" int ""  ""
    | ""  ":" ""  ":" int
    | ""  ":" ""  ""  ""
}
`;

MyArray.prototype.grammar.ohmGrammar = ohm.grammar(MyArray.prototype.grammar.grammar);


MyArray.prototype.grammar.__makeSemantics = () => {

  const semanticVariables = [];
  const semantics = {
    Instruction_sliceAssignment($tgt, _open, $slicesSpec, _close, $symbol, $src) {
      // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
      const _tgt = $tgt.parse();
      const _src = $src.parse();
      const symbol = $symbol.sourceString;
      const slicesSpec = $slicesSpec.parse();
      const { asarray, op_assign, to_js_array } = MyArray.prototype;
      let tgt = asarray(_tgt);
      op_assign[symbol](_tgt, _src, slicesSpec);
      tgt = to_js_array(tgt);
      while (_tgt.length) _tgt.pop();
      // @ts-ignore
      _tgt.push(..._tgt);
      return null;
    },
    Instruction_expression($arr) {
      const arr = $arr.parse();
      if (typeof arr === "number") return arr;
      if (Array.isArray(arr)) return arr;
      return MyArray.prototype.to_js_array(arr);
    },
    Precedence11: BinaryOperation,
    Precedence10: BinaryOperation,
    Precedence09: BinaryOperation,
    Precedence08: BinaryOperation,
    Precedence07: BinaryOperation,
    Precedence06: BinaryOperation,
    Precedence05: BinaryOperation,
    Precedence04: UnaryOperation,
    Precedence03: UnaryOperation,
    Precedence02: BinaryOperation,
    number: function (arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
      return parseFloat(this.sourceString)
    },
    Arr_slice($arr, _open, $slicesSpec, _close) {
      const arr = $arr.parse();
      const slicesSpec = $slicesSpec.parse();
      return MyArray.prototype.slice(arr, slicesSpec);
    },
    SliceTerm_constant($x) {
      return $x.sourceString;
    },
    Arr_call($name, $names, _, $callArgs) {
      let name = $name.sourceString + $names.sourceString;
      if (name.slice(0, 3) == "np.") name = name.slice(3);
      const func = MyArray.prototype[name];
      if (func === undefined) throw new Error(`Unrecognized function ${name}`)
      const { args, kwArgs } = $callArgs.parse();
      return func.bind(kwArgs)(...args);
    },
    Arr_method($arr, _dot, $name, $callArgs) {
      let arr = $arr.parse();
      let name = $name.sourceString;
      if (name.slice(0, 3) == "np.") name = name.slice(3);
      const func = MyArray.prototype[name];
      if (func === undefined) throw new Error(`Unrecognized method ${name}`)
      const { args, kwArgs } = $callArgs.parse();
      return func.bind(kwArgs)(arr, ...args);
    },
    Parenthesis(_, $arr, __) { return $arr.parse(); },
    Arr_attribute($arr, _, $name) { return $arr.parse()[$name.sourceString]; },
    Variable(_, $i, __) {
      const i = parseInt($i.sourceString);
      const arr = semanticVariables[i];
      return arr;
    },
    int($sign, $value) {
      const value = parseInt($value.sourceString);
      if ($sign.sourceString == '-') return -value;
      else return value
    },
    SliceRange($start, _, $stop, __, $step) {
      const start = $start.parse();
      const stop = $stop.parse();
      const step = $step.parse();
      return { start, stop, step, isRange: true };
    },

    CallArgs(_open, $args, _comma, $kwArgs, _trailing, _close) {
      const args = $args.parse() || [];
      let parse_constants = (s) => {
        switch (s) {
          case "True": return true;
          case "False": return false;
          case "None": return null;
        }
        if (s.length && s[0] == '"' && s[s.length - 1] == '"') return s;
        if (s.length && s[0] == "'" && s[s.length - 1] == "'") return s;
        return s;
      }
      let entries = $kwArgs.parse() || [];
      entries = entries.map(([k, v]) => [k, parse_constants(v)]);
      let kwArgs = Object.fromEntries(entries);
      return { args, kwArgs };
    },
    KwArg($key, _equals, $value) {
      const key = $key.sourceString;
      const value = $value.sourceString;
      return [key, value];
    },
    NonemptyListOf(first, _, more) {
      return [first, ...more.children].map(c => c.parse());
    },
    JsArray(_open, $list, _trailing, _close) {
      const list = $list.parse();
      // Downcast arrays (needed because, e.g., for [-1, 3, -2], -1 and -2 are interpreted as MyArray rather than int)
      for (let i in list) if (list[i] instanceof MyArray) list[i] = MyArray.prototype.to_js_array(list[i]);
      return list;
    },
    _terminal() { return null; },
  };

  function BinaryOperation($A, $symbol, $B) {
    const { opx } = MyArray.prototype;
    const A = $A.parse();
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "" && A === null) return B;
    return MyArray.prototype.op[symbol](A, B);
  }
  function UnaryOperation(_, $symbol, $B) {
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "") return B;
    switch (symbol) {
      case "+": return B;
      case "-": return MyArray.prototype.multiply(-1, B);
      case "^": return MyArray.prototype.boolean_not(B);
    }
    throw new Error(`Programming Error: ${symbol}`);
  }

  const { ohmGrammar } = MyArray.prototype.grammar;

  const ohmSemantics = ohmGrammar.createSemantics();
  ohmSemantics.addOperation('parse', semantics);
  /**
   * @param {TemplateStringsArray} template
   * @param {any[]} variables
   */
  function parse(template, ...variables) {
    // Replace variables in template with #0# #1# #2#, ...
    let idx = 0;
    const template_with_placeholders = template.join('###').replace(/###/g, () => `#${idx++}#`);
    semanticVariables.length = 0;
    semanticVariables.push(...variables);
    const match = ohmGrammar.match(template_with_placeholders);
    if (!match.succeeded()) throw new Error(match.message);
    return ohmSemantics(match).parse();
  }

  return { parse, ohmSemantics, semantics, semanticVariables, busy: 0 };
}


MyArray.prototype.grammar.__parser_pool = [MyArray.prototype.grammar.__makeSemantics()];

/**
 * @param {TemplateStringsArray} template
 * @param {any[]} variables
 */
MyArray.prototype.grammar.parse = function (template, ...variables) {
  // Thread control, because the parser depends on semanticVariables,
  // but we don't want to waste CPU time recreating the parser on each call
  // No cleaning is done (we assume that the number of threads is negligible compared to the memory size)
  const { __parser_pool: pool } = MyArray.prototype.grammar;
  for (let i = 0; i < pool.length; i++) {
    const parser = pool[i];
    if (parser.busy++ == 0) {
      try { return parser.parse(template, ...variables); }
      finally { parser.busy = 0; }
    }
    if (i == pool.length) pool.push(MyArray.prototype.grammar.__makeSemantics());
  }
}

MyArray.prototype.linspace = function (start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  const { multiply, arange, add, __as_number } = MyArray.prototype;
  start = __as_number(start);
  stop = __as_number(stop);
  let n = (num - (endpoint ? 1 : 0))
  let arr = add(multiply(arange(num), (stop - start) / n), start);
  return arr;
}

MyArray.prototype.geomspace = function (start, stop, num = 50, endpoint = true) {
  ({ stop, num, endpoint } = Object.assign({ stop, num, endpoint }, this));
  const { exp, log, linspace } = MyArray.prototype;
  start = log(start);
  stop = log(stop);
  return exp(linspace(start, stop, num, endpoint));
}


MyArray.prototype.reshape = function (A, shape, ...more_shape) {
  const { __parse_shape, __as_number } = MyArray.prototype;
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
  return new MyArray(A.flat, shape, A.dtype);
};


module.exports = MyArray;