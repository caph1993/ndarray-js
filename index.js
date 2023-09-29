//@ts-check
var ohm = require('ohm-js');

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
  /** @type {AxesIndex|null} */ _simpleIndexes;

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
  toString() {
    return NDArray.prototype.humanReadable(this);
  }

  /**
   * @param {GeneralSliceSpec[]} slicesSpec
   */
  slice(...slicesSpec) {
    // This can result either in a value, a view, a copy.
    // The index is simple if there are only ranges, numbers, ":" and at most one "..."
    // If index is simple, don't call ".indices" and make view
    // If index is advanced, get indices and make copy
    const axesIndex = AxesIndex.prototype.parse(this.shape, slicesSpec);
    if (axesIndex.isConstant) {
      let [index] = axesIndex.indices;
      return this.flat[index];
    } else if (axesIndex.isSimple) {
      const composition = NDArray.prototype.__compose_simpleIndexes(this._simpleIndexes, axesIndex);
      const out = new NDArray(this._flat, axesIndex.shape, this.dtype);
      out._simpleIndexes = composition;
      return out;
    } else {
      const src_flat = this.flat;
      const flat = axesIndex.indices.map(i => src_flat[i]);
      return new NDArray(flat, axesIndex.shape, this.dtype);
    }
  }
}


/**
 * @param {NDArray} arr 
 * @returns {string}
 */
NDArray.prototype.humanReadable = function (arr) {
  if (arr.shape.length == 0) return arr.flat[0].toString();
  let budgets = arr.shape.map(_ => 1);
  let lBudget = 30;
  for (let i = 0; i < arr.shape.length; i++) {
    let before = budgets[i];
    budgets[i] = Math.min(arr.shape[i], lBudget);
    if (budgets[i] > before) lBudget = Math.floor(lBudget / (budgets[i] - before));
  }
  let rBudget = 30;
  for (let i = arr.shape.length - 1; i >= 0; i--) {
    let before = budgets[i];
    budgets[i] = Math.min(arr.shape[i], rBudget);
    if (budgets[i] > before) rBudget = Math.floor(rBudget / (budgets[i] - before));
  }
  function simplify(list, depth = 0) {
    if (depth == arr.shape.length) return list;
    if (2 * budgets[depth] >= list.length) {
      return list.map(l => simplify(l, depth + 1));
    }
    const left = list.slice(0, budgets[depth]).map(l => simplify(l, depth + 1));
    const right = list.slice(-budgets[depth]).map(l => simplify(l, depth + 1));
    return [...left, '...', ...right];
  }
  let rLimit = arr.shape.length - 1;
  while (rLimit > 0 && arr.shape[rLimit] == 1) {
    rLimit--;
  }
  if (arr.dtype == Number) arr = NDArray.prototype.round(arr, 2);
  let list = NDArray.prototype.toJS(arr);

  function str(list, indent = 0, depth = 0) {
    if (list == '...' || depth >= arr.shape.length) return list;
    if (depth == arr.shape.length - 1) return `[${list.join(', ')}]`;
    let sep = depth >= rLimit ? ' ' : '\n' + ' '.repeat(indent + 1);
    const out = [];
    for (let i = 0; i < list.length; i++) {
      let s = str(list[i], indent + 1, depth + 1) + ',';
      out.push(i < list.length - 1 ? s : s.slice(0, -1));
    }
    return `[${out.join(sep)}]`;
  }

  let prefix = 'Arr';
  let suffix = `, shape=(${arr.shape}), dtype=${arr.dtype.name}`;
  let out = str(simplify(list), 1 + prefix.length);
  function alignColumns(inputString, delimiter = ',') {
    // Split the input string into rows
    const rows = inputString.split('\n');
    // Initialize an array to store the maximum width of each column
    const columnWidths = Array(rows[0].split(delimiter).length).fill(0);
    // Find the maximum width for each column
    for (const row of rows) {
      let columns = row.split(delimiter);
      for (let i = 0; i < columns.length; i++) {
        columnWidths[i] = Math.max(columnWidths[i], columns[i].trim().length);
      }
    }
    // Build the formatted outputs
    let formattedString = '';
    for (const row of rows) {
      let columns = row.split(delimiter);
      columns = columns.map((s, i) => i == columns.length - 1 ? s : s + delimiter);
      for (let i = 0; i < columns.length; i++) {
        const column = columns[i].trim();
        formattedString += column.padStart(columnWidths[i] + 1).padEnd(columnWidths[i] + 2); // Add 1 for padding
      }
      formattedString += '\n';
    }
    return formattedString;
  }
  out = out.replace(/.*?(\n|$)/g, (match) => {
    // Split with a newline every 0 characters, but only after a comma,
    return match.replace(/(.{60,}?,)/g, '$1\n');
  }).replace(/\n+/g, '\n');
  out = alignColumns(`${prefix}(${out}`).trim();
  out = `${out}${suffix})`;
  return out;

}


/** @typedef {null|{size:Number, ranges:{refSize:number, range:number|[number,number,number]|null}[], indices:null|number[]}} SimpleIndexes */

class AxesIndex {
  /**
   * @param {AxisIndex[]} axisIndexes
   */
  constructor(apparentShape, internalShape, axisIndexes) {
    this.shape = apparentShape;
    this.internalShape = internalShape;
    this.axisIndexes = axisIndexes;
    this._indices = null;
    this._size = null;
    this.isSimple = this.axisIndexes.map(idx => idx.isSimple).reduce((a, b) => a && b, true);
    this.isConstant = this.axisIndexes.map(idx => idx.isConstant).reduce((a, b) => a && b, true);
  }
  get indices() {
    if (this._indices) return this._indices;
    let indices = NDArray.prototype.__slices_to_indices(this.internalShape, this.__slices);
    return this._indices = indices;
  }
  get __slices() {
    return this.axisIndexes.map(idx => idx.indices);
  }
  get size() {
    if (this._size) return this._size;
    return this._size = this.axisIndexes.map(idx => idx.size).reduce((a, b) => a * b, 0);
  }
}




/** @typedef {{size:Number, ranges:{refSize:number, range:number|[number,number,number]|null}[], indices:null|number[]}} SimpleIndex */

NDArray.prototype.__simpleIndex_to_slices = function (/** @type {SimpleIndexes} */ simpleIndex) {
  if (!simpleIndex) throw new Error(`This function can only be called on views`);
  if (simpleIndex.indices) return simpleIndex.indices;
  const { ranges } = simpleIndex;
  const slices = ranges.map(({ refSize, range }) => {
    let [start, stop, step] = range == null ? [0, refSize, 1] : typeof range == "number" ? [range, range + 1, 1] : range;
    let indices = [];
    for (let i = start; i < stop; i += step) indices.push(i);
    return indices;
  });
  const indices = NDArray.prototype.__slices_to_indices(this.shape, slices);
  return simpleIndex.indices = indices;
}

/**
 * 
 * @param {AxesIndex|null} first 
 * @param {AxesIndex} second 
 * @returns {AxesIndex}
 */
NDArray.prototype.__compose_simpleIndexes = function (first, second) {
  if (first == null) return second;
  const axisIndexes = [];
  // console.log({ first, second })
  let j = 0;
  for (let i = 0; i < first.axisIndexes.length; i++) {
    let { spec: specA } = first.axisIndexes[i];
    let { spec: specB } = second.axisIndexes[j];
    if (specA.type == "array") throw new Error(`Expected simple index. Found advanced: ${specA.type}`);
    if (specB.type == "array") throw new Error(`Expected simple index. Found advanced: ${specB.type}`);
    let /**@type {AxisIndexSpec} */ spec;
    if (specA.type == "number") spec = specA
    else {
      j++;
      if (specA.type == ":") spec = specB;
      else if (specB.type == ":") spec = specA;
      else {
        let { start: startA, step: stepA, nSteps: nStepsA } = specA.range;
        if (specB.type == "number") {
          let { index } = specB;
          if (index < 0) index = nStepsA + index;
          if (index < 0 || index >= nStepsA) throw new Error(`Index ${index} out of bounds [0..${nStepsA})`);
          index = startA + index * stepA;
          spec = { type: "number", index };
        } else {
          let { start: startB, step: stepB, nSteps: nStepsB } = specB.range;
          let sub = AxisIndex.prototype.parse_range(nStepsA, startB, startB + nStepsB * stepB, stepB);
          let step = sub.step * stepA;
          let start = startA + sub.start * step;
          let nSteps = sub.nSteps;
          spec = { type: "range", range: { start, step, nSteps } };
        }
      }
    }
    axisIndexes.push(new AxisIndex(spec));
  }
  if (j < second.axisIndexes.length) throw new Error(`Index too long. Expected ${j} axes. Found ${second.axisIndexes.length}`)
  const apparentShape = second.shape;
  const internalShape = first.internalShape;
  return new AxesIndex(apparentShape, internalShape, axisIndexes);
}


/**
 * Computes the indices wr to shape of the cartesian products of the slices.
 * We have shape.length==slices.length, and the elements in slices[axis] are
 * integers between 0 and shape[axis]-1
 * @param {number[]} shape 
 * @param {number[][]} slices 
 * @returns {number[]}
 */
NDArray.prototype.__slices_to_indices = function (shape, slices) {
  const { __shape_shifts } = NDArray.prototype;
  for (let slice of slices) if (slice.length == 0) return [];
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
      let { indices } = AxesIndex.prototype.parse(tgt.shape, slicesSpec);
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


// ==============================
//    array instantiation and reshaping
// ==============================

NDArray.prototype.asarray = function (A) {
  if (A instanceof NDArray) return A;
  else return NDArray.prototype.fromJS(A);
}
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


/**@typedef {{type:':', size:number}|{type:'number', index:number}|{type:'range', range:{start:number, step:number, nSteps:number}}|{type:'array', indices:number[]}} AxisIndexSpec */

class AxisIndex {
  /**
   * Invariant: Immutable
   * @param {AxisIndexSpec} spec
   */
  constructor(spec) {
    this.spec = spec;
    this._indices = null;
    this.isSimple = (this.spec.type != "array");
    this.isConstant = (this.spec.type == "number");
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



/**@typedef {':'|number|{isRange:boolean, start:null|number, stop:null|number, step:null|number}|NDArray|number[]} SliceSpec */
/**@typedef {':'|'...'|'None'|SliceSpec} GeneralSliceSpec */


/**
 * We are reading `slicesSpec` and `shape` in parallel, in the reading direction readDir.
 * With respect to `shape` we are at the given `axis`.
 * With respect to `slicesSpec`, we found `sliceSpec`, which we should process.
 * @param {SliceSpec} sliceSpec
 */
AxisIndex.prototype.parse = function (sliceSpec, size) {
  /**
   * 
   * span (virtual shape) matches shape unless there are boolean masks spanning
   * over several axes/dimensions.
   * For example, in `np.ones((2,3,4,5))[:, np.arange(12).reshape((3,4))>5, 1]`,
   * the boolean mask is spanning over axes 1 and 2. In this case, the output should
   * merge these axes, resulting in an a vShape of (2, 12, 5).
   * The boolean mask is then converted to indices in the flattened merged axis.
   */
  /**@type {AxisIndexSpec} */
  let spec;
  let span = 1;

  if (sliceSpec == ':' || sliceSpec === undefined) {
    spec = { type: ':', size: size };
  }
  else if (typeof sliceSpec === "number") {
    let index = sliceSpec
    if (index < 0) index = size + index;
    if (index < 0 || index >= size) throw new Error(`Index ${index} out of bounds [0..${size})`);
    spec = { type: 'number', index };
  }
  else if (sliceSpec instanceof NDArray || Array.isArray(sliceSpec)) {
    let indices;
    let arr = NDArray.prototype.asarray(sliceSpec)
    if (arr.dtype == Number) {
      // Array of indices
      if (arr.shape.length > 1) throw new Error(
        `Expected 1D array of indices or nD array of booleans. ` +
        `Found shape=${arr.shape} and dtype=${arr.dtype}`
      );
      indices = arr.flat;
    } else {
      // Boolean mask
      indices = [];
      arr.flat.forEach((if_value, i) => if_value && indices.push(i));
      // Next lines: the boolean mask spans over more than 1 axis
      span = Math.max(1, arr.shape.length);
      // Multiply the (possibly inverted) interval
    }
    spec = { type: 'array', indices };
  }
  else if (sliceSpec.isRange) {
    let { start, stop, step } = sliceSpec;
    const range = AxisIndex.prototype.parse_range(size, start, stop, step);
    if (range.start == 0 && range.nSteps == size && range.step == 1) {
      // Small optimization: all of these are just ":": ["::","0::1", ":axisSize:", etc.]
      spec = { type: ':', size: size };
    } else {
      spec = { type: 'range', range };
    }
  }
  else throw new Error(`Unknown index type. Found ${typeof sliceSpec}: ${sliceSpec}`);

  const axisIndex = new AxisIndex(spec);
  return { axisIndex, span };
}


/**
 * @param {GeneralSliceSpec[]} slicesSpec
 * @returns {AxesIndex}
 */
AxesIndex.prototype.parse = function (shape, slicesSpec) {
  const buffers = {
    axisIndexes: /**@type {AxisIndex[]}*/([]),
    apparentShape: /**@type {number[]}*/([]),
    internalShape: /**@type {number[]}*/([]),
  }
  let /**@type {1|-1}*/ readDir = 1;
  const reversedAfter = { axisIndexes: NaN, apparentShape: NaN, internalShape: NaN };
  let axis = 0, j = 0, remainingAxes = shape.length;
  while (remainingAxes > 0) {
    let generalSpec = slicesSpec[j];
    //@ts-ignore
    slicesSpec[j] = undefined; // For ellipsis to avoid reading twice
    j += readDir;
    if (generalSpec == "None") {
      buffers.apparentShape.push(1);
      continue;
    } else if (generalSpec == "...") {
      if (readDir == -1) throw new Error(`Index can only have a single ellipsis ('...')`)
      readDir = -1;
      for (let key in reversedAfter) reversedAfter[key] = buffers[key].length;
      j = slicesSpec.length - 1;
      axis = shape.length - 1;
      continue;
    }
    const { axisIndex, span } = AxisIndex.prototype.parse(generalSpec, shape[axis]);
    // Advance the axis cursor span axes in readDir and compute the total size of consumed axes
    remainingAxes -= span;
    let refSize = 1;
    for (let i = 0; i < span; i++) {
      if (axis < 0 || axis >= shape.length) throw new Error(`Index spans over more dimensions than available`);
      refSize *= shape[axis];
      axis += readDir;
    }
    buffers.axisIndexes.push(axisIndex);
    if (axisIndex.spec.type != "number") buffers.apparentShape.push(axisIndex.size);
    buffers.internalShape.push(refSize);
  }
  if (readDir == -1) { // reverse the right to left elements
    for (let key in buffers) buffers[key].splice(0, reversedAfter[key]).concat(buffers[key].reverse());
  }
  return new AxesIndex(buffers.apparentShape, buffers.internalShape, buffers.axisIndexes);
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



NDArray.prototype.grammar = {}
NDArray.prototype.grammar.grammar = String.raw`
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

NDArray.prototype.grammar.ohmGrammar = ohm.grammar(NDArray.prototype.grammar.grammar);


NDArray.prototype.grammar.__makeSemantics = () => {

  const semanticVariables = [];
  const semantics = {
    Instruction_sliceAssignment($tgt, _open, $slicesSpec, _close, $symbol, $src) {
      const _tgt = $tgt.parse();
      const _src = $src.parse();
      const symbol = $symbol.sourceString;
      const slicesSpec = $slicesSpec.parse();
      const { asarray, op_assign, toJS: to_js_array } = NDArray.prototype;
      let tgt = asarray(_tgt);
      op_assign[symbol](_tgt, _src, slicesSpec);
      if (tgt !== _tgt) {
      // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
        tgt = to_js_array(tgt);
        while (_tgt.length) _tgt.pop();
        // @ts-ignore
        _tgt.push(..._tgt);
      }
      return null;
    },
    Instruction_expression($arr) {
      const arr = $arr.parse();
      if (typeof arr === "number") return arr;
      if (Array.isArray(arr)) return arr;
      return NDArray.prototype.__number_collapse(arr);
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
      return arr.slice(...slicesSpec);
    },
    SliceTerm_constant($x) {
      return $x.sourceString;
    },
    Arr_call($name, $names, _, $callArgs) {
      let name = $name.sourceString + $names.sourceString;
      if (name.slice(0, 3) == "np.") name = name.slice(3);
      const func = NDArray.prototype[name];
      if (func === undefined) throw new Error(`Unrecognized function ${name}`)
      const { args, kwArgs } = $callArgs.parse();
      return func.bind(kwArgs)(...args);
    },
    Arr_method($arr, _dot, $name, $callArgs) {
      let arr = $arr.parse();
      let name = $name.sourceString;
      if (name.slice(0, 3) == "np.") name = name.slice(3);
      const func = NDArray.prototype[name];
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
      for (let i in list) if (list[i] instanceof NDArray) list[i] = NDArray.prototype.toJS(list[i]);
      return list;
    },
    _terminal() { return null; },
  };

  function BinaryOperation($A, $symbol, $B) {
    const A = $A.parse();
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "" && A === null) return B;
    return NDArray.prototype.opx[symbol](A, B);
  }
  function UnaryOperation(_, $symbol, $B) {
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "") return B;
    switch (symbol) {
      case "+": return B;
      case "-": return NDArray.prototype.multiply(-1, B);
      case "^": return NDArray.prototype.boolean_not(B);
    }
    throw new Error(`Programming Error: ${symbol}`);
  }

  const { ohmGrammar } = NDArray.prototype.grammar;

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


NDArray.prototype.grammar.__parser_pool = [NDArray.prototype.grammar.__makeSemantics()];

/**
 * @param {TemplateStringsArray} template
 * @param {any[]} variables
 */
NDArray.prototype.grammar.parse = function (template, ...variables) {
  // Thread control, because the parser depends on semanticVariables,
  // but we don't want to waste CPU time recreating the parser on each call
  // No cleaning is done (we assume that the number of threads is negligible compared to the memory size)
  const { __parser_pool: pool } = NDArray.prototype.grammar;
  for (let i = 0; i < pool.length; i++) {
    const parser = pool[i];
    if (parser.busy++ == 0) {
      try { return parser.parse(template, ...variables); }
      finally { parser.busy = 0; }
    }
    if (i == pool.length) pool.push(NDArray.prototype.grammar.__makeSemantics());
  }
}
/**
 * @param {TemplateStringsArray} template
 * @param {any[]} variables
 */
NDArray.prototype.grammar.parseJS = function (template, ...variables) {
  const arr = NDArray.prototype.grammar.parse(template, ...variables)
  return NDArray.prototype.toJS(arr);
}

NDArray.prototype.parse = NDArray.prototype.grammar.parse;
NDArray.prototype.parseJS = NDArray.prototype.grammar.parseJS;


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



module.exports = NDArray;