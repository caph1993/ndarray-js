//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./globals").GLOBALS;

/**
 * @param {NDArray} arr
 * @param {GeneralSliceSpec[]} slicesSpec
 */
function slice(arr, ...slicesSpec) {
  // This can result either in a value, a view, a copy.
  // The index is simple if there are only ranges, numbers, ":" and at most one "..."
  // If index is simple, don't call ".indices" and make view
  // If index is advanced, get indices and make copy
  // @ts-ignore
  if (this !== NDArray.prototype) return NDArray.prototype.slice(this, ...arguments);
  if (!(arr instanceof NDArray)) throw new Error(`Expected NDArray. Found ${typeof arr}: ${arr}`);
  const axesIndex = AxesIndex.prototype.parse(arr.shape, slicesSpec);
  if (axesIndex.isConstant) {
    let [index] = axesIndex.indices;
    return arr.flat[index];
  } else if (axesIndex.isSimple) {
    const composition = __compose_simpleIndexes(arr._simpleIndexes, axesIndex);
    const out = new NDArray(arr._flat, axesIndex.shape, arr.dtype);
    out._simpleIndexes = composition;
    return out;
  } else {
    const src_flat = arr.flat;
    const flat = axesIndex.indices.map(i => src_flat[i]);
    return new NDArray(flat, axesIndex.shape, arr.dtype);
  }
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
    let indices = __slices_to_indices(this.internalShape, this.__slices);
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


/**
 * 
 * @param {AxesIndex|null} first 
 * @param {AxesIndex} second 
 * @returns {AxesIndex}
 */
function __compose_simpleIndexes(first, second) {
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
function __slices_to_indices(shape, slices) {
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


// =========================================
//     Slicing
// =========================================

module.exports.__parse_sliceRange = function __parse_sliceRange(axis_size, { start, stop, step }) {
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
    let arr = NDArray.prototype.asarray(sliceSpec)
    let indices;
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
  } //@ts-ignore
  else if (sliceSpec.isRange) {
    //@ts-ignore
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

module.exports = {
  slice, AxesIndex, AxisIndex, __slices_to_indices,
} 