//@ts-check
import { isarray, asarray, new_NDArray, shape_shifts } from './basic';
import type NDArray from "../NDArray";


export type RangeSpec = string;
export type indexSpec = ':' | number | RangeSpec | NDArray | number[];
export type GeneralIndexSpec = ':' | '...' | 'None' | null | indexSpec;
export type Where = null | GeneralIndexSpec[];



export function index(arr: NDArray, where: Where) {
  // This can result either in a value, a view, a copy.
  // The index is simple if there are only ranges, numbers, ":" and at most one "..."
  // If index is simple, don't call ".indices" and make view
  // If index is advanced, get indices and make copy
  let copy = false;
  if (!(isarray(arr))) throw new Error(`Expected NDArray. Found ${typeof arr}: ${arr}`);
  const axesIndex = AxesIndex.prototype.parse(arr.shape, where);
  if (axesIndex.isConstant) {
    let [index] = axesIndex.indices;
    return arr.flat[index];
  } else if (axesIndex.isSimple) {
    const composition = __compose_simpleIndexes(arr._simpleIndexes, axesIndex);
    const out = new_NDArray(arr._flat, axesIndex.shape, arr.dtype);
    out._simpleIndexes = composition;
    if (arr['__warnAssignments']) out['__warnAssignments'] = true;
    return copy ? out.copy() : out;
  } else {
    const src_flat = arr.flat;
    const flat = axesIndex.indices.map(i => src_flat[i]);
    const out = new_NDArray(flat, axesIndex.shape, arr.dtype);
    if (!copy) out['__warnAssignments'] = true;
    return out;
  }
}


// type SimpleIndexes = null | { size: number, ranges: { refSize: number, range: null | number | [number, number, number] }[], indices: null | number[] };


export class AxesIndex {
  shape: any;
  internalShape: any;
  axisIndexes: AxisIndex[];
  private _indices: null | number[];
  private _size: null | number;
  isSimple: boolean;
  isConstant: boolean;
  parse: (shape: number[], where: Where) => AxesIndex;
  /**
   * @param {AxisIndex[]} axisIndexes
   */
  constructor(apparentShape, internalShape, axisIndexes: AxisIndex[]) {
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



function __compose_simpleIndexes(first: AxesIndex | null, second: AxesIndex): AxesIndex {
  if (first == null) return second;
  const axisIndexes = [];
  // console.log({ first, second })
  let j = 0;
  for (let i = 0; i < first.axisIndexes.length; i++) {
    let { spec: specA } = first.axisIndexes[i];
    let { spec: specB } = second.axisIndexes[j];
    if (specA.type == "array") throw new Error(`Expected simple index. Found advanced: ${specA.type}`);
    if (specB.type == "array") throw new Error(`Expected simple index. Found advanced: ${specB.type}`);
    let /**@type {AxisIndexSpec} */ spec: AxisIndexSpec;
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
function __slices_to_indices(shape: number[], slices: number[][]): number[] {
  for (let slice of slices) if (slice.length == 0) return [];
  const shifts = shape_shifts(shape);
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

export function __parse_sliceRange(axis_size, { start, stop, step }) {
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


export type AxisIndexSpec = { type: ':', size: number } | { type: 'number', index: number } | { type: 'range', range: { start: number, step: number, nSteps: number } } | { type: 'array', indices: number[] };

export class AxisIndex {
  spec: AxisIndexSpec;
  private _indices: null;
  isSimple: boolean;
  isConstant: boolean;
  parse: (indexSpec: indexSpec | undefined, size: number) => { axisIndex: AxisIndex; span: number; };
  parse_range: (size: number, start?: number, stop?: number, step?: number) => { start: number; step: number; nSteps: number; };
  parse_range_spec: (rangeString: string) => { start: number; stop: number; step: number; };

  /**
   * Invariant: Immutable
   * @param {AxisIndexSpec} spec
   */
  constructor(spec: AxisIndexSpec) {
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


AxisIndex.prototype.parse_range = function (size: number, start: number | null = null, stop: number | null = null, step: number | null = null) {
  if (step == null) step = 1;
  else if (step == 0) throw new Error(`Index specification error. Step must be different from zero.`);
  /**
   *  @param {number|null} i @param {number} ifNull @param {number} min @param {number} max */
  const parse = (i: number | null, ifNull: number, min: number, max: number) => {
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

/**
 * 
 * @param {string} rangeString
 * @returns {{start:(number|null), stop:(number|null), step:(number|null)}}
 */
AxisIndex.prototype.parse_range_spec = function (rangeString: string): { start: (number | null); stop: (number | null); step: (number | null); } {
  const numbers = rangeString.split(':').map(s => {
    s = s.trim();
    if (s == "") return null;
    const n = parseInt(s)
    if (!Number.isInteger(n)) throw new Error(`Wrong input. Slice index unrecognized: ${s}`);
    return n;
  });
  if (numbers.length == 0) throw new Error('Unexpected empty index. Expected colons.');
  if (numbers.length > 3) throw new Error(`Too many colons in index ${rangeString}`);
  let [start, stop, step, ..._] = [...numbers, null, null, null];
  return { start, stop, step };
}




/**
 * We are reading `indexSpec` and `shape` in parallel, in the reading direction readDir.
 * With respect to `shape` we are at the given `axis`.
 * With respect to `indexSpec`, we found `indexSpec`, which we should process.
 * @param {indexSpec|undefined} indexSpec
 */
AxisIndex.prototype.parse = function (indexSpec: indexSpec | undefined, size) {
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
  let spec: AxisIndexSpec;
  let span = 1;

  if (indexSpec == ':' || indexSpec === undefined) {
    spec = { type: ':', size: size };
  }
  else if (typeof indexSpec === "number") {
    let index = indexSpec
    if (index < 0) index = size + index;
    if (index < 0 || index >= size) throw new Error(`Index ${index} out of bounds [0..${size})`);
    spec = { type: 'number', index };
  }
  else if (isarray(indexSpec) || Array.isArray(indexSpec)) {
    let arr = asarray(indexSpec);
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
  }
  else if (typeof indexSpec == "string") {
    let { start, stop, step } = AxisIndex.prototype.parse_range_spec(indexSpec);
    const range = AxisIndex.prototype.parse_range(size, start, stop, step);
    if (range.start == 0 && range.nSteps == size && range.step == 1) {
      // Small optimization: all of these are just ":": ["::","0::1", ":axisSize:", etc.]
      spec = { type: ':', size: size };
    } else {
      spec = { type: 'range', range };
    }
  }
  else throw new Error(`Unknown index type. Found ${typeof indexSpec}: ${indexSpec}`);

  const axisIndex = new AxisIndex(spec);
  return { axisIndex, span };
}


/**
 * @param {Where} where
 * @returns {AxesIndex}
 */
AxesIndex.prototype.parse = function (shape, where: Where): AxesIndex {
  /**@type {Array<GeneralIndexSpec|undefined>}*/
  const _where: Array<GeneralIndexSpec | undefined> = where == null ? [] : [...where];

  const buffers = {
    axisIndexes: /**@type {AxisIndex[]}*/([]),
    apparentShape: /**@type {number[]}*/([]),
    internalShape: /**@type {number[]}*/([]),
  }
  let readDir = 1;
  const reversedAfter = { axisIndexes: NaN, apparentShape: NaN, internalShape: NaN };
  let axis = 0, j = 0, remainingAxes = shape.length, remainingWhere = _where.length;
  while (remainingWhere > 0 || remainingAxes > 0) {
    let axisWhere = _where[j];
    if (remainingWhere > 0) {
      if (j < 0 || j >= _where.length) axisWhere = ":";
      remainingWhere--;
      //else _where[j] = undefined; // For ellipsis to avoid reading twice in opposite reading directions
      j += readDir;
      if (axisWhere == "None" || axisWhere === null) {
        buffers.apparentShape.push(1);
        continue;
      } else if (axisWhere == "...") {
        if (readDir == -1) throw new Error(`Index can only have a single ellipsis. Found index(${where})`)
        readDir = -1;
        for (let key in reversedAfter) reversedAfter[key] = buffers[key].length;
        j = _where.length - 1;
        axis = shape.length - 1;
        continue;
      }
    } else {
      axisWhere = ":"; // If there are no more axes, fill with ":"
    }
    // if (remainingAxes <= 0) throw Error(`Too many axes`);
    const { axisIndex, span } = AxisIndex.prototype.parse(axisWhere, shape[axis]);
    // Advance the axis cursor span axes in readDir and compute the total size of consumed axes
    remainingAxes -= span;
    let refSize = 1;
    for (let i = 0; i < span; i++) {
      if (axis < 0 || axis >= shape.length) throw new Error(`Index spans over more dimensions than available in shape [${shape}]: index(${where})`);
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
  const axesIndex = new AxesIndex(buffers.apparentShape, buffers.internalShape, buffers.axisIndexes)
  return axesIndex;
}
