//@ts-check
import { NDArray, isarray, asarray, shape_shifts } from "../NDArray";
import { dtype_is_boolean, dtype_is_integer, new_buffer } from '../dtypes';

export type RangeSpec = string;
export type IndexSpec = ':' | number | RangeSpec | number[] | NDArray;
export type GeneralIndexSpec = '...' | 'None' | null | boolean | IndexSpec;
export type Where = null | GeneralIndexSpec[];

export function index(arr: NDArray, where: Where) {
  // This can result either in a value, a view, a copy.
  // The index is simple if there are only ranges, numbers, ":" and at most one "..."
  // If index is simple, don't call ".indices" and make view
  // If index is advanced, get indices and make copy
  let copy = false;
  if (!(isarray(arr))) throw new Error(`Expected NDArray. Found ${typeof arr}: ${arr}`);
  const axesIndex = AxesIndex.prototype.parse(arr.shape, where);
  if (axesIndex.isSimple) {
    const composition = __compose_simpleIndexes(arr._simpleIndexes, axesIndex);
    if (composition.isConstant && composition.shape.length == 0) {
      // Full integer index: return a plain value
      const value = arr._flat[composition.indices[0]];
      return (arr.dtype === "bool" ? !!value : value) as any as NDArray;
    }
    const out = new NDArray(arr._flat, axesIndex.shape, arr.dtype);
    out._simpleIndexes = composition;
    if (arr['__warnAssignments']) out['__warnAssignments'] = true;
    return copy ? out.copy() : out;
  } else {
    const src_flat = arr.flat;
    const buffer = new_buffer(axesIndex.size, arr.dtype);
    const indices = axesIndex.indices;
    for (let i = 0; i < axesIndex.size; i++) buffer[i] = src_flat[indices[i]];
    const out = new NDArray(buffer, axesIndex.shape, arr.dtype);
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
    if (this._size !== null) return this._size;
    return this._size = this.axisIndexes.map(idx => idx.size).reduce((a, b) => a * b, 1);
  }
}



function __as_range(spec: AxisIndexSpec): { start: number, step: number, nSteps: number } {
  if (spec.type === ':') return { start: 0, step: 1, nSteps: spec.size };
  if (spec.type === 'range') return spec.range;
  if (spec.type === 'new') return { start: 0, step: 1, nSteps: 1 };
  if (spec.type === false) return { start: 0, step: 1, nSteps: 0 };
  throw new Error(`Expected simple range. Found ${spec.type}`);
}

function __compose_simpleIndexes(first: AxesIndex | null, second: AxesIndex): AxesIndex {
  // `second` was parsed against the apparent shape of `first`.
  if (first == null) return second;
  const axisIndexes = [];
  let j = 0;
  for (let i = 0; i < first.axisIndexes.length; i++) {
    const specA = first.axisIndexes[i].spec;
    if (specA.type == "array") throw new Error(`Expected simple index. Found advanced: ${specA.type}`);
    let spec: AxisIndexSpec;
    if (specA.type == "number") spec = specA; // axis already removed from the apparent shape
    else {
      if (j >= second.axisIndexes.length) throw new Error(`Programming error: index too short`);
      const specB = second.axisIndexes[j++].spec;
      if (specB.type == "array") throw new Error(`Expected simple index. Found advanced: ${specB.type}`);
      if (specB.type == ":") spec = specA;
      else if (specB.type === false) spec = specB;
      else if (specB.type == "new") spec = specB;
      else {
        const { start: startA, step: stepA, nSteps: nStepsA } = __as_range(specA);
        if (specB.type == "number") {
          const index = specB.index;
          if (index < 0 || index >= nStepsA) throw new Error(`Index ${index} out of bounds [0..${nStepsA})`);
          spec = { type: "number", index: startA + index * stepA };
        } else {
          const { start: startB, step: stepB, nSteps } = specB.range;
          spec = { type: "range", range: { start: startA + startB * stepA, step: stepA * stepB, nSteps } };
        }
      }
    }
    axisIndexes.push(new AxisIndex(spec));
  }
  if (j != second.axisIndexes.length) throw new Error(`Programming error: composed ${j} of ${second.axisIndexes.length} axes`);
  return new AxesIndex(second.shape, first.internalShape, axisIndexes);
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


export type AxisIndexSpec = { type: ':', size: number } | { type: 'number', index: number } | { type: 'range', range: { start: number, step: number, nSteps: number } } | { type: 'array', indices: number[] } | { type: 'new' } | { type: false };

export class AxisIndex {
  spec: AxisIndexSpec;
  private _indices: number[] | null;
  isSimple: boolean;
  isConstant: boolean;
  parse: (indexSpec: IndexSpec | undefined, size: number) => { axisIndex: AxisIndex; span: number; };
  parse_range: (size: number, start?: number, stop?: number, step?: number) => { start: number; step: number; nSteps: number; };
  parse_range_spec: (rangeString: string) => { start: number; stop: number; step: number; };

  /**
   * Invariant: Immutable
   */
  constructor(spec: AxisIndexSpec) {
    this.spec = spec;
    this._indices = null;
    this.isSimple = (this.spec.type != "array");
    this.isConstant = (this.spec.type == "number");
  }
  get indices() {
    if (this._indices) return this._indices;
    let indices: number[];
    // if (this.spec.type == 'false') indices = []; else
    if (this.spec.type === ':') indices = Array.from({ length: this.spec.size }, (_, i) => i);
    else if (this.spec.type === "number") indices = [this.spec.index];
    else if (this.spec.type === "array") indices = this.spec.indices;
    else if (this.spec.type === false) indices = [];
    else if (this.spec.type === "new") indices = [0];
    else if (this.spec.type == "range") {
      const { nSteps, step, start } = this.spec.range;
      indices = Array.from({ length: nSteps }, (_, i) => start + i * step);
    } else throw new Error(`Unknown spec type ${this.spec['type']}`);
    return this._indices = indices;
  }
  get size() {
    if (this.spec.type === ':') return this.spec.size;
    else if (this.spec.type === "number") return 1;
    else if (this.spec.type === "array") return this.spec.indices.length;
    else if (this.spec.type == "range") return this.spec.range.nSteps;
    else if (this.spec.type === false) return 0;
    else if (this.spec.type === "new") return 1;
    else throw new Error(`Unknown spec type ${this.spec['type']}`);
  }
}


AxisIndex.prototype.parse_range = function (size: number, start: number | null = null, stop: number | null = null, step: number | null = null) {
  // Same semantics as python's slice(start, stop, step).indices(size)
  if (step == null) step = 1;
  else if (step == 0) throw new Error(`Index specification error. Step must be different from zero.`);
  const clamp = (i: number | null, ifNull: number, lo: number, hi: number) => {
    if (i == null) return ifNull;
    if (i < 0) i += size;
    return Math.min(hi, Math.max(lo, i));
  }
  let nSteps;
  if (step > 0) {
    start = clamp(start, 0, 0, size);
    stop = clamp(stop, size, 0, size);
    nSteps = Math.max(0, Math.ceil((stop - start) / step));
  } else {
    start = clamp(start, size - 1, -1, size - 1);
    stop = clamp(stop, -1, -1, size - 1);
    nSteps = Math.max(0, Math.ceil((start - stop) / -step));
  }
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
 * @param {IndexSpec|undefined} indexSpec
 */
AxisIndex.prototype.parse = function (indexSpec: IndexSpec | undefined, size) {
  /**
   * 
   * span (virtual shape) matches shape unless there are boolean masks spanning
   * over several axes/dimensions.
   * For example, in `np.ones((2,3,4,5))[:, np.arange(12).reshape((3,4))>5, 1]`,
   * the boolean mask is spanning over axes 1 and 2. In this case, the output should
   * merge these axes, resulting in an a vShape of (2, 12, 5).
   * The boolean mask is then converted to indices in the flattened merged axis.
   */
  let spec: AxisIndexSpec;
  let span = 1;

  if (isarray(indexSpec) && indexSpec.shape.length == 0) {
    // Unwrap scalar
    //@ts-ignore
    indexSpec = indexSpec.item();
  }

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
    if (dtype_is_boolean(arr.dtype)) {
      // Boolean mask
      indices = [];
      arr.flat.forEach((if_value, i) => if_value && indices.push(i));
      // Next lines: the boolean mask spans over more than 1 axis
      span = Math.max(1, arr.shape.length);
      // Multiply the (possibly inverted) interval
    } else {
      // Array of indices (negative values count from the end)
      const raw = arr.flat;
      if (!dtype_is_integer(arr.dtype) && arr.size > 0) {
        for (let i = 0; i < raw.length; i++) if (!Number.isInteger(raw[i])) throw new Error(
          `Expected 1D array of indices or nD array of booleans. ` +
          `Found shape=${arr.shape} and dtype=${arr.dtype}`
        );
      }
      indices = new Array(raw.length);
      for (let i = 0; i < raw.length; i++) {
        let v = +raw[i];
        if (v < 0) v += size;
        if (v < 0 || v >= size) throw new Error(`Index ${raw[i]} out of bounds [0..${size})`);
        indices[i] = v;
      }
    }
    spec = { type: "array", indices };
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
  // else if (typeof indexSpec == "boolean" && indexSpec === false) spec = { type: 'false' };
  else throw new Error(`Unknown index type. Found ${typeof indexSpec}: ${indexSpec}`);

  const axisIndex = new AxisIndex(spec);
  return { axisIndex, span };
}


function __is_new_axis(item: any) {
  return item === null || item === "None" || item === true;
}

/** Number of array axes consumed by an item of the index tuple */
function __consumed_axes(item: any): number {
  if (__is_new_axis(item) || item === false) return 0;
  if (isarray(item)) {
    if (item.shape.length == 0) return typeof item.item() === "boolean" ? 0 : 1;
    return dtype_is_boolean(item.dtype) ? item.shape.length : 1;
  }
  if (Array.isArray(item)) {
    const arr = asarray(item);
    return dtype_is_boolean(arr.dtype) ? Math.max(1, arr.shape.length) : 1;
  }
  return 1;
}

AxesIndex.prototype.parse = function (shape: number[], where: Where): AxesIndex {
  const items: any[] = where == null ? [] : where.map(w => (isarray(w) && w.shape.length == 0) ? w.item() : w);
  const ellipsis = items.indexOf("...");
  if (ellipsis != items.lastIndexOf("...")) throw new Error(`Index can only have a single ellipsis. Found index(${where})`);
  const consumed = items.reduce((n, item) => n + (item === "..." ? 0 : __consumed_axes(item)), 0);
  const nFill = shape.length - consumed;
  if (nFill < 0) throw new Error(`Too many indices for array: array is ${shape.length}-dimensional, but ${consumed} were indexed`);
  const fill = Array.from({ length: nFill }, () => ":");
  if (ellipsis >= 0) items.splice(ellipsis, 1, ...fill);
  else items.push(...fill);

  const axisIndexes: AxisIndex[] = [];
  const apparentShape: number[] = [];
  const internalShape: number[] = [];
  let axis = 0;
  for (const item of items) {
    let axisIndex: AxisIndex;
    let refSize = 1;
    if (__is_new_axis(item)) axisIndex = new AxisIndex({ type: 'new' });
    else if (item === false) axisIndex = new AxisIndex({ type: false });
    else {
      const span = __consumed_axes(item);
      if (axis + span > shape.length) throw new Error(`Index where=${JSON.stringify(where)} spans over more dimensions than available in shape=${JSON.stringify(shape)}`);
      if (span > 1 || (span == 1 && (isarray(item) || Array.isArray(item)) && dtype_is_boolean(asarray(item).dtype))) {
        const maskShape = asarray(item).shape;
        for (let i = 0; i < span; i++) if (maskShape[i] !== shape[axis + i]) {
          throw new Error(`Boolean index of shape [${maskShape}] does not match array axes [${shape.slice(axis, axis + span)}]`);
        }
      }
      axisIndex = AxisIndex.prototype.parse(item, shape[axis]).axisIndex;
      for (let i = 0; i < span; i++) refSize *= shape[axis + i];
      axis += span;
    }
    axisIndexes.push(axisIndex);
    if (axisIndex.spec.type != "number") apparentShape.push(axisIndex.size);
    internalShape.push(refSize);
  }
  return new AxesIndex(apparentShape, internalShape, axisIndexes);
}


NDArray.prototype.index = function (...where: GeneralIndexSpec[]) {
  return index(this, where);
}