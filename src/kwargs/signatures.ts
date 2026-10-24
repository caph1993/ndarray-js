//@ts-check
/**
 * Keyword-argument signatures (see `signature` in kwargs.ts) for the functions registered in index.ts.
 * Each signature lists the parameters in numpy's positional order with their default values
 * (`undefined` = required). Every function accepts positional arguments, an object with named
 * arguments, or a mixture of both.
 */
import { signature, frequently_used_parsers as P, Parser } from "./kwargs";
import * as math from "../modules/math-functions";
import * as stats from "../modules/statistics";
import * as sets from "../modules/sets";
import * as indexing from "../modules/indexing";
import * as transform from "../array/transform";
import * as elementwise from "../array/elementwise";
import * as operators from "../array/operators";
import * as reduce from "../array/reduce";
import { asarray } from "../NDArray";

const asarray_if_not_null = P.asarray_or_null;
const check_mode = (kwargs: any) => {
  if (!['full', 'same', 'valid'].includes(kwargs.mode)) throw new Error(`mode must be one of 'full', 'same', or 'valid' (got ${JSON.stringify(kwargs.mode)})`);
};
const boolean = (...keys: string[]): Parser => (kwargs: any) => { for (const k of keys) kwargs[k] = !!kwargs[k]; };
const axis_or_null: Parser = (kwargs: any) => {
  const ax = kwargs.axis;
  if (ax != null && !(typeof ax === "number" && Number.isInteger(ax))) throw new Error(`Invalid "axis" argument. Expected integer or null. Found ${JSON.stringify(ax)}`);
};
const axis_int: Parser = (kwargs: any) => {
  if (typeof kwargs.axis !== "number" || !Number.isInteger(kwargs.axis)) throw new Error(`Invalid "axis" argument. Expected integer. Found ${JSON.stringify(kwargs.axis)}`);
};

// ==============================
//   Unary / binary with two outputs
// ==============================
export const a_to_pair = signature<(a: import("../NDArray").NDArray) => import("../NDArray").NDArray[]>()(
  [["a", undefined]] as const, [P.asarray('a')]);

// ==============================
//   Elementwise extras
// ==============================
export const nan_to_num = signature<typeof elementwise.nan_to_num>()(
  [["a", undefined], ["nan", 0.0], ["posinf", null], ["neginf", null], ["out", null]] as const,
  [P.asarray('a'), P.isarray_or_null('out')]);

export const isclose = signature<typeof operators.isclose>()(
  [["a", undefined], ["b", undefined], ["rtol", 1e-5], ["atol", 1e-8], ["equal_nan", false]] as const,
  [P.asarray('a'), P.asarray('b'), boolean('equal_nan')]);
export const allclose = signature<typeof operators.allclose>()(
  [["a", undefined], ["b", undefined], ["rtol", 1e-5], ["atol", 1e-8], ["equal_nan", false]] as const,
  [P.asarray('a'), P.asarray('b'), boolean('equal_nan')]);
export const array_equal = signature<typeof operators.array_equal>()(
  [["a1", undefined], ["a2", undefined], ["equal_nan", false]] as const,
  [P.asarray('a1'), P.asarray('a2'), boolean('equal_nan')]);
export const array_equiv = signature<typeof operators.array_equiv>()(
  [["a1", undefined], ["a2", undefined]] as const,
  [P.asarray('a1'), P.asarray('a2')]);

// ==============================
//   Cumulative, differences, integration
// ==============================
export const cumulative = signature<typeof math.cumsum>()(
  [["a", undefined], ["axis", null], ["dtype", null], ["out", null]] as const,
  [P.asarray('a'), axis_or_null, P.dtype_or_null, P.isarray_or_null('out')]);
export const cumulative_xxx = signature<typeof math.cumulative_sum>()(
  [["x", undefined], ["axis", null], ["dtype", null], ["out", null], ["include_initial", false]] as const,
  [P.asarray('x'), axis_or_null, P.dtype_or_null, P.isarray_or_null('out'), boolean('include_initial')]);
export const diff = signature<typeof math.diff>()(
  [["a", undefined], ["n", 1], ["axis", -1], ["prepend", null], ["append", null]] as const,
  [P.asarray('a'), P.is_integer_non_neg('n'), axis_int, asarray_if_not_null('prepend'), asarray_if_not_null('append')]);
export const ediff1d = signature<typeof math.ediff1d>()(
  [["ary", undefined], ["to_end", null], ["to_begin", null]] as const,
  [P.asarray('ary'), asarray_if_not_null('to_end'), asarray_if_not_null('to_begin')]);
export const gradient = signature<typeof math.gradient>()(
  [["f", undefined], ["...varargs", []], ["axis", null], ["edge_order", 1]] as const,
  [P.asarray('f'), P.axis_or_axes]);
export const cross = signature<typeof math.cross>()(
  [["a", undefined], ["b", undefined], ["axisa", -1], ["axisb", -1], ["axisc", -1], ["axis", null]] as const,
  [P.asarray('a'), P.asarray('b')]);
export const trapezoid = signature<typeof math.trapezoid>()(
  [["y", undefined], ["x", null], ["dx", 1.0], ["axis", -1]] as const,
  [P.asarray('y'), asarray_if_not_null('x'), axis_int]);
export const convolve = signature<typeof math.convolve>()(
  [["a", undefined], ["v", undefined], ["mode", 'full']] as const,
  [P.asarray('a'), P.asarray('v'), check_mode]);
export const correlate = signature<typeof math.correlate>()(
  [["a", undefined], ["v", undefined], ["mode", 'valid']] as const,
  [P.asarray('a'), P.asarray('v'), check_mode]);
export const interp = signature<typeof math.interp>()(
  [["x", undefined], ["xp", undefined], ["fp", undefined], ["left", null], ["right", null], ["period", null]] as const,
  [P.asarray('x'), P.asarray('xp'), P.asarray('fp')]);
export const unwrap = signature<typeof math.unwrap>()(
  [["p", undefined], ["discont", null], ["axis", -1], ["period", 2 * Math.PI]] as const,
  [P.asarray('p'), axis_int]);

// ==============================
//   Sets
// ==============================
export const unique = signature<typeof sets.unique>()(
  [["ar", undefined], ["return_index", false], ["return_inverse", false], ["return_counts", false], ["axis", null], ["equal_nan", true]] as const,
  [P.asarray('ar'), boolean('return_index', 'return_inverse', 'return_counts', 'equal_nan'), axis_or_null]);
export const intersect1d = signature<typeof sets.intersect1d>()(
  [["ar1", undefined], ["ar2", undefined], ["assume_unique", false], ["return_indices", false]] as const,
  [P.asarray('ar1'), P.asarray('ar2'), boolean('assume_unique', 'return_indices')]);
export const isin = signature<typeof sets.isin>()(
  [["element", undefined], ["test_elements", undefined], ["assume_unique", false], ["invert", false]] as const,
  [P.asarray('element'), P.asarray('test_elements'), boolean('assume_unique', 'invert')]);
export const setdiff1d = signature<typeof sets.setdiff1d>()(
  [["ar1", undefined], ["ar2", undefined], ["assume_unique", false]] as const,
  [P.asarray('ar1'), P.asarray('ar2'), boolean('assume_unique')]);
export const union1d = signature<typeof sets.union1d>()(
  [["ar1", undefined], ["ar2", undefined]] as const,
  [P.asarray('ar1'), P.asarray('ar2')]);
export const x_only = signature<typeof sets.unique_values>()(
  [["x", undefined]] as const, [P.asarray('x')]);

// ==============================
//   Sorting and searching
// ==============================
export const lexsort = signature<typeof transform.lexsort>()(
  [["keys", undefined], ["axis", -1]] as const, [axis_int]);
export const partition = signature<typeof transform.partition>()(
  [["a", undefined], ["kth", undefined], ["axis", -1]] as const,
  [P.asarray('a'), axis_int]);
export const a_only = signature<typeof indexing.argwhere>()(
  [["a", undefined]] as const, [P.asarray('a')]);
export const where = signature<(condition: any, x: any, y: any, out: any) => any>()(
  [["condition", undefined], ["x", null], ["y", null], ["out", null]] as const,
  [(kwargs: any) => {
    if ((kwargs.x === null) !== (kwargs.y === null)) throw new Error(`either both or neither of x and y should be given`);
    if (typeof kwargs.condition !== "boolean") kwargs.condition = asarray(kwargs.condition);
    if (kwargs.x !== null) { kwargs.x = asarray(kwargs.x); kwargs.y = asarray(kwargs.y); }
    if (kwargs.out != null) kwargs.out = asarray(kwargs.out);
  }]);
export const searchsorted = signature<typeof stats.searchsorted>()(
  [["a", undefined], ["v", undefined], ["side", 'left'], ["sorter", null]] as const,
  [P.asarray('a'), P.asarray('v'), asarray_if_not_null('sorter')]);
export const extract = signature<typeof indexing.extract>()(
  [["condition", undefined], ["arr", undefined]] as const,
  [P.asarray('condition'), P.asarray('arr')]);

// ==============================
//   Statistics
// ==============================
const quantile_parsers = [P.asarray('a'), P.asarray('q'), P.axis_or_axes, boolean('keepdims'), P.isarray_or_null('out')];
export const quantile = signature<typeof stats.quantile>()(
  [["a", undefined], ["q", undefined], ["axis", null], ["keepdims", false], ["method", 'linear'], ["out", null]] as const,
  quantile_parsers);
export const average = signature<typeof stats.average>()(
  [["a", undefined], ["axis", null], ["weights", null], ["returned", false], ["keepdims", false]] as const,
  [P.asarray('a'), P.axis_or_axes, asarray_if_not_null('weights'), boolean('returned', 'keepdims')]);
export const cov = signature<typeof stats.cov>()(
  [["m", undefined], ["y", null], ["rowvar", true], ["bias", false], ["ddof", null], ["fweights", null], ["aweights", null]] as const,
  [P.asarray('m'), asarray_if_not_null('y'), boolean('rowvar', 'bias'), asarray_if_not_null('fweights'), asarray_if_not_null('aweights')]);
export const corrcoef = signature<typeof stats.corrcoef>()(
  [["x", undefined], ["y", null], ["rowvar", true]] as const,
  [P.asarray('x'), asarray_if_not_null('y'), boolean('rowvar')]);
export const histogram = signature<typeof stats.histogram>()(
  [["a", undefined], ["bins", 10], ["range", null], ["density", null], ["weights", null]] as const,
  [P.asarray('a'), asarray_if_not_null('weights')]);
export const histogram_bin_edges = signature<typeof stats.histogram_bin_edges>()(
  [["a", undefined], ["bins", 10], ["range", null], ["weights", null]] as const,
  [P.asarray('a'), asarray_if_not_null('weights')]);
export const histogram2d = signature<typeof stats.histogram2d>()(
  [["x", undefined], ["y", undefined], ["bins", 10], ["range", null], ["density", null], ["weights", null]] as const,
  [P.asarray('x'), P.asarray('y'), asarray_if_not_null('weights')]);
export const histogramdd = signature<typeof stats.histogramdd>()(
  [["sample", undefined], ["bins", 10], ["range", null], ["density", null], ["weights", null]] as const,
  [P.asarray('sample'), asarray_if_not_null('weights')]);
export const bincount = signature<typeof stats.bincount>()(
  [["x", undefined], ["weights", null], ["minlength", 0]] as const,
  [P.asarray('x'), asarray_if_not_null('weights'), P.is_integer_non_neg('minlength')]);
export const digitize = signature<typeof stats.digitize>()(
  [["x", undefined], ["bins", undefined], ["right", false]] as const,
  [P.asarray('x'), P.asarray('bins'), boolean('right')]);

export const take = signature<typeof indexing.take>()(
  [["a", undefined], ["indices", undefined], ["axis", undefined]] as const,
  [P.asarray('a'), P.asarray('indices'), axis_int]);

// ==============================
//   Array creation and shape manipulation
// ==============================
import * as jsInterface from "../array/js-interface";
import * as shapeOps from "../array/shape_operations";
export const array = signature<(object: any, dtype: import("../dtypes").DType | null, copy: boolean, ndmin: number) => import("../NDArray").NDArray>()(
  [["object", undefined], ["dtype", null], ["copy", true], ["ndmin", 0]] as const,
  [P.dtype_or_null, boolean('copy'), P.is_integer_non_neg('ndmin')]);
export const asarray_sig = signature<(a: any, dtype: import("../dtypes").DType | null) => import("../NDArray").NDArray>()(
  [["a", undefined], ["dtype", null]] as const, [P.dtype_or_null]);
export const reshape = signature<(a: import("../NDArray").NDArray, shape: any[]) => import("../NDArray").NDArray>()(
  [["a", undefined], ["...shape", []]] as const, [P.asarray('a')]);
export const transpose = signature<typeof transform.transpose>()(
  [["a", undefined], ["axes", null]] as const, [P.asarray('a')]);
export const ravel = signature<typeof shapeOps.ravel>()(
  [["a", undefined]] as const, [P.asarray('a')]);
export const concatenate = signature<typeof transform.concatenate>()(
  [["arrays", undefined], ["axis", 0]] as const, [axis_or_null]);
export const stack = signature<typeof transform.stack>()(
  [["arrays", undefined], ["axis", 0]] as const, [axis_int]);

export const apply_along_axis = signature<(func1d: (lane: any, ...args: any[]) => any, axis: number, arr: import("../NDArray").NDArray, args: any[]) => import("../NDArray").NDArray>()(
  [["func1d", undefined], ["axis", undefined], ["arr", undefined], ["...args", []]] as const,
  [P.asarray('arr'), axis_int]);
