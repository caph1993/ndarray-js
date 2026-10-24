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

// ==============================
//   Group 2: array creation and manipulation
// ==============================
import * as manip from "../modules/manipulation";
import { float64 as _float64 } from "../dtypes";
const nonneg_int = (...keys: string[]): Parser => (kwargs: any) => {
  for (const k of keys) if (kwargs[k] !== null && (typeof kwargs[k] !== "number" || kwargs[k] < 0 || !Number.isInteger(kwargs[k]))) throw new Error(`Invalid "${k}" argument. Expected non-negative integer. Found ${kwargs[k]}`);
};
const int_arg = (...keys: string[]): Parser => (kwargs: any) => {
  for (const k of keys) if (typeof kwargs[k] !== "number" || !Number.isInteger(kwargs[k])) throw new Error(`Invalid "${k}" argument. Expected integer. Found ${kwargs[k]}`);
};
const shape_arg: Parser = (kwargs: any) => { if (kwargs.shape !== null && kwargs.shape !== undefined) kwargs.shape = Array.isArray(kwargs.shape) ? kwargs.shape : (typeof kwargs.shape === "number" ? [kwargs.shape] : Array.from(asarray(kwargs.shape).flat as any)); };
const as_array_list = (key: string): Parser => (kwargs: any) => { kwargs[key] = (kwargs[key] as any[]).map(a => asarray(a)); };
const kwargs_list = (key: string): Parser => (kwargs: any) => { /* varargs collected by the engine */ };

export const eye = signature<typeof manip.eye>()(
  [["N", undefined], ["M", null], ["k", 0], ["dtype", _float64]] as const, [nonneg_int("N", "M"), int_arg("k"), P.dtype_or_null]);
export const identity = signature<typeof manip.identity>()(
  [["n", undefined], ["dtype", _float64]] as const, [nonneg_int("n"), P.dtype_or_null]);
export const tri = signature<typeof manip.tri>()(
  [["N", undefined], ["M", null], ["k", 0], ["dtype", _float64]] as const, [nonneg_int("N", "M"), int_arg("k"), P.dtype_or_null]);
export const tril = signature<typeof manip.tril>()([["m", undefined], ["k", 0]] as const, [P.asarray('m'), int_arg("k")]);
export const diag = signature<typeof manip.diag>()([["v", undefined], ["k", 0]] as const, [P.asarray('v'), int_arg("k")]);
export const full = signature<typeof manip.full>()(
  [["shape", undefined], ["fill_value", undefined], ["dtype", null]] as const, [shape_arg, P.dtype_or_null], { keep_weak: ["fill_value"] });
export const full_like = signature<typeof manip.full_like>()(
  [["a", undefined], ["fill_value", undefined], ["dtype", null], ["shape", null]] as const, [P.asarray('a'), P.dtype_or_null, shape_arg], { keep_weak: ["fill_value"] });
export const like = signature<typeof manip.zeros_like>()(
  [["a", undefined], ["dtype", null], ["shape", null]] as const, [P.asarray('a'), P.dtype_or_null, shape_arg]);
export const logspace = signature<typeof manip.logspace>()(
  [["start", undefined], ["stop", undefined], ["num", 50], ["endpoint", true], ["base", 10.0], ["dtype", null]] as const,
  [P.as_number('start'), P.as_number('stop'), P.dtype_or_null, boolean('endpoint')]);
export const meshgrid = signature<typeof manip.meshgrid>()(
  [["...xi", []], ["indexing", 'xy'], ["sparse", false], ["copy", true]] as const, [as_array_list('xi'), boolean('sparse', 'copy')]);
export const grid = signature<(specs: any[]) => any>()(
  [["...specs", []]] as const, []);
export const ndim_ = signature<typeof manip.ndim>()([["a", undefined]] as const, [P.asarray('a')]);
export const size_ = signature<typeof manip.size>()([["a", undefined], ["axis", null]] as const, [P.asarray('a'), axis_or_null]);
export const astype = signature<typeof manip.astype>()(
  [["x", undefined], ["dtype", undefined], ["copy", true]] as const, [P.asarray('x'), P.dtype_or_null, boolean('copy')]);
export const copy_ = signature<typeof manip.copy>()([["a", undefined]] as const, [P.asarray('a')]);
export const copyto = signature<typeof manip.copyto>()(
  [["dst", undefined], ["src", undefined], ["casting", 'same_kind'], ["where", true]] as const, [P.isarray('dst'), P.asarray('src')]);
export const moveaxis = signature<typeof manip.moveaxis>()(
  [["a", undefined], ["source", undefined], ["destination", undefined]] as const, [P.asarray('a')]);
export const rollaxis = signature<typeof manip.rollaxis>()(
  [["a", undefined], ["axis", undefined], ["start", 0]] as const, [P.asarray('a'), int_arg("axis", "start")]);
export const swapaxes = signature<typeof transform.swapAxes>()(
  [["a", undefined], ["axis1", undefined], ["axis2", undefined]] as const, [P.asarray('a'), int_arg("axis1", "axis2")]);
export const a_only_arr = signature<typeof manip.matrix_transpose>()([["x", undefined]] as const, [P.asarray('x')]);
export const atleast = signature<(...arys: any[]) => any>()([["...arys", []]] as const, [as_array_list('arys')]);
export const broadcast_to = signature<typeof manip.broadcast_to>()([["array", undefined], ["shape", undefined]] as const, [P.asarray('array'), shape_arg]);
export const broadcast_arrays = signature<(...args: any[]) => any>()([["...args", []]] as const, [as_array_list('args')]);
export const expand_dims = signature<typeof manip.expand_dims>()([["a", undefined], ["axis", undefined]] as const, [P.asarray('a')]);
export const squeeze = signature<typeof manip.squeeze>()([["a", undefined], ["axis", null]] as const, [P.asarray('a')]);
export const require_ = signature<(a: any, dtype: any, requirements: any) => any>()(
  [["a", undefined], ["dtype", null], ["requirements", null]] as const, [P.dtype_or_null]);
export const stack_like = signature<(tup: any) => any>()([["tup", undefined]] as const, [(kwargs: any) => { kwargs.tup = Array.from(kwargs.tup as any).map(a => asarray(a)); }]);
export const block = signature<typeof manip.block>()([["arrays", undefined]] as const, []);
export const split = signature<typeof manip.split>()(
  [["ary", undefined], ["indices_or_sections", undefined], ["axis", 0]] as const, [P.asarray('ary'), int_arg("axis")]);
export const hsplit = signature<typeof manip.hsplit>()(
  [["ary", undefined], ["indices_or_sections", undefined]] as const, [P.asarray('ary')]);
export const unstack = signature<typeof manip.unstack>()([["x", undefined], ["axis", 0]] as const, [P.asarray('x'), int_arg("axis")]);
export const tile = signature<typeof manip.tile>()([["A", undefined], ["reps", undefined]] as const, [P.asarray('A')]);
export const repeat = signature<typeof manip.repeat>()([["a", undefined], ["repeats", undefined], ["axis", null]] as const, [P.asarray('a'), axis_or_null]);
export const del = signature<typeof manip.del>()([["arr", undefined], ["obj", undefined], ["axis", null]] as const, [P.asarray('arr'), axis_or_null]);
export const insert = signature<typeof manip.insert>()([["arr", undefined], ["obj", undefined], ["values", undefined], ["axis", null]] as const, [P.asarray('arr'), axis_or_null]);
export const append = signature<typeof manip.append>()([["arr", undefined], ["values", undefined], ["axis", null]] as const, [P.asarray('arr'), P.asarray('values'), axis_or_null]);
export const resize = signature<typeof manip.resize>()([["a", undefined], ["new_shape", undefined]] as const, [P.asarray('a')]);
export const trim_zeros = signature<typeof manip.trim_zeros>()([["filt", undefined], ["trim", 'fb'], ["axis", null]] as const, [P.asarray('filt')]);
export const pad = signature<(array: any, pad_width: any, mode: manip.PadMode, constant_values: any, end_values: any, stat_length: any, reflect_type: any) => any>()(
  [["array", undefined], ["pad_width", undefined], ["mode", 'constant'], ["constant_values", null], ["end_values", null], ["stat_length", null], ["reflect_type", 'even']] as const, [P.asarray('array')]);
export const flip = signature<typeof manip.flip>()([["m", undefined], ["axis", null]] as const, [P.asarray('m')]);
export const flip1 = signature<typeof manip.fliplr>()([["m", undefined]] as const, [P.asarray('m')]);
export const roll = signature<typeof manip.roll>()([["a", undefined], ["shift", undefined], ["axis", null]] as const, [P.asarray('a')]);
export const rot90 = signature<typeof manip.rot90>()([["m", undefined], ["k", 1], ["axes", [0, 1]]] as const, [P.asarray('m'), int_arg("k")]);
export const bitwise_count = signature<typeof manip.bitwise_count>()([["x", undefined]] as const, [P.asarray('x')]);

// ==============================
//   Group 3: indexing extras, functional tools
// ==============================
import * as ix from "../modules/indexing-extra";
import * as fn from "../modules/functional";
import * as la from "../modules/linalg";
const mode_arg: Parser = (kwargs: any) => { if (!["raise", "wrap", "clip"].includes(kwargs.mode)) throw new Error(`clipmode must be one of 'clip', 'raise', or 'wrap' (got ${JSON.stringify(kwargs.mode)})`); };
const dims_arg = (key: string): Parser => (kwargs: any) => { if (typeof kwargs[key] === "number") kwargs[key] = [kwargs[key]]; else kwargs[key] = Array.from(asarray(kwargs[key]).flat as any); };

export const take_ = signature<typeof ix.take>()(
  [["a", undefined], ["indices", undefined], ["axis", null], ["out", null], ["mode", 'raise']] as const,
  [P.asarray('a'), P.asarray('indices'), axis_or_null, P.isarray_or_null('out'), mode_arg]);
export const take_along_axis = signature<typeof ix.take_along_axis>()(
  [["arr", undefined], ["indices", undefined], ["axis", undefined]] as const, [P.asarray('arr'), P.asarray('indices'), axis_or_null]);
export const put_along_axis = signature<typeof ix.put_along_axis>()(
  [["arr", undefined], ["indices", undefined], ["values", undefined], ["axis", undefined]] as const, [P.isarray('arr'), P.asarray('indices'), P.asarray('values'), axis_or_null]);
export const choose = signature<typeof ix.choose>()(
  [["a", undefined], ["choices", undefined], ["out", null], ["mode", 'raise']] as const, [P.asarray('a'), P.isarray_or_null('out'), mode_arg]);
export const compress = signature<typeof ix.compress>()(
  [["condition", undefined], ["a", undefined], ["axis", null], ["out", null]] as const, [P.asarray('condition'), P.asarray('a'), axis_or_null, P.isarray_or_null('out')]);
export const diagonal = signature<typeof ix.diagonal>()(
  [["a", undefined], ["offset", 0], ["axis1", 0], ["axis2", 1]] as const, [P.asarray('a'), int_arg("offset", "axis1", "axis2")]);
export const select = signature<typeof ix.select>()(
  [["condlist", undefined], ["choicelist", undefined], ["default", 0]] as const, []);
export const place = signature<typeof ix.place>()(
  [["arr", undefined], ["mask", undefined], ["vals", undefined]] as const, [P.isarray('arr'), P.asarray('mask'), P.asarray('vals')]);
export const put = signature<typeof ix.put>()(
  [["a", undefined], ["ind", undefined], ["v", undefined], ["mode", 'raise']] as const, [P.isarray('a'), P.asarray('ind'), P.asarray('v'), mode_arg]);
export const putmask = signature<typeof ix.putmask>()(
  [["a", undefined], ["mask", undefined], ["values", undefined]] as const, [P.isarray('a'), P.asarray('mask'), P.asarray('values')]);
export const fill_diagonal = signature<typeof ix.fill_diagonal>()(
  [["a", undefined], ["val", undefined], ["wrap", false]] as const, [P.isarray('a'), P.asarray('val'), boolean('wrap')]);
export const indices = signature<typeof ix.indices>()(
  [["dimensions", undefined], ["dtype", 'int32'], ["sparse", false]] as const, [dims_arg("dimensions"), P.dtype_or_null, boolean('sparse')]);
export const ix_ = signature<typeof ix.ix_>()([["...args", []]] as const, [as_array_list('args')]);
export const unravel_index = signature<typeof ix.unravel_index>()(
  [["indices", undefined], ["shape", undefined], ["order", 'C']] as const, [P.asarray('indices')]);
export const ravel_multi_index = signature<typeof ix.ravel_multi_index>()(
  [["multi_index", undefined], ["dims", undefined], ["mode", 'raise'], ["order", 'C']] as const, []);
export const diag_indices = signature<typeof ix.diag_indices>()([["n", undefined], ["ndim", 2]] as const, [int_arg("n", "ndim")]);
export const diag_indices_from = signature<typeof ix.diag_indices_from>()([["arr", undefined]] as const, [P.asarray('arr')]);
export const mask_indices = signature<typeof ix.mask_indices>()([["n", undefined], ["mask_func", undefined], ["k", 0]] as const, [int_arg("n", "k")]);
export const tri_indices = signature<typeof ix.tril_indices>()([["n", undefined], ["k", 0], ["m", null]] as const, [int_arg("n", "k")]);
export const tri_indices_from = signature<typeof ix.tril_indices_from>()([["arr", undefined], ["k", 0]] as const, [P.asarray('arr'), int_arg("k")]);
export const apply_over_axes = signature<typeof fn.apply_over_axes>()([["func", undefined], ["a", undefined], ["axes", undefined]] as const, [P.asarray('a')]);
export const dot = signature<typeof la.dot>()([["a", undefined], ["b", undefined], ["out", null]] as const, [P.asarray('a'), P.asarray('b'), P.isarray_or_null('out')]);

// ==============================
//   Group 5: linear algebra
// ==============================
const two_arrays = (a: string, b: string): Parser[] => [P.asarray(a), P.asarray(b)];
export const matmul = signature<typeof la.matmul>()([["x1", undefined], ["x2", undefined], ["out", null]] as const, [...two_arrays("x1", "x2"), P.isarray_or_null('out')]);
export const arr2 = signature<typeof la.vdot>()([["a", undefined], ["b", undefined]] as const, two_arrays("a", "b"));
export const outer = signature<typeof la.outer>()([["a", undefined], ["b", undefined], ["out", null]] as const, [...two_arrays("a", "b"), P.isarray_or_null('out')]);
export const x1x2 = signature<typeof la.matvec>()([["x1", undefined], ["x2", undefined]] as const, two_arrays("x1", "x2"));
export const vecdot = signature<typeof la.vecdot>()([["x1", undefined], ["x2", undefined], ["axis", -1]] as const, [...two_arrays("x1", "x2"), axis_int]);
export const tensordot = signature<typeof la.tensordot>()([["a", undefined], ["b", undefined], ["axes", 2]] as const, two_arrays("a", "b"));
export const kron = signature<typeof la.kron>()([["a", undefined], ["b", undefined]] as const, two_arrays("a", "b"));
export const matrix_power = signature<typeof la.matrix_power>()([["a", undefined], ["n", undefined]] as const, [P.asarray('a')]);
export const multi_dot = signature<typeof la.multi_dot>()([["arrays", undefined]] as const, []);
export const trace = signature<typeof la.trace>()([["a", undefined], ["offset", 0], ["axis1", 0], ["axis2", 1], ["dtype", null], ["out", null]] as const, [P.asarray('a'), int_arg("offset", "axis1", "axis2"), P.dtype_or_null, P.isarray_or_null('out')]);
export const einsum = signature<(subscripts: string, operands: any[], dtype: any, out: any) => any>()(
  [["subscripts", undefined], ["...operands", []], ["dtype", null], ["out", null]] as const, [P.dtype_or_null, P.isarray_or_null('out')]);
export const x_only_la = signature<(x: any) => any>()([["a", undefined]] as const, [P.asarray('a')]);
export const cholesky = signature<typeof la.cholesky>()([["a", undefined], ["upper", false]] as const, [P.asarray('a'), boolean('upper')]);
export const qr = signature<typeof la.qr>()([["a", undefined], ["mode", 'reduced']] as const, [P.asarray('a')]);
export const svd = signature<typeof la.svd>()([["a", undefined], ["full_matrices", true], ["compute_uv", true], ["hermitian", false]] as const, [P.asarray('a'), boolean('full_matrices', 'compute_uv', 'hermitian')]);
export const eigh = signature<typeof la.eigh>()([["a", undefined], ["UPLO", 'L']] as const, [P.asarray('a')]);
export const solve = signature<typeof la.solve>()([["a", undefined], ["b", undefined]] as const, two_arrays("a", "b"));
export const lstsq = signature<typeof la.lstsq>()([["a", undefined], ["b", undefined], ["rcond", null]] as const, two_arrays("a", "b"));
export const pinv = signature<typeof la.pinv>()([["a", undefined], ["rcond", null], ["hermitian", false], ["rtol", null]] as const, [P.asarray('a')]);
export const matrix_rank = signature<typeof la.matrix_rank>()([["A", undefined], ["tol", null], ["hermitian", false], ["rtol", null]] as const, [P.asarray('A')]);
export const cond = signature<typeof la.cond>()([["x", undefined], ["p", null]] as const, [P.asarray('x')]);
export const tensorsolve = signature<typeof la.tensorsolve>()([["a", undefined], ["b", undefined], ["axes", null]] as const, two_arrays("a", "b"));
export const tensorinv = signature<typeof la.tensorinv>()([["a", undefined], ["ind", 2]] as const, [P.asarray('a')]);
export const linalg_norm = signature<typeof la.norm>()([["x", undefined], ["ord", null], ["axis", null], ["keepdims", false]] as const, [P.asarray('x'), P.axis_or_axes, boolean('keepdims')]);
export const matrix_norm = signature<typeof la.matrix_norm>()([["x", undefined], ["ord", 'fro'], ["keepdims", false]] as const, [P.asarray('x'), boolean('keepdims')]);
export const vector_norm = signature<typeof la.vector_norm>()([["x", undefined], ["axis", null], ["keepdims", false], ["ord", 2]] as const, [P.asarray('x'), P.axis_or_axes, boolean('keepdims')]);
export const linalg_cross = signature<(a: any, b: any, axis: number) => any>()([["x1", undefined], ["x2", undefined], ["axis", -1]] as const, [...two_arrays("x1", "x2"), axis_int]);
export const eigvalsh = signature<typeof la.eigvalsh>()([["a", undefined], ["UPLO", 'L']] as const, [P.asarray('a')]);

export const angle = signature<typeof elementwise.angle>()([["z", undefined], ["deg", false]] as const, [P.asarray('z'), boolean('deg')]);
export const real_if_close = signature<typeof elementwise.real_if_close>()([["a", undefined], ["tol", 100]] as const, [P.asarray('a')]);

// ==============================
//   FFT
// ==============================
import * as fftm from "../modules/fft";
const norm_arg: Parser = (kwargs: any) => { if (kwargs.norm !== null && !["backward", "ortho", "forward"].includes(kwargs.norm)) throw new Error(`Invalid norm value ${kwargs.norm}; should be "backward", "ortho" or "forward".`); };
const n_arg: Parser = (kwargs: any) => { if (kwargs.n !== null && (!Number.isInteger(kwargs.n) || kwargs.n < 1)) throw new Error(`Invalid number of FFT data points (${kwargs.n}) specified.`); };
export const fft1 = signature<typeof fftm.fft>()([["a", undefined], ["n", null], ["axis", -1], ["norm", null]] as const, [P.asarray('a'), n_arg, int_arg("axis"), norm_arg]);
export const fftn = signature<typeof fftm.fftn>()([["a", undefined], ["s", null], ["axes", null], ["norm", null]] as const, [P.asarray('a'), norm_arg]);
export const fft2 = signature<typeof fftm.fft2>()([["a", undefined], ["s", null], ["axes", [-2, -1]], ["norm", null]] as const, [P.asarray('a'), norm_arg]);
export const fftfreq = signature<typeof fftm.fftfreq>()([["n", undefined], ["d", 1.0]] as const, []);
export const fftshift = signature<typeof fftm.fftshift>()([["x", undefined], ["axes", null]] as const, [P.asarray('x')]);

export const vander = signature<typeof manip.vander>()([["x", undefined], ["N", null], ["increasing", false]] as const, [P.asarray('x'), boolean('increasing')]);
