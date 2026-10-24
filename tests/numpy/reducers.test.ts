import { numpy_cases } from "../helpers/cases";

const M = "np.array([[1.5, 2.0, -3.0], [4.0, np.nan, 6.5]])";
const N = "np.array([[1.5, 2.0, -3.0], [4.0, 5.0, 6.5]])";
const I3 = "np.arange(24).reshape(2, 3, 4)";
const IM = "np.array([[3, 1, 2], [9, 7, 8]])";
const BM = "np.array([[True, False, True], [False, False, True]])";

const axes = ["", ", axis=0", ", axis=1", ", axis=-1", ", axis=None", ", axis=0, keepdims=True", ", axis=1, keepdims=True", ", keepdims=True", ", axis=(0, 1)"];
const reducers = ["sum", "prod", "mean", "max", "min", "argmax", "argmin", "std", "var", "ptp", "any", "all", "count_nonzero", "median", "amax", "amin"];

numpy_cases("reducers on 2D floats", reducers.flatMap(f => axes.filter(a => !(["argmax", "argmin", "count_nonzero"].includes(f) && a.includes("(0, 1)"))).filter(a => !(f == "count_nonzero" && a.includes("None") && false)).map(a => `np.${f}(${N}${a})`)));
numpy_cases("reducers on 3D ints", ["sum", "mean", "max", "min", "std", "prod", "argmax", "median", "ptp"].flatMap(f => ["", ", axis=0", ", axis=1", ", axis=2", ", axis=-2", ", axis=(0, 2)", ", axis=(1, 2), keepdims=True", ", axis=(0, 1, 2)"].filter(a => !(["argmax"].includes(f) && a.includes("("))).map(a => `np.${f}(${I3}${a})`)));
numpy_cases("reducers with ints and bools", [
  `np.sum(${IM})`, `np.sum(${IM}, axis=0)`, `np.mean(${IM}, axis=1)`, `np.max(${IM}, axis=0)`, `np.prod(${IM}, axis=1)`, `np.sum(${BM})`, `np.sum(${BM}, axis=1)`, `np.mean(${BM})`,
  `np.max(${BM}, axis=0)`, `np.any(${BM}, axis=0)`, `np.all(${BM}, axis=1)`, `np.argmax(${BM}, axis=1)`, `np.prod(${BM})`, `np.sum(np.array([1, 2, 3]), dtype='float64')`, `np.sum(np.array([1.5, 2.5]), dtype='int32')`,
  `np.mean(np.array([1, 2, 4]), dtype='float32')`, `np.sum(np.array([], dtype='float64'))`, `np.prod(np.array([], dtype='float64'))`, `np.mean(np.array([1, 2]), axis=0, keepdims=True)`,
  `np.any(np.array([0, 0.0, np.nan]))`, `np.all(np.array([1, 2, np.nan]))`, `np.all(np.array([]))`, `np.any(np.array([]))`, `np.sum([1, 2, 3])`, `np.sum(5)`, `np.max([[1, 5], [7, 2]], axis=1)`,
  `np.array([1, 2, 3]).sum()`, `${IM}.mean(axis=0)`, `${IM}.sum(1)`, `${IM}.max(axis=0, keepdims=True)`, `${IM}.argmax()`, `${IM}.argmin(axis=1)`, `${IM}.std(ddof=1)`, `${IM}.var(axis=0, ddof=1)`, `${IM}.prod(axis=0)`, `${IM}.any()`, `${IM}.all(axis=0)`,
  "!np.max(np.array([]))", "!np.min(np.array([], dtype='float64'))", "!np.argmax(np.array([]))", "!np.sum(np.array([1, 2]), axis=1)", "!np.sum(np.array([[1, 2]]), axis=(0, 0))", "!np.sum(np.array([1, 2]), axis=-2)",
]);
numpy_cases("nan handling in reducers", [
  `np.sum(${M})`, `np.mean(${M}, axis=0)`, `np.max(${M})`, `np.min(${M}, axis=1)`, `np.argmax(${M}, axis=1)`, `np.argmin(${M})`, `np.std(${M}, axis=1)`,
  `np.nansum(${M})`, `np.nansum(${M}, axis=0)`, `np.nansum(${M}, axis=1, keepdims=True)`, `np.nanprod(${M}, axis=0)`, `np.nanprod(${M})`, `np.nanmean(${M}, axis=0)`, `np.nanmean(${M})`, `np.nanmean(${M}, axis=1, keepdims=True)`,
  `np.nanmax(${M})`, `np.nanmax(${M}, axis=0)`, `np.nanmin(${M}, axis=1)`, `np.nanmin(${M})`, `np.nanargmax(${M}, axis=1)`, `np.nanargmin(${M})`, `np.nanargmax(${M})`, `np.nanstd(${M})`, `np.nanstd(${M}, axis=1)`, `np.nanvar(${M}, axis=0)`,
  `np.nanvar(${M}, ddof=1)`, `np.nanstd(${M}, axis=0, ddof=1)`, `np.nanmedian(${M})`, `np.nanmedian(${M}, axis=1)`, `np.nanmedian(${M}, axis=0, keepdims=True)`,
  `np.median(${M}, axis=1)`, "np.nansum(np.array([np.nan, np.nan]))", "np.nanprod(np.array([np.nan]))", "np.nanmean(np.array([np.nan, np.nan]))", "np.nanmax(np.array([np.nan, np.nan]))", "!np.nanargmax(np.array([np.nan, np.nan]))",
  "np.nanmedian(np.array([np.nan, np.nan]))", "np.nanstd(np.array([np.nan, 1.0]))", "np.nanvar(np.array([np.nan, 1.0, 3.0]))",
  "np.max(np.array([1.0, np.inf, np.nan]))", "np.min(np.array([-np.inf, 1.0]))", "np.nanmax(np.array([-np.inf, np.nan]))", "np.ptp(np.array([1.0, np.nan]))",
]);
numpy_cases("std/var parameters", [
  `np.std(${N}, ddof=1)`, `np.var(${N}, axis=0, ddof=1)`, `np.std(${N}, axis=1, keepdims=True)`, `np.var(np.array([1.0]), ddof=1)`, `np.std(np.array([1.0, 2.0]), ddof=2)`,
  `np.std(np.array([1, 2, 3, 4]))`, `np.var(np.array([True, False, True]))`, `np.std(np.array([1, 2, 3, 4], dtype='float32'))`, `np.var(${I3}, axis=(0, 2))`,
]);
numpy_cases("count_nonzero, ptp, average", [
  "np.count_nonzero(np.array([[0, 1, 2], [0, 0, 3]]))", "np.count_nonzero(np.array([[0, 1, 2], [0, 0, 3]]), axis=0)", "np.count_nonzero(np.array([[0, 1, 2], [0, 0, 3]]), axis=1)",
  "np.count_nonzero(np.array([[0, 1, 2], [0, 0, 3]]), axis=1, keepdims=True)", "np.count_nonzero(np.array([0.0, np.nan]))", "np.count_nonzero(np.array([True, False]))",
  "np.ptp(np.array([[1, 5], [3, 2]]))", "np.ptp(np.array([[1, 5], [3, 2]]), axis=1)", "np.ptp(np.array([[1, 5], [3, 2]]), axis=0, keepdims=True)", "np.ptp(np.array([1.5, -2.0]))",
  "np.average(np.array([1, 2, 3, 4]))", "np.average(np.array([1, 2, 3, 4]), weights=np.array([4, 3, 2, 1]))", "np.average(np.array([[1, 2], [3, 4]]), axis=0, weights=np.array([1, 3]))",
  "np.average(np.array([[1, 2], [3, 4]]), axis=1, weights=np.array([1, 3]))", "np.average(np.array([[1, 2], [3, 4]]), weights=np.array([[1, 0], [0, 1]]))", "np.average(np.array([[1, 2], [3, 4]]), axis=0)",
  "np.average(np.array([[1, 2], [3, 4]]), axis=1, keepdims=True)", "np.average(np.array([1, 2, 3]), weights=np.array([1, 1, 2]), returned=True)", "np.average(np.array([1, 2, 3]), returned=True)",
  "!np.average(np.array([1, 2]), weights=np.array([0, 0]))", "!np.average(np.array([1, 2, 3]), weights=np.array([1, 2]))",
], { check_dtype: true });
