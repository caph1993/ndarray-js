import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const A = "np.array([3, 1, 2, 3, 1, 5])";
const M = "np.array([[3, 1, 2], [9, 7, 8], [0, 5, 4]])";
const F = "np.array([2.5, np.nan, 1.5, np.nan, -1.0, 2.5])";

numpy_cases("unique and set operations", [
  `np.unique(${A})`, `np.unique(${A}, return_index=True)`, `np.unique(${A}, return_inverse=True)`, `np.unique(${A}, return_counts=True)`,
  `np.unique(${A}, return_index=True, return_inverse=True, return_counts=True)`, `np.unique(${M})`, `np.unique(${M}, return_inverse=True)`,
  `np.unique(${F})`, `np.unique(${F}, equal_nan=False)`, `np.unique(${F}, return_counts=True)`, `np.unique(np.array([True, False, True]))`, `np.unique(np.array([]))`,
  `np.unique(np.array([[1, 2], [3, 4], [1, 2]]), axis=0)`, `np.unique(np.array([[1, 2], [3, 4], [1, 2]]), axis=0, return_counts=True)`, `np.unique(np.array([[1, 1, 2], [3, 3, 4]]), axis=1)`,
  `np.unique(np.array([[1, 1, 2], [3, 3, 4]]), axis=1, return_inverse=True)`, "np.unique([3, 1, 3])",
  `np.sort(np.unique_values(${A}))`, [() => np.unique_counts(np.array([3, 1, 2, 3, 1, 5])).counts, "np.unique_counts(np.array([3, 1, 2, 3, 1, 5])).counts"],
  [() => np.unique_inverse(np.array([3, 1, 2, 3, 1, 5]))[1], "np.unique_inverse(np.array([3, 1, 2, 3, 1, 5])).inverse_indices"],
  [() => np.sort(np.unique_all(np.array([3, 1, 2, 3, 1, 5])).values), "np.sort(np.unique_all(np.array([3, 1, 2, 3, 1, 5])).values)"],
  `np.intersect1d(np.array([1, 3, 4, 3]), np.array([3, 1, 2, 1]))`, `np.intersect1d(np.array([1, 3, 4, 3]), np.array([3, 1, 2, 1]), return_indices=True)`, `np.intersect1d(np.array([1, 2]), np.array([3, 4]))`,
  `np.intersect1d(np.array([[1, 2], [3, 4]]), np.array([4, 1]))`, `np.intersect1d(np.array([1.5, 2.0]), np.array([2, 1.5]))`,
  `np.union1d(np.array([-1, 0, 1]), np.array([-2, 0, 2]))`, `np.union1d(np.array([[1, 2]]), np.array([2, 5]))`, `np.union1d(np.array([1.5]), np.array([1]))`,
  `np.setdiff1d(np.array([1, 2, 3, 2, 4, 1]), np.array([3, 4, 5, 6]))`, `np.setdiff1d(np.array([1, 2, 3]), np.array([]))`, `np.setdiff1d(np.array([1, 2, 3]), np.array([1, 2, 3]))`,
  `np.setxor1d(np.array([1, 2, 3, 2, 4]), np.array([2, 3, 5, 7, 5]))`, `np.setxor1d(np.array([1, 2]), np.array([1, 2]))`,
  `np.isin(np.array([[1, 2], [3, 4]]), np.array([2, 4]))`, `np.isin(np.array([1, 2, 3]), np.array([2]), invert=True)`, `np.isin(np.array([1.0, np.nan]), np.array([np.nan, 1.0]))`, `np.isin(5, np.array([1, 5]))`, `np.isin(np.array([1, 2]), np.array([[1, 3]]))`,
]);
numpy_cases("sorting", [
  `np.sort(${A})`, `np.sort(${M})`, `np.sort(${M}, axis=0)`, `np.sort(${M}, axis=-1)`, `np.sort(${M}, axis=None)`, `np.sort(${F})`, `np.sort(np.array([True, False, True]))`, `np.sort(np.array([[2.5, 1.5], [0.5, 3.5]]), axis=0)`,
  `np.argsort(${A})`, `np.argsort(${M}, axis=0)`, `np.argsort(${M})`, `np.argsort(${M}, axis=None)`, `np.argsort(${F})`, `np.argsort(np.array([1, 1, 1, 0]))`,
  `np.sort(np.arange(24).reshape(2, 3, 4)[::-1, :, ::-1], axis=1)`, `np.argsort(np.arange(24).reshape(2, 3, 4)[:, ::-1], axis=1)`, "np.sort([3, 1, 2])", "np.sort(np.array([-0.0, 0.0, -1.0]))",
  `np.array([3, 1, 2]).argsort()`, `np.array([[3, 1], [2, 0]]).argsort(axis=0)`,
  [() => { const a = np.array([3, 1, 2]); a.sort(); return a; }, "np.sort(np.array([3, 1, 2]))"],
  [() => { const a = np.array([[3, 1], [2, 0]]); a.sort(0); return a; }, "np.sort(np.array([[3, 1], [2, 0]]), axis=0)"],
  [() => { const a = np.arange(6).reshape(2, 3).index(':', '::-1'); a.sort(); return a; }, "np.sort(np.arange(6).reshape(2, 3)[:, ::-1], axis=-1)"],
  "np.lexsort((np.array([1, 5, 1, 4, 3, 4, 4]), np.array([9, 4, 0, 4, 0, 2, 1])))", "np.lexsort(np.array([[1, 5, 1, 4], [9, 4, 0, 4]]))", "np.lexsort((np.array([[1, 2], [2, 1]]), np.array([[0, 0], [0, 0]])), axis=0)",
  "np.lexsort((np.array([1.0, np.nan, 1.0]), np.array([2, 2, 1])))",
  "!np.lexsort((np.array([1, 2]), np.array([1, 2, 3])))",
]);
numpy_cases("partition", [
  // The order inside each side of the partition is unspecified: compare through properties
  "np.sort(np.partition(np.array([3, 4, 2, 1]), 2))", "np.partition(np.array([3, 4, 2, 1]), 2)[2]", "np.partition(np.array([3, 4, 2, 1]), 0)[0]", "np.partition(np.array([3, 4, 2, 1]), -1)[-1]",
  "np.partition(np.array([3, 4, 2, 1, 7, 6]), [1, 4])[[1, 4]]", "np.sort(np.partition(np.array([[3, 4, 2], [1, 7, 6]]), 1, axis=1), axis=1)", "np.partition(np.array([[3, 4, 2], [1, 7, 6]]), 1, axis=0)[1]",
  "np.array([3, 4, 2, 1])[np.argpartition(np.array([3, 4, 2, 1]), 2)][2]", "np.sort(np.argpartition(np.array([3, 4, 2, 1]), 1))", "!np.partition(np.array([1, 2]), 5)",
]);
numpy_cases("searching", [
  "np.where(np.array([True, False, True]), np.array([1, 2, 3]), np.array([10, 20, 30]))", "np.where(np.array([[True, False], [False, True]]), 1, 0)", "np.where(np.array([1, 0, 2]) > 0, np.array([1.5, 2.5, 3.5]), 0)",
  "np.where(np.array([True, False]), np.array([[1], [2]]), np.array([10, 20]))", "np.where(np.array([1, 0, np.nan]))", "np.where(np.array([[0, 1], [1, 1]]))", "np.where(True, 1, 2)",
  "np.where(np.array([True, False]), np.array([1, 2], dtype='float32'), 2.5)", "np.where(np.array([True, False]), np.array([1, 2]), 2.5)", "!np.where(np.array([True]), 1)",
  "np.nonzero(np.array([[0, 1], [2, 0]]))", "np.nonzero(np.array([0, 3, 0, 5]))", "np.nonzero(np.array([0.0, np.nan]))", "np.array([[0, 1], [2, 0]]).nonzero()", "!np.nonzero(np.array(5))",
  "np.argwhere(np.array([[0, 1], [2, 0]]))", "np.argwhere(np.array([0, 3, 0, 5]))", "np.argwhere(np.zeros((2, 2)))", "np.flatnonzero(np.array([[0, 1], [2, 0]]))", "np.flatnonzero(np.array([0, 0]))",
  "np.extract(np.array([[1, 0], [0, 1]]) > 0, np.array([[1, 2], [3, 4]]))", "np.extract(np.array([True, False, True]), np.array([5.5, 6.5, 7.5]))",
  "np.searchsorted(np.array([1, 2, 2, 3]), 2)", "np.searchsorted(np.array([1, 2, 2, 3]), 2, side='right')", "np.searchsorted(np.array([1, 2, 2, 3]), np.array([0, 2, 5, 3]))", "np.searchsorted(np.array([1, 2, 2, 3]), np.array([[0, 2], [5, 3]]), 'right')",
  "np.searchsorted(np.array([1.0, 2.0, np.nan]), np.array([np.nan, 1.5]))", "np.searchsorted(np.array([3, 1, 2]), 2, sorter=np.array([1, 2, 0]))", "!np.searchsorted(np.array([1, 2]), 1, side='middle')",
  "np.array([1, 2, 2, 3]).searchsorted(2)", "np.array([1, 2, 2, 3]).searchsorted(np.array([2, 3]), side='right')",
  "np.digitize(np.array([0.2, 6.4, 3.0, 1.6]), np.array([0.0, 1.0, 2.5, 4.0, 10.0]))", "np.digitize(np.array([0.2, 6.4, 3.0, 1.6]), np.array([0.0, 1.0, 2.5, 4.0, 10.0]), right=True)",
  "np.digitize(np.array([1.2, 10.0, 12.4, 15.5, 20.0]), np.array([0.0, 5.0, 10.0, 15.0, 20.0]), right=True)", "np.digitize(np.array([1.2, 10.0, 12.4, 15.5, 20.0]), np.array([0.0, 5.0, 10.0, 15.0, 20.0]))",
  "np.digitize(np.array([1.2, 4.0, 6.0]), np.array([10.0, 5.0, 1.0]))", "np.digitize(np.array([1.2, 4.0, 6.0]), np.array([10.0, 5.0, 1.0]), right=True)", "np.digitize(np.array([[1, 5], [9, 11]]), np.array([2, 8]))", "!np.digitize(np.array([1]), np.array([1, 3, 2]))",
]);
