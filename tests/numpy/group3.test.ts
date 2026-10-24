import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const A = "np.arange(6).reshape(2, 3)";
const B = "np.arange(24).reshape(2, 3, 4)";
const SQ = "np.arange(16).reshape(4, 4)";

numpy_cases("take / take_along_axis / choose / compress / diagonal / select", [
  `np.take(${A}, 1)`, `np.take(${A}, [0, 2])`, `np.take(${A}, [0, 2], axis=1)`, `np.take(${A}, [[0, 1], [1, 0]], axis=0)`, `np.take(${A}, np.array([[0, 1], [2, 3]]))`, `np.take(${A}, -1, axis=1)`, `np.take(${B}, [1], axis=2)`,
  `np.take(${A}, [5, 6, -7], mode='wrap')`, `np.take(${A}, [5, -3], mode='clip')`, `!np.take(${A}, 6)`, `!np.take(${A}, [0, 3], axis=1)`, `np.take(${B}, 2, axis=1)`, "np.take(5, 0)",
  `np.take_along_axis(${A}, np.array([[2, 0], [1, 1]]), axis=1)`, `np.take_along_axis(${A}, np.array([[1, 0, 1]]), axis=0)`, `np.take_along_axis(${A}, np.array([[0], [2]]), axis=1)`, `np.take_along_axis(${A}, np.array([3, 0, 5]), axis=None)`,
  `np.take_along_axis(${A}, np.argsort(${A}, axis=1)[:, ::-1], axis=1)`, `!np.take_along_axis(${A}, np.array([[3]]), axis=1)`, `!np.take_along_axis(${A}, np.array([1, 2]), axis=1)`,
  [() => { const a = np.arange(6).reshape(2, 3); np.put_along_axis(a, np.array([[1], [2]]), 99, 1); return a; }, "np.array([[0, 99, 2], [3, 4, 99]])"],
  [() => { const a = np.arange(6).reshape(2, 3); np.put_along_axis(a, np.array([[0, 1, 0]]), np.array([[7, 8, 9]]), 0); return a; }, "np.array([[7, 1, 9], [3, 8, 5]])"],
  "np.choose(np.array([0, 1, 2, 1]), [np.array([10, 11, 12, 13]), np.array([20, 21, 22, 23]), np.array([30, 31, 32, 33])])", "np.choose(np.array([[1, 0], [0, 1]]), [10, 20])", "np.choose(np.array([2, -1]), [np.array([1, 2]), np.array([3, 4])], mode='wrap')",
  "np.choose(np.array([5, -2]), [np.array([1, 2]), np.array([3, 4])], mode='clip')", "!np.choose(np.array([2]), [np.array([1]), np.array([2])])", "np.choose(np.array([0, 1]), np.array([[1, 2], [3, 4]]))",
  "np.compress(np.array([True, False, True]), np.array([1, 2, 3]))", `np.compress(np.array([False, True]), ${A}, axis=0)`, `np.compress(np.array([True, False, True]), ${A}, axis=1)`, `np.compress(np.array([True, True]), ${A})`, "np.compress(np.array([0, 1, 1]), np.array([10, 20, 30]))",
  `np.diagonal(${SQ})`, `np.diagonal(${SQ}, 1)`, `np.diagonal(${SQ}, -2)`, `np.diagonal(${A})`, `np.diagonal(${B})`, `np.diagonal(${B}, axis1=1, axis2=2)`, `np.diagonal(${B}, 1, 0, 2)`, `np.diagonal(${SQ}, 9)`, `!np.diagonal(np.arange(3))`, `!np.diagonal(${SQ}, 0, 1, 1)`,
  "np.select([np.array([True, False, False]), np.array([False, True, False])], [np.array([1, 2, 3]), np.array([10, 20, 30])])", "np.select([np.array([True, False, False]), np.array([True, True, False])], [np.array([1, 2, 3]), np.array([10, 20, 30])], default=-1)",
  "np.select([np.arange(5) < 2, np.arange(5) > 3], [np.arange(5), np.arange(5) ** 2], default=0.5)", "!np.select([np.array([1, 0])], [np.array([1, 2])])",
]);

const py = (setup: string, stmt: string) => `(lambda a: (${stmt}, a)[1])(${setup})`;
numpy_cases("scatter operations", [
  [() => { const a = np.arange(6).reshape(2, 3); np.place(a, a.greater(2), np.array([-1, -2])); return a; }, py("np.arange(6).reshape(2, 3)", "np.place(a, a > 2, np.array([-1, -2]))")],
  [() => { const a = np.arange(6); np.place(a, np.array([1, 0, 1, 0, 0, 1]), [7, 8]); return a; }, py("np.arange(6)", "np.place(a, np.array([1, 0, 1, 0, 0, 1]), [7, 8])")],
  [() => { const a = np.arange(6); np.put(a, [0, 2], [-44, -55]); return a; }, py("np.arange(6)", "np.put(a, [0, 2], [-44, -55])")],
  [() => { const a = np.arange(6); np.put(a, [0, 2, 4], [9]); return a; }, py("np.arange(6)", "np.put(a, [0, 2, 4], [9])")],
  [() => { const a = np.arange(6); np.put(a, [7, -9], [1, 2], 'wrap'); return a; }, py("np.arange(6)", "np.put(a, [7, -9], [1, 2], 'wrap')")],
  [() => { const a = np.arange(6); np.put(a, [7, -9], [1, 2], 'clip'); return a; }, py("np.arange(6)", "np.put(a, [7, -9], [1, 2], 'clip')")],
  [() => { const a = np.arange(6).reshape(2, 3); np.put(a, 4, 99); return a; }, py("np.arange(6).reshape(2, 3)", "np.put(a, 4, 99)")],
  [() => { const a = np.arange(6); np.putmask(a, a.greater(2), np.array([-1, -2])); return a; }, py("np.arange(6)", "np.putmask(a, a > 2, np.array([-1, -2]))")],
  [() => { const a = np.arange(6).reshape(2, 3); np.putmask(a, a.equal(4), 0); return a; }, py("np.arange(6).reshape(2, 3)", "np.putmask(a, a == 4, 0)")],
  [() => { const a = np.zeros([3, 3]); np.fill_diagonal(a, 5); return a; }, py("np.zeros((3, 3))", "np.fill_diagonal(a, 5)")],
  [() => { const a = np.zeros([4, 2]); np.fill_diagonal(a, [1, 2]); return a; }, py("np.zeros((4, 2))", "np.fill_diagonal(a, [1, 2])")],
  [() => { const a = np.zeros([5, 2]); np.fill_diagonal(a, 1, true); return a; }, py("np.zeros((5, 2))", "np.fill_diagonal(a, 1, True)")],
  [() => { const a = np.zeros([2, 2, 2]); np.fill_diagonal(a, 3); return a; }, py("np.zeros((2, 2, 2))", "np.fill_diagonal(a, 3)")],
  [() => { const a = np.arange(6).reshape(2, 3); np.place(a.index(':', '::2'), np.array([[true, false], [false, true]]), [100]); return a; }, py("np.arange(6).reshape(2, 3)", "np.place(a[:, ::2], np.array([[True, False], [False, True]]), [100])")],
]);

numpy_cases("index generators", [
  "np.indices((2, 3))", "np.indices((2, 3), dtype='float64')", "np.indices((2, 3), sparse=True)", "np.indices((3,))", "np.indices((2, 2, 2)).shape",
  "np.ix_(np.array([0, 1]), np.array([2, 4, 5]))", "np.ix_(np.array([True, False, True]), np.array([1, 2]))", "np.ix_([1, 2], [3], [4, 5])", "np.ix_([])",
  "np.unravel_index(np.array([22, 41, 37]), (7, 6))", "np.unravel_index(np.array([31, 41, 13]), (7, 6), order='F')", "np.unravel_index(5, (2, 3))", "np.unravel_index(np.array([[1, 5], [3, 4]]), (2, 3))", "!np.unravel_index(np.array([6]), (2, 3))", "np.unravel_index(0, ())", "np.unravel_index(np.array([5]), 6)",
  "np.ravel_multi_index((np.array([3, 6, 6]), np.array([4, 5, 1])), (7, 6))", "np.ravel_multi_index((np.array([3, 6, 6]), np.array([4, 5, 1])), (7, 6), order='F')", "np.ravel_multi_index((np.array([3, 1]), np.array([4, 7])), (7, 6), mode='wrap')", "np.ravel_multi_index((np.array([3, 1]), np.array([4, 7])), (7, 6), mode='clip')",
  "np.ravel_multi_index(np.array([[1, 2], [3, 0]]), (4, 4))", "np.ravel_multi_index((np.array([1, 5]), np.array([1, 2])), (4, 4), mode=('clip', 'wrap'))", "!np.ravel_multi_index((np.array([9]), np.array([1])), (4, 4))",
  "np.diag_indices(3)", "np.diag_indices(2, 3)", `np.diag_indices_from(${SQ})`, "!np.diag_indices_from(np.arange(4))", "!np.diag_indices_from(np.zeros((2, 3)))",
  "np.tril_indices(3)", "np.tril_indices(3, 1)", "np.tril_indices(3, -1)", "np.tril_indices(3, 0, 4)", "np.triu_indices(3)", "np.triu_indices(3, 1)", "np.triu_indices(3, -1, 4)", "np.triu_indices(4, 0, 2)",
  `np.tril_indices_from(${SQ})`, `np.triu_indices_from(${SQ}, 1)`, `np.tril_indices_from(${A}, -1)`, `!np.tril_indices_from(${B})`,
  [() => np.mask_indices(3, np.triu), "np.mask_indices(3, np.triu)"], [() => np.mask_indices(3, np.tril, -1), "np.mask_indices(3, np.tril, -1)"],
]);

numpy_cases("functional tools", [
  [() => np.apply_over_axes(np.sum, np.arange(24).reshape(2, 3, 4), [0, 2]), "np.apply_over_axes(np.sum, np.arange(24).reshape(2, 3, 4), [0, 2])"],
  [() => np.apply_over_axes(np.sum, np.arange(24).reshape(2, 3, 4), [-1, 0]), "np.apply_over_axes(np.sum, np.arange(24).reshape(2, 3, 4), [-1, 0])"],
  [() => np.apply_over_axes((a: any, ax: number) => np.max(a, ax), np.arange(24).reshape(2, 3, 4), 1), "np.apply_over_axes(np.max, np.arange(24).reshape(2, 3, 4), 1)"],
  [() => np.vectorize((a: number, b: number) => a > b ? a - b : a + b)(np.array([1, 2, 3, 4]), 2), "np.vectorize(lambda a, b: a - b if a > b else a + b)(np.array([1, 2, 3, 4]), 2)"],
  [() => np.vectorize((a: number) => a * 0.5)(np.arange(6).reshape(2, 3)), "np.vectorize(lambda a: a * 0.5)(np.arange(6).reshape(2, 3))"],
  [() => np.vectorize((a: number) => a % 2 == 0)(np.arange(4)), "np.vectorize(lambda a: a % 2 == 0)(np.arange(4))"],
  [() => np.vectorize((a: number, k: number) => a * k, { excluded: [1] })(np.array([1, 2, 3]), 10), "np.vectorize(lambda a, k: a * k, excluded=[1])(np.array([1, 2, 3]), 10)"],
  [() => np.vectorize((a: number) => a + 1, { otypes: ['float64'] })(np.array([1, 2])), "np.vectorize(lambda a: a + 1, otypes=[float])(np.array([1, 2]))"],
  [() => np.vectorize((a: number, b: number) => a + b)(np.array([[1], [2]]), np.array([10, 20])), "np.vectorize(lambda a, b: a + b)(np.array([[1], [2]]), np.array([10, 20]))"],
  [() => np.frompyfunc((a: number, b: number) => a + b, 2, 1)(np.array([1, 2]), np.array([10, 20])).tolist(), "np.frompyfunc(lambda a, b: a + b, 2, 1)(np.array([1, 2]), np.array([10, 20])).tolist()"],
  [() => np.frompyfunc((a: number) => [a, a * 2], 1, 1)(np.array([1, 2])).tolist(), "np.frompyfunc(lambda a: [a, a * 2], 1, 1)(np.array([1, 2])).tolist()"],
  [() => np.frompyfunc((a: number) => [a, a * 2], 1, 2)(np.array([1, 2])).map((x: any) => x.tolist()), "[x.tolist() for x in np.frompyfunc(lambda a: (a, a * 2), 1, 2)(np.array([1, 2]))]"],
  [() => (() => { const x = np.array([-2, -1, 0, 1, 2], 'float64'); return np.piecewise(x, [x.less(0), x.greater_equal(0)], [-1, 1]); })(), "np.piecewise(np.array([-2.0, -1, 0, 1, 2]), [np.array([-2.0, -1, 0, 1, 2]) < 0, np.array([-2.0, -1, 0, 1, 2]) >= 0], [-1, 1])"],
  [() => (() => { const x = np.array([-2, -1, 0, 1, 2], 'float64'); return np.piecewise(x, [x.less(0)], [(v: any) => np.abs(v), (v: any) => np.multiply(v, 10)]); })(), "np.piecewise(np.array([-2.0, -1, 0, 1, 2]), [np.array([-2.0, -1, 0, 1, 2]) < 0], [lambda x: np.abs(x), lambda x: x * 10])"],
  [() => np.piecewise(np.arange(6), [np.arange(6).less(2), np.arange(6).greater(3)], [0, (x: any) => np.multiply(x, x), 5]), "np.piecewise(np.arange(6), [np.arange(6) < 2, np.arange(6) > 3], [0, lambda x: x * x, 5])"],
  [() => np.piecewise(np.arange(6), [np.arange(6).less(2)], [7, 8]), "np.piecewise(np.arange(6), [np.arange(6) < 2], [7, 8])"],
  [() => [...np.ndindex(2, 2)], "[list(i) for i in np.ndindex(2, 2)]"], [() => [...np.ndindex([2, 3])].length, "len(list(np.ndindex(2, 3)))"],
  [() => [...np.ndenumerate(np.array([[1, 2], [3, 4]]))].map(([i, v]: any) => [i, v]), "[[list(i), int(v)] for i, v in np.ndenumerate(np.array([[1, 2], [3, 4]]))]"],
  [() => [...np.nditer(np.arange(6).reshape(2, 3))], "[int(x) for x in np.nditer(np.arange(6).reshape(2, 3))]"],
  [() => [...np.nditer([np.arange(3), np.array([[10], [20]])])].map((p: any) => [...p]), "[[int(a), int(b)] for a, b in np.nditer([np.arange(3), np.array([[10], [20]])])]"],
  [() => { const out: any[] = []; for (const row of np.nested_iters(np.arange(6).reshape(2, 3), [[0], [1]])[0]) out.push([...row]); return out; }, "[[0, 1, 2], [3, 4, 5]]"],
  [() => { const it = new np.flatiter(np.arange(6).reshape(2, 3)); return [it.get(4), it.length, [...it]]; }, "(lambda it: [int(it[4]), len(list(it)), [int(x) for x in np.arange(6).reshape(2, 3).flat]])(np.arange(6).reshape(2, 3).flat)"],
  [() => [np.iterable([1]), np.iterable(5), np.iterable(np.arange(3)), np.iterable('abc')], "[np.iterable([1]), np.iterable(5), np.iterable(np.arange(3)), np.iterable('abc')]"],
  "np.emath.sqrt(np.array([4.0, 9.0]))", "np.emath.log(np.array([1.0, np.e]))", "np.emath.log2(np.array([1.0, 8.0]))", "np.emath.log10(np.array([10.0, 1000.0]))", "np.emath.logn(2, np.array([2.0, 8.0]))", "np.emath.power(np.array([4.0, 9.0]), 0.5)", "np.emath.arccos(np.array([0.5, 1.0]))", "np.emath.arcsin(np.array([0.5, 1.0]))", "np.emath.arctanh(np.array([0.5, 0.0]))",
]);

numpy_cases("dot", [
  "np.dot(3, 4)", "np.dot(3, np.array([1, 2]))", "np.dot(np.array([1, 2]), 3)", "np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))", "np.dot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))",
  "np.dot(np.array([[1, 2], [3, 4]]), np.array([1, 1]))", "np.dot(np.array([1, 1]), np.array([[1, 2], [3, 4]]))", "np.dot(np.array([[1.5, 2], [3, 4]]), np.array([[0.5], [2]]))",
  `np.dot(${B}, np.arange(4))`, `np.dot(${B}, np.arange(8).reshape(4, 2))`, `np.dot(${A}, ${B}.reshape(2, 12)[:, :3].T.reshape(3, 4)[:, :2].T)`.replace(/np\.dot\(.*\)/, "np.dot(np.arange(6).reshape(2, 3), np.arange(24).reshape(2, 3, 4))"),
  "np.dot(np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 4, 3))", "np.dot(np.array([True, False]), np.array([True, True]))", "np.dot(np.array([1, 2], dtype='float32'), np.array([1, 2], dtype='float32'))",
  "!np.dot(np.array([1, 2, 3]), np.array([1, 2]))", "!np.dot(np.ones((2, 3)), np.ones((2, 3)))", "np.dot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))[0, 1]",
  "np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])", "np.array([1, 2]) @ np.array([3, 4])",
]);
