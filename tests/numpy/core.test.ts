import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const X = "np.array([[1,2,3],[4,5,6]])";
const Y = "np.arange(24).reshape(2,3,4)";

numpy_cases("indexing", [
  `${X}[0]`, `${X}[1, 2]`, `${X}[:, 1]`, `${X}[-1]`, `${X}[:, -1]`, `${X}[0:1]`, `${X}[:, ::2]`, `${X}[:, ::-1]`,
  `${X}[::-1]`, `np.arange(10)[-1:-1:-2]`, `np.arange(10)[8:2:-3]`, `np.arange(10)[::3]`, `np.arange(10)[2:-2]`, `np.arange(10)[-3:]`,
  `np.arange(10)[:-3]`, `np.arange(10)[100:]`, `np.arange(10)[5:2]`, `np.arange(10)[-100:3]`,
  `${Y}[..., 1]`, `${Y}[0, ...]`, `${Y}[1, ..., 2]`, `${Y}[:, 1]`, `${Y}[:, 1:3, ::2]`, `${Y}[::-1, 1, ::-2]`,
  `${Y}[1][2]`, `${Y}[1][2][3]`, `${Y}[1, 2, 3]`, `${Y}[:, None, 1]`, `${Y}[None]`,
  `np.arange(5)[np.arange(5) > 2]`, `${X}[np.array([True, False])]`,
  `${X}[:, np.array([True, False, True])]`, `np.arange(5)[[0, 2, -1]]`,
  "!np.arange(5)[5]", "!np.arange(5)[-6]", "!np.arange(5)[[0, 7]]",
  `${X}[0][1:]`, `${X}[:, 1:][0]`, `${X}[:, 1:][:, ::-1]`, `${X}[:, ::-1][0, 1:]`,
  `${X}[:, ::-1][:, 1:][0]`, `${Y}[:, ::-1][1][:, 1:]`,
]);

numpy_cases("broadcasting and dtypes", [
  `${X} + 1`, `${X} + 1.5`, `${X} * np.array([1, 2, 3])`, `${X} - np.array([[1], [2]])`, `np.array([[1],[2]]) + np.array([1,2,3])`,
  `${X} / 2`, `${X} // 2`, `${X} % 4`, `-${X} % 4`, `np.array([-7, 7]) // 2`, `np.array([-7., 7.]) % 3`, `np.array([-7., 7.]) % -3`,
  `${X} ** 2`, `np.array([2, 3]) ** np.array([3, 2])`, `np.array([2., 3.]) ** 0.5`,
  `np.array([True, False]) + np.array([True, True])`, `np.array([True, False]) * 3`, `np.array([True, False]) & np.array([True, True])`,
  `np.array([1, 2]) | np.array([4, 8])`, `np.array([1, 2]) ^ np.array([3, 3])`, `np.array([1, 2]) << 2`, `np.array([8, 16]) >> 2`,
  `np.array([1, 2, 3]) > 2`, `np.array([1, 2, 3]) == np.array([1, 5, 3])`, `np.array([1.0, np.nan]) == np.array([1.0, np.nan])`,
  "np.array([1, 2], dtype='float32') + 1.5", "np.array([1, 2], dtype='float32') + np.array([1, 2])",
  "!np.array([1, 2]) + np.array([1, 2, 3])", "!np.array([True]) - np.array([True])",
  "!np.array([1.5]) | np.array([1])", "!np.array([2]) ** np.array([-1])",
  `np.maximum(${X}, 3)`, `np.minimum(${X}, np.array([2, 4, 6]))`, `np.maximum(np.array([1., np.nan]), np.array([np.nan, 2.]))`,
  `np.fmax(np.array([1., np.nan]), np.array([np.nan, 2.]))`, `np.fmin(np.array([1., np.nan]), np.array([np.nan, 2.]))`,
  `np.add(${X}, ${X})`, `np.abs(np.array([-1, 2]))`, `np.negative(np.array([1, -2]))`,
  `np.bitwise_not(np.array([0, 1, 5]))`, `np.logical_not(np.array([0, 1, 5]))`,
]);

numpy_cases("reshape and transpose", [
  `${Y}.reshape(6, 4)`, `${Y}.reshape(-1, 4)`, `${Y}.reshape(2, -1)`, `${Y}.reshape([4, 6])`, `${Y}.ravel()`, `np.ravel(${Y})`,
  `${Y}.T`, `np.transpose(${Y})`, `np.transpose(${Y}, [1, 0, 2])`, `${Y}.transpose([2, 0, 1])`, `${Y}.transpose()`,
  `${Y}[:, ::-1].reshape(-1)`, `${Y}[:, :, 1].T`, `${Y}[0].T.reshape(-1)`,
  "!np.arange(5).reshape(2, 3)", "!np.arange(5).reshape(-1, 2)",
  `np.concatenate([${X}, ${X}])`, `np.concatenate([${X}, ${X}], axis=1)`, `np.concatenate([${X}, ${X}], axis=None)`, `np.stack([${X}, ${X}])`,
  `np.stack([${X}, ${X}], axis=1)`, `np.stack([${X}, ${X}], axis=-1)`, "np.concatenate([np.array([1, 2]), np.array([1.5])])",
  "!np.concatenate([np.array([[1, 2]]), np.array([1, 2])])",
]);

const A = "np.arange(6).reshape(2, 3)";
numpy_cases("assignment (in place)", [
  ["np.arange(6).reshape(2, 3).assign(9, 0)", "(lambda a: (a.__setitem__(0, 9), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6).reshape(2, 3).assign(9, ':', 1)", "(lambda a: (a.__setitem__((slice(None), 1), 9), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6).reshape(2, 3).add_assign(10, 1)", "(lambda a: (a.__setitem__(1, a[1] + 10), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6).reshape(2, 3).multiply_assign(np.array([1, 10, 100]))", "(lambda a: (a.__imul__(np.array([1, 10, 100])), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6).reshape(2, 3).assign(np.array([7, 8, 9]), 1)", "(lambda a: (a.__setitem__(1, np.array([7, 8, 9])), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6).reshape(2, 3).assign(0, '...', '::2')", "(lambda a: (a.__setitem__((Ellipsis, slice(None, None, 2)), 0), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6).assign(-1, np.arange(6) > 3)", "(lambda a: (a.__setitem__(np.arange(6) > 3, -1), a)[1])(np.arange(6))"],
  ["np.arange(6).assign(-1, [0, 5])", "(lambda a: (a.__setitem__([0, 5], -1), a)[1])(np.arange(6))"],
  [() => { const a = np.arange(6).reshape(2, 3); a.index(':', 1).add_assign(100); return a; }, "(lambda a: (a.__setitem__((slice(None), 1), a[:, 1] + 100), a)[1])(np.arange(6).reshape(2, 3))"],
  [() => { const a = np.arange(6).reshape(2, 3); a.index(1).index('::-1').assign(0); return a; }, "(lambda a: (a.__setitem__((1, slice(None, None, -1)), 0), a)[1])(np.arange(6).reshape(2, 3))"],
  ["np.arange(6, dtype='float64').divide_assign(2)", "(lambda a: (a.__itruediv__(2), a)[1])(np.arange(6, dtype='float64'))"],
  ["!np.arange(6).divide_assign(2)", "(lambda a: a.__itruediv__(2))(np.arange(6))"],
  ["!np.arange(6).add_assign(1.5)", "(lambda a: a.__iadd__(1.5))(np.arange(6))"],
]);

numpy_cases("apply_along_axis", [
  [() => np.apply_along_axis((r: any) => r.sum(), 1, np.arange(6).reshape(2, 3)), "np.apply_along_axis(np.sum, 1, np.arange(6).reshape(2, 3))"],
  [() => np.apply_along_axis((r: any) => r.sum(), 0, np.arange(6).reshape(2, 3)), "np.apply_along_axis(np.sum, 0, np.arange(6).reshape(2, 3))"],
  [() => np.apply_along_axis((r: any) => np.sort(r), 0, np.array([[3, 1], [2, 5]])), "np.apply_along_axis(np.sort, 0, np.array([[3, 1], [2, 5]]))"],
  [() => np.apply_along_axis((r: any) => np.array([r.min(), r.max()]), 1, np.array([[3, 1, 2], [5, 9, 7]])), "np.apply_along_axis(lambda r: np.array([r.min(), r.max()]), 1, np.array([[3, 1, 2], [5, 9, 7]]))"],
  [() => np.apply_along_axis((r: any) => np.array([r.min(), r.max()]), 0, np.arange(12).reshape(3, 4)), "np.apply_along_axis(lambda r: np.array([r.min(), r.max()]), 0, np.arange(12).reshape(3, 4))"],
  [() => np.apply_along_axis((r: any, k: any) => r.sum() * k, 1, np.arange(6).reshape(2, 3), 10), "np.apply_along_axis(lambda r, k: r.sum() * k, 1, np.arange(6).reshape(2, 3), 10)"],
  // Integer-valued floats returned as JS numbers can not be told apart from ints
  [() => np.apply_along_axis((r: any) => r.mean(), 1, np.arange(24).reshape(2, 3, 4)), "np.apply_along_axis(np.mean, 1, np.arange(24).reshape(2, 3, 4))"],
], { check_dtype: false });

numpy_cases("slice assignment in the template parser", [
  [() => { const a = np.arange(6); np`${a}[1:3] = 9`; return a; }, "np.array([0, 9, 9, 3, 4, 5])"],
  [() => { const a = np.arange(6).reshape(2, 3); np`${a}[:, 0] += 10`; return a; }, "np.array([[10, 1, 2], [13, 4, 5]])"],
  [() => { const a = np.arange(6).reshape(2, 3); np`${a}[1, ...] *= 2`; return a; }, "np.array([[0, 1, 2], [6, 8, 10]])"],
]);
