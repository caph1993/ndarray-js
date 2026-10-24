import { numpy_cases } from "../helpers/cases";

const M = "np.array([[1.5, 2.0, -3.0, 4.0], [4.0, 5.0, 6.5, -1.0], [0.5, 0.25, 8.0, 2.0]])";
const I = "np.array([[1, 2, 3], [4, 5, 6]])";
const NANS = "np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]])";

numpy_cases("cumulative", [
  ...["cumsum", "cumprod", "nancumsum", "nancumprod"].flatMap(f => [
    `np.${f}(${I})`, `np.${f}(${I}, axis=0)`, `np.${f}(${I}, axis=1)`, `np.${f}(${I}, axis=-1)`, `np.${f}(${M}, axis=1)`, `np.${f}(${NANS}, axis=0)`, `np.${f}(${NANS})`,
    `np.${f}(np.array([True, True, False]))`, `np.${f}(np.array([1, 2, 3]), dtype='float64')`, `np.${f}(np.array([1.5, 2.5, 3.5]))`, `np.${f}(np.array([]))`,
  ]),
  `${I}.cumsum(axis=1)`, `${I}.cumprod(1)`, "np.cumsum(5)", "np.cumsum([1, 2, 3])",
  `np.cumulative_sum(np.array([1, 2, 3]))`, `np.cumulative_sum(${I}, axis=1)`, `np.cumulative_sum(${I}, axis=0, include_initial=True)`, `np.cumulative_prod(np.array([1, 2, 3, 4]), include_initial=True)`,
  "!np.cumulative_sum(np.array([[1, 2], [3, 4]]))", "np.cumsum(np.array([2**40, 2**40]))",
]);
numpy_cases("diff and friends", [
  `np.diff(np.array([1, 4, 9, 16]))`, `np.diff(np.array([1, 4, 9, 16]), n=2)`, `np.diff(np.array([1, 4, 9, 16]), n=3)`, `np.diff(np.array([1, 4, 9, 16]), n=5)`, `np.diff(np.array([1, 4, 9, 16]), n=0)`,
  `np.diff(${M})`, `np.diff(${M}, axis=0)`, `np.diff(${M}, n=2, axis=1)`, `np.diff(${M}, axis=-2)`, `np.diff(np.array([1, 2, 4]), prepend=0)`, `np.diff(np.array([1, 2, 4]), append=10)`, `np.diff(np.array([1, 2, 4]), prepend=np.array([-1, 0]), append=np.array([7]))`,
  `np.diff(np.array([True, False, False, True]))`, `np.diff(np.array([1.5, np.nan, 3.0]))`, `np.diff(${I}, prepend=0, axis=1)`, `np.diff(np.array([3.5]))`,
  "!np.diff(np.array(5))", "!np.diff(np.array([1, 2]), n=-1)",
  `np.ediff1d(np.array([1, 2, 4, 7]))`, `np.ediff1d(np.array([1, 2, 4, 7]), to_end=np.array([99]), to_begin=np.array([-99, -98]))`, `np.ediff1d(${I})`, `np.ediff1d(np.array([1, 2, 4]), to_begin=-1)`,
  `np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]))`, `np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]), 2.0)`, `np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]), edge_order=2)`,
  `np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]), np.array([0.0, 1.0, 1.5, 3.5, 4.0]))`, `np.gradient(np.array([1.0, 2.0, 4.0, 7.0, 11.0]), np.array([0.0, 1.0, 1.5, 3.5, 4.0]), edge_order=2)`,
  `np.gradient(${M})`, `np.gradient(${M}, axis=1)`, `np.gradient(${M}, axis=0)`, `np.gradient(${M}, 2.0, 3.0)`, `np.gradient(${M}, 0.5, axis=1)`, `np.gradient(np.array([1, 2, 4, 8]))`, `np.gradient(np.arange(5) ** 2, edge_order=2)`,
  "!np.gradient(np.array([1.0]))",
  `np.cross(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))`, `np.cross(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [1, 0, 2]]))`, `np.cross(np.array([1, 2, 3]), np.array([[7, 8, 9], [1, 0, 2]]))`,
  "!np.cross(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))",
  "!np.cross(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))",
  `np.trapezoid(np.array([1.0, 2.0, 3.0]))`, `np.trapezoid(np.array([1.0, 2.0, 3.0]), dx=0.5)`, `np.trapezoid(np.array([1.0, 2.0, 3.0]), x=np.array([0.0, 1.0, 3.0]))`,
  `np.trapezoid(${M})`, `np.trapezoid(${M}, axis=0)`, `np.trapezoid(${M}, x=np.array([0.0, 1.0, 3.0, 3.5]))`, `np.trapezoid(np.array([1, 2, 3]))`, `np.trapezoid(np.array([[1.0, 2.0], [3.0, 4.0]]), x=np.array([[0.0, 1.0], [0.0, 3.0]]))`,
]);
numpy_cases("convolve, correlate, interp, unwrap", [
  ...["full", "same", "valid"].flatMap(m => [
    `np.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]), mode='${m}')`, `np.convolve(np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1, 1]), mode='${m}')`, `np.convolve(np.array([1, 2]), np.array([1, 2, 3, 4, 5]), mode='${m}')`,
    `np.convolve(np.array([1, 2, 3, 4]), np.array([1, 1]), mode='${m}')`, `np.convolve(np.array([1, 2, 3]), np.array([5]), mode='${m}')`,
    `np.correlate(np.array([1, 2, 3]), np.array([0, 1, 0.5]), mode='${m}')`, `np.correlate(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4]), mode='${m}')`, `np.correlate(np.array([1, 2]), np.array([1, 2, 3, 4, 5]), mode='${m}')`,
    `np.correlate(np.array([1, 2, 3, 4]), np.array([1, 2]), mode='${m}')`, `np.correlate(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5, 6]), mode='${m}')`,
  ]),
  "np.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]))", "np.correlate(np.array([1, 2, 3]), np.array([0, 1, 0.5]))", "!np.convolve(np.array([]), np.array([1]))", "!np.convolve(np.array([1]), np.array([1]), mode='foo')",
"np.convolve([1, 2, 3], [1, 1])",
  "np.interp(np.array([0.5, 1.5, 2.5]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 10.0, 20.0, 40.0]))", "np.interp(np.array([-1.0, 0.0, 3.0, 5.0]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 10.0, 20.0, 40.0]))",
  "np.interp(np.array([-1.0, 5.0]), np.array([0.0, 3.0]), np.array([1.0, 2.0]), left=-5.0, right=99.0)", "np.interp(2.5, np.array([1, 2, 3]), np.array([3, 2, 0]))", "np.interp(np.array([[0.5, 1.5]]), np.array([0, 1, 2]), np.array([0, 1, 4]))",
  "np.interp(np.array([np.nan, 1.0]), np.array([0.0, 2.0]), np.array([0.0, 4.0]))", "np.interp(np.array([-180.0, 190.0, 370.0]), np.array([0.0, 90.0, 180.0]), np.array([0.0, 1.0, 0.0]), period=360)",
  "np.interp(np.array([1.0, 2.0]), np.array([1.0, 1.0, 2.0]), np.array([0.0, 5.0, 6.0]))", "!np.interp(1.0, np.array([1.0, 2.0]), np.array([1.0]))", "np.interp(0.5, np.array([0.0, 1.0]), np.array([np.inf, 1.0]))",
  "np.unwrap(np.array([0.0, 3.0, 6.0, 9.0, 12.0]))", "np.unwrap(np.array([0.0, 4.0, 8.0, -1.0, 2.0]))", "np.unwrap(np.array([0, 1, 2, 3, 4, 5, 6]) * 1.5)", "np.unwrap(np.array([0.0, 100.0, 200.0, 300.0]), period=360)",
  "np.unwrap(np.array([0.0, 7.0, 14.0]), discont=3.0)", "np.unwrap(np.array([[0.0, 4.0, 8.0], [1.0, 5.0, 9.0]]))", "np.unwrap(np.array([[0.0, 4.0, 8.0], [1.0, 5.0, 9.0]]), axis=0)",
]);
numpy_cases("sinc, i0, gcd...", [
  "np.sinc(np.array([-1.0, -0.5, 0.0, 0.25, 1.0, 2.5]))", "np.i0(np.array([0.0, 0.5, 1.0, 5.0, 20.0, -3.0]))", "np.sinc(0)", "np.i0(2.0)",
  "np.unwrap(np.array([0, 3, 6]))",
]);
