import { numpy_cases } from "../helpers/cases";

const F = "np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.7])";
const I = "np.array([-3, -1, 0, 2, 5])";
const I_NZ = "np.array([-3, -1, 2, 5])";
const B = "np.array([True, False, True])";
const S = "np.array([[0.1, 0.5], [0.9, -0.3]])";
const SPECIAL = "np.array([np.nan, np.inf, -np.inf, 0.0, 1.0])";

const unary_float = ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "arcsinh", "asinh",
  "arctanh", "atanh", "exp", "exp2", "expm1", "log1p", "sqrt", "cbrt", "degrees", "radians", "deg2rad", "rad2deg", "sinc", "i0",
  "floor", "ceil", "trunc", "fix", "rint", "round", "around", "square", "sign", "absolute", "abs", "fabs", "negative", "positive",
  "reciprocal", "log", "log2", "log10", "arccosh", "acosh", "spacing", "signbit", "isnan", "isfinite", "isinf", "isposinf", "isneginf",
  "real", "imag", "conj", "conjugate", "angle", "iscomplex", "isreal", "real_if_close", "nan_to_num"];

numpy_cases("unary on floats", [
  ...unary_float.flatMap(f => [`np.${f}(${F})`, `np.${f}(${S})`, `np.${f}(${SPECIAL})`]),
]);
numpy_cases("unary on ints", [
  ...["floor", "ceil", "trunc", "rint", "round", "square", "sign", "abs", "negative", "positive", "sqrt", "exp", "sin", "reciprocal", "isnan", "signbit", "real", "imag", "conj", "nan_to_num", "fix", "degrees", "cbrt", "log1p", "expm1", "exp2", "sinc", "i0"]
    .filter(f => f != "reciprocal").map(f => `np.${f}(${I})`), `np.reciprocal(${I_NZ})`,
]);
numpy_cases("unary on bools and float32", [
  ...["abs", "floor", "square", "logical_not", "isnan", "bitwise_not", "invert", "real"].map(f => `np.${f}(${B})`),
  "!np.negative(np.array([True, False]))", "!np.sign(np.array([True]))", "!np.bitwise_not(np.array([1.5]))",
  ...["sin", "exp", "sqrt", "floor", "abs", "negative", "square", "sign", "reciprocal", "rint", "cbrt", "spacing"].map(f => `np.${f}(np.array([0.5, 1.5, -2.25], dtype='float32'))`),
  ...["sin", "exp", "sqrt", "abs", "cumsum", "sum", "mean", "max", "std"].map(f => `np.${f}(np.array([0.5, 1.5, 2.25], dtype='float32'))`),
]);
numpy_cases("round with decimals", [
  `np.round(np.array([1.2345, 2.5, 3.5, -2.5, 0.125, 1234.5678]), 2)`, `np.round(np.array([1.2345, 2.5, 3.5, -2.5, 0.125]), 0)`,
  `np.round(np.array([1234.5678, 15.5, 25.5]), -1)`, `np.round(np.array([1234, 1250, 1350, 1450]), -2)`, `np.round(np.array([1, 2, 3]), 2)`,
  `np.around(np.array([0.5, 1.5, 2.5]))`, `np.round(np.array([2.675, 1.005]), 2)`, `np.array([1.55, 2.45]).round(1)`,
  `np.round(2.5)`, `np.round(-0.4)`, `np.round(np.array([1.5, 2.5]), decimals=0)`,
]);
numpy_cases("nan_to_num", [
  `np.nan_to_num(np.array([np.nan, np.inf, -np.inf, 1.5]))`, `np.nan_to_num(np.array([np.nan, np.inf, -np.inf]), nan=1.0, posinf=2.0, neginf=3.0)`,
  `np.nan_to_num(np.array([np.nan, np.inf]), nan=-1)`, `np.nan_to_num(np.array([1, 2]))`,
  `np.nan_to_num(np.array([np.inf], dtype='float32'))`,
]);
numpy_cases("modf frexp divmod", [
  `np.modf(np.array([-2.5, 0.0, 3.25, np.inf]))`, `np.frexp(np.array([8.0, 0.5, -3.0, 0.0, 1.0, 1e300]))`, `np.frexp(np.array([1, 2, 3]))`,
  `np.divmod(np.array([7, -7, 7]), np.array([2, 2, -2]))`, `np.divmod(np.array([7.5, -7.5]), 2)`, `np.divmod(7, 3)`,
  `np.ldexp(np.array([1.0, 0.5]), np.array([3, 2]))`, `np.ldexp(1.5, 4)`,
]);
numpy_cases("binary float functions", [
  "np.hypot(np.array([3., 5.]), np.array([4., 12.]))", "np.hypot(3, 4)", "np.arctan2(np.array([1., -1., 0.]), np.array([1., -1., -1.]))", "np.atan2(1, 2)",
  "np.arctan2(np.array([0.0, -0.0]), np.array([-1.0, -1.0]))", "np.copysign(np.array([1., 2., -3.]), np.array([-1., 1., -0.0]))", "np.copysign(1.5, -2)",
  "np.logaddexp(np.array([1., 2.]), np.array([3., 1.]))", "np.logaddexp2(np.array([1., 2.]), np.array([3., 1.]))",
  "np.logaddexp(-np.inf, -np.inf)", "np.logaddexp(np.inf, np.inf)", "np.logaddexp(1000., 1000.)",
  "np.nextafter(np.array([1.0, 0.0, -1.0]), np.array([2.0, 1.0, 0.0]))", "np.nextafter(np.array([1.0], dtype='float32'), np.array([2.0], dtype='float32'))",
  "np.spacing(np.array([1.0, 1e10, 0.0, -1.0]))", "np.spacing(np.array([1.0, 1e5], dtype='float32'))",
  "np.heaviside(np.array([-1.5, 0.0, 2.0]), 0.5)", "np.heaviside(np.array([-1.5, 0.0, 2.0, np.nan]), np.array([1., 2., 3., 4.]))",
  "np.float_power(np.array([2, 3]), 2)", "np.float_power(np.array([2, 3]), np.array([0.5, 3]))", "np.float_power(np.array([-8.]), 1/3)",
  "np.power(np.array([2, 3, 4]), 3)", "np.power(2.5, np.array([1, 2]))", "np.power(np.array([2, 3]), np.array([True, False]))",
  "np.fmod(np.array([-7, 7, -7]), np.array([3, -3, -3]))", "np.fmod(np.array([-7.5, 7.5]), 2)",
  "np.mod(np.array([-7, 7, -7, 7]), np.array([3, -3, -3, 3]))", "np.remainder(np.array([-7.5, 7.5]), 2)", "np.mod(5, 0)", "np.mod(5., 0)", "np.floor_divide(5, 0)", "np.floor_divide(-5., 0)",
  "np.floor_divide(np.array([7, -7, 7.5]), 2)", "np.true_divide(np.array([1, 2]), np.array([4, 4]))", "np.divide(1, 0.)", "np.divide(np.array([0.0]), 0.0)",
]);
numpy_cases("gcd lcm and bitwise", [
  "np.gcd(np.array([12, 18, 0, -4]), np.array([8, 27, 5, 6]))", "np.lcm(np.array([4, 6, 0, -3]), np.array([6, 8, 5, 4]))", "np.gcd(12, 18)", "np.lcm(np.array([2]), np.array([3, 4, 5]))",
  "!np.gcd(np.array([1.5]), np.array([2]))",
  "np.bitwise_and(np.array([12, 10]), 6)", "np.bitwise_or(np.array([12, 10]), 1)", "np.bitwise_xor(np.array([12, 10]), 5)",
  "np.bitwise_left_shift(np.array([1, 2]), 3)", "np.bitwise_right_shift(np.array([16, 32]), 2)",
  "np.bitwise_and(np.array([True, False]), np.array([True, True]))",
]);
numpy_cases("comparison and logical", [
  "np.greater(np.array([1, 2, 3]), 2)", "np.less_equal(np.array([1., np.nan, 3.]), 2)", "np.equal(np.array([1, 2]), np.array([1., 2.5]))", "np.not_equal(np.array([1, 2]), np.array([[1], [2]]))",
  "np.logical_and(np.array([1, 0, 2]), np.array([1, 1, 0]))", "np.logical_or(np.array([1.5, 0]), np.array([0, 0]))", "np.logical_xor(np.array([True, False]), np.array([True, True]))",
  "np.logical_not(np.array([0, 1, 2.5]))", "np.logical_and(np.array([[True], [False]]), np.array([True, False]))",
  "np.isclose(np.array([1.0, 1.00001, 1.1]), 1.0)", "np.isclose(np.array([1e10, 1e-8]), np.array([1.00001e10, 1e-9]))", "np.isclose(np.array([np.nan, np.inf, -np.inf]), np.array([np.nan, np.inf, np.inf]))",
  "np.isclose(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), equal_nan=True)", "np.isclose(np.array([1.0, 2.0]), np.array([1.1, 2.1]), rtol=0.1, atol=0)", "np.isclose(1.0, 1.0 + 1e-9)",
  "np.isclose(np.array([100.0, 200.0]), np.array([101.0, 202.0]), 0.0, 1.5)",
  "np.allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-9]))", "np.allclose(np.array([1.0, 2.0]), np.array([1.0, 2.1]))", "np.allclose(np.array([1.0, np.nan]), np.array([1.0, np.nan]), equal_nan=True)",
  "np.allclose(np.array([1.0, np.nan]), np.array([1.0, np.nan]))", "np.allclose([1, 2], [1, 2])", "np.allclose(np.array([[1.0, 2.0]]), np.array([[1.0], [2.0]]))",
  "np.array_equal(np.array([1, 2]), np.array([1, 2]))", "np.array_equal(np.array([1, 2]), np.array([1, 3]))", "np.array_equal(np.array([1, 2]), np.array([[1, 2]]))", "np.array_equal(np.array([np.nan]), np.array([np.nan]))",
  "np.array_equal(np.array([np.nan]), np.array([np.nan]), equal_nan=True)", "np.array_equal([1, 2], [1.0, 2.0])",
  "np.array_equiv(np.array([1, 2]), np.array([[1, 2], [1, 2]]))", "np.array_equiv(np.array([1, 2]), np.array([[1, 2], [1, 3]]))", "np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))",
  "np.isscalar(5)", "np.isscalar(np.array([1]))", "np.isscalar([1])", "np.isscalar(True)",
]);
numpy_cases("clip and sign handling", [
  "np.clip(np.array([1, 5, 10]), 2, 8)", "np.clip(np.array([1., 5., 10.]), 2, None)", "np.clip(np.array([1., 5., 10.]), None, 8)", "np.clip(np.array([1., 5., 10.]), 2.5, 7.5)",
  "np.clip(np.array([1, 5, 10]), np.array([2, 6, 6]), 8)", "np.clip(np.array([np.nan, 1.]), 0, 0.5)", "np.clip(np.array([[1, 5], [10, 0]]), np.array([2, 3]), np.array([4, 6]))",
  "np.array([1, 5, 10]).clip(2, 8)", "np.clip(np.array([1.5, 2.5]), a_min=2, a_max=2.2)",
]);
