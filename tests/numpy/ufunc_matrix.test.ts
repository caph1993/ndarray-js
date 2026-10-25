import { numpy_cases } from "../helpers/cases";

// Every unary / binary ufunc against many dtypes, special values and broadcasting shapes, compared with numpy.

const UNARY = "sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh exp exp2 expm1 log log2 log10 log1p sqrt cbrt square abs absolute fabs negative positive sign floor ceil trunc rint fix isnan isinf isfinite signbit reciprocal deg2rad rad2deg degrees radians i0 nan_to_num spacing logical_not conj conjugate real imag angle".split(" ");
const U_IN: { [k: string]: string } = {
  f64: "np.array([-2.5, -1., 0., 0.5, 1., 2., 3.5, np.inf, -np.inf, np.nan])",
  f32: "np.array([-2.5, -1., 0.5, 1., 3.5], dtype='float32')",
  i32: "np.array([-3, -1, 0, 1, 2, 100])",
  i8: "np.array([-3, -1, 0, 1, 2, 100], dtype='int8')",
  u8: "np.array([0, 1, 2, 200, 255], dtype='uint8')",
  bool: "np.array([True, False, True])",
  f64_2d: "np.array([[0.1, 0.5, 0.9], [-0.3, 0.2, 0.7]])",
  c128: "np.array([1+2j, -1.5+0.5j, 0j, 3-1j])",
  empty: "np.array([], dtype='float64')",
};
const U_SKIP_COMPLEX = ["i0", "spacing", "signbit", "fix"];
const unary: string[] = [];
for (const u of UNARY) for (const [k, v] of Object.entries(U_IN)) {
  if (k === "c128" && U_SKIP_COMPLEX.includes(u)) continue;
  if (k === "i32" && u === "reciprocal") continue; // 1/0 gives the smallest int: int32 here, int64 in numpy
  unary.push(`np.${u}(${v})`);
}
numpy_cases("ufunc matrix: unary functions x dtypes", unary, { rtol: 1e-6, auto_errors: true });

const BINARY = "add subtract multiply divide true_divide floor_divide mod remainder fmod power float_power maximum minimum fmax fmin hypot arctan2 copysign nextafter logaddexp logaddexp2 gcd lcm bitwise_and bitwise_or bitwise_xor left_shift right_shift logical_and logical_or logical_xor equal not_equal less less_equal greater greater_equal heaviside divmod ldexp".split(" ");
const B_IN: { [k: string]: string } = {
  f: "np.array([-2.5, -1., 0., 0.5, 2., 3.5])",
  fn: "np.array([-2.5, np.nan, np.inf, 0.5, -np.inf, 3.5])",
  i: "np.array([-7, -1, 0, 3, 5, 12])",
  i8: "np.array([-7, -1, 0, 3, 5, 12], dtype='int8')",
  u8: "np.array([7, 1, 0, 3, 5, 250], dtype='uint8')",
  u16: "np.array([7, 1, 0, 3, 5, 65000], dtype='uint16')",
  b: "np.array([True, False, True, True, False, False])",
  f32: "np.array([-2.5, -1., 0., 0.5, 2., 3.5], dtype='float32')",
  col: "np.array([[1], [2]])",
  m: "np.array([[1., -2., 3., 4., 5., 6.], [0., 1., 2., 3., 4., 5.]])",
  s: "3",
  sf: "2.5",
  c: "np.array([1+1j, 2-1j, 0j, 3j, -1+0j, 2+2j])",
};
// Known differences: int32 default integers (numpy: int64) overflow earlier; complex division/power by special values.
const B_SKIP = (f: string, a: string, b: string) =>
  (f === "power" && a === "u16" && b === "col") ||
  ((f === "divide" || f === "true_divide") && a === "fn" && b === "c") ||
  ((f === "power" || f === "float_power") && b === "c" && (a === "b" || a === "fn"));
const binary: string[] = [];
for (const f of BINARY) for (const a of Object.keys(B_IN)) for (const b of Object.keys(B_IN)) {
  if ((a === "s" || a === "sf") && (b === "s" || b === "sf")) continue;
  if (B_SKIP(f, a, b)) continue;
  binary.push(`np.${f}(${B_IN[a]}, ${B_IN[b]})`);
}
numpy_cases("ufunc matrix: binary functions x dtypes x broadcasting", binary, { rtol: 1e-6, auto_errors: true });
