import { describe, it, expect } from "vitest";
import { np } from "../../src/index";
import { HAS_NUMPY, numpy_batch } from "../helpers/numpy";

// For each case, numpy and ndarray-js must agree on whether the assertion passes
const cases: [string, () => any][] = [
  ["np.testing.assert_allclose([1.0, 2.0], [1.0, 2.0 + 1e-9])", () => np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-9]))],
  ["np.testing.assert_allclose([1.0, 2.0], [1.0, 2.1])", () => np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.1]))],
  ["np.testing.assert_allclose([1.0, 2.0], [1.0, 2.1], rtol=0.1)", () => np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.1]), 0.1)],
  ["np.testing.assert_allclose([1.0, np.nan], [1.0, np.nan])", () => np.testing.assert_allclose(np.array([1.0, NaN]), np.array([1.0, NaN]))],
  ["np.testing.assert_allclose([1.0, np.nan], [1.0, np.nan], equal_nan=False)", () => np.testing.assert_allclose(np.array([1.0, NaN]), np.array([1.0, NaN]), 1e-7, 0, false)],
  ["np.testing.assert_allclose([1.0, 2.0], [1.0, 2.0, 3.0])", () => np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))],
  ["np.testing.assert_allclose([1.0, 1.0], 1.0)", () => np.testing.assert_allclose(np.array([1.0, 1.0]), 1.0)],
  ["np.testing.assert_allclose(0.0, 1e-12, atol=1e-10)", () => np.testing.assert_allclose(0.0, 1e-12, 1e-7, 1e-10)],
  ["np.testing.assert_array_equal([1, 2], [1, 2])", () => np.testing.assert_array_equal(np.array([1, 2]), np.array([1, 2]))],
  ["np.testing.assert_array_equal([1, 2], [1, 3])", () => np.testing.assert_array_equal(np.array([1, 2]), np.array([1, 3]))],
  ["np.testing.assert_array_equal([1, np.nan], [1, np.nan])", () => np.testing.assert_array_equal(np.array([1, NaN]), np.array([1, NaN]))],
  ["np.testing.assert_array_equal([1, 2], [[1, 2]])", () => np.testing.assert_array_equal(np.array([1, 2]), np.array([[1, 2]]))],
  ["np.testing.assert_array_equal([5, 5], 5)", () => np.testing.assert_array_equal(np.array([5, 5]), 5)],
  ["np.testing.assert_array_less([1, 2], [2, 3])", () => np.testing.assert_array_less(np.array([1, 2]), np.array([2, 3]))],
  ["np.testing.assert_array_less([1, 2], [2, 2])", () => np.testing.assert_array_less(np.array([1, 2]), np.array([2, 2]))],
  ["np.testing.assert_equal(3, 3)", () => np.testing.assert_equal(3, 3)],
  ["np.testing.assert_equal(3, 4)", () => np.testing.assert_equal(3, 4)],
  ["np.testing.assert_equal([1, 2], [1, 2])", () => np.testing.assert_equal([1, 2], [1, 2])],
  ["np.testing.assert_equal(np.array([1, 2]), np.array([1, 2]))", () => np.testing.assert_equal(np.array([1, 2]), np.array([1, 2]))],
  ["np.testing.assert_equal({'a': 1}, {'a': 1})", () => 'skip'],
  ["np.testing.assert_almost_equal(1.0, 1.0 + 1e-8)", () => np.testing.assert_almost_equal(1.0, 1.0 + 1e-8)],
  ["np.testing.assert_almost_equal(1.0, 1.001)", () => np.testing.assert_almost_equal(1.0, 1.001)],
  ["np.testing.assert_almost_equal(1.0, 1.001, decimal=2)", () => np.testing.assert_almost_equal(1.0, 1.001, 2)],
  ["np.testing.assert_array_almost_equal([1.0, 2.0], [1.0000001, 2.0])", () => np.testing.assert_array_almost_equal(np.array([1.0, 2.0]), np.array([1.0000001, 2.0]))],
  ["np.testing.assert_array_almost_equal([1.0, 2.0], [1.001, 2.0])", () => np.testing.assert_array_almost_equal(np.array([1.0, 2.0]), np.array([1.001, 2.0]))],
  ["np.testing.assert_array_almost_equal([1.0, 2.0], [1.001, 2.0], decimal=2)", () => np.testing.assert_array_almost_equal(np.array([1.0, 2.0]), np.array([1.001, 2.0]), 2)],
  ["np.testing.assert_approx_equal(123456.0, 123457.0, significant=5)", () => np.testing.assert_approx_equal(123456.0, 123457.0, 5)],
  ["np.testing.assert_approx_equal(123456.0, 123457.0, significant=7)", () => np.testing.assert_approx_equal(123456.0, 123457.0, 7)],
  ["np.testing.assert_string_equal('a', 'a')", () => np.testing.assert_string_equal('a', 'a')],
  ["np.testing.assert_string_equal('a', 'b')", () => np.testing.assert_string_equal('a', 'b')],
  ["np.testing.assert_(True)", () => np.testing.assert_(true)],
  ["np.testing.assert_(False)", () => np.testing.assert_(false)],
  ["np.testing.assert_array_almost_equal_nulp(np.array([1.0]), np.array([1.0 + 2.2e-16]))", () => np.testing.assert_array_almost_equal_nulp(np.array([1.0]), np.array([1.0 + 2.2e-16]))],
  ["np.testing.assert_array_almost_equal_nulp(np.array([1.0]), np.array([1.0 + 1e-10]))", () => np.testing.assert_array_almost_equal_nulp(np.array([1.0]), np.array([1.0 + 1e-10]))],
  ["np.testing.assert_array_max_ulp(np.array([1.0]), np.array([1.0 + 2.2e-16]))", () => np.testing.assert_array_max_ulp(np.array([1.0]), np.array([1.0 + 2.2e-16]))],
  ["np.testing.assert_array_max_ulp(np.array([1.0]), np.array([1.0 + 1e-10]))", () => np.testing.assert_array_max_ulp(np.array([1.0]), np.array([1.0 + 1e-10]))],
  ["np.testing.assert_raises(ValueError, int, 'x')", () => np.testing.assert_raises(Error, () => { throw new Error('x'); })],
  ["np.testing.assert_raises(ValueError, int, '5')", () => np.testing.assert_raises(Error, () => 5)],
  ["np.testing.assert_equal(np.nan, np.nan)", () => np.testing.assert_equal(NaN, NaN)],
];


describe.skipIf(!HAS_NUMPY)("numpy.testing agrees with numpy on whether assertions pass", () => {
  const runnable = cases.filter(([, f]) => f.toString() !== "() => 'skip'");
  // python side: evaluate each assertion and report pass/fail
  const results = numpy_batch(runnable.map(([code]) => `(lambda f: f())(lambda: (${code}, True)[1])`));
  runnable.forEach(([code, fn], i) => {
    it(code, () => {
      const passes_py = results[i].ok;
      let passes_js = true;
      try { fn(); } catch (e: any) { passes_js = false; expect(e.name === "AssertionError" || e instanceof Error).toBe(true); }
      expect(passes_js, `numpy ${passes_py ? "passes" : "fails"} but ndarray-js ${passes_js ? "passes" : "fails"}`).toBe(passes_py);
    });
  });
});

describe("numpy.testing helpers", () => {
  it("assert_raises_regex", () => {
    np.testing.assert_raises_regex(Error, /boom/, () => { throw new Error("big boom"); });
    expect(() => np.testing.assert_raises_regex(Error, /zzz/, () => { throw new Error("big boom"); })).toThrow();
  });
  it("assert_warns and assert_no_warnings", () => {
    np.testing.assert_warns(Error, () => console.warn("x"));
    expect(() => np.testing.assert_warns(Error, () => 1)).toThrow();
    np.testing.assert_no_warnings(() => 1);
    expect(() => np.testing.assert_no_warnings(() => console.warn("x"))).toThrow();
  });
});
