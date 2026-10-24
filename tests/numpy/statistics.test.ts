import { numpy_cases } from "../helpers/cases";

const X = "np.array([[10.0, 7.0, 4.0, 3.5], [3.0, 2.0, 1.0, 8.0], [5.5, 6.0, 0.5, 9.0]])";
const V = "np.array([1.0, 7.0, 3.0, 9.0, 5.0, 2.0])";
const W = "np.array([1.0, np.nan, 3.0, 4.0, np.nan, 2.0])";
const methods = ["linear", "lower", "higher", "midpoint", "nearest", "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "median_unbiased", "normal_unbiased"];

numpy_cases("quantile / percentile", [
  ...methods.flatMap(m => [`np.quantile(${V}, 0.3, method='${m}')`, `np.quantile(${V}, np.array([0.0, 0.25, 0.5, 0.9, 1.0]), method='${m}')`, `np.percentile(${X}, 40, axis=1, method='${m}')`]),
  `np.quantile(np.array([1, 5, 2, 8]), 0.5)`, `np.quantile(np.array([1, 5, 2, 8]), 0.5, method='lower')`, `np.quantile(np.array([1, 5, 2, 8]), np.array([0.2, 0.7]), method='nearest')`,
  `np.quantile(${X}, 0.5)`, `np.quantile(${X}, 0.5, axis=0)`, `np.quantile(${X}, np.array([0.1, 0.9]), axis=0)`, `np.quantile(${X}, np.array([0.1, 0.9]), axis=1, keepdims=True)`, `np.quantile(${X}, 0.25, axis=(0, 1))`,
  `np.quantile(${X}, np.array([[0.1, 0.2], [0.3, 0.4]]))`, `np.percentile(${X}, 50)`, `np.percentile(${X}, [25, 75], axis=1)`, `np.percentile(${X}, 90, axis=0, keepdims=True)`, `np.percentile(${V}, 0)`, `np.percentile(${V}, 100)`,
  `np.quantile(${W}, 0.5)`, `np.quantile(${W}, 0.5, method='lower')`, `np.nanquantile(${W}, 0.5)`, `np.nanquantile(${W}, [0.1, 0.9])`, `np.nanpercentile(${W}, 75)`, `np.nanpercentile(${X}, 75, axis=0)`,
  `np.nanquantile(np.array([np.nan, np.nan]), 0.5)`, `np.quantile(np.array([3.0]), 0.7)`, `np.quantile([1, 2, 3, 4], 0.5)`,
  `np.quantile(np.array([1, 2, 3], dtype='float32'), 0.3)`, `np.quantile(a=${V}, q=0.5)`,
  "!np.quantile(np.array([1.0, 2.0]), 1.5)", "!np.quantile(np.array([1.0, 2.0]), -0.1)", "!np.percentile(np.array([1.0, 2.0]), 101)", "!np.quantile(np.array([1.0, 2.0]), 0.5, method='foo')",
  `np.median(${V})`, `np.median(${X})`, `np.median(${X}, axis=0)`, `np.median(${X}, axis=1, keepdims=True)`, `np.median(${W})`, `np.median(np.array([1, 2, 3, 4]))`, `np.median(np.array([5]))`, `np.median(np.array([1, 3]))`,
  `np.median(np.arange(24).reshape(2, 3, 4), axis=(0, 2))`, `np.nanmedian(${W})`, `np.nanmedian(np.array([[1.0, np.nan], [3.0, 4.0]]), axis=0)`, `np.median(np.array([]))`,
]);
numpy_cases("cov and corrcoef", [
  `np.cov(${X})`, `np.cov(${X}, rowvar=False)`, `np.cov(${X}, bias=True)`, `np.cov(${X}, ddof=0)`, `np.cov(${X}, ddof=2)`, `np.cov(${V})`, `np.cov(${V}, np.array([1.0, 2.0, 4.0, 3.0, 8.0, 5.0]))`,
  `np.cov(np.array([1.0, 2.0, 3.0, 4.0]), fweights=np.array([1, 2, 1, 3]))`, `np.cov(np.array([1.0, 2.0, 3.0, 4.0]), aweights=np.array([0.5, 1.0, 1.5, 1.0]))`,
  `np.cov(${X}, fweights=np.array([1, 2, 1, 3]), aweights=np.array([0.5, 1.0, 1.5, 1.0]))`, `np.cov(${X}, fweights=np.array([1, 2, 1, 3]), ddof=0)`, `np.cov(np.array([1, 2, 3, 4]), np.array([2, 4, 6, 9]))`,
  `np.cov(m=${X}[:2], y=${X}[2:])`, `np.cov(np.array([[1, 2, 3]]))`, `np.cov(np.array([2.0]), ddof=0)`,
  `np.corrcoef(${X})`, `np.corrcoef(${X}, rowvar=False)`, `np.corrcoef(${V}, np.array([1.0, 2.0, 4.0, 3.0, 8.0, 5.0]))`, `np.corrcoef(np.array([1.0, 2.0, 3.0]))`, `np.corrcoef(np.array([1, 2, 3]), np.array([3, 2, 1]))`,
  `np.corrcoef(np.array([[1, 2, 3], [1, 2, 3]]))`,
]);
numpy_cases("histogram", [
  `np.histogram(${V})`, `np.histogram(${V}, bins=3)`, `np.histogram(${V}, bins=np.array([0.0, 2.0, 5.0, 10.0]))`, `np.histogram(${V}, bins=4, range=(0, 8))`, `np.histogram(${V}, bins=4, range=(2, 6))`, `np.histogram(${V}, bins=3, density=True)`,
  `np.histogram(${V}, bins=3, weights=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))`, `np.histogram(${V}, bins=3, weights=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), density=True)`,
  `np.histogram(np.array([1, 2, 1]), bins=np.array([0, 1, 2, 3]))`, `np.histogram(np.array([1, 2, 1]), bins=[0, 1, 2, 3])`, `np.histogram(np.array([3, 3, 3]), bins=2)`, `np.histogram(np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0]), bins=5, range=(1, 3))`,
  `np.histogram(${X})`, `np.histogram(np.array([1.0, 2.0, 2.0, 3.0, 5.0, 9.0, 10.0]), bins=np.array([1, 2.5, 10]))`, `np.histogram(np.array([]))`, `np.histogram(np.array([]), bins=3, range=(0, 1))`,
  ...["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"].map(b => `np.histogram(np.array([1.0, 2.0, 2.5, 3.0, 3.1, 4.0, 5.5, 6.0, 8.0, 9.5, 10.0, 12.0, 15.0, 3.3, 2.2, 4.4, 7.7]), bins='${b}')`),
  ...["auto", "fd", "doane", "scott", "rice", "sturges", "sqrt"].map(b => `np.histogram_bin_edges(np.array([1.0, 2.0, 2.5, 3.0, 3.1, 4.0, 5.5, 6.0, 8.0, 9.5, 10.0, 12.0, 15.0, 3.3, 2.2, 4.4, 7.7]), bins='${b}')`),
  `np.histogram_bin_edges(${V}, bins=4)`, `np.histogram_bin_edges(${V}, bins=4, range=(0, 20))`, `np.histogram_bin_edges(${V}, bins=np.array([0.0, 3.0, 10.0]))`, `np.histogram_bin_edges(np.array([2.0, 2.0]), bins=2)`,
  "!np.histogram(np.array([1.0, np.nan]))", "!np.histogram(np.array([1.0, 2.0]), bins=np.array([3.0, 1.0]))", "!np.histogram(np.array([1.0]), bins=0)", "!np.histogram(np.array([1.0]), bins='bogus')",
  `np.histogram(np.array([1.0, np.nan, 3.0]), bins=2, range=(0, 4))`,
  `np.histogram2d(np.array([1.0, 2.0, 3.0, 4.0, 2.0]), np.array([1.0, 3.0, 2.0, 4.0, 2.0]), bins=2)`, `np.histogram2d(np.array([1.0, 2.0, 3.0, 4.0, 2.0]), np.array([1.0, 3.0, 2.0, 4.0, 2.0]), bins=(2, 3))`,
  `np.histogram2d(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]), bins=[np.array([0.0, 2.0, 4.0]), np.array([0.0, 1.5, 4.0])])`, `np.histogram2d(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]), bins=2, range=[[0, 4], [0, 4]], density=True)`,
  `np.histogram2d(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]), bins=2, weights=np.array([1.0, 2.0, 3.0]))`,
  `np.histogramdd(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [3.0, 3.0, 2.0], [0.0, 0.5, 1.0]]), bins=2)`, `np.histogramdd(np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]), bins=(2, 3), density=True)`,
  `np.histogramdd(np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]), bins=[np.array([0.0, 2.0, 4.0]), np.array([0.0, 1.5, 4.0])])`, `np.histogramdd(np.array([1.0, 2.0, 3.0, 2.5]), bins=3)`,
]);
numpy_cases("bincount", [
  "np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))", "np.bincount(np.array([0, 1, 1, 3]), weights=np.array([0.5, 1.0, 2.0, 0.25]))", "np.bincount(np.array([1, 1]), minlength=5)", "np.bincount(np.array([1, 1, 7]), minlength=3)",
  "np.bincount(np.array([], dtype='int32'))", "np.bincount(np.array([], dtype='int32'), minlength=2)", "np.bincount(np.array([True, True, False]))", "!np.bincount(np.array([2.0, 3.0]))",
  "!np.bincount(np.array([-1, 2]))", "!np.bincount(np.array([1.5]))", "!np.bincount(np.array([[1, 2]]))", "!np.bincount(np.array([1]), weights=np.array([1.0, 2.0]))",
]);
numpy_cases("misc", [
  "np.broadcast_shapes((2, 3), (3,))", "np.broadcast_shapes((5, 1, 4), (1, 3, 1), (4,))", "np.broadcast_shapes((1,), ())", "np.broadcast_shapes(3, (2, 3))", "!np.broadcast_shapes((2,), (3,))",
  "np.shares_memory(np.arange(6), np.arange(6))", "np.may_share_memory(np.arange(6), np.arange(6))",
]);
