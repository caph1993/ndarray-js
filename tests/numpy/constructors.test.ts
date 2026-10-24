import { numpy_cases } from "../helpers/cases";

numpy_cases("arange / linspace / geomspace", [
  "np.arange(5)", "np.arange(2, 7)", "np.arange(2, 10, 3)", "np.arange(10, 0, -3)", "np.arange(0.0, 1.0, 0.25)", "np.arange(0, 1, 0.1)", "np.arange(1.5, 4)", "np.arange(5, dtype='float64')", "np.arange(0, 5, 2, dtype='float32')", "np.arange(0)", "np.arange(3, 3)", "np.arange(5, 2)",
  "np.arange(-3, 3)", "np.arange(5.0)", "np.arange(0, 1, 0.3)", "!np.arange(0, 5, 0)",
  "np.linspace(0, 1, 5)", "np.linspace(0, 1)", "np.linspace(0, 1, 5, endpoint=False)", "np.linspace(2.0, 3.0, 1)", "np.linspace(2.0, 3.0, 0)", "np.linspace(5, 1, 5)", "np.linspace(0, 10, 4, dtype='int32')", "np.linspace(0, 1, num=3)", "np.linspace(-1, 1, 7)", "np.linspace(0, 1, 1, endpoint=False)",
  "np.geomspace(1, 1000, 4)", "np.geomspace(1, 1000, 4, endpoint=False)", "np.geomspace(1, 256, 9)", "np.geomspace(-1, -1000, 4)", "np.geomspace(0.5, 8, 5)", "!np.geomspace(0, 10, 3)",
  "np.zeros(3)", "np.zeros((2, 3))", "np.ones((2, 2), dtype='int32')", "np.zeros(0)", "np.zeros([2, 0])", "np.ones(3, 'bool')",
  "np.array([1, 2, 3], dtype='float64')", "np.array([1.5, 2.5], dtype='int32')", "np.array([[1, 2], [3, 4]], dtype='bool')", "np.array([1, 2, 3], ndmin=2)", "np.array(np.array([1, 2]), dtype='float32')", "np.array([])", "np.array([[], []])", "np.array([True, 2, 3.5])", "np.array([True, 2])",
  "np.array([2**40, 1])", "np.array([2**31, 1])", "np.array([2**31 - 1, -2**31])",
  "np.asarray([1, 2, 3])", "np.asarray([1, 2], dtype='float64')",
]);
