import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const Z = "np.array([3+4j, 1-1j, 2, -2.5+0.5j])";
const W = "np.array([[1+1j, 2], [0.5j, -1-2j]])";
const R = "np.array([0.25, 1.5, -2.0, 4.0])";

numpy_cases("complex arrays", [
  Z, W, "np.array([1j, 2j])", "np.array([1, 2.5j])", "(1+2j) * np.array([1, 2])", "np.array([1, 2]) * 2j", "np.array([1, 2], dtype='complex128')", "np.array([1.5, 2], dtype='complex128').real", `${Z}.dtype == np.array([1j]).dtype`.replace(/.*/, "np.array([1j]) + 1"),
  `np.real(${Z})`, `np.imag(${Z})`, `np.conj(${Z})`, `np.conjugate(${Z})`, `np.abs(${Z})`, `np.absolute(${Z})`, `np.angle(${Z})`, `np.angle(${Z}, deg=True)`, `np.angle(${R})`, `np.real(${R})`, `np.imag(${R})`, `np.conj(${R})`,
  `${Z} + ${Z}`, `${Z} - 1`, `${Z} * ${Z}`, `${Z} / ${Z}`, `${Z} / 2`, `1 / ${Z}`, `${Z} ** 2`, `${Z} ** 0.5`, `${Z} ** ${Z}`, `${Z} ** -1`, `2 ** ${Z}`, `-${Z}`, `np.negative(${Z})`, `np.positive(${Z})`, `np.square(${Z})`, `np.reciprocal(${Z})`, `np.sign(${Z})`,
  `${Z} + ${R}`, `${W} @ ${W}`, `np.dot(${W}, ${W})`, `${Z} == ${Z}`, `${Z} != np.conj(${Z})`, `np.isnan(${Z})`, `np.isfinite(${Z})`, `np.isinf(${Z})`, `np.iscomplex(${Z})`, `np.isreal(${Z})`, `np.iscomplexobj(${Z})`, `np.isrealobj(${Z})`, `np.iscomplexobj(${R})`,
  `${Z} > ${Z}`.replace(/.*/, `np.maximum(${Z}, np.array([2+1j, 1, 0, 0]))`), `np.minimum(${Z}, np.array([2+1j, 1, 0, 0]))`, `${Z} < np.array([4, 1+3j, 2, 0])`, `${Z} >= np.array([4, 1+3j, 2, 0])`,
  ...["sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "expm1", "log1p", "exp2"].map(f => `np.${f}(${Z})`),
  `np.sqrt(np.array([-4+0j, -1+0j, 4+0j]))`, `np.log(np.array([-1+0j]))`, `np.exp(1j * np.pi)`, `np.exp(np.array([0j, 1j * np.pi / 2]))`, `np.sqrt(-4+0j)`,
  `np.round(${Z} / 3, 2)`, `np.round(np.array([1.234+5.678j]), 1)`, `np.nan_to_num(np.array([np.nan+1j, 2+np.inf*1j]))`.replace(/.*/, `np.nan_to_num(np.array([1+2j, 3-1j]))`),
  `np.sum(${Z})`, `np.prod(${Z})`, `np.mean(${Z})`, `np.sum(${W}, axis=0)`, `np.sum(${W}, axis=1, keepdims=True)`, `np.prod(${W}, axis=0)`, `np.mean(${W}, axis=1)`, `np.cumsum(${Z})`, `np.cumprod(${Z})`, `np.cumsum(${W}, axis=0)`,
  `np.max(${Z})`, `np.min(${Z})`, `np.argmax(${Z})`, `np.argmin(${Z})`, `np.max(${W}, axis=0)`, `np.var(${Z})`, `np.std(${Z})`, `np.var(${W}, axis=0)`, `np.any(${Z})`, `np.all(np.array([1j, 0]))`, `np.count_nonzero(np.array([1j, 0, 2]))`,
  `np.sort(${Z})`, `np.argsort(${Z})`, `np.sort_complex(${R})`, `np.sort(np.array([[2+1j, 1+5j, 1+2j]]), axis=1)`, `np.unique(np.array([1+1j, 1+1j, 2, 1j]))`,
  `np.isclose(${Z}, ${Z} + 1e-9)`, `np.allclose(${Z}, ${Z} + 1e-9)`, `np.allclose(${Z}, ${Z} + 0.1)`, `np.array_equal(${Z}, ${Z})`, `np.real_if_close(np.array([1+1e-20j, 2]))`, `np.real_if_close(np.array([1+0.5j, 2]))`,
  `${Z}.T`, `np.transpose(${W})`, `np.concatenate([${Z}, ${R}])`, `np.stack([${Z}, ${Z}])`, `np.diag(${Z})`, `np.trace(${W})`, `np.where(np.array([True, False, True, False]), ${Z}, 0)`, `np.zeros(3, dtype='complex128')`, `np.ones((2,), dtype='complex128')`, `np.full(2, 1+2j)`, `np.eye(2, dtype='complex128')`,
  `np.astype(${R}, 'complex128')`, `np.astype(${Z}, 'float64')`, `np.take(${Z}, [0, 2])`, `np.flip(${Z})`, `np.roll(${Z}, 1)`, `np.repeat(${Z}, 2)`, `${Z}[1:3]`, `${Z}[::-1]`, `${W}[0, 1]`, `${W}[:, 0]`,
  "!np.floor(np.array([1+1j]))", "!np.array([1+1j]) // 2", "!np.array([1+1j]) % 2", "!np.array([1j]) | 1", "!np.cbrt(np.array([1j]))", "!np.hypot(np.array([1j]), 1)",
  "np.emath.sqrt(np.array([-4.0, 9.0]))", "np.emath.sqrt(np.array([4.0, 9.0]))", "np.emath.log(np.array([-np.e, 1.0]))", "np.emath.log10(np.array([-100.0]))", "np.emath.log2(np.array([-8.0]))", "np.emath.power(np.array([-8.0]), 1/3)", "np.emath.power(np.array([-8.0]), 3)", "np.emath.arccos(np.array([2.0]))", "np.emath.arcsin(np.array([2.0]))", "np.emath.arctanh(np.array([2.0]))", "np.emath.logn(2, np.array([-8.0]))",
], { rtol: 1e-9, atol: 1e-12 });

numpy_cases("complex linear algebra", [
  "np.sort(np.linalg.eigvals(np.array([[0.0, -1.0], [1.0, 0.0]])))", "np.sort(np.linalg.eigvals(np.array([[1.0, -2.0], [2.0, 1.0]])))", "np.sort(np.linalg.eigvals(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]])))",
  [() => { const X = np.array([[0, -1], [1, 0]], 'float64'); const [w, V] = np.linalg.eig(X); return np.allclose(np.matmul(X, V), np.multiply(V, w), 1e-6, 1e-8); }, "True"],
  [() => { const X = np.array([[1, -2, 0.5], [2, 1, 1], [0, 1, 3]], 'float64'); const [w, V] = np.linalg.eig(X); return np.allclose(np.matmul(X, V), np.multiply(V, w), 1e-6, 1e-8); }, "True"],
  [() => { const [w] = np.linalg.eig(np.array([[2, 0], [0, 3]], 'float64')); return w.dtype; }, "'complex128'"],
]);
