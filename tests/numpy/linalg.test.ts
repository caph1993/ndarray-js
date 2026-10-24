import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const M = "np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]])";
const N = "np.array([[2.0, -1.0, 0.5], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]])";
const NR = "np.array([[4.0, 1.0, 0.0], [0.5, 3.0, 1.0], [0.0, 0.2, 1.0]])"; // real eigenvalues
const R = "np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])";
const W = "np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])";
const STACK = "np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 0.5], [0.5, 1.0]]])";
const A = "np.arange(6).reshape(2, 3)";

numpy_cases("products", [
  "np.matmul(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))", "np.matmul(np.array([1, 2]), np.array([[1, 2], [3, 4]]))", "np.matmul(np.array([[1, 2], [3, 4]]), np.array([1, 2]))", "np.matmul(np.array([1, 2]), np.array([3, 4]))",
  "np.matmul(np.arange(24).reshape(2, 3, 4), np.arange(8).reshape(4, 2))", "np.matmul(np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 4, 3))", "np.matmul(np.arange(24).reshape(2, 3, 4), np.arange(4))", "np.matmul(np.arange(3), np.arange(24).reshape(2, 3, 4))",
  "np.matmul(np.arange(6).reshape(1, 2, 3), np.arange(12).reshape(4, 3, 1))", "np.matmul(np.array([[True, False]]), np.array([[True], [True]]))", "!np.matmul(np.ones((2, 3)), np.ones((2, 3)))", "!np.matmul(2, np.ones((2, 3)))", `np.matmul(${W}, ${R})`,
  "np.vdot(np.array([1, 2, 3]), np.array([4, 5, 6]))", "np.vdot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))", "!np.vdot(np.array([1, 2]), np.array([1, 2, 3]))",
  "np.inner(np.array([1, 2, 3]), np.array([0, 1, 0]))", "np.inner(np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]]))", "np.inner(np.arange(6).reshape(2, 3), np.arange(12).reshape(2, 2, 3))", "np.inner(3, np.array([1, 2]))", "!np.inner(np.array([1, 2]), np.array([1, 2, 3]))",
  "np.outer(np.array([1, 2, 3]), np.array([0, 1, 2]))", "np.outer(np.array([[1, 2], [3, 4]]), np.array([1, 2]))", "np.outer(np.array([1.5, 2.5]), np.array([2, 4]))",
  "np.matvec(np.array([[1, 2], [3, 4]]), np.array([1, 1]))", "np.matvec(np.arange(24).reshape(2, 3, 4), np.arange(4))", "np.vecmat(np.array([1, 1]), np.array([[1, 2], [3, 4]]))", "np.vecmat(np.arange(3), np.arange(24).reshape(2, 3, 4))",
  "np.vecdot(np.array([1, 2, 3]), np.array([4, 5, 6]))", "np.vecdot(np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3))", "np.vecdot(np.arange(6).reshape(2, 3), np.arange(3))", "!np.vecdot(np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2), axis=0)",
  "np.tensordot(np.arange(60).reshape(3, 4, 5), np.arange(24).reshape(4, 3, 2), axes=([1, 0], [0, 1]))", "!np.tensordot(np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2))", "np.tensordot(np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3), axes=2)",
  "np.tensordot(np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), axes=1)", "np.tensordot(np.arange(3), np.arange(4), axes=0)", "np.tensordot(np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2), axes=([1], [0]))", "!np.tensordot(np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3), axes=1)",
  "np.kron(np.array([1, 10, 100]), np.array([5, 6, 7]))", "np.kron(np.eye(2), np.ones((2, 2)))", "np.kron(np.array([[1, 2], [3, 4]]), np.array([[0, 1], [1, 0]]))", "np.kron(np.array([1, 2]), np.array([[1, 2], [3, 4]]))",
  "np.trace(np.arange(9).reshape(3, 3))", "np.trace(np.arange(9).reshape(3, 3), offset=1)", "np.trace(np.arange(24).reshape(2, 3, 4))", "np.trace(np.arange(24).reshape(2, 3, 4), 0, 1, 2)", "np.trace(np.arange(6).reshape(2, 3), -1)", "np.trace(np.eye(3), dtype='int32')", "np.trace(np.array([[1.5, 2], [3, 4.5]]))",
  "np.linalg.matrix_power(np.array([[1, 1], [0, 1]]), 5)", "np.linalg.matrix_power(np.array([[1, 1], [0, 1]]), 0)", "np.linalg.matrix_power(np.array([[2.0, 0], [0, 4.0]]), -1)", "np.linalg.matrix_power(np.array([[2.0, 1], [1, 3.0]]), 3)", "np.linalg.matrix_power(np.arange(8).reshape(2, 2, 2), 2)", "!np.linalg.matrix_power(np.ones((2, 3)), 2)",
  "np.linalg.multi_dot([np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), np.arange(8).reshape(4, 2)])", "np.linalg.multi_dot([np.arange(3), np.arange(12).reshape(3, 4), np.arange(4)])", "!np.linalg.multi_dot([np.arange(3)])",
  "np.linalg.cross(np.array([1, 2, 3]), np.array([4, 5, 6]))", "np.linalg.matmul(np.array([[1, 2]]), np.array([[3], [4]]))", "np.linalg.outer(np.array([1, 2]), np.array([3, 4]))", "np.linalg.vecdot(np.array([1, 2]), np.array([3, 4]))",
  "np.linalg.tensordot(np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2), axes=1)", "np.linalg.trace(np.arange(9).reshape(3, 3))", "np.linalg.diagonal(np.arange(9).reshape(3, 3))", "np.linalg.matrix_transpose(np.arange(6).reshape(2, 3))",
]);

numpy_cases("einsum", [
  "np.einsum('ij,jk->ik', np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))", "np.einsum('ij,jk', np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))", "np.einsum('i,i->', np.arange(4), np.arange(4))", "np.einsum('i,i', np.arange(4), np.arange(4))",
  "np.einsum('ii', np.arange(9).reshape(3, 3))", "np.einsum('ii->i', np.arange(9).reshape(3, 3))", "np.einsum('ij->ji', np.arange(6).reshape(2, 3))", "np.einsum('ij->', np.arange(6).reshape(2, 3))", "np.einsum('ij->j', np.arange(6).reshape(2, 3))", "np.einsum('ij', np.arange(6).reshape(2, 3))",
  "np.einsum('i,j->ij', np.arange(3), np.arange(4))", "np.einsum('ijk,ikl->ijl', np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 4, 3))", "np.einsum('...ij,...jk->...ik', np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 4, 3))", "np.einsum('...j->...', np.arange(6).reshape(2, 3))",
  "np.einsum('ba', np.arange(6).reshape(2, 3))", "np.einsum('i,ij,j', np.arange(2), np.arange(6).reshape(2, 3), np.arange(3))", "np.einsum('ij,ij->ij', np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3))", "np.einsum('ij,ij->i', np.arange(6).reshape(2, 3), np.ones((2, 3)))",
  "np.einsum('ijk->kji', np.arange(24).reshape(2, 3, 4))", "np.einsum('i->', np.array([1.5, 2.5]))", "np.einsum('ij,j', np.array([[1.5, 2], [3, 4]]), np.array([1, 2]))", "np.einsum('ij,kj->ik', np.arange(6).reshape(2, 3), np.arange(9).reshape(3, 3))",
  "!np.einsum('ij,jk', np.arange(6).reshape(2, 3), np.arange(8).reshape(2, 4))", "!np.einsum('ij->k', np.arange(6).reshape(2, 3))", "!np.einsum('ijk', np.arange(6).reshape(2, 3))", "np.einsum('ij,jk->ik', np.arange(4).reshape(2, 2), np.eye(2))",
  "np.einsum_path('ij,jk->ik', np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))[0][0]",
]);

numpy_cases("det, inv, solve, norms", [
  `np.linalg.det(${M})`, `np.linalg.det(${N})`, "np.linalg.det(np.eye(3))", "np.linalg.det(np.array([[1.0, 2.0], [2.0, 4.0]]))", `np.linalg.det(${STACK})`, "np.linalg.det(np.array([[3.0]]))", "!np.linalg.det(np.ones((2, 3)))", "!np.linalg.det(np.array([1.0, 2.0]))",
  `np.linalg.slogdet(${M})`, `np.linalg.slogdet(${N})`, `np.linalg.slogdet(np.array([[0.0, 1.0], [1.0, 0.0]]))`, `np.linalg.slogdet(np.zeros((2, 2)))`, `np.linalg.slogdet(${STACK})`,
  `np.linalg.inv(${M})`, `np.linalg.inv(${N})`, `np.linalg.inv(${STACK})`, "np.linalg.inv(np.array([[2.0]]))", "np.linalg.inv(np.eye(2))", "!np.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]]))", "!np.linalg.inv(np.zeros((2, 2)))", "np.linalg.inv(np.array([[1, 2], [3, 4]]))",
  `np.linalg.solve(${M}, np.array([1.0, 2.0, 3.0]))`, `np.linalg.solve(${N}, np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 4.0]]))`, `np.linalg.solve(${STACK}, np.array([[1.0, 2.0], [3.0, 4.0]]))`, `np.linalg.solve(${STACK}, np.arange(4.0).reshape(2, 2, 1))`,
  "np.linalg.solve(np.array([[3, 1], [1, 2]]), np.array([9, 8]))", "!np.linalg.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))", "!np.linalg.solve(np.eye(2), np.arange(3.0))",
  `np.linalg.norm(${M})`, `np.linalg.norm(${M}, 'fro')`, `np.linalg.norm(${M}, 1)`, `np.linalg.norm(${M}, np.inf)`, `np.linalg.norm(${M}, -1)`, `np.linalg.norm(${M}, -np.inf)`, `np.linalg.norm(${M}, 2)`, `np.linalg.norm(${M}, -2)`, `np.linalg.norm(${M}, 'nuc')`,
  "np.linalg.norm(np.array([3.0, 4.0]))", "np.linalg.norm(np.array([3.0, 4.0]), 1)", "np.linalg.norm(np.array([3.0, 4.0]), np.inf)", "np.linalg.norm(np.array([3.0, 4.0, 0.0]), 0)", "np.linalg.norm(np.array([3.0, 4.0]), 3)", "np.linalg.norm(np.array([3.0, -4.0]), -np.inf)", "np.linalg.norm(np.array([3, 4]))",
  `np.linalg.norm(${M}, axis=0)`, `np.linalg.norm(${M}, axis=1, keepdims=True)`, `np.linalg.norm(${M}, ord=1, axis=1)`, `np.linalg.norm(${M}, axis=(0, 1))`, `np.linalg.norm(np.arange(24.0).reshape(2, 3, 4), axis=(1, 2))`, `np.linalg.norm(np.arange(24.0).reshape(2, 3, 4), ord=2, axis=(1, 2))`,
  `np.linalg.norm(np.arange(24.0).reshape(2, 3, 4), axis=(0, 2), keepdims=True)`, `!np.linalg.norm(np.arange(24.0).reshape(2, 3, 4))`.replace("!", ""), "!np.linalg.norm(np.array([1.0, 2.0]), 'fro')",
  `np.linalg.vector_norm(${M})`, `np.linalg.vector_norm(${M}, axis=1, ord=1)`, `np.linalg.vector_norm(${M}, axis=None, keepdims=True, ord=np.inf)`, `np.linalg.vector_norm(np.arange(24.0).reshape(2, 3, 4), axis=(0, 1))`,
  `np.linalg.matrix_norm(${M})`, `np.linalg.matrix_norm(${M}, ord=1)`, `np.linalg.matrix_norm(${STACK}, ord='nuc')`, `np.linalg.matrix_norm(${STACK}, keepdims=True)`, `np.linalg.matrix_norm(${M}, ord=np.inf)`, "!np.linalg.matrix_norm(np.arange(3.0))",
  `np.linalg.cond(${M})`, `np.linalg.cond(${N}, 1)`, `np.linalg.cond(${N}, np.inf)`, `np.linalg.cond(${N}, 'fro')`, `np.linalg.cond(${N}, -2)`, `np.linalg.cond(${R})`,
  `np.linalg.matrix_rank(${M})`, "np.linalg.matrix_rank(np.array([[1.0, 2.0], [2.0, 4.0]]))", "np.linalg.matrix_rank(np.zeros((3, 3)))", `np.linalg.matrix_rank(${R})`, `np.linalg.matrix_rank(${STACK})`, "np.linalg.matrix_rank(np.array([1.0, 0.0]))", "np.linalg.matrix_rank(np.eye(3), tol=2)", "np.linalg.matrix_rank(np.arange(12.0).reshape(3, 4))",
  "np.linalg.tensorinv(np.eye(4).reshape(4, 2, 2), ind=1)", "np.linalg.tensorinv(np.arange(1, 17.0).reshape(4, 4) + 3 * np.eye(4), ind=1).shape", "np.linalg.tensorsolve(np.eye(6).reshape(2, 3, 6), np.arange(6.0).reshape(2, 3)).shape",
  "np.linalg.tensorsolve(np.eye(6).reshape(2, 3, 6) * 2, np.arange(6.0).reshape(2, 3))",
]);

numpy_cases("cholesky, qr", [
  `np.linalg.cholesky(${M})`, `np.linalg.cholesky(${M}, upper=True)`, `np.linalg.cholesky(${STACK})`, "np.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 3.0]]))", "np.linalg.cholesky(np.array([[2.0]]))", "!np.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))", "!np.linalg.cholesky(np.ones((2, 3)))",
  `np.linalg.qr(${N})`, `np.linalg.qr(${R})`, `np.linalg.qr(${W})`, `np.linalg.qr(${R}, mode='complete')`, `np.linalg.qr(${W}, mode='complete')`, `np.linalg.qr(${R}, mode='r')`, `np.linalg.qr(${STACK})`, "np.linalg.qr(np.array([[1.0, 0.0], [0.0, 1.0]]))", "np.linalg.qr(np.array([[0.0, 1.0], [1.0, 0.0]]))", "np.linalg.qr(np.array([[3.0], [4.0]]))",
  "np.linalg.qr(np.array([[1.0, 2.0], [-1.0, 3.0]]))", "!np.linalg.qr(np.array([1.0, 2.0]))", `!np.linalg.qr(${N}, mode='bogus')`,
]);

// SVD / eig have sign and ordering ambiguities: compare invariant quantities
const prop = (js: () => any, py: string): [() => any, string] => [js, py];
const sv = (X: string) => `np.linalg.svd(${X}, compute_uv=False)`;
numpy_cases("svd and eigen decompositions", [
  `np.linalg.svd(${M}, compute_uv=False)`, `np.linalg.svd(${R}, compute_uv=False)`, `np.linalg.svd(${W}, compute_uv=False)`, `np.linalg.svdvals(${N})`, `np.linalg.svd(${STACK}, compute_uv=False)`, "np.linalg.svd(np.zeros((2, 2)), compute_uv=False)", "np.linalg.svd(np.array([[3.0, 0.0], [0.0, -4.0]]), compute_uv=False)",
  `np.linalg.svd(${R})[1]`, `np.linalg.svd(${R})[0].shape`, `np.linalg.svd(${R})[2].shape`, `np.linalg.svd(${R}, full_matrices=False)[0].shape`, `np.linalg.svd(${R}, full_matrices=False)[2].shape`, `np.linalg.svd(${W})[0].shape`, `np.linalg.svd(${W}, full_matrices=False)[2].shape`, `np.linalg.svd(${STACK})[0].shape`,
  prop(() => { const [U, s, Vt] = np.linalg.svd(np.array([[1, 2], [3, 4], [5, 6]], 'float64'), false); return np.allclose(np.matmul(np.multiply(U, s), Vt), np.array([[1, 2], [3, 4], [5, 6]], 'float64')); }, `(lambda U, s, Vt: np.allclose((U * s) @ Vt, ${R}))(*np.linalg.svd(${R}, full_matrices=False))`),
  prop(() => { const X = np.array([[1, 2, 3], [4, 5, 6]], 'float64'); const [U, s, Vt] = np.linalg.svd(X); const S = np.zeros([2, 3]); np.fill_diagonal(S, s); return np.allclose(np.matmul(np.matmul(U, S), Vt), X); }, `(lambda U, s, Vt: np.allclose(U @ np.hstack([np.diag(s), np.zeros((2, 1))]) @ Vt, ${W}))(*np.linalg.svd(${W}))`),
  prop(() => { const [U] = np.linalg.svd(np.array([[1, 2, 3], [4, 5, 6]], 'float64')); return np.allclose(np.matmul(U, U.T), np.eye(2)); }, "True"),
  prop(() => { const [, , Vt] = np.linalg.svd(np.array([[1, 2, 3], [4, 5, 6]], 'float64')); return np.allclose(np.matmul(Vt, Vt.T), np.eye(3)); }, "True"),
  prop(() => { const [, , Vt] = np.linalg.svd(np.array([[1, 2], [3, 4], [5, 6]], 'float64')); return np.allclose(np.matmul(Vt, Vt.T), np.eye(2)); }, "True"),
  prop(() => { const [U] = np.linalg.svd(np.array([[1, 2], [3, 4], [5, 6]], 'float64')); return np.allclose(np.matmul(U.T, U), np.eye(3)); }, "True"),
  prop(() => np.linalg.pinv(np.array([[1, 2], [3, 4], [5, 6]], 'float64')), `np.linalg.pinv(${R})`),
  `np.linalg.pinv(${W})`, `np.linalg.pinv(${M})`, "np.linalg.pinv(np.array([[1.0, 2.0], [2.0, 4.0]]))", "np.linalg.pinv(np.zeros((2, 3)))", `np.linalg.pinv(${STACK})`, `np.linalg.pinv(${W}, rcond=0.5)`, "np.linalg.pinv(np.array([[1.0, 0.0], [0.0, 1e-20]]))",
  `np.linalg.eigvalsh(${M})`, `np.linalg.eigvalsh(${M}, 'U')`, `np.linalg.eigh(${M})[0]`, `np.linalg.eigvalsh(${STACK})`, "np.linalg.eigvalsh(np.array([[2.0, 1.0], [0.0, 3.0]]))", "np.linalg.eigvalsh(np.array([[2.0, 1.0], [0.0, 3.0]]), 'U')", "np.linalg.eigvalsh(np.diag([3.0, 1.0, 2.0]))", "np.linalg.eigvalsh(np.array([[1.0]]))",
  prop(() => { const X = np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]], 'float64'); const [w, V] = np.linalg.eigh(X); return np.allclose(np.matmul(X, V), np.multiply(V, w)); }, "True"),
  prop(() => { const X = np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]], 'float64'); const [, V] = np.linalg.eigh(X); return np.allclose(np.matmul(V.T, V), np.eye(3)); }, "True"),
  prop(() => { const X = np.array([[2, 1], [0, 3]], 'float64'); const [w] = np.linalg.eigh(X, 'U'); return w; }, "np.linalg.eigh(np.array([[2.0, 1.0], [0.0, 3.0]]), 'U')[0]"),
  prop(() => np.sort(np.linalg.eigvals(np.array([[4, 1, 0], [0.5, 3, 1], [0, 0.2, 1]], 'float64')).real), `np.sort(np.linalg.eigvals(${NR}).real)`),
  prop(() => np.sort(np.linalg.eigvals(np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]], 'float64')).real), `np.sort(np.linalg.eigvals(${M}).real)`),
  prop(() => np.sort(np.linalg.eigvals(np.array([[2, 0], [0, 3]], 'float64')).real), "np.sort(np.linalg.eigvals(np.array([[2.0, 0.0], [0.0, 3.0]])).real)"),
  prop(() => np.sort(np.linalg.eigvals(np.array([[1, 2], [3, 4]], 'float64')).real), "np.sort(np.linalg.eigvals(np.array([[1.0, 2.0], [3.0, 4.0]])).real)"),
  prop(() => np.sort(np.linalg.eigvals(np.array([[0, 1, 0], [0, 0, 1], [6, -11, 6]], 'float64')).real), "np.sort(np.linalg.eigvals(np.array([[0.0, 1, 0], [0, 0, 1], [6, -11, 6]])).real)"),
  prop(() => { const X = np.array([[4, 1, 0], [0.5, 3, 1], [0, 0.2, 1]], 'float64'); const [w, V] = np.linalg.eig(X); return np.allclose(np.matmul(X, V), np.multiply(V, w), 1e-6, 1e-8); }, "True"),
  prop(() => { const X = np.array([[1, 2], [3, 4]], 'float64'); const [w, V] = np.linalg.eig(X); return np.allclose(np.matmul(X, V), np.multiply(V, w), 1e-6, 1e-8); }, "True"),
  prop(() => { const [, V] = np.linalg.eig(np.array([[2, 1], [1, 3]], 'float64')); return np.allclose(np.linalg.norm(V, 2, 0), 1); }, "True"),
  
  "np.linalg.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]))[0]", "np.linalg.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]))[1]", "np.linalg.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]))[2]", "np.linalg.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]))[3]",
  "np.linalg.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([[1.0, 0.0], [2.0, 1.0], [2.0, 5.0]]))[0]", "np.linalg.lstsq(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([1.0, 2.0]))[0]", "np.linalg.lstsq(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([1.0, 2.0]))[1]", "np.linalg.lstsq(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), np.array([1.0, 2.0, 3.0]))[2]",
  "np.linalg.lstsq(np.eye(2), np.array([3.0, 4.0]))[0]", "np.linalg.lstsq(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.array([1.0, 2.0, 2.0]), rcond=None)[0]",
]);
