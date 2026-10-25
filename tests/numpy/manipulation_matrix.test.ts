import { numpy_cases } from "../helpers/cases";

// Shape / sorting / indexing / numeric functions against arrays of different dimensions and several positional arguments.
const A: { [k: string]: string } = {
  a1: "np.arange(6)", a2: "np.arange(12).reshape(3, 4)", a3: "np.arange(24).reshape(2, 3, 4)",
  f2: "np.arange(12.).reshape(3, 4) / 5", b2: "(np.arange(12).reshape(3, 4) % 3 == 0)", e: "np.zeros((0, 3))", s: "np.array(5)",
};
const ONE_ARG = "flip fliplr flipud rot90 squeeze ravel transpose atleast_1d atleast_2d atleast_3d diag triu tril trace copy shape ndim size argwhere nonzero flatnonzero count_nonzero sort argsort unique cumsum diff gradient msort trim_zeros sort_complex asfortranarray ascontiguousarray diagflat vander ediff1d isfinite real_if_close nan_to_num round around signbit argmax median".split(" ");
const ONE_SKIP = (f: string, k: string) => k === "s" && (f === "argsort" || f === "gradient");
const cases: string[] = [];
for (const f of ONE_ARG) for (const k of Object.keys(A)) if (!ONE_SKIP(f, k)) cases.push(`np.${f}(${A[k]})`);

const KW = [", 1", ", 2", ", -1", ", (1, -1)", ", axis=0", ", axis=-1", ", axis=1", ", shift=2", ", 2, axis=0", ", 1, axis=1", ", (1, 2), axis=(0, 1)", ", 1, 0", ", 0, 1"];
const FUNCS = "roll flip rot90 repeat tile expand_dims squeeze swapaxes moveaxis rollaxis take delete append split array_split diff cumsum sort argsort unique diag triu tril trace diagonal partition argpartition choose insert resize pad concatenate stack hstack vstack dstack column_stack tensordot inner outer kron cross dot matmul searchsorted isin where take_along_axis put_along_axis select apply_along_axis convolve correlate interp bincount digitize percentile nanpercentile cov corrcoef gradient average msort vdot einsum tri indices meshgrid fliplr flipud atleast_2d".split(" ");
// numpy argument conventions that differ on purpose, or nonsensical calls where only the error type differs
const SKIP = (f: string, kw: string) =>
  (f === "convolve" || f === "correlate") && /, [01]$/.test(kw) ||
  (f === "gradient" || f === "cross" || f === "put_along_axis" || f === "select" || f === "apply_along_axis" || f === "einsum" || f === "meshgrid" || f === "indices" || f === "tri" || f === "pad") && false;
for (const kw of KW) for (const f of FUNCS) for (const k of ["a1", "a2", "a3"]) {
  if (SKIP(f, kw)) continue;
  cases.push(`np.${f}(${A[k]}${kw})`);
}
// Known divergences (documented): numpy-specific weird signatures
const KNOWN_BAD = [
  /^np\.(mean|ptp)\(.*, 1, 0\)$/, /^np\.histogram/, /^np\.clip/, /^np\.compress/, /^np\.unwrap/, /^np\.quantile/, /^np\.trapezoid/,
];
numpy_cases("manipulation matrix", cases.filter(c => !KNOWN_BAD.some(r => r.test(c))), { rtol: 1e-6, auto_errors: true });
