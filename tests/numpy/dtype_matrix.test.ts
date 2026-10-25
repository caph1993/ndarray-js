import { numpy_cases } from "../helpers/cases";

// All dtypes: casts, promotion in arithmetic (array/array and array/python scalar), constructors, reducers.
const DT = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"];
const V: { [k: string]: string } = {
  fl: "np.array([-1.5, 0., 2.7, 300., -129., 65504., 0.1])", // float -> int casts of huge values are undefined behaviour in C: not tested
  int: "np.array([-1, 0, 2, 127, 128, 255, 256, -129, 65535, 70000])",
  bl: "np.array([True, False, True])",
  ci: "np.array([1+2j, -3.5j, 0j])",
  fn: "np.array([np.nan, np.inf, -np.inf, 1.])",
};
const isInt = (d: string) => /^u?int/.test(d);
const cases: string[] = [];
for (const [k, v] of Object.entries(V)) for (const d of DT) {
  if (k === "fn" && isInt(d)) continue;
  if (k === "ci" && (isInt(d) || d.startsWith("float"))) continue;
  if ((k === "fl" || k === "int") && isInt(d) && d !== "int64" && d !== "uint64") {
    // out of range values wrap around (astype), numpy raises when building from python lists
    cases.push(`${v}.astype('${d}')`);
    continue;
  }
  cases.push(`${v}.astype('${d}')`);
}
const NUM = DT.slice(0, -2);
for (const a of NUM) for (const b of NUM) {
  cases.push(`np.array([1, 0, 3], dtype='${a}') + np.array([2, 1, 0], dtype='${b}')`);
  cases.push(`np.array([1, 0, 3], dtype='${a}') * np.array([2, 1, 0], dtype='${b}')`);
  cases.push(`np.array([1, 0, 3], dtype='${a}') == np.array([1, 1, 0], dtype='${b}')`);
  cases.push(`np.maximum(np.array([1, 0, 3], dtype='${a}'), np.array([2, 1, 0], dtype='${b}'))`);
  if (a !== "bool" && b !== "bool") {
    cases.push(`np.array([1, 7, 3], dtype='${a}') / np.array([2, 1, 5], dtype='${b}')`);
    cases.push(`np.array([5, 7, 3], dtype='${a}') - np.array([2, 1, 3], dtype='${b}')`);
    cases.push(`np.array([5, 7, 3], dtype='${a}') // np.array([2, 1, 3], dtype='${b}')`);
    cases.push(`np.array([5, 7, 3], dtype='${a}') % np.array([2, 1, 3], dtype='${b}')`);
  }
}
for (const a of DT) {
  for (const sc of ["1", "2.5", "True", "1j", "300", "-1"]) {
    cases.push(`np.array([1, 2, 3], dtype='${a}') + ${sc}`);
    cases.push(`${sc} * np.array([1, 2, 3], dtype='${a}')`);
    if (sc !== "1j") cases.push(`np.array([1, 2, 3], dtype='${a}') > ${sc}`);
  }
  cases.push(`np.zeros(3, dtype='${a}')`, `np.ones((2,2), dtype='${a}')`, `np.full(3, 7, dtype='${a}')`, `np.eye(3, dtype='${a}')`, `np.empty(0, dtype='${a}')`);
  if (!a.startsWith("complex") && a !== "bool") cases.push(`np.arange(5, dtype='${a}')`);
  if (a !== "bool") cases.push(`np.linspace(0, 10, 5, dtype='${a}')`);
  for (const f of ["sum", "prod", "mean", "cumsum", "max", "sort"]) cases.push(`np.${f}(np.array([3, 1, 2], dtype='${a}'))`);
  cases.push(`np.dtype('${a}').itemsize`);
  if (isInt(a) && !a.endsWith("64")) cases.push(`np.iinfo('${a}').max`, `np.iinfo('${a}').min`);
  if (a.startsWith("float")) cases.push(`np.finfo('${a}').eps`, `np.finfo('${a}').max`);
}
numpy_cases("dtype matrix", cases, { rtol: 1e-6, auto_errors: true });
numpy_cases("python ints must fit the integer dtype", [
  "!np.array([300], dtype='int8')", "!np.array([-1], dtype='uint8')", "!np.array([1, 2, 3], dtype='int8') + 300", "!np.array([1], dtype='uint16') + -1",
  "np.array([1, 2, 3], dtype='int8') > 300", "np.array([100], dtype='int8') + 27", "np.array([1.5, 2.5], dtype='int8')",
]);
