import { numpy_cases } from "../helpers/cases";

const types = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"];
const pairs = types.flatMap(a => types.map(b => [a, b]));

numpy_cases("dtype utilities", [
  ...pairs.map(([a, b]) => `np.promote_types('${a}', '${b}')`),
  ...pairs.flatMap(([a, b]) => ["safe", "same_kind", "no", "unsafe"].map(c => `np.can_cast('${a}', '${b}', '${c}')`)),
  "np.can_cast('int32', 'float64')", "np.can_cast('float64', 'int32')",
  "np.result_type('int8', 'uint8')", "np.result_type('float32', 'int64')", "np.result_type(np.array([1, 2]), 2.5)", "np.result_type(np.array([1, 2], dtype='float32'), 2.5)", "np.result_type(np.array([1, 2], dtype='int8'), 100)", "np.result_type(np.array([True]), 1)",
  "np.result_type(1, 2.5)", "np.result_type(1, 2)", "np.result_type(1j, 2.5)", "np.result_type(np.array([1.5], dtype='float32'), 1j)", "np.result_type('float32', 'complex64', 'int8')", "np.result_type(np.array([1]), 'float32')",
  ...[10, 255, 256, 70000, 5000000000, -1, -128, -129, -40000, -5000000000, 0, 3.5, 0.1, 1e10, 1e300, true, 65504.0, 1e-8].map(v => `np.min_scalar_type(${v === true ? "True" : v})`),
  "np.common_type(np.array([1, 2], dtype='float32'))", "np.common_type(np.array([1, 2]))", "np.common_type(np.array([1.5]), np.array([1, 2], dtype='float32'))", "np.common_type(np.array([1j]))", "np.common_type(np.array([1j]), np.array([1.5]))", "!np.common_type(np.array([True]))",
  ...types.map(t => `np.finfo('${t}').eps`).filter(c => /float|complex/.test(c)).filter(c => !/complex64/.test(c) || true),
  ...["float16", "float32", "float64", "complex128"].flatMap(t => ["eps", "epsneg", "max", "min", "tiny", "smallest_normal", "smallest_subnormal", "bits", "precision", "resolution", "nmant", "nexp", "minexp", "maxexp", "machep", "negep", "iexp"].map(a => `np.finfo('${t}').${a}`)),
  ...["int8", "uint8", "int16", "uint16", "int32", "uint32"].flatMap(t => ["min", "max", "bits", "kind"].map(a => `np.iinfo('${t}').${a}`)),
  "np.iinfo('int64').bits", "!np.iinfo('float32')", "!np.finfo('int32')",
  ...["bool", "signed integer", "unsigned integer", "integral", "real floating", "complex floating", "numeric"].flatMap(k => ["bool", "int8", "uint32", "float32", "complex128", "int64"].map(t => `np.isdtype(np.dtype('${t}'), '${k}')`)),
  "np.isdtype(np.dtype('int32'), ('bool', 'integral'))", "np.isdtype(np.dtype('float32'), np.dtype('float32'))", "!np.isdtype(np.dtype('int32'), 'bogus')",
  ...["number", "integer", "signedinteger", "unsignedinteger", "inexact", "floating", "complexfloating", "generic"].flatMap(k => ["bool", "int8", "uint32", "float32", "complex128"].map(t => `np.issubdtype('${t}', np.${k})`)),
  "np.issubdtype('int32', 'int32')", "np.issubdtype('int32', 'int64')", "np.issubdtype('float32', 'float64')",
  ...["f", "d", "i", "?", "B", "D", "F", "b", "h", "l", "q", "O", "H", "I", "L", "Q", "g", "G", "S1", "S", "U", "V"].map(c => `np.typename('${c}')`),
  "!np.typename('z')",
  ...[["f", "d"], ["f"], ["D", "f"], ["F", "d"], ["i", "f"], ["?"], ["F", "D"], ["g", "G"], ["i"]].map(l => `np.mintypecode(${JSON.stringify(l).replace(/"/g, "'")})`),
  "np.mintypecode('fd')", "np.mintypecode(['f', 'd'], typeset='fd')", "np.mintypecode(['i'], default='f')",
  ...["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"].flatMap(t => ["name", "kind", "itemsize", "char", "num"].map(a => `np.dtype('${t}').${a}`)),
  "np.dtype('i4').name", "np.dtype('f8').name", "np.dtype('?').name", "np.dtype('c16').name", "np.dtype('float').name", "np.dtype('int').name", "np.dtype('complex').name", "!np.dtype('bogus')",
]);

numpy_cases("number formatting", [
  ...[0, 1, 5, 255, 256, 12345, -1, -5, -128].map(n => `np.binary_repr(${n})`),
  ...[[5, 8], [0, 4], [255, 10], [-5, 8], [-1, 4], [-128, 8]].map(([n, w]) => `np.binary_repr(${n}, ${w})`),
  "!np.binary_repr(1.5)", "!np.binary_repr(-200, 8)",
  ...[[255, 16], [255, 2], [0, 2], [-10, 2], [123456, 36], [35, 36], [100, 8], [-255, 16]].map(([n, b]) => `np.base_repr(${n}, ${b})`),
  "np.base_repr(5, 2, 8)", "np.base_repr(5, base=3, padding=2)", "np.base_repr(-5, 2, 6)", "!np.base_repr(5, 1)", "!np.base_repr(5, 37)",
  "np.packbits(np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1]))", "np.packbits(np.array([1, 0, 1, 1, 0, 0, 0, 1]))", "np.packbits(np.array([[1, 0, 1], [0, 1, 1]]), axis=1)", "np.packbits(np.array([[1, 0, 1], [0, 1, 1]]), axis=0)", "np.packbits(np.array([1, 0, 1]), bitorder='little')",
  "np.packbits(np.array([[1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0, 0, 0, 1]]))", "np.packbits(np.array([True, False, True]))", "!np.packbits(np.array([]))", "np.packbits(np.array([2, 0, 1]))", "!np.packbits(np.array([1.5]))",
  "np.unpackbits(np.array([177, 192], dtype='uint8'))", "np.unpackbits(np.array([5], dtype='uint8'), count=4)", "np.unpackbits(np.array([5], dtype='uint8'), bitorder='little')", "np.unpackbits(np.array([[5, 255], [1, 2]], dtype='uint8'), axis=1)", "np.unpackbits(np.array([[5, 255], [1, 2]], dtype='uint8'), axis=0)", "np.unpackbits(np.array([5], dtype='uint8'), count=-3)", "np.unpackbits(np.array([5], dtype='uint8'), count=10)", "!np.unpackbits(np.array([5]))",
  "np.unpackbits(np.packbits(np.array([1, 0, 1, 1, 0, 0, 0, 1, 1])).astype('uint8'), count=9)",
  ...[1.5, 0.1, 1e-5, 100.0, 123456.789, -0.5, 1e22, 2.0, 0.0, 1e-300, 1 / 3].flatMap(x => [`np.format_float_positional(${x})`, `np.format_float_positional(${x}, precision=3)`, `np.format_float_positional(${x}, trim='-')`, `np.format_float_positional(${x}, trim='0')`, `np.format_float_positional(${x}, trim='.')`, `np.format_float_positional(${x}, unique=False, precision=4)`,
    `np.format_float_scientific(${x})`, `np.format_float_scientific(${x}, precision=2)`, `np.format_float_scientific(${x}, trim='-')`, `np.format_float_scientific(${x}, unique=False, precision=3)`, `np.format_float_scientific(${x}, exp_digits=3)`]),
  "np.format_float_positional(1.5, sign=True)", "np.format_float_positional(1.5, pad_left=5)", "np.format_float_positional(1.5, pad_right=4)", "np.format_float_positional(2.0, min_digits=3)", "np.format_float_scientific(2.5, sign=True)", "np.format_float_scientific(2.5, pad_left=12)", "np.format_float_scientific(2.0, min_digits=4)",
  "np.format_float_positional(np.nan)", "np.format_float_positional(np.inf)", "np.format_float_positional(-np.inf)", "np.format_float_scientific(np.nan)", "!np.format_float_positional(1.5, trim='x')",
]);
