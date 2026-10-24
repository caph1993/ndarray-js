import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const arrays = [
  "np.arange(3)", "np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])", "np.array([-1, 10, 100])", "np.array([[1, 2], [3, 4]])", "np.arange(24).reshape(2, 3, 4)", "np.arange(120).reshape(2, 3, 4, 5)",
  "np.array([1.5, 2.0])", "np.array([1.0, 2.0, 3.0])", "np.array([0.1, 0.25, 100.5])", "np.array([1.123456789, 2.5, -3.0])", "np.array([[1.5, 2.25], [3.125, 4.0]])", "np.array([1e-9, 1.0])", "np.array([1e10, 1.0])", "np.array([1e-5, 1.5])",
  "np.array([1e16, 1.0])", "np.array([0.0001, 0.001])", "np.array([123456789.0, 1.5])", "np.array([np.nan, 1.5, np.inf, -np.inf])", "np.array([np.nan, np.inf])", "np.array([-0.0, 0.0])", "np.array([1/3, 2/3])", "np.array([100000.0, 200000.5])",
  "np.array([True, False])", "np.array([[True, False], [False, True]])", "np.array([True])", "np.zeros((2, 3))", "np.array([1.5, 2.5], dtype='float32')", "np.array([1, 2], dtype='uint8')", "np.array([1, 2], dtype='int16')", "np.array([0.5, 1.5], dtype='float16')",
  "np.array([1+2j, 3.5-1j])", "np.array([1j, 2])", "np.array([[1+1j, 2], [3, 4j]])", "np.array([1.5+0.25j])", "np.array([1+2j], dtype='complex64')", "np.array(5)", "np.array(2.5)", "np.array(True)", "np.array(1+2j)",
  "np.arange(1000)", "np.arange(1001)", "np.arange(2000).reshape(40, 50)", "np.linspace(0, 1, 2000)", "np.arange(100.0)", "np.arange(30)", "np.arange(30.0) * 1.5", "np.arange(40).reshape(4, 10) * 1000",
  "np.array([[1.5, 22.25, 333.125], [4.0, 5.0, 6.0]])", "np.array([1, 2, 3]) / 7", "np.array([[0.1, 0.2], [0.3, 0.4]]) * 1e-3", "np.linspace(0, 100, 7)", "np.array([1e300, 1e-300])",
];

numpy_cases("array_repr / array_str", [
  ...arrays.map(a => `np.array_repr(${a})`),
  ...arrays.map(a => `np.array_str(${a})`),
  "np.array_repr(np.zeros(0))", "np.array_repr(np.zeros((0, 3)))", "np.array_repr(np.array([], dtype='int32'))", "np.array_str(np.zeros(0))", "np.array_repr(np.arange(3), max_line_width=5)", "np.array_repr(np.arange(20), max_line_width=30)", "np.array_str(np.arange(20), max_line_width=30)",
  "np.array_repr(np.array([1.123456789]), precision=3)", "np.array_str(np.array([1.123456789, 2.5]), precision=2)", "np.array_repr(np.array([1e-9, 1]), suppress_small=True)", "np.array_str(np.array([0.00001, 1.5]), suppress_small=True)",
  "np.array_repr(np.arange(60.0).reshape(3, 4, 5))",
]);

numpy_cases("array2string", [
  "np.array2string(np.array([1.123456789, 2.5]), precision=3)", "np.array2string(np.arange(3), separator=',')", "np.array2string(np.arange(3), separator=', ')", "np.array2string(np.arange(10), threshold=5)", "np.array2string(np.arange(10), threshold=5, edgeitems=2)",
  "np.array2string(np.array([1e-9, 1]))", "np.array2string(np.array([1.5, 2.5]), prefix='x = ')", "np.array2string(np.array([1.5, 2.5]), suffix=')')", "np.array2string(np.arange(30), max_line_width=40)", "np.array2string(np.arange(30), max_line_width=40, separator=', ')",
  "np.array2string(np.array([1.5, -2.5]), sign='+')", "np.array2string(np.array([1.5, -2.5]), sign=' ')", "np.array2string(np.array([1.0, 2.5, 3.14159]), floatmode='fixed', precision=3)", "np.array2string(np.array([1.0, 2.5, 3.14159]), floatmode='unique')", "np.array2string(np.array([1.0, 2.5, 3.14159]), floatmode='maxprec_equal', precision=3)",
  "np.array2string(np.array([1.0, 2.5, 3.14159]), floatmode='maxprec', precision=2)", "np.array2string(np.array([0.5, 0.25]), floatmode='fixed', precision=1)", "np.array2string(np.array([100000.0, 0.00012]), suppress_small=True)", "np.array2string(np.array([100000.0, 0.00012]))",
  "np.array2string(np.arange(24).reshape(2, 3, 4), separator=', ')", "np.array2string(np.arange(24).reshape(2, 3, 4), max_line_width=20)", "np.array2string(np.array(5))", "np.array2string(np.array([True, False]))", "np.array2string(np.array([1+2j, 3-4j]))", "np.array2string(np.array([1.5+2j, 3-4.25j]), precision=1)",
  "np.array2string(np.array([[1, 22], [333, 4]]), prefix='array(', separator=', ', suffix=')')",
  [() => np.array2string(np.array([1, 2, 3]), { formatter: { int: (x: number) => String(x * 2) } } as any), "np.array2string(np.array([1, 2, 3]), formatter={'int': lambda x: str(x * 2)})"],
  [() => np.array2string(np.array([1.5, 2.5]), { formatter: { float_kind: (x: number) => x.toFixed(1) + 'f' } } as any), "np.array2string(np.array([1.5, 2.5]), formatter={'float_kind': lambda x: '%.1ff' % x})"],
  [() => np.array2string(np.array([true, false]), { formatter: { bool: (x: boolean) => x ? 'Y' : 'N' } } as any), "np.array2string(np.array([True, False]), formatter={'bool': lambda x: 'Y' if x else 'N'})"],
  [() => np.array2string(np.array([1, 2]), { formatter: { all: (x: any) => '<' + x + '>' } } as any), "np.array2string(np.array([1, 2]), formatter={'all': lambda x: '<' + str(x) + '>'})"],
]);

numpy_cases("print options", [
  [() => { np.set_printoptions({ precision: 3 }); const r = np.array_repr(np.array([1.123456789, 2.0])); np.set_printoptions({ precision: 8 }); return r; }, "(lambda: (np.set_printoptions(precision=3), np.array_repr(np.array([1.123456789, 2.0])), np.set_printoptions(precision=8))[1])()"],
  [() => np.printoptions({ precision: 2 }, () => np.array_repr(np.array([1.123456789]))), "(lambda: (np.set_printoptions(precision=2), np.array_repr(np.array([1.123456789])), np.set_printoptions(precision=8))[1])()"],
  [() => { np.set_printoptions({ threshold: 5, edgeitems: 1 }); const r = np.array_repr(np.arange(10)); np.set_printoptions({ threshold: 1000, edgeitems: 3 }); return r; }, "(lambda: (np.set_printoptions(threshold=5, edgeitems=1), np.array_repr(np.arange(10)), np.set_printoptions(threshold=1000, edgeitems=3))[1])()"],
  [() => { np.set_printoptions({ linewidth: 20 }); const r = np.array_repr(np.arange(12)); np.set_printoptions({ linewidth: 75 }); return r; }, "(lambda: (np.set_printoptions(linewidth=20), np.array_repr(np.arange(12)), np.set_printoptions(linewidth=75))[1])()"],
  [() => { np.set_printoptions({ suppress: true }); const r = np.array_repr(np.array([1e-9, 1.5])); np.set_printoptions({ suppress: false }); return r; }, "(lambda: (np.set_printoptions(suppress=True), np.array_repr(np.array([1e-9, 1.5])), np.set_printoptions(suppress=False))[1])()"],
  [() => { np.set_printoptions({ nanstr: 'NAN', infstr: 'INF' }); const r = np.array_repr(np.array([np.nan, np.inf])); np.set_printoptions({ nanstr: 'nan', infstr: 'inf' }); return r; }, "(lambda: (np.set_printoptions(nanstr='NAN', infstr='INF'), np.array_repr(np.array([np.nan, np.inf])), np.set_printoptions(nanstr='nan', infstr='inf'))[1])()"],
  [() => { np.set_printoptions({ sign: '+' }); const r = np.array_repr(np.array([1.5, -2.5])); np.set_printoptions({ sign: '-' }); return r; }, "(lambda: (np.set_printoptions(sign='+'), np.array_repr(np.array([1.5, -2.5])), np.set_printoptions(sign='-'))[1])()"],
  [() => { np.set_printoptions({ floatmode: 'fixed' }); const r = np.array_repr(np.array([1.5, 2.0])); np.set_printoptions({ floatmode: 'maxprec' }); return r; }, "(lambda: (np.set_printoptions(floatmode='fixed'), np.array_repr(np.array([1.5, 2.0])), np.set_printoptions(floatmode='maxprec'))[1])()"],
  [() => np.get_printoptions().precision, "np.get_printoptions()['precision']"], [() => np.get_printoptions().threshold, "np.get_printoptions()['threshold']"], [() => np.get_printoptions().linewidth, "np.get_printoptions()['linewidth']"], [() => np.get_printoptions().floatmode, "np.get_printoptions()['floatmode']"],
  [() => { try { np.set_printoptions({ floatmode: 'bogus' }); return false; } catch (e) { return true; } }, "True"],
]);
