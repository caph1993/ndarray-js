import { numpy_cases } from "../helpers/cases";

// Every reduction / accumulation against several dtypes, shapes, axis forms and keepdims, compared with numpy.
const FUNCS = "sum prod mean std var min max amin amax argmin argmax ptp all any median nansum nanmean nanmax nanmin nanstd nanvar nanprod nanargmax nanargmin nanmedian cumsum cumprod nancumsum nancumprod count_nonzero average".split(" ");
const IN: { [k: string]: string } = {
  i: "np.arange(24).reshape(2, 3, 4)",
  f: "(np.arange(24.).reshape(2, 3, 4) - 7.5) / 3",
  nan: "np.where(np.arange(24).reshape(2, 3, 4) % 5 == 0, np.nan, np.arange(24.).reshape(2, 3, 4))",
  b: "(np.arange(24).reshape(2, 3, 4) % 3 == 0)",
  u8: "np.arange(24, dtype='uint8').reshape(2, 3, 4) * 10",
  f32: "np.arange(24, dtype='float32').reshape(2, 3, 4) / 7",
  c: "(np.arange(24).reshape(2, 3, 4) + 1j * np.arange(24).reshape(2, 3, 4)[::-1])",
  e: "np.zeros((0, 3))",
  "1d": "np.array([3., 1., 2.])",
};
const AX = ["", ", axis=0", ", axis=1", ", axis=-1", ", axis=(0, 2)", ", axis=1, keepdims=True", ", axis=None, keepdims=True", ", axis=(0, 1), keepdims=True", ", axis=5"];
const cases: string[] = [];
for (const f of FUNCS) for (const [k, v] of Object.entries(IN)) for (const a of AX) {
  if (k === "e" && ![ "", ", axis=0", ", axis=1" ].includes(a)) continue;
  if (k === "1d" && ![ "", ", axis=0", ", axis=-1", ", axis=None, keepdims=True" ].includes(a)) continue;
  if (f === "average" && k === "e" && a === ", axis=1") continue; // numpy raises ZeroDivisionError
  cases.push(`np.${f}(${v}${a})`);
}
numpy_cases("reduction matrix", cases, { rtol: 1e-6, auto_errors: true });

numpy_cases("parenthesised call arguments", [
  "np.sum((np.arange(6.) - 7.5) / 3)",
  "np.abs((np.arange(3) - 1) * 2)",
  "np.sum((np.arange(6) % 3 == 0))",
  "np.add((1), 2)",
  "np.abs((-3))",
  "np.sum(((np.arange(4) + 1) * 2) ** 2)",
  "np.maximum((np.arange(4) - 2), (np.arange(4)[::-1] - 1))",
  "np.arange(10)[(np.arange(3) + 1) * 2]",
  "np.arange(10)[(1 + 1):(2 * 3)]",
]);
