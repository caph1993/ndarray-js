import { describe, it, expect } from "vitest";
import { HAS_NUMPY, numpy_batch, js_eval, expect_same } from "./numpy";

export type Case = string | [string | (() => any), string];

/**
 * Registers one test per case. The same string is evaluated by real numpy (python)
 * and by ndarray-js (through its numpy-syntax parser) and the results are compared.
 *  - Prefix a case with "!" to assert that numpy raises an error and ndarray-js raises too.
 *  - Prefix a case with "~" for features that ndarray-js does not support (e.g. complex results): ndarray-js must raise,
 *    whatever numpy returns.
 *  - A case can be a pair [ndarray-js code or function, python code] when the syntaxes differ
 *    (e.g. in-place operations, which are statements in python).
 * Options per group: rtol/atol for float comparison, check_dtype (compare dtype kind),
 * auto_errors (every case where numpy raises must also raise in ndarray-js, no "!" prefix needed).
 */
export function numpy_cases(name: string, cases: Case[], opts: { rtol?: number, atol?: number, check_dtype?: boolean, auto_errors?: boolean } = {}) {
  const norm = cases.map(c => {
    const [js, py] = Array.isArray(c) ? c : [c, c as string];
    const throws = typeof js === "string" && js.startsWith("!");
    const unsupported = typeof js === "string" && js.startsWith("~");
    const strip = (s: string) => (s.startsWith("!") || s.startsWith("~")) ? s.slice(1) : s;
    return { js: typeof js === "string" ? strip(js) : js, py: strip(py), throws, unsupported };
  });
  const expected = HAS_NUMPY ? numpy_batch(norm.map(c => c.py)) : [];
  describe.skipIf(!HAS_NUMPY)(name, () => {
    norm.forEach((c, i) => {
      it(typeof c.js === "string" ? c.js : `${c.py} [function]`, () => {
        const want = expected[i];
        const run = () => typeof c.js === "string" ? js_eval(c.js) : c.js();
        if (c.unsupported) { expect(run).toThrow(); return; }
        if (c.throws) {
          expect(want.ok, `numpy was expected to raise but returned ${JSON.stringify((want as any).value)}`).toBe(false);
          expect(run).toThrow();
          return;
        }
        if (!want.ok && opts.auto_errors) { expect(run, `numpy raised (${(want as any).error.slice(0, 80)}) but ndarray-js returned a value`).toThrow(); return; }
        if (!want.ok) throw new Error(`numpy raised: ${want.error} (mark the case with "!" if the error is expected)`);
        expect_same(run(), want.value, opts);
      });
    });
  });
}
