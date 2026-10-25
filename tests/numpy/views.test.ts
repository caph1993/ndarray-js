import { numpy_cases, Case } from "../helpers/cases";
import { np } from "../../src/index";
import { describe, it, expect } from "vitest";

const N: any = np;
// [ndarray-js function, python expression]; both return the base array after mutating through a view
const W = (js: (x: any) => any, py: string, base = "np.arange(24).reshape(2, 3, 4)"): Case =>
  [() => { const x = N(Object.assign([base.replace(/np\./g, "np.")], { raw: [base] })); js(x); return x; }, `(lambda x: (${py}, x)[1])(${base})`];

numpy_cases("transposed views share memory", [
  W(x => { const t = x.T; N`${t}[3, 1, 1] = 99`; }, "x.T.__setitem__((3, 1, 1), 99)"),
  W(x => { const t = x.transpose(1, 0, 2); N`${t}[1:, 0, ::2] = -1`; }, "x.transpose(1, 0, 2).__setitem__((slice(1, None), 0, slice(None, None, 2)), -1)"),
  W(x => { const t = x.swapaxes(0, 2); N`${t}[:, 1, :] = 7`; }, "x.swapaxes(0, 2).__setitem__((slice(None), 1, slice(None)), 7)"),
  W(x => { const t = np.moveaxis(x, 0, -1); N`${t}[..., 1] += 5`; }, "np.moveaxis(x, 0, -1).__setitem__((Ellipsis, 1), np.moveaxis(x, 0, -1)[..., 1] + 5)"),
  W(x => { const t = x.T.T; N`${t}[1, 1, 1] = 42`; }, "x.T.T.__setitem__((1, 1, 1), 42)"),
  W(x => { const t = x.index(':', '1:', ':').T; N`${t}[0, :, 1] = 11`; }, "x[:, 1:, :].T.__setitem__((0, slice(None), 1), 11)"),
  W(x => { const t = x.T.index('::2', '::-1'); N`${t}[:] = 3`; }, "x.T[::2, ::-1].__setitem__(slice(None), 3)"),
  W(x => { const t = x.transpose(2, 0, 1).index(1); N`${t}[0, :] = 0`; }, "x.transpose(2, 0, 1)[1].__setitem__((0, slice(None)), 0)"),
  W(x => { const t = x.index(':', 2).T; N`${t}[...] = 8`; }, "x[:, 2].T.__setitem__(Ellipsis, 8)"),
  W(x => { np.add(x, 1, { out: x.T.T }); }, "np.add(x, 1, out=x.T.T)"),
  W(x => { const t = x.T; np.multiply(t, 2, { out: t }); }, "np.multiply(x.T, 2, out=x.T)"),
  W(x => { np.copyto(x.T, np.zeros([4, 3, 2], 'int64')); }, "np.copyto(x.T, np.zeros((4, 3, 2), dtype=np.int64))"),
  W(x => { x.T.fill(5); }, "x.T.fill(5)"),
  W(x => { x.T.sort(); }, "x.T.sort()"),
  W(x => { np.fill_diagonal(x.T.index(':', ':', 0), 9); }, "np.fill_diagonal(x.T[:, :, 0], 9)"),
]);

numpy_cases("transposed views read like numpy", [
  "np.arange(24).reshape(2, 3, 4).T",
  "np.arange(24).reshape(2, 3, 4).transpose(1, 2, 0)[1:, ::-1]",
  "np.arange(24).reshape(2, 3, 4).T[2, 1:, ::2]",
  "np.arange(24).reshape(2, 3, 4).T.sum(axis=0)",
  "np.arange(24).reshape(2, 3, 4).T.sum(axis=(1, 2))",
  "np.arange(24).reshape(2, 3, 4).T.reshape(4, 6)",
  "np.arange(24).reshape(2, 3, 4).T.ravel()",
  "np.arange(24).reshape(2, 3, 4).T.T.shape",
  "np.arange(6.).reshape(2, 3).T @ np.arange(6.).reshape(2, 3)",
  "np.arange(6).reshape(2, 3).T + np.arange(2)",
  "np.arange(12).reshape(3, 4).T[:, None, :]",
  "np.sort(np.arange(12)[::-1].reshape(3, 4).T, axis=0)",
  "np.linalg.inv(np.array([[1., 2.], [3., 5.]]).T)",
  "np.concatenate([np.arange(6).reshape(2, 3).T, np.arange(6).reshape(2, 3).T])",
  "np.arange(24).reshape(2, 3, 4).T.copy()",
  "np.transpose(np.arange(24).reshape(2, 3, 4), (2, 0, 1)).tolist()",
  "np.arange(0).reshape(0, 3).T",
  "np.arange(4).T",
]);

describe("reshape shares memory when it can", () => {
  it("reshape/ravel of a contiguous array is a view", () => {
    const x = np.arange(6);
    const r = x.reshape(2, 3);
    N`${r}[1, 2] = 50`;
    expect(x.tolist()).toEqual([0, 1, 2, 3, 4, 50]);
    const q = r.ravel();
    N`${q}[0] = -1`;
    expect(x.tolist()[0]).toBe(-1);
  });
  it("reshape of a transposed view copies (numpy copies too)", () => {
    const x = np.arange(6).reshape(2, 3);
    const r = x.T.reshape(6);
    N`${r}[0] = 99`;
    expect(x.tolist()).toEqual([[0, 1, 2], [3, 4, 5]]);
  });
  it("transposing does not copy", () => {
    const x = np.arange(6).reshape(2, 3);
    const t = x.T;
    N`${x}[0, 1] = 77`;
    expect(t.tolist()).toEqual([[0, 3], [77, 4], [2, 5]]);
  });
});

numpy_cases("composition of views with new axes", [
  "np.arange(24).reshape(2, 3, 4)[1:][None]",
  "np.arange(24).reshape(2, 3, 4)[:, 1:][:, None, :, None]",
  "np.arange(24).reshape(2, 3, 4)[1][None, ..., None]",
  "np.arange(24).reshape(2, 3, 4)[::-1, 1][:, None, 1:]",
  "np.arange(24).reshape(2, 3, 4).T[None, 1:, :, None]",
  "np.arange(24).reshape(2, 3, 4).T[1][:, None][0]",
  "np.arange(24).reshape(2, 3, 4).transpose(2, 0, 1)[::2, 1][None]",
]);
