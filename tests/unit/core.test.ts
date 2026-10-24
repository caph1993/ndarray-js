import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

describe("kwargs handling", () => {
  const x = np.arange(12).reshape(3, 4);
  it("accepts positional, object and mixed arguments", () => {
    const want = np.tolist(np.sum(x, 1, true));
    expect(np.tolist(x.sum(1, true))).toEqual(want);
    expect(np.tolist(x.sum({ axis: 1, keepdims: true }))).toEqual(want);
    expect(np.tolist(x.sum(1, { keepdims: true }))).toEqual(want);
    expect(np.tolist(np.sum(x, { axis: 1, keepdims: true }))).toEqual(want);
    expect(np.tolist(np.sum({ a: x, axis: 1, keepdims: true }))).toEqual(want);
  });
  it("rejects wrong usages", () => {
    expect(() => x.sum({ axis: 1 }, { axis: 0 })).toThrow(/Duplicate/);
    expect(() => x.sum({ axes: 1 })).toThrow(/Invalid argument/);
    expect(() => (np.sum as any)(x, 0, true, null, null, 5)).toThrow(/Too many/);
    expect(() => (np.add as any)(x)).toThrow(/Missing/);
  });
  it("does not treat functions or typed arrays as kwargs", () => {
    expect(np.tolist(np.array(new Float64Array([1.5, 2.5])))).toEqual([1.5, 2.5]);
  });
  it("follows numpy argument order for std/var", () => {
    expect(np.std(np.array([1, 2, 3, 4]), 0, 1)).toBeCloseTo(Math.sqrt(5 / 3));
  });
});

describe("out arguments", () => {
  it("writes binary operations into out", () => {
    const out = np.zeros(3);
    const res = np.add(np.array([1, 2, 3]), 0.5, { out });
    expect(res).toBe(out);
    expect(np.tolist(out)).toEqual([1.5, 2.5, 3.5]);
  });
  it("writes unary operations into out and checks dtypes", () => {
    const out = np.zeros(2);
    np.sqrt(np.array([4, 9]), out);
    expect(np.tolist(out)).toEqual([2, 3]);
    expect(() => np.sqrt(np.array([4, 9]), np.zeros(2, 'int32'))).toThrow(/cannot hold/);
    expect(() => np.add(np.array([1, 2]), 1, { out: np.zeros(3) })).toThrow(/shape/);
  });
  it("writes reductions into out", () => {
    const out = np.zeros(3);
    np.sum(np.arange(12).reshape(4, 3), 0, false, null, out);
    expect(np.tolist(out)).toEqual([18, 22, 26]);
  });
});

describe("views and in-place operations", () => {
  it("modifies the parent through views", () => {
    const a = np.arange(12).reshape(3, 4);
    a.index(1).assign(0);
    a.index(':', 3).add_assign(100);
    a.index('::2', '::2').multiply_assign(-1);
    expect(np.tolist(a)).toEqual([[0, 1, -2, 103], [0, 0, 0, 100], [-8, 9, -10, 111]]);
  });
  it("supports views of views, None and ellipsis", () => {
    const a = np.arange(24).reshape(2, 3, 4);
    const v = a.index(':', 'None', '::-1', '1:3').index(1, 0).index('...', 1);
    expect(np.tolist(v)).toEqual(np.tolist(a.index(1, '::-1', 2)));
    const w = a.index('None', '...', 'None').index(0, 1, 2, ':', 0);
    expect(np.tolist(w)).toEqual(np.tolist(a.index(1, 2)));
  });
  it("returns plain values for full integer indices", () => {
    const a = np.array([true, false]);
    expect(a.index(0)).toBe(true);
    expect(np.arange(6).reshape(2, 3).index(1, 2)).toBe(5);
    expect(np.arange(6).reshape(2, 3).index(':', '::-1').index(1, 0)).toBe(5);
  });
  it("copies on advanced indexing", () => {
    const a = np.arange(5);
    const b = a.index([0, 1]);
    b.assign(99);
    expect(np.tolist(a)).toEqual([0, 1, 2, 3, 4]);
  });
  it("assigns with boolean masks and index lists", () => {
    const a = np.arange(6);
    a.assign(-1, a.greater(3));
    a.assign(np.array([7, 8]), [0, 1]);
    expect(np.tolist(a)).toEqual([7, 8, 2, 3, -1, -1]);
  });
  it("iterates over the first axis", () => {
    const rows = [...np.arange(6).reshape(2, 3)].map(r => np.tolist(r));
    expect(rows).toEqual([[0, 1, 2], [3, 4, 5]]);
    expect([...np.arange(3)]).toEqual([0, 1, 2]);
  });
});

describe("dtypes", () => {
  it("infers and promotes dtypes like numpy", () => {
    expect(np.array([1, 2]).dtype).toBe("int32");
    expect(np.array([1.5, 2]).dtype).toBe("float64");
    expect(np.array([true]).dtype).toBe("bool");
    expect(np.add(np.array([1, 2]), 0.5).dtype).toBe("float64");
    expect(np.add(np.array([1, 2], 'float32'), 0.5).dtype).toBe("float32");
    expect(np.add(np.array([1, 2], 'float32'), np.array([1, 2])).dtype).toBe("float64");
    expect(np.divide(np.array([1, 2]), 2).dtype).toBe("float64");
    expect(np.sum(np.array([1, 2])).valueOf()).toBe(3);
    expect(np.exp(np.array([1, 2])).dtype).toBe("float64");
    expect(np.cumsum(np.array([1, 2])).dtype).toBe("int64");
    expect(np.greater(np.array([1, 2]), 1).dtype).toBe("bool");
  });
  it("does not overflow int32 in sums and products", () => {
    expect(np.sum(np.arange(100000))).toBe(4999950000);
    expect(np.prod(np.arange(1, 21))).toBe(2432902008176640000);
  });
  it("keeps shape [1] arrays as arrays", () => {
    const r = np.add(np.array([1]), np.array([2]));
    expect(np.isscalar(r)).toBe(false);
    expect(np.tolist(r)).toEqual([3]);
  });
});

describe("printing", () => {
  it("prints arrays", () => {
    expect(String(np.arange(3))).toBe("np([0, 1, 2], shape=(3,), dtype=int32)");
    expect(String(np.array([[1.5, 2], [3, 4]]))).toBe("np([[1.5,   2],\n    [  3,   4]], shape=(2, 2), dtype=float64)");
    expect(String(np.array([true, false]))).toBe("np([ true, false], shape=(2,), dtype=bool)");
    expect(String(np.arange(2000))).toContain("...");
    expect(String(np.array([0.1 + 0.2, NaN, Infinity]))).toBe("np([0.3, nan, inf], shape=(3,), dtype=float64)");
  });
});

describe("template parser", () => {
  it("evaluates slices, assignments and float literals", () => {
    const a = np.arange(6).reshape(2, 3);
    expect(np.tolist(np`${a}[:, 1:]`)).toEqual([[1, 2], [4, 5]]);
    expect(np.tolist(np`${a}[-1, ::-1]`)).toEqual([5, 4, 3]);
    expect(np`np.array([1, 2]) * 2.0`.dtype).toBe("float64");
    expect(np`np.array([1., 2])`.dtype).toBe("float64");
    expect(np`np.array([1, 2]) * 2`.dtype).toBe("int32");
    expect(np.tolist(np`np.arange(10)[2:8:2] // 2`)).toEqual([1, 2, 3]);
    expect(np.tolist(np`np.linspace(0., 1., 3)`)).toEqual([0, 0.5, 1]);
  });
  it("refuses to walk dangerous paths", () => {
    expect(() => np`np.constructor.constructor("return 1")()`).toThrow();
  });
});

describe("sorting in place", () => {
  it("sorts along an axis", () => {
    const a = np.array([[3, 1, 2], [9, 8, 7]]);
    a.sort();
    expect(np.tolist(a)).toEqual([[1, 2, 3], [7, 8, 9]]);
    a.index(':', '::-1').sort();
    expect(np.tolist(a)).toEqual([[3, 2, 1], [9, 8, 7]]);
  });
});
