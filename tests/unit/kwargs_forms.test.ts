import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

const N: any = np;
const x = () => N.arange(24).reshape(2, 3, 4);
const f = () => N.arange(12).reshape(3, 4).divide(5);
const eq = (a: any, b: any) => expect(N.array_equal(a, b) || (typeof a === 'number' && a === b)).toBe(true);

// [function name, positional args, equivalent keyword form, array factory]
const forms: [string, any[], any, () => any][] = [
  ["sum", [1], { axis: 1 }, x], ["sum", [1, true], { axis: 1, keepdims: true }, x], ["mean", [-1, true], { axis: -1, keepdims: true }, x], ["std", [0, 1], { axis: 0, ddof: 1 }, f],
  ["var", [1, 0, true], { axis: 1, ddof: 0, keepdims: true }, f], ["max", [2], { axis: 2 }, x], ["min", [null, true], { axis: null, keepdims: true }, x], ["argmax", [1], { axis: 1 }, x], ["prod", [0], { axis: 0 }, f],
  ["cumsum", [1], { axis: 1 }, x], ["cumprod", [1], { axis: 1 }, f], ["sort", [0], { axis: 0 }, f], ["argsort", [1], { axis: 1 }, f], ["diff", [1, 1], { n: 1, axis: 1 }, x], ["median", [0], { axis: 0 }, f],
  ["percentile", [50, 1], { q: 50, axis: 1 }, f], ["quantile", [0.25, 0], { q: 0.25, axis: 0 }, f], ["clip", [2, 7], { a_min: 2, a_max: 7 }, x], ["round", [1], { decimals: 1 }, f],
  ["flip", [1], { axis: 1 }, x], ["roll", [2, 1], { shift: 2, axis: 1 }, x], ["repeat", [2, 0], { repeats: 2, axis: 0 }, x], ["tile", [[2, 1]], { reps: [2, 1] }, f], ["squeeze", [], {}, () => N.zeros([1, 3, 1])],
  ["expand_dims", [1], { axis: 1 }, f], ["swapaxes", [0, 2], { axis1: 0, axis2: 2 }, x], ["moveaxis", [0, -1], { source: 0, destination: -1 }, x], ["take", [[0, 2], 1], { indices: [0, 2], axis: 1 }, x],
  ["triu", [1], { k: 1 }, f], ["tril", [-1], { k: -1 }, f], ["trace", [1], { offset: 1 }, f], ["diagonal", [1], { offset: 1 }, f], ["unique", [], {}, () => N.array([3, 1, 3])],
  ["searchsorted", [[2, 5]], { v: [2, 5] }, () => N.array([1, 3, 5, 7])], ["reshape", [[4, 6]], { shape: [4, 6] }, x], ["transpose", [[2, 0, 1]], { axes: [2, 0, 1] }, x],
  ["nansum", [0], { axis: 0 }, f], ["any", [1], { axis: 1 }, f], ["all", [0], { axis: 0 }, f], ["ptp", [1], { axis: 1 }, f], ["average", [1], { axis: 1 }, f], ["count_nonzero", [0], { axis: 0 }, f],
];

// numpy names the array argument of some functions differently
const FIRST: { [name: string]: string } = { flip: 'm', tile: 'A', triu: 'm', tril: 'm', unique: 'ar' };

describe("keyword forms of the API give the same result", () => {
  for (const [name, args, kw, make] of forms) {
    const label = `${name}(${args.map(a => JSON.stringify(a)).join(', ')}) == ${name}(${JSON.stringify(kw)})`;
    it(label, () => {
      const a = make();
      const want = N[name](a, ...args);
      // function, options object
      eq(N[name](a, kw), want);
      // function, everything in one object
      eq(N[name]({ [FIRST[name] || "a"]: a, ...kw }), want);
      // first positional then the rest in an object
      if (args.length && Object.keys(kw).length === args.length) eq(N[name](a, args[0], ...(args.length > 1 ? [Object.fromEntries(Object.entries(kw).slice(1))] : [])), want);
    });
  }
  it("methods accept the same forms", () => {
    const a = x();
    eq(a.sum(1, true), a.sum({ axis: 1, keepdims: true }));
    eq(a.sum(1, { keepdims: true }), np.sum(a, 1, true));
    eq(a.mean({ axis: [0, 2] }), np.mean(a, [0, 2]));
    eq(a.reshape(4, 6), a.reshape([4, 6]));
    eq(a.transpose(2, 0, 1), np.transpose(a, [2, 0, 1]));
    eq(a.max(-1), np.max(a, { axis: -1 }));
    eq(a.argmax({ axis: 1 }), a.argmax(1));
  });
});

describe("invalid arguments are rejected", () => {
  it("unknown keyword", () => {
    expect(() => N.sum(x(), { axiss: 1 })).toThrow();
    expect(() => x().mean({ keepdim: true })).toThrow();
    expect(() => N.sort(x(), { axis: 0, order: null, bogus: 1 })).toThrow();
  });
  it("duplicated argument", () => {
    expect(() => N.sum(x(), 1, { axis: 2 })).toThrow();
    expect(() => N.clip(x(), 1, { a_min: 2 })).toThrow();
  });
  it("too many positional arguments", () => {
    expect(() => N.argmax(x(), 1, true, null, 5, 6)).toThrow();
    expect(() => N.transpose(x(), [0, 1, 2], 4)).toThrow();
  });
  it("axis out of range / repeated axis", () => {
    expect(() => N.sum(x(), 3)).toThrow();
    expect(() => N.sum(x(), [0, 0])).toThrow();
    expect(() => N.transpose(x(), [0, 0, 1])).toThrow();
    expect(() => x().swapaxes(0, 5)).toThrow();
  });
  it("shape errors", () => {
    expect(() => x().reshape(5, 5)).toThrow();
    expect(() => N.add(N.ones([2, 3]), N.ones([4]))).toThrow();
    expect(() => N.concatenate([N.ones([2, 3]), N.ones([4, 3])], 1)).toThrow();
    expect(() => N.dot(N.ones([2, 3]), N.ones([2, 3]))).toThrow();
  });
  it("missing required argument", () => {
    expect(() => N.sum()).toThrow();
    expect(() => N.reshape(x())).toThrow();
  });
});

describe("scalars and arrays are interchangeable", () => {
  it("0-d results are plain JS values", () => {
    expect(typeof N.sum(N.arange(5))).toBe("number");
    expect(N.any(N.array([0, 1]))).toBe(true);
    expect(typeof N.argmax(N.arange(5))).toBe("number");
    expect(N.mean([1, 2, 3])).toBe(2);
  });
  it("JS arrays and typed arrays are accepted wherever arrays are", () => {
    expect(N.sum([[1, 2], [3, 4]], 0).tolist()).toEqual([4, 6]);
    expect(N.add([1, 2], [3, 4]).tolist()).toEqual([4, 6]);
    expect(N.sort(new Float64Array([3, 1, 2])).tolist()).toEqual([1, 2, 3]);
    expect(N.concatenate([[1], [2, 3]]).tolist()).toEqual([1, 2, 3]);
  });
  it("python scalars are weak (NEP 50)", () => {
    expect(N.add(N.array([1, 2], 'int8'), 1).dtype).toBe('int8');
    expect(N.add(N.array([1, 2], 'float32'), 1.5).dtype).toBe('float32');
    expect(N.add(N.array([1, 2], 'int8'), 1.5).dtype).toBe('float64');
    expect(N.add(N.array([1, 2], 'int8'), N.array([1], 'int16')).dtype).toBe('int16');
  });
});
