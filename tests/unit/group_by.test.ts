import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

describe("np.group_by", () => {
  const keys = ['b', 'a', 'b', 'c', 'a', 'b'];
  it("returns sorted keys and the positions of each group", () => {
    const g = (np as any).group_by(keys);
    expect(g.keys.tolist()).toEqual(['a', 'b', 'c']);
    expect(g.indices.map((x: any) => x.tolist())).toEqual([[1, 4], [0, 2, 5], [3]]);
    expect(g.groups.map((x: any) => x.tolist())).toEqual([[1, 4], [0, 2, 5], [3]]);
  });
  it("splits values along an axis", () => {
    const v = np.arange(12).reshape(6, 2);
    const g = (np as any).group_by([1, 0, 1, 2, 0, 1], v);
    expect(g.keys.tolist()).toEqual([0, 1, 2]);
    expect(g.groups[0].tolist()).toEqual([[2, 3], [8, 9]]);
    expect(g.groups[2].tolist()).toEqual([[6, 7]]);
    const h = (np as any).group_by([0, 1, 0], np.arange(6).reshape(2, 3), 1);
    expect(h.groups[0].tolist()).toEqual([[0, 2], [3, 5]]);
  });
  it("applies a function per group and stacks the results", () => {
    const g = (np as any).group_by([1, 0, 1, 0], [10, 20, 30, 40], { func: (x: any) => np.sum(x) });
    expect(g.keys.tolist()).toEqual([0, 1]);
    expect(g.values.tolist()).toEqual([60, 40]);
    const m = (np as any).group_by([1, 0, 1, 0], np.arange(8).reshape(4, 2), 0, (x: any) => np.mean(x, 0));
    expect(m.values.tolist()).toEqual([[4, 5], [2, 3]]);
  });
  it("validates lengths", () => {
    expect(() => (np as any).group_by([1, 2], [1, 2, 3])).toThrow();
    expect(() => (np as any).group_by([[1, 2]])).toThrow();
  });
});
