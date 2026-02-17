import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

describe("constructors", () => {
  it("creates arange correctly", () => {
    const arr = np.arange(5);
    expect(np.tolist(arr)).toEqual([0, 1, 2, 3, 4]);
  });

  it("creates zeros and ones", () => {
    const zeros = np.zeros([2, 2]);
    const ones = np.ones([2, 2]);
    expect(np.tolist(zeros)).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(np.tolist(ones)).toEqual([
      [1, 1],
      [1, 1],
    ]);
  });

  it("creates linspace", () => {
    const arr = np.linspace(0, 1, 5);
    expect(np.tolist(arr)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });
});
