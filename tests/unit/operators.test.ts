import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

describe("operators", () => {
  it("adds scalar to array", () => {
    const arr = np.arange(4);
    const result = np.add(arr, 2);
    expect(np.tolist(result)).toEqual([2, 3, 4, 5]);
  });

  it("multiplies arrays elementwise", () => {
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const result = np.multiply(a, b);
    expect(np.tolist(result)).toEqual([4, 10, 18]);
  });

  it("compares arrays elementwise", () => {
    const a = np.array([1, 2, 3]);
    const result = np.greater_equal(a, 2);
    expect(np.tolist(result)).toEqual([0, 1, 1]);
  });
});
