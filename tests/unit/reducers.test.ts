import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

describe("reducers", () => {
  it("sums and averages", () => {
    const arr = np.arange(5);
    expect(np.sum(arr)).toBe(10);
    expect(np.mean(arr)).toBe(2);
  });

  it("finds min and max", () => {
    const arr = np.array([3, 1, 4, 2]);
    expect(np.min(arr)).toBe(1);
    expect(np.max(arr)).toBe(4);
  });
});
