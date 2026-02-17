import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

describe("basic flow", () => {
  it("reshapes and transposes", () => {
    const a = np.arange(6);
    const reshaped = np.reshape(a, [2, 3]);
    const transposed = np.transpose(reshaped);
    expect(np.tolist(transposed)).toEqual([
      [0, 3],
      [1, 4],
      [2, 5],
    ]);
  });

  it("uses template parser", () => {
    const result = np`np.arange(3) + ${2}`;
    expect(np.tolist(result)).toEqual([2, 3, 4]);
  });
});
