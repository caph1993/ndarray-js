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

  it("floor_divide", () => {
    const result = np`np.floor_divide([10, 20], [3, 7])`;
    expect(np.tolist(result)).toEqual([3, 2]);
  });

  it("power operations", () => {
    expect(np.tolist(np`np.power([2, 3], [3, 2])`)).toEqual([8, 9]);
    expect(np.tolist(np`np.pow([2, 3], [3, 2])`)).toEqual([8, 9]);
  });

  it("float_power", () => {
    const result = np.float_power([2.5, 3.5], [2, 1.5]);
    const expected = [6.25, 6.547900426854397];

    expect(np.max(np.abs(np.subtract(result, expected)))).toBeLessThan(1e-5);
  });

  it("positive and sign", () => {
    expect(np.tolist(np`np.positive([1, -2, 3])`)).toEqual([1, -2, 3]);
    expect(np.tolist(np`np.sign([-3, 0, 2])`)).toEqual([-1, 0, 1]);
  });

  it("maximum and minimum", () => {
    expect(np.tolist(np`np.maximum([1, 5, 3], [2, 3, 4])`)).toEqual([2, 5, 4]);
    expect(np.tolist(np`np.minimum([1, 5, 3], [2, 3, 4])`)).toEqual([1, 3, 3]);
    expect(np.tolist(np`np.amax([1, 5, 3, 2])`)).toEqual(5);
    expect(np.tolist(np`np.amin([1, 5, 3, 2])`)).toEqual(1);
  });

  it("sqrt, cbrt, square", () => {
    expect(np.tolist(np`np.sqrt([1, 4, 9, 16])`)).toEqual([1, 2, 3, 4]);
    expect(np.tolist(np`np.cbrt([1, 8, 27])`)).toEqual([1, 2, 3]);
    expect(np.tolist(np`np.square([1, 2, 3, 4])`)).toEqual([1, 4, 9, 16]);
  });

  it("absolute and fabs", () => {
    expect(np.tolist(np`np.absolute([-1, -2, 3])`)).toEqual([1, 2, 3]);
    expect(np.tolist(np`np.fabs([-1, -2, 3])`)).toEqual([1, 2, 3]);
  });

  it("all and any", () => {
    expect(np.tolist(np`np.all([${true}, ${true}, ${false}])`)).toEqual(false);
    expect(np.tolist(np`np.any([${false}, ${false}, ${true}])`)).toEqual(true);
  });

  it("isfinite, isinf, isnan", () => {
    expect(np.tolist(np`np.isfinite([0, ${np.inf}, ${np.nan}])`)).toEqual([true, false, false]);
    expect(np.tolist(np`np.isinf([0, ${np.inf}, -${np.inf}])`)).toEqual([false, true, true]);
    expect(np.tolist(np`np.isposinf([0, ${np.inf}, -${np.inf}])`)).toEqual([false, true, false]);
    expect(np.tolist(np`np.isneginf([0, ${np.inf}, -${np.inf}])`)).toEqual([false, false, true]);
    expect(np.tolist(np`np.isnan([0, ${np.nan}])`)).toEqual([false, true]);
  });

  it("logical operations", () => {
    expect(np.tolist(np`np.logical_and([1, 0], [1, 1])`)).toEqual([true, false]);
    expect(np.tolist(np`np.logical_or([0, 0], [1, 0])`)).toEqual([true, false]);
    expect(np.tolist(np`np.logical_not([1, 0])`)).toEqual([false, true]);
    expect(np.tolist(np`np.logical_xor([1, 0, 1], [1, 1, 0])`)).toEqual([false, true, true]);
  });

  it("comparison operations", () => {
    expect(np.tolist(np`np.greater([1, 2], [2, 1])`)).toEqual([false, true]);
    expect(np.tolist(np`np.greater_equal([1, 2], [1, 3])`)).toEqual([true, false]);
    expect(np.tolist(np`np.less([1, 2], [2, 1])`)).toEqual([true, false]);
    expect(np.tolist(np`np.less_equal([1, 2], [1, 1])`)).toEqual([true, false]);
    expect(np.tolist(np`np.equal([1, 2], [1, 3])`)).toEqual([true, false]);
    expect(np.tolist(np`np.not_equal([1, 2], [1, 3])`)).toEqual([false, true]);
  });

  it("trigonometric functions", () => {
    const sinResult = np.tolist(np`np.sin([0, 1.5707963267948966])`);
    expect(sinResult[0]).toBeCloseTo(0);
    expect(sinResult[1]).toBeCloseTo(1);

    const cosResult = np.tolist(np`np.cos([0, 1.5707963267948966])`);
    expect(cosResult[0]).toBeCloseTo(1);
    expect(cosResult[1]).toBeCloseTo(0);
  });

  it("hypot and arctan2", () => {
    expect(np.tolist(np`np.hypot([3, 5], [4, 12])`)).toEqual([5, 13]);

    const atan2Result = np.tolist(np`np.arctan2([1, 2], [3, 4])`);
    expect(atan2Result[0]).toBeCloseTo(0.32175055439664219);
    expect(atan2Result[1]).toBeCloseTo(0.46364760900080609);
  });

  it("arithmetic operations", () => {
    expect(np.tolist(np`np.add([1, 2], [3, 4])`)).toEqual([4, 6]);
    expect(np.tolist(np`np.subtract([5, 6], [1, 2])`)).toEqual([4, 4]);
    expect(np.tolist(np`np.multiply([2, 3], [4, 5])`)).toEqual([8, 15]);
    expect(np.tolist(np`np.divide([10, 20], [2, 4])`)).toEqual([5, 5]);
    expect(np.tolist(np`np.true_divide([10, 20], [2, 4])`)).toEqual([5, 5]);
    expect(np.tolist(np`np.mod([10, 20], [3, 7])`)).toEqual([1, 6]);
  });

});
