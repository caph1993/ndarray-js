import { describe, it, expect } from "vitest";
import { np } from "../../src/index";

const N: any = np;
// Generous time limits: they only catch accidental quadratic behaviour or per-element slow paths.
function within(ms: number, f: () => any) {
  const t0 = Date.now();
  f();
  const dt = Date.now() - t0;
  expect(dt, `took ${dt} ms (limit ${ms} ms)`).toBeLessThan(ms);
}

describe("performance sanity", () => {
  const big = N.arange(1_000_000);
  const bigf = N.divide(big, 7);
  it("elementwise and reductions on 1e6 elements", () => {
    within(1500, () => N.add(big, 1));
    within(1500, () => N.sqrt(bigf));
    within(1500, () => N.sum(bigf));
    within(1500, () => N.cumsum(big));
    within(1500, () => N.max(bigf));
    within(1500, () => N.sum(N.reshape(bigf, [1000, 1000]), 0));
    within(2000, () => N.sum(N.reshape(bigf, [1000, 1000]), 1));
  });
  it("broadcasting", () => {
    const m = N.reshape(bigf, [1000, 1000]);
    within(2000, () => N.add(m, N.arange(1000)));
    within(2000, () => N.multiply(m, N.reshape(N.arange(1000), [1000, 1]) ));
  });
  it("sorting and sets on 2e5 elements", () => {
    const r = N.random.rand(200_000);
    within(3000, () => N.sort(r));
    within(3000, () => N.argsort(r));
    within(3000, () => N.unique(N.floor(N.multiply(r, 1000))));
    within(3000, () => N.searchsorted(N.sort(r), N.random.rand(50_000)));
  });
  it("views and indexing", () => {
    const m = N.reshape(bigf, [1000, 1000]);
    within(1000, () => { for (let i = 0; i < 300; i++) m.index(i, ':'); });
    within(1000, () => { for (let i = 0; i < 2000; i++) N.transpose(m); });
    within(2500, () => N.sum(N.transpose(m), 0));
    within(2500, () => m.index('::2', '::3').tolist());
    within(1500, () => N.where(N.greater(bigf, 5000), bigf, 0));
    within(1500, () => N.take(bigf, N.arange(0, 1_000_000, 3)));
  });
  it("linear algebra and fft", () => {
    const a = N.random.rand(150, 150);
    within(3000, () => N.matmul(a, a));
    within(4000, () => N.linalg.inv(a));
    within(4000, () => N.linalg.svd(N.random.rand(60, 60)));
    within(3000, () => N.fft.fft(N.random.rand(1 << 15)));
  });
  it("many small operations", () => {
    within(3000, () => { let x = N.arange(10); for (let i = 0; i < 20000; i++) x = N.add(x, 1); });
    within(3000, () => { for (let i = 0; i < 2000; i++) N`np.sum(np.arange(10) * 2 + 1)`; });
  });
  it("random numbers", () => {
    within(2000, () => N.random.randn(500_000));
    within(3000, () => N.random.default_rng(0).standard_normal(300_000));
    within(3000, () => N.random.default_rng(0).integers(0, 100, 300_000));
  });
});
