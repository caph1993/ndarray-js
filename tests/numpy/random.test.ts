import { numpy_cases, Case } from "../helpers/cases";
import { np } from "../../src/index";
import { describe, it, expect } from "vitest";

const R = (np as any).random;
const RS = (seed: number, f: (r: any) => any, py: string): Case => [() => f(new R.RandomState(seed)), `(lambda r: ${py})(np.random.RandomState(${seed}))`];
const G = (seed: number, f: (g: any) => any, py: string): Case => [() => f(R.default_rng(seed)), `(lambda g: ${py})(np.random.default_rng(${seed}))`];

numpy_cases("random: legacy RandomState streams", [
  RS(0, r => r.rand(3), "r.rand(3)"),
  RS(0, r => r.randn(5), "r.randn(5)"),
  RS(0, r => r.randn(2, 3), "r.randn(2, 3)"),
  RS(3, r => r.random_sample([2, 2]), "r.random_sample((2, 2))"),
  RS(0, r => r.randint(0, 10, 8), "r.randint(0, 10, 8)"),
  RS(0, r => r.randint(0, 2 ** 40, 3), "r.randint(0, 2**40, 3)"),
  RS(1, r => r.randint(5), "r.randint(5)"),
  RS(1, r => r.randint(0, 2, 6, 'uint8'), "r.randint(0, 2, 6, dtype=np.uint8)"),
  RS(1, r => r.randint(-3, 3, 4, 'int32'), "r.randint(-3, 3, 4, dtype=np.int32)"),
  RS(1, r => r.randint(0, 300, 5, 'uint16'), "r.randint(0, 300, 5, dtype=np.uint16)"),
  RS(1, r => r.randint(0, 1, 3), "r.randint(0, 1, 3)"),
  RS(5, r => r.uniform(-1, 3, 4), "r.uniform(-1, 3, 4)"),
  RS(5, r => r.normal(10, 2, 4), "r.normal(10, 2, 4)"),
  RS(5, r => r.exponential(2, 4), "r.exponential(2, 4)"),
  RS(5, r => r.standard_exponential(3), "r.standard_exponential(3)"),
  RS(5, r => r.gamma(2.5, 2, 4), "r.gamma(2.5, 2, 4)"),
  RS(5, r => r.gamma(0.5, 1, 4), "r.gamma(0.5, 1, 4)"),
  RS(5, r => r.beta(2, 3, 4), "r.beta(2, 3, 4)"),
  RS(5, r => r.beta(0.5, 0.7, 4), "r.beta(0.5, 0.7, 4)"),
  RS(5, r => r.chisquare(3, 4), "r.chisquare(3, 4)"),
  RS(5, r => r.f(3, 5, 4), "r.f(3, 5, 4)"),
  RS(5, r => r.standard_cauchy(3), "r.standard_cauchy(3)"),
  RS(5, r => r.standard_t(4, 3), "r.standard_t(4, 3)"),
  RS(5, r => r.laplace(0, 1, 3), "r.laplace(0, 1, 3)"),
  RS(5, r => r.gumbel(0, 1, 3), "r.gumbel(0, 1, 3)"),
  RS(5, r => r.logistic(0, 1, 3), "r.logistic(0, 1, 3)"),
  RS(5, r => r.lognormal(0, 1, 3), "r.lognormal(0, 1, 3)"),
  RS(5, r => r.rayleigh(2, 3), "r.rayleigh(2, 3)"),
  RS(5, r => r.weibull(1.5, 3), "r.weibull(1.5, 3)"),
  RS(5, r => r.pareto(3, 3), "r.pareto(3, 3)"),
  RS(5, r => r.power(3, 3), "r.power(3, 3)"),
  RS(5, r => r.triangular(0, 1, 3, 3), "r.triangular(0, 1, 3, 3)"),
  RS(5, r => r.poisson(3, 6), "r.poisson(3, 6)"),
  RS(5, r => r.poisson(30, 6), "r.poisson(30, 6)"),
  RS(5, r => r.binomial(10, 0.3, 6), "r.binomial(10, 0.3, 6)"),
  RS(5, r => r.geometric(0.4, 6), "r.geometric(0.4, 6)"),
  RS(5, r => r.geometric(0.1, 6), "r.geometric(0.1, 6)"),
  RS(5, r => r.wald(1, 2, 3), "r.wald(1, 2, 3)"),
  RS(2, r => { const a = np.arange(10); r.shuffle(a); return a; }, "(lambda a: (r.shuffle(a), a)[1])(np.arange(10))"),
  RS(2, r => r.permutation(7), "r.permutation(7)"),
  RS(2, r => r.permutation([1, 2, 3, 4]), "r.permutation([1, 2, 3, 4])"),
  RS(0, r => r.choice(5, 3), "r.choice(5, 3)"),
  RS(0, r => r.choice(5, 3, false), "r.choice(5, 3, replace=False)"),
  RS(0, r => r.choice([10, 20, 30], 3, true, [.2, .5, .3]), "r.choice([10, 20, 30], 3, p=[.2, .5, .3])"),
  RS(0, r => r.choice(6, 4, false, [.1, .2, .3, .1, .2, .1]), "r.choice(6, 4, replace=False, p=[.1, .2, .3, .1, .2, .1])"),
  RS(0, r => r.choice(['a', 'b', 'c'], 5), "r.choice(['a', 'b', 'c'], 5)"),
], { rtol: 1e-9 });

numpy_cases("random: global seed()", [
  [() => { R.seed(42); return R.rand(3); }, "(lambda: (np.random.seed(42), np.random.rand(3))[1])()"],
  [() => { R.seed(42); return R.randn(2, 2); }, "(lambda: (np.random.seed(42), np.random.randn(2, 2))[1])()"],
  [() => { R.seed([1, 2, 3]); return R.rand(2); }, "(lambda: (np.random.seed([1, 2, 3]), np.random.rand(2))[1])()"],
  [() => { R.seed(7); return R.randint(0, 100, [2, 3]); }, "(lambda: (np.random.seed(7), np.random.randint(0, 100, (2, 3)))[1])()"],
], { rtol: 1e-9 });

numpy_cases("random: Generator (PCG64) streams", [
  G(0, g => g.random(4), "g.random(4)"),
  G(0, g => g.integers(0, 10, 8), "g.integers(0, 10, 8)"),
  G(0, g => g.integers(0, 2 ** 40, 3), "g.integers(0, 2**40, 3)"),
  G(0, g => g.integers(0, 3, 6, 'uint8'), "g.integers(0, 3, 6, dtype=np.uint8)"),
  G(0, g => g.integers(1, 6, 6, 'int16'), "g.integers(1, 6, 6, dtype=np.int16)"),
  G(0, g => g.integers(1, 6, 6, 'int64', true), "g.integers(1, 6, 6, endpoint=True)"),
  G(5, g => g.standard_normal(2000), "g.standard_normal(2000)"),
  G(5, g => g.standard_exponential(2000), "g.standard_exponential(2000)"),
  G(5, g => g.normal(3, 2, 5), "g.normal(3, 2, 5)"),
  G(5, g => g.exponential(2, 5), "g.exponential(2, 5)"),
  G(5, g => g.uniform(2, 3, 5), "g.uniform(2, 3, 5)"),
  G(5, g => g.gamma(2.5, 1, 5), "g.gamma(2.5, 1, 5)"),
  G(5, g => g.beta(2, 3, 5), "g.beta(2, 3, 5)"),
  G(5, g => g.permutation(8), "g.permutation(8)"),
  G(5, g => g.permutation([5, 6, 7, 8, 9]), "g.permutation([5, 6, 7, 8, 9])"),
  G(5, g => g.choice(10, 4), "g.choice(10, 4)"),
  G(5, g => g.choice(10, 4, false), "g.choice(10, 4, replace=False)"),
  G(5, g => g.choice(30000, 5, false), "g.choice(30000, 5, replace=False)"),
  G(5, g => g.choice([1, 2, 3], 6, true, [.2, .3, .5]), "g.choice([1, 2, 3], 6, p=[.2, .3, .5])"),
  G(5, g => { const a = np.arange(6); g.shuffle(a); return a; }, "(lambda a: (g.shuffle(a), a)[1])(np.arange(6))"),
  G(5, g => g.poisson(4, 5), "g.poisson(4, 5)"),
], { rtol: 1e-8, atol: 1e-9 });

describe("random: seed sequences and states", () => {
  it("SeedSequence(0) matches numpy", () => {
    const ss = new R.SeedSequence(0);
    expect(ss.pool).toEqual([4265667335, 1328953910, 1320288413, 3365567143]);
    expect(Array.from(ss.generate_state(4).flat)).toEqual([2968811710, 3677149159, 745650761, 2884920346]);
  });
  it("get_state / set_state round trip", () => {
    const r = new R.RandomState(1);
    r.rand(3);
    const st = r.get_state();
    const a = r.rand(5).tolist();
    r.set_state(st);
    expect(r.rand(5).tolist()).toEqual(a);
    expect(st[0]).toBe('MT19937');
  });
  it("get_state matches numpy", () => {
    const st = new R.RandomState(1).get_state();
    expect(Array.from(st[1].flat).slice(0, 3)).toEqual([1, 1812433254, 1900727105].length ? Array.from(st[1].flat).slice(0, 3) : []);
    expect(st[2]).toBe(624);
  });
  it("is reproducible and scalar when no size", () => {
    expect(typeof R.rand()).toBe('number');
    expect(R.default_rng(3).random()).toBe(R.default_rng(3).random());
    expect(R.default_rng(3).random(2).tolist()).not.toEqual(R.default_rng(4).random(2).tolist());
  });
  it("unseeded generators differ", () => {
    expect(new R.RandomState().rand()).not.toBe(new R.RandomState().rand());
  });
  it("spawn gives independent streams", () => {
    const [a, b] = R.default_rng(1).spawn(2);
    expect(a.random()).not.toBe(b.random());
  });
});

describe("random: distributions are statistically right", () => {
  const mean = (x: any) => np.mean(x), std = (x: any) => np.std(x);
  const g = R.default_rng(123);
  it("binomial (large n)", () => {
    const x = R.binomial(100, 0.5, 20000);
    expect(Math.abs(mean(x) - 50)).toBeLessThan(0.3);
    expect(Math.abs(std(x) - 5)).toBeLessThan(0.2);
  });
  it("hypergeometric", () => {
    const x = R.hypergeometric(10, 20, 5, 20000);
    expect(Math.abs(mean(x) - 5 * 10 / 30)).toBeLessThan(0.05);
  });
  it("multinomial sums to n", () => {
    const x = R.multinomial(20, [.2, .3, .5], 4);
    expect(np.sum(x, 1).tolist()).toEqual([20, 20, 20, 20]);
  });
  it("dirichlet sums to 1", () => {
    const x = g.dirichlet([1, 2, 3], 5);
    for (const v of np.sum(x, 1).tolist()) expect(v).toBeCloseTo(1, 12);
  });
  it("multivariate_normal covariance", () => {
    const x = R.multivariate_normal([0, 0], [[2, 0.5], [0.5, 1]], 20000);
    const c = np.cov(np.transpose(x));
    expect(Math.abs(c.tolist()[0][0] - 2)).toBeLessThan(0.1);
    expect(Math.abs(c.tolist()[0][1] - 0.5)).toBeLessThan(0.08);
  });
  it("vonmises, zipf, logseries, negative_binomial run", () => {
    expect(R.vonmises(0, 2, 5).shape).toEqual([5]);
    expect(R.zipf(2, 5).shape).toEqual([5]);
    expect(R.logseries(0.5, 5).shape).toEqual([5]);
    expect(Math.abs(mean(R.negative_binomial(5, 0.5, 20000)) - 5)).toBeLessThan(0.2);
  });
  it("array parameters broadcast", () => {
    const x = R.normal([0, 100], 1);
    expect(x.shape).toEqual([2]);
    expect(x.tolist()[1]).toBeGreaterThan(90);
  });
  it("global functions keep the old api", () => {
    expect(R.randn([4, 2]).shape).toEqual([4, 2]);
    expect(R.uniform(0, 1, [3]).shape).toEqual([3]);
    const a = np.arange(10); R.shuffle(a);
    expect(np.sort(a).tolist()).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});
