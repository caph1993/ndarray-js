// Loads the webpack bundle (dist/index.js) as a plain node module and runs a few calls.
// Usage: npm run build && node tests/bundle_smoke.js
const assert = require("assert");
const m = require("../dist/index.js");
const np = m.np || m.default || m;
assert.strictEqual(typeof np, "function");
assert.deepStrictEqual(np.arange(6).reshape(2, 3).T.tolist(), [[0, 3], [1, 4], [2, 5]]);
assert.strictEqual(np`np.sum(np.arange(10) ** 2)`, 285);
assert.deepStrictEqual(np.random.default_rng(0).integers(0, 10, 5).tolist(), [8, 6, 5, 2, 3]);
np.random.seed(0);
assert.ok(Math.abs(np.random.rand(1).tolist()[0] - 0.5488135039273248) < 1e-15);
assert.deepStrictEqual(np.logspace(0, 2, 3).tolist(), [1, 10, 100]);
assert.deepStrictEqual(np.add.reduce(np.arange(5)), 10);
assert.deepStrictEqual(np.linalg.solve([[2, 0], [0, 4]], [2, 4]).tolist(), [1, 1]);
assert.deepStrictEqual(np.fft.fft(np.array([1, 0, 0, 0])).tolist().map(z => z.re), [1, 1, 1, 1]);
console.log("bundle ok");
