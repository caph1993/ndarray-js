import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { spawnSync } from "child_process";
import { describe, it, expect } from "vitest";
import { PYTHON, HAS_NUMPY } from "../helpers/numpy";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ndarrayjs-"));
const f = (name: string) => path.join(dir, name);
const py = (code: string) => { const r = spawnSync(PYTHON, ["-c", `import numpy as np\n${code}`]); if (r.status !== 0) throw new Error(r.stderr.toString()); return r.stdout.toString().trim(); };

const samples: [string, () => any, string][] = [
  ["int32 2d", () => np.arange(6).reshape(2, 3), "np.arange(6, dtype=np.int32).reshape(2, 3)"],
  ["float64 3d", () => np.arange(24).reshape(2, 3, 4).divide(7), "np.arange(24.0).reshape(2, 3, 4) / 7"],
  ["bool", () => np.array([true, false, true]), "np.array([True, False, True])"],
  ["float32", () => np.array([1.5, 2.5, -3.25], 'float32'), "np.array([1.5, 2.5, -3.25], dtype=np.float32)"],
  ["uint8", () => np.array([0, 127, 255], 'uint8'), "np.array([0, 127, 255], dtype=np.uint8)"],
  ["int16", () => np.array([-5, 300], 'int16'), "np.array([-5, 300], dtype=np.int16)"],
  ["int64", () => np.array([2 ** 40, -7], 'int64'), "np.array([2**40, -7], dtype=np.int64)"],
  ["complex128", () => np.array([1.5, 2], 'complex128'), "np.array([1.5, 2], dtype=np.complex128)"],
  ["float16", () => np.array([0.5, 1.5, -2], 'float16'), "np.array([0.5, 1.5, -2], dtype=np.float16)"],
  ["scalar", () => np.array(5.5), "np.array(5.5)"],
  ["empty", () => np.zeros([0, 3]), "np.zeros((0, 3))"],
];

describe.skipIf(!HAS_NUMPY)("npy / npz interoperability", () => {
  for (const [name, mk, pyexpr] of samples) {
    it(`${name}: numpy reads what ndarray-js saves`, () => {
      np.save(f(`a_${name}.npy`), mk());
      const out = py(`a = np.load(${JSON.stringify(f(`a_${name}.npy`))})\nb = ${pyexpr}\nprint(a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b)))`);
      expect(out).toBe("True");
    });
    it(`${name}: ndarray-js reads what numpy saves`, () => {
      py(`np.save(${JSON.stringify(f(`b_${name}.npy`))}, ${pyexpr})`);
      const got = np.load(f(`b_${name}.npy`)) as any;
      const want = mk();
      expect(got.shape).toEqual(want.shape);
      expect(got.dtype).toBe(want.dtype);
      expect(np.tolist(got)).toEqual(np.tolist(want));
    });
  }
  it("save without extension, in memory and fortran order", () => {
    np.save(f("noext"), np.arange(3));
    expect(fs.existsSync(f("noext.npy"))).toBe(true);
    const bytes = np.save(null, np.arange(4)) as Uint8Array;
    expect(np.tolist(np.load(bytes) as any)).toEqual([0, 1, 2, 3]);
    py(`np.save(${JSON.stringify(f("fortran.npy"))}, np.asfortranarray(np.arange(6).reshape(2, 3)))`);
    expect(np.tolist(np.load(f("fortran.npy")) as any)).toEqual([[0, 1, 2], [3, 4, 5]]);
  });
  it("npz: savez / savez_compressed / NpzFile", () => {
    np.savez(f("z.npz"), np.arange(3), np.ones(2), { named: np.zeros(2) });
    np.savez_compressed(f("zc.npz"), np.arange(1000), { other: np.linspace(0, 1, 5) });
    expect(py(`z = np.load(${JSON.stringify(f("z.npz"))})\nprint(sorted(z.files), z['arr_0'].tolist(), z['named'].tolist())`)).toBe("['arr_0', 'arr_1', 'named'] [0, 1, 2] [0.0, 0.0]");
    expect(py(`z = np.load(${JSON.stringify(f("zc.npz"))})\nprint(sorted(z.files), int(z['arr_0'].sum()), z['other'].tolist())`)).toBe("['arr_0', 'other'] 499500 [0.0, 0.25, 0.5, 0.75, 1.0]");
    py(`np.savez(${JSON.stringify(f("p.npz"))}, a=np.arange(4), b=np.eye(2))\nnp.savez_compressed(${JSON.stringify(f("pc.npz"))}, c=np.arange(2000.0))`);
    const z = np.load(f("p.npz")) as any;
    expect([...z.files].sort()).toEqual(["a", "b"]);
    expect(np.tolist(z.get("a"))).toEqual([0, 1, 2, 3]);
    expect(np.tolist(z.f.b)).toEqual([[1, 0], [0, 1]]);
    expect(np.sum((np.load(f("pc.npz")) as any).get("c"))).toBe(1999000);
    expect(() => z.get("missing")).toThrow();
  });
  it("tofile / fromfile / frombuffer / tobytes", () => {
    np.array([0, 1, 2, 3, 4], 'float64').tofile(f("raw.bin"));
    expect(py(`print(np.fromfile(${JSON.stringify(f("raw.bin"))}).tolist())`)).toBe("[0.0, 1.0, 2.0, 3.0, 4.0]");
    py(`np.arange(6, dtype=np.int32).tofile(${JSON.stringify(f("raw32.bin"))})`);
    expect(np.tolist(np.fromfile(f("raw32.bin"), 'int32'))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(np.tolist(np.fromfile(f("raw32.bin"), 'int32', 2, '', 8))).toEqual([2, 3]);
    np.array([1.5, 2.5]).tofile(f("t.txt"), ',', '%.2f');
    expect(fs.readFileSync(f("t.txt"), "utf8")).toBe("1.50,2.50");
    expect(np.tolist(np.fromfile(f("t.txt"), 'float64', -1, ','))).toEqual([1.5, 2.5]);
    const bytes = np.array([1, 2, 3], 'int16').tobytes();
    expect(Array.from(bytes)).toEqual([1, 0, 2, 0, 3, 0]);
    expect(np.tolist(np.frombuffer(bytes, 'int16'))).toEqual([1, 2, 3]);
    expect(np.tolist(np.frombuffer(bytes, 'int16', 2, 2))).toEqual([2, 3]);
    expect(np.tolist(np.frombuffer(new Float64Array([1.5, 2.5]).buffer))).toEqual([1.5, 2.5]);
    expect(() => np.frombuffer(bytes, 'float64')).toThrow();
  });
  it("memmap", () => {
    py(`np.arange(10, dtype=np.int32).tofile(${JSON.stringify(f("mm.bin"))})`);
    const m = np.memmap(f("mm.bin"), 'int32', 'r+', 0, [2, 5]);
    expect(np.tolist(m)).toEqual([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]);
    m.assign(100, 0, 0);
    (m as any).flush();
    expect(py(`print(np.fromfile(${JSON.stringify(f("mm.bin"))}, dtype=np.int32)[:3].tolist())`)).toBe("[100, 1, 2]");
    const w = np.memmap(f("mm2.bin"), 'float64', 'w+', 0, 3);
    w.assign(np.array([1.5, 2.5, 3.5]));
    (w as any).flush();
    expect(py(`print(np.fromfile(${JSON.stringify(f("mm2.bin"))}).tolist())`)).toBe("[1.5, 2.5, 3.5]");
    const om = (np.lib.format as any).open_memmap(f("om.npy"), 'w+', 'float32', [2, 2]);
    om.assign(7); om.flush();
    expect(py(`print(np.load(${JSON.stringify(f("om.npy"))}).tolist())`)).toBe("[[7.0, 7.0], [7.0, 7.0]]");
  });
});

describe.skipIf(!HAS_NUMPY)("text files", () => {
  const text = "# comment\n1 2 3\n4 5 6\n7 8 9 # trailing\n";
  it("loadtxt", () => {
    fs.writeFileSync(f("a.txt"), text);
    for (const [js, pyc] of [
      [() => np.loadtxt(f("a.txt")), "np.loadtxt(F)"],
      [() => np.loadtxt(f("a.txt"), 'int32'), "np.loadtxt(F, dtype=np.int32)"],
      [() => np.loadtxt(f("a.txt"), { usecols: [0, 2] }), "np.loadtxt(F, usecols=(0, 2))"],
      [() => np.loadtxt(f("a.txt"), { skiprows: 1 }), "np.loadtxt(F, skiprows=1)"],
      [() => np.loadtxt(f("a.txt"), { max_rows: 2 }), "np.loadtxt(F, max_rows=2)"],
      [() => np.loadtxt(f("a.txt"), { usecols: 1 }), "np.loadtxt(F, usecols=1)"],
      [() => np.loadtxt(f("a.txt"), { ndmin: 2, usecols: 1 }), "np.loadtxt(F, usecols=1, ndmin=2)"],
      [() => np.loadtxt(f("a.txt"), { converters: { 0: (s: string) => parseFloat(s) * 10 } }), "np.loadtxt(F, converters={0: lambda s: float(s) * 10})"],
      [() => np.array((np.loadtxt as any)(f("a.txt"), { unpack: true }).map((r: any) => np.tolist(r))), "np.array(np.loadtxt(F, unpack=True))"],
    ] as [() => any, string][]) {
      const want = JSON.parse(py(`import json\nF = ${JSON.stringify(f("a.txt"))}\nprint(json.dumps(${pyc}.tolist()))`));
      expect(np.tolist(js())).toEqual(want);
    }
    fs.writeFileSync(f("c.csv"), "a,b\n1.5,2\n3,4e2\n");
    expect(np.tolist(np.loadtxt(f("c.csv"), { delimiter: ",", skiprows: 1 }))).toEqual([[1.5, 2], [3, 400]]);
    expect(() => np.loadtxt(["1 2", "3"])).toThrow();
    expect(np.tolist(np.loadtxt(["1 2", "3 4"]))).toEqual([[1, 2], [3, 4]]);
    expect(np.tolist(np.loadtxt("5\n6\n7"))).toEqual([5, 6, 7]);
  });
  it("genfromtxt", () => {
    fs.writeFileSync(f("g.csv"), "1,2,3\n4,,6\n7,8,nan\n");
    const want = JSON.parse(py(`import json\nprint(json.dumps(np.genfromtxt(${JSON.stringify(f("g.csv"))}, delimiter=',').tolist()).replace('NaN', 'null'))`));
    const got = np.tolist((np as any).genfromtxt(f("g.csv"), { delimiter: "," }));
    expect(got.map((r: number[]) => r.map(v => Number.isNaN(v) ? null : v))).toEqual(want);
    expect(np.tolist(np.genfromtxt(f("g.csv"), { delimiter: ",", filling_values: -1 }))[1]).toEqual([4, -1, 6]);
    expect(np.tolist(np.genfromtxt(f("g.csv"), { delimiter: ",", skip_header: 1, usecols: [0, 2] }))).toEqual([[4, 6], [7, NaN]].map(r => r));
  });
  it("savetxt", () => {
    const X = np.array([[1.5, 2], [3, 4]]);
    for (const [args, pyargs] of [
      [[], ""], [["%d"], ", fmt='%d'"], [["%.3f", ","], ", fmt='%.3f', delimiter=','"], [["%.2e", " ", "\n", "head"], ", fmt='%.2e', header='head'"],
      [[["%d", "%.1f"]], ", fmt=['%d', '%.1f']"], [["%5.1f|%5.1f"], ", fmt='%5.1f|%5.1f'"], [["%g", ";", "\n", "", "end", "% "], ", fmt='%g', delimiter=';', footer='end', comments='% '"],
    ] as [any[], string][]) {
      const got = (np.savetxt as any)(null, X, ...args);
      const want = py(`import io, json\nb = io.StringIO()\nnp.savetxt(b, np.array([[1.5, 2], [3, 4]])${pyargs})\nprint(json.dumps(b.getvalue()))`);
      expect(JSON.stringify(got)).toBe(want);
    }
    np.savetxt(f("s.txt"), np.array([1, 2, 3]), '%d');
    expect(fs.readFileSync(f("s.txt"), "utf8")).toBe("1\n2\n3\n");
    expect(py(`print(np.loadtxt(${JSON.stringify(f("s.txt"))}).tolist())`)).toBe("[1.0, 2.0, 3.0]");
  });
  it("fromstring / fromregex", () => {
    expect(np.tolist(np.fromstring("1 2 3", 'float64', -1, ' '))).toEqual([1, 2, 3]);
    expect(np.tolist(np.fromstring("1,2,3,4", 'int32', 3, ','))).toEqual([1, 2, 3]);
    expect(() => np.fromstring("1 2")).toThrow();
    expect(np.tolist(np.fromregex("a1 b22 c3", /(\d)/, 'int32'))).toEqual([[1], [2], [2], [3]]);
  });
});

numpy_cases("index tricks", [
  [() => np.r_(1, 2, np.array([3, 4])), "np.r_[1, 2, np.array([3, 4])]"], [() => np.r_('0:5'), "np.r_[0:5]"], [() => np.r_('0:1:5j'), "np.r_[0:1:5j]"], [() => np.r_('1:10:3'), "np.r_[1:10:3]"], [() => np.r_(np.array([1, 2]), 0, 0, np.array([3])), "np.r_[np.array([1, 2]), 0, 0, np.array([3])]"],
  [() => np.r_('0,2', [1, 2], [3, 4]), "np.r_['0,2', [1, 2], [3, 4]]"], [() => np.r_('1', np.array([[1, 2]]), np.array([[3, 4]])), "np.r_['1', np.array([[1, 2]]), np.array([[3, 4]])]"], [() => np.r_('-1,2,0', [1, 2], [3, 4]), "np.r_['-1,2,0', [1, 2], [3, 4]]"],
  [() => np.r_(np.array([1.5]), '0:3'), "np.r_[np.array([1.5]), 0:3]"], [() => np.c_(np.array([1, 2]), np.array([3, 4])), "np.c_[np.array([1, 2]), np.array([3, 4])]"], [() => np.c_(np.array([[1, 2], [3, 4]]), np.array([5, 6])), "np.c_[np.array([[1, 2], [3, 4]]), np.array([5, 6])]"],
  [() => np.c_(np.array([1, 2, 3]), np.array([[4, 5], [6, 7], [8, 9]])), "np.c_[np.array([1, 2, 3]), np.array([[4, 5], [6, 7], [8, 9]])]"], [() => np.c_(1, 2), "np.c_[1, 2]"],
  "np.r_[1, 2, np.array([3, 4])]", "np.r_[0:5]", "np.r_[0:1:5j]", "np.c_[np.array([1, 2]), np.array([3, 4])]", "np.r_[np.array([1, 2]), 0, 0, np.array([3])]",
  [() => np.s_.index(1, 2), "(1, 2)"], [() => np.index_exp.index('1:3'), "('1:3',)"],
  "np.mgrid[0:3, 0:2]", "np.ogrid[0:3, 0:2]", "np.mgrid[0:1:5j]",
]);

numpy_cases("less common constructors", [
  "np.asmatrix([[1, 2], [3, 4]])", "np.asmatrix([1, 2, 3])", "np.asmatrix(np.array([[1, 2]]))", [() => np.bmat('1 2; 3 4'), "np.array([[1, 2], [3, 4]])"], [() => np.bmat([[np.eye(2), np.zeros([2, 1])], [np.ones([1, 2]), np.array([[5.0]])]]), "np.bmat([[np.eye(2), np.zeros((2, 1))], [np.ones((1, 2)), np.array([[5.0]])]])"],
  "np.ascontiguousarray(5)", "np.ascontiguousarray([[1, 2], [3, 4]])", "np.ascontiguousarray([[1, 2], [3, 4]], dtype='float64')", "np.asfortranarray([[1, 2], [3, 4]])", "np.asarray_chkfinite([1.0, 2.0])", "!np.asarray_chkfinite([1.0, np.nan])", "!np.asarray_chkfinite([np.inf])",
  "np.fromiter([1, 2, 3], dtype='float64')", "np.fromiter([0, 1, 2, 3, 4], dtype='int32', count=3)", "!np.fromiter([1, 2], dtype='int32', count=3)",
  [() => np.fromfunction((i: any, j: any) => np.add(i, j), [2, 3]), "np.fromfunction(lambda i, j: i + j, (2, 3))"], [() => np.fromfunction((i: any, j: any) => np.multiply(i, 10 as any).add(j), [2, 2], 'int32'), "np.fromfunction(lambda i, j: i * 10 + j, (2, 2), dtype=int)"],
  [() => np.fromfunction((i: any) => np.square(i), [4]), "np.fromfunction(lambda i: i ** 2, (4,))"],
  "np.fromstring('1 2 3', dtype='int32', sep=' ')", "np.fromstring('1.5,2.5', sep=',')",
  [() => { try { np.from_dlpack(np.arange(3)); return false; } catch (e) { return true; } }, "True"],
  [() => np.rec.format_parser('i4, f8', 'a, b').dtype.names, "np.rec.format_parser('i4, f8', 'a, b', None).dtype.names"],
]);
