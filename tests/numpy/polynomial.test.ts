import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const bases: [string, string, boolean][] = [
  ["polynomial", "poly", false], ["chebyshev", "cheb", true], ["hermite", "herm", true], ["hermite_e", "herme", true], ["laguerre", "lag", true], ["legendre", "leg", true],
];
const C1 = "[1.0, 2.0, 3.0]", C2 = "[0.5, -1.0, 2.0, 0.25]", X = "np.array([-0.7, 0.1, 0.5, 1.3])";
const C2D = "np.array([[1.0, 2.0, 0.5], [0.25, -1.0, 3.0]])", C3D = "np.arange(1.0, 9.0).reshape(2, 2, 2) / 4";

for (const [mod, p, hasGauss] of bases) {
  const f = (name: string) => `np.polynomial.${mod}.${p}${name}`;
  numpy_cases(`polynomial.${mod}`, [
    `np.polynomial.${mod}.${p}domain`, `np.polynomial.${mod}.${p}zero`, `np.polynomial.${mod}.${p}one`, `np.polynomial.${mod}.${p}x`,
    `${f("line")}(1.5, 2.0)`, `${f("line")}(1.5, 0)`,
    `${f("add")}(${C1}, ${C2})`, `${f("sub")}(${C1}, ${C2})`, `${f("sub")}(${C1}, ${C1})`, `${f("mulx")}(${C1})`, `${f("mulx")}([0.0])`, `${f("mul")}(${C1}, ${C2})`, `${f("mul")}([2.0], ${C2})`, `${f("mul")}(${C1}, [0.0, 1.0])`,
    `${f("div")}(${C2}, ${C1})`, `${f("div")}(${C1}, ${C2})`, `${f("div")}(${C2}, [2.0])`, `${f("div")}([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0])`, `!${f("div")}(${C1}, [0.0])`,
    `${f("pow")}(${C1}, 3)`, `${f("pow")}(${C1}, 0)`, `${f("pow")}([1.0, 2.0], 2)`, ...(p == "poly" ? [`${f("pow")}(${C1}, 20)`] : [`!${f("pow")}(${C1}, 20)`, `${f("pow")}(${C1}, 20, maxpower=25)`]), `!${f("pow")}(${C1}, -1)`,
    `${f("val")}(${X}, ${C2})`, `${f("val")}(0.3, ${C1})`, `${f("val")}(${X}, [4.0])`, `${f("val")}(np.array([[0.1, 0.2], [0.3, 0.4]]), ${C2})`, `${f("val")}(${X}, np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.0]]))`,
    `${f("val")}(${X}[:2], np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.0]]), tensor=False)`, `${f("val")}(np.array([0.5+0.5j, 0.1j]), ${C2})`, `${f("val")}(${X}, np.zeros(1))`,
    `${f("val2d")}(np.array([0.1, 0.5]), np.array([-0.3, 0.7]), ${C2D})`, `${f("val3d")}(np.array([0.1, 0.5]), np.array([-0.3, 0.7]), np.array([0.2, 0.9]), ${C3D})`,
    `${f("grid2d")}(np.array([0.1, 0.5, 0.2]), np.array([-0.3, 0.7]), ${C2D})`, `${f("grid3d")}(np.array([0.1, 0.5]), np.array([-0.3, 0.7, 0.2]), np.array([0.2, 0.9]), ${C3D})`,
    `${f("der")}(${C2})`, `${f("der")}(${C2}, 2)`, `${f("der")}(${C2}, 2, 3.0)`, `${f("der")}(${C2}, 0)`, `${f("der")}(${C2}, 10)`, `${f("der")}([5.0])`, `${f("der")}(${C2}, m=3)`,
    `${f("int")}(${C2})`, `${f("int")}(${C2}, 2)`, `${f("int")}(${C2}, 1, [3.0])`, `${f("int")}(${C2}, 2, [1.0, 2.0], 0.5)`, `${f("int")}(${C2}, 1, [], 0.5, 2.0)`, `${f("int")}([0.0], 2, [1.0, 1.0])`, `${f("int")}(${C2}, 0)`, `!${f("int")}(${C2}, 1, [1.0, 2.0])`,
    `${f("int")}(${f("der")}(${C2}))`, `${f("der")}(${f("int")}(${C2}, 2), 2)`,
    `${f("fromroots")}([-1.0, 0.5, 2.0])`, `${f("fromroots")}([])`, `${f("fromroots")}([1.5])`, `${f("fromroots")}([2.0, 2.0, -1.0, 0.25])`,
    `${f("roots")}(${C1})`, `${f("roots")}([1.0, -3.0, 2.0])`, `${f("roots")}(${f("fromroots")}([-1.0, 0.5, 2.0]))`, `${f("roots")}([3.0])`, `${f("roots")}([1.0, 2.0])`, `${f("roots")}([1.0, 0.0, 1.0])`.replace(/.*/, `${f("roots")}([1.0, 0.0, 1.0, 0.5])`),
    `${f("companion")}([1.0, 2.0, 3.0])`, `${f("companion")}([1.0, 2.0, 3.0, 4.0])`, `${f("companion")}(${C2})`, `${f("companion")}([1.0, 2.0])`, `!${f("companion")}([1.0])`,
    `${f("vander")}(np.array([-0.5, 0.2, 0.9]), 4)`, `${f("vander")}(np.array([[0.1, 0.2], [0.3, 0.4]]), 2)`, `${f("vander")}(np.array([0.5]), 0)`, `!${f("vander")}(np.array([0.5]), -1)`,
    `${f("vander2d")}(np.array([0.1, 0.5]), np.array([-0.3, 0.7]), [1, 2])`, `${f("vander3d")}(np.array([0.1, 0.5]), np.array([-0.3, 0.7]), np.array([0.2, 0.9]), [1, 1, 2])`,
    `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2)`, `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 4)`,
    `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2, w=np.array([1.0, 2.0, 3.0, 2.0, 1.0]))`, `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), [0, 2])`,
    `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([[1.0, 2.0], [0.3, 0.1], [0.1, 0.2], [0.4, 0.9], [1.2, 0.3]]), 2)`,
    `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2, full=True)[1][1]`, `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2, full=True)[1][0]`, `${f("fit")}(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2, full=True)[1][2]`,
    `!${f("fit")}(np.array([1.0, 2.0]), np.array([1.0]), 1)`, `!${f("fit")}(np.array([1.0, 2.0]), np.array([1.0, 2.0]), -1)`,
    `${f("trim")}([1.0, 0.0, 2.0, 0.0, 0.0])`, `${f("trim")}([1.0, 0.0, 2.0, 1e-6, 0.0], tol=1e-5)`, `${f("trim")}([0.0, 0.0])`, `!${f("trim")}([1.0], tol=-1)`,
    ...(p == "poly" ? [] : [`${f("weight")}(np.array([-0.5, 0.1, 0.9]))`]),
    ...(hasGauss ? [`${f("2poly")}(${C2})`, `np.polynomial.${mod}.poly2${p}([1.0, -2.0, 0.5, 3.0])`, `np.polynomial.${mod}.poly2${p}(${f("2poly")}(${C2}))`, `${f("2poly")}([1.0])`,
      ...[1, 2, 3, 5, 8].flatMap(d => [`${f("gauss")}(${d})`]), `!${f("gauss")}(0)`] : []),
  ], { rtol: 1e-8, atol: 1e-10 });
}

numpy_cases("chebyshev extras", [
  "np.polynomial.chebyshev.chebpts1(5)", "np.polynomial.chebyshev.chebpts1(1)", "np.polynomial.chebyshev.chebpts2(5)", "np.polynomial.chebyshev.chebpts2(2)", "!np.polynomial.chebyshev.chebpts2(1)", "!np.polynomial.chebyshev.chebpts1(0)",
  [() => np.polynomial.chebyshev.chebinterpolate((x: any) => np.exp(x), 6), "np.polynomial.chebyshev.chebinterpolate(np.exp, 6)"],
  [() => np.polynomial.chebyshev.chebinterpolate((x: any, a: number) => np.multiply(np.sin(x), a), 5, [2.0]), "np.polynomial.chebyshev.chebinterpolate(lambda x, a: np.sin(x) * a, 5, (2.0,))"],
], { rtol: 1e-8, atol: 1e-10 });

numpy_cases("polyutils", [
  "np.polynomial.polyutils.trimseq(np.array([1.0, 0.0, 2.0, 0.0, 0.0]))", "np.polynomial.polyutils.trimseq(np.array([0.0, 0.0]))", "np.polynomial.polyutils.trimcoef(np.array([1.0, 0.0, 2.0, 1e-9]), 1e-6)", "np.polynomial.polyutils.getdomain(np.array([0.5, 3.0, -1.0, 2.0]))",
  "np.polynomial.polyutils.mapparms(np.array([0.0, 4.0]), np.array([-1.0, 1.0]))", "np.polynomial.polyutils.mapdomain(np.array([0.0, 2.0, 4.0]), np.array([0.0, 4.0]), np.array([-1.0, 1.0]))",
  "np.polynomial.polyutils.as_series([[1, 2, 0, 0], [3]])", "np.polynomial.polyutils.as_series([[1, 2, 0, 0], [3]], trim=False)", "!np.polynomial.polyutils.as_series([[]])",
]);

const mk = (cls: string, args: string) => `np.polynomial.${cls}(${args})`;
const classes: [string, string, string][] = [["polynomial", "Polynomial", "poly"], ["chebyshev", "Chebyshev", "cheb"], ["hermite", "Hermite", "herm"], ["hermite_e", "HermiteE", "herme"], ["laguerre", "Laguerre", "lag"], ["legendre", "Legendre", "leg"]];
for (const [mod, cls, p] of classes) {
  const Cls = (np.polynomial as any)[cls];
  const py = `np.polynomial.${mod}.${cls}`;
  numpy_cases(`class ${cls}`, [
    [() => Cls([1, 2, 3])(np.array([0.1, 0.5])), `${py}([1, 2, 3])(np.array([0.1, 0.5]))`],
    [() => Cls([1, 2, 3], [0, 4], [-1, 1]).call(np.array([0.5, 3.0])), `${py}([1, 2, 3], [0, 4], [-1, 1])(np.array([0.5, 3.0]))`],
    [() => Cls([1, 2, 3]).add(Cls([0.5, 1])).coef, `(${py}([1, 2, 3]) + ${py}([0.5, 1])).coef`],
    [() => Cls([1, 2, 3]).sub(Cls([0.5, 1])).coef, `(${py}([1, 2, 3]) - ${py}([0.5, 1])).coef`],
    [() => Cls([1, 2, 3]).mul(Cls([0.5, 1])).coef, `(${py}([1, 2, 3]) * ${py}([0.5, 1])).coef`],
    [() => Cls([1, 2, 3]).mul(2).coef, `(${py}([1, 2, 3]) * 2).coef`],
    [() => Cls([1, 2, 3, 4]).floordiv(Cls([1, 1])).coef, `(${py}([1, 2, 3, 4]) // ${py}([1, 1])).coef`],
    [() => Cls([1, 2, 3, 4]).mod(Cls([1, 1])).coef, `(${py}([1, 2, 3, 4]) % ${py}([1, 1])).coef`],
    [() => Cls([1, 2, 3]).div(2).coef, `(${py}([1, 2, 3]) / 2).coef`],
    [() => Cls([1, 2, 3]).pow(2).coef, `(${py}([1, 2, 3]) ** 2).coef`],
    [() => Cls([1, 2, 3]).neg().coef, `(-${py}([1, 2, 3])).coef`],
    [() => Cls([1, 2, 3, 0, 0]).trim().coef, `${py}([1, 2, 3, 0, 0]).trim().coef`],
    [() => Cls([1, 2, 3, 4]).truncate(2).coef, `${py}([1, 2, 3, 4]).truncate(2).coef`],
    [() => Cls([1, 2, 3, 4]).cutdeg(1).coef, `${py}([1, 2, 3, 4]).cutdeg(1).coef`],
    [() => Cls([1, 2, 3, 4]).degree(), `${py}([1, 2, 3, 4]).degree()`],
    [() => Cls([1, 2, 3, 4]).deriv().coef, `${py}([1, 2, 3, 4]).deriv().coef`],
    [() => Cls([1, 2, 3, 4], [0, 2], [-1, 1]).deriv(2).coef, `${py}([1, 2, 3, 4], [0, 2], [-1, 1]).deriv(2).coef`],
    [() => Cls([1, 2, 3, 4]).integ().coef, `${py}([1, 2, 3, 4]).integ().coef`],
    [() => Cls([1, 2, 3, 4], [0, 2], [-1, 1]).integ(1, 2.0).coef, `${py}([1, 2, 3, 4], [0, 2], [-1, 1]).integ(1, 2.0).coef`],
    [() => Cls([1, 2, 3, 4], [0, 2], [-1, 1]).integ(2, [1, 2], 0.5).coef, `${py}([1, 2, 3, 4], [0, 2], [-1, 1]).integ(2, [1, 2], 0.5).coef`],
    [() => Cls([2, -3, 1]).roots(), `${py}([2, -3, 1]).roots()`],
    [() => Cls([2, -3, 1], [0, 4], [-1, 1]).roots(), `${py}([2, -3, 1], [0, 4], [-1, 1]).roots()`],
    [() => Cls.fromroots([-1, 0.5, 2]).coef, `${py}.fromroots([-1, 0.5, 2]).coef`],
    [() => Cls.fromroots([-1, 0.5, 2], [-3, 3]).coef, `${py}.fromroots([-1, 0.5, 2], [-3, 3]).coef`],
    [() => Cls.fit(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2).coef, `${py}.fit(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([1.0, 0.3, 0.1, 0.4, 1.2]), 2).coef`],
    [() => Cls.fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0]), 2).domain, `${py}.fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0]), 2).domain`],
    [() => Cls.fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0]), 2).coef, `${py}.fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0]), 2).coef`],
    [() => Cls.fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0]), 2, [0, 5]).coef, `${py}.fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0]), 2, [0, 5]).coef`],
    [() => Cls.identity().coef, `${py}.identity().coef`],
    [() => Cls.identity([0, 3]).coef, `${py}.identity([0, 3]).coef`],
    [() => Cls.basis(3).coef, `${py}.basis(3).coef`],
    [() => Cls([1, 2, 3], [0, 4], [-1, 1]).mapparms(), `${py}([1, 2, 3], [0, 4], [-1, 1]).mapparms()`],
    [() => Cls([1, 2, 3]).linspace(5)[1], `${py}([1, 2, 3]).linspace(5)[1]`],
    [() => Cls([1, 2, 3]).linspace(4, [0, 1])[0], `${py}([1, 2, 3]).linspace(4, [0, 1])[0]`],
    [() => Cls([1, 2, 3]).equals(Cls([1, 2, 3])), `${py}([1, 2, 3]) == ${py}([1, 2, 3])`],
    [() => Cls([1, 2, 3]).equals(Cls([1, 2, 4])), `${py}([1, 2, 3]) == ${py}([1, 2, 4])`],
    ...["Polynomial", "Chebyshev", "Legendre", "Laguerre", "Hermite", "HermiteE"].map((k): [() => any, string] => [
      () => Cls([1, 2, 3, 0.5], [0, 2], [-1, 1]).convert(null, (np.polynomial as any)[k]).coef, `${py}([1, 2, 3, 0.5], [0, 2], [-1, 1]).convert(kind=np.polynomial.${k}).coef`]),
    [() => Cls([1, 2, 3]).convert([0, 1], Cls, [-1, 1]).coef, `${py}([1, 2, 3]).convert([0, 1], np.polynomial.${cls}, [-1, 1]).coef`],
    [() => Cls([1, 2, 3], [0, 2], [-1, 1]).convert([0, 5]).call(np.array([1.0, 2.5])), `${py}([1, 2, 3], [0, 2], [-1, 1]).convert([0, 5])(np.array([1.0, 2.5]))`],
    [() => { try { Cls([1, 2]).add(Cls([1, 2], [0, 2])); return false; } catch (e) { return true; } }, "True"],
  ], { rtol: 1e-8, atol: 1e-9, check_dtype: false });
}
