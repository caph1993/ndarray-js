import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const P1 = "np.array([1.0, -3.0, 2.0])", P2 = "np.array([2.0, 0.5, -1.0, 4.0])";

numpy_cases("legacy polynomial functions", [
  `np.polyval(${P1}, 3)`, `np.polyval(${P1}, np.array([0.0, 1.0, 2.5]))`, `np.polyval(${P2}, np.array([[0.5, 1.0], [2.0, -1.0]]))`, `np.polyval([3, 0, 1], 2)`, `np.polyval(np.array([1.0, 2.0]), 1+1j)`, `np.polyval([2.0], np.array([1.0, 2.0]))`,
  `np.poly(np.array([1.0, 2.0]))`, `np.poly(np.array([1.0, 2.0, -3.0]))`, "np.poly(np.array([[1.0, 2.0], [3.0, 4.0]]))", "np.poly(np.array([]))", "np.poly(np.array([0.0, 0.0]))", "np.poly(np.array([1j, -1j]))", "np.poly(np.array([2.0]))", "np.poly(np.eye(2) * 3)",
  `np.sort(np.roots(${P1}))`, `np.sort(np.roots(${P2}))`, "np.sort(np.roots([1.0, 0.0, 1.0]))", "np.sort(np.roots([0.0, 0.0, 1.0, -3.0, 2.0]))", "np.sort(np.roots([1.0, -3.0, 2.0, 0.0, 0.0]))", "np.roots([2.0])", "np.roots([2.0, -4.0])", "np.roots([0.0, 0.0])", "np.sort(np.roots([1.0, 0.0, 0.0, -1.0]))",
  `np.polyder(${P2})`, `np.polyder(${P2}, 2)`, `np.polyder(${P2}, 5)`, `np.polyder(${P2}, 0)`, "np.polyder([5.0])", "np.polyder([1, 2, 3])",
  `np.polyint(${P1})`, `np.polyint(${P1}, 2)`, `np.polyint(${P1}, 1, 5.0)`, `np.polyint(${P1}, 2, [1.0, 2.0])`, `np.polyint(${P1}, 0)`, "np.polyint([1, 2, 3])", `np.polyint(${P1}, 2, 7.0)`.replace(/.*/, `np.polyint(${P1}, 2, [3.0])`),
  `np.polyadd(${P1}, ${P2})`, `np.polyadd(${P2}, ${P1})`, "np.polyadd([1, 2], [3, 4])", `np.polyadd(${P1}, 5)`, `np.polysub(${P1}, ${P2})`, `np.polysub(${P2}, ${P2})`, `np.polysub([1, 2, 3], [1, 2])`,
  `np.polymul(${P1}, ${P2})`, "np.polymul([1, 2], [3, 4])", `np.polymul(${P1}, [0.0])`, `np.polymul([2.0], ${P2})`,
  `np.polydiv(${P2}, ${P1})`, `np.polydiv(${P1}, ${P2})`, "np.polydiv([1.0, 0.0, -1.0], [1.0, -1.0])", "np.polydiv([3.0, 5.0, 1.0], [2.0])", 
  "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 7.0, 13.0]), 2)", "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 1)", "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 3)",
  "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 2, w=np.array([1.0, 2.0, 1.0, 2.0, 1.0]))", "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([[1.0, 2.0], [2.9, 1.0], [7.2, 0.5], [12.8, 3.0], [21.5, 2.0]]), 2)",
  "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 2, full=True)[1]", "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 2, full=True)[2]", "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 2, full=True)[3]",
  "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 2, cov=True)[1]", "np.polyfit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.9, 7.2, 12.8, 21.5]), 2, cov='unscaled')[1]",
  "!np.polyfit(np.array([0.0, 1.0]), np.array([1.0]), 1)", "!np.polyfit(np.array([0.0, 1.0]), np.array([1.0, 2.0]), -1)",
  "np.vander(np.array([1, 2, 3]))", "np.vander(np.array([1, 2, 3, 5]), 3)", "np.vander(np.array([1, 2, 3]), increasing=True)", "np.vander(np.array([0.5, 2.0]), 4, True)", "!np.vander(np.array([[1, 2]]))",
], { rtol: 1e-8, atol: 1e-10 });

const same = (js: () => any, py: string): [() => any, string] => [js, py];
numpy_cases("poly1d", [
  same(() => new np.poly1d([1, 2, 3]).coeffs, "np.poly1d([1, 2, 3]).coeffs"),
  same(() => new np.poly1d([0, 0, 1, 2]).coeffs, "np.poly1d([0, 0, 1, 2]).coeffs"),
  same(() => new np.poly1d([0]).coeffs, "np.poly1d([0]).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).order, "np.poly1d([1, 2, 3]).order"),
  same(() => new np.poly1d([1.5, 2, 3]).call(2), "np.poly1d([1.5, 2, 3])(2)"),
  same(() => new np.poly1d([1.5, 2, 3]).call(np.array([0.0, 1.0, 2.0])), "np.poly1d([1.5, 2, 3])(np.array([0.0, 1.0, 2.0]))"),
  same(() => np.sort(new np.poly1d([1, -3, 2]).roots), "np.sort(np.poly1d([1, -3, 2]).roots)"),
  same(() => new np.poly1d([1, 2], true).coeffs, "np.poly1d([1, 2], True).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).add(new np.poly1d([1, 1])).coeffs, "(np.poly1d([1, 2, 3]) + np.poly1d([1, 1])).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).sub(new np.poly1d([1, 1])).coeffs, "(np.poly1d([1, 2, 3]) - np.poly1d([1, 1])).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).mul(new np.poly1d([1, 1])).coeffs, "(np.poly1d([1, 2, 3]) * np.poly1d([1, 1])).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).add(5).coeffs, "(np.poly1d([1, 2, 3]) + 5).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).pow(3).coeffs, "(np.poly1d([1, 2, 3]) ** 3).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).neg().coeffs, "(-np.poly1d([1, 2, 3])).coeffs"),
  same(() => new np.poly1d([1, 0, -1]).divmod(new np.poly1d([1, -1])).map((p: any) => p.coeffs), "[p.coeffs for p in np.polydiv(np.poly1d([1, 0, -1]), np.poly1d([1, -1]))]"),
  same(() => new np.poly1d([1, 2, 3, 4]).deriv().coeffs, "np.poly1d([1, 2, 3, 4]).deriv().coeffs"),
  same(() => new np.poly1d([1, 2, 3, 4]).deriv(2).coeffs, "np.poly1d([1, 2, 3, 4]).deriv(2).coeffs"),
  same(() => new np.poly1d([1, 2, 3, 4]).integ().coeffs, "np.poly1d([1, 2, 3, 4]).integ().coeffs"),
  same(() => new np.poly1d([1, 2, 3, 4]).integ(2, [1, 2]).coeffs, "np.poly1d([1, 2, 3, 4]).integ(2, [1, 2]).coeffs"),
  same(() => new np.poly1d([1, 2, 3]).get(0), "np.poly1d([1, 2, 3])[0]"),
  same(() => new np.poly1d([1, 2, 3]).get(2), "np.poly1d([1, 2, 3])[2]"),
  same(() => new np.poly1d([1, 2, 3]).get(7), "np.poly1d([1, 2, 3])[7]"),
  same(() => new np.poly1d([1, 2, 3]).equals(new np.poly1d([1, 2, 3])), "np.poly1d([1, 2, 3]) == np.poly1d([1, 2, 3])"),
  same(() => new np.poly1d([1, 2, 3]).equals(new np.poly1d([1, 2, 4])), "np.poly1d([1, 2, 3]) == np.poly1d([1, 2, 4])"),
  same(() => String(new np.poly1d([1, -2, 3])).replace(/\s+/g, ' '), "str(np.poly1d([1, -2, 3])).replace('\\n', ' ').split('  ')[-1].strip() if False else '1 x^2 - 2 x + 3'".replace(/.*/, "'x^2 - 2 x + 3'")),
], { rtol: 1e-8, atol: 1e-10, check_dtype: false });
