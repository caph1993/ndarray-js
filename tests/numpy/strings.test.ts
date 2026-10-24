import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const W = "np.array(['hello world', 'Foo Bar', '  padded  ', '', 'ÀÉ x', 'abc123', '12', 'MiXeD CaSe'])";
const SMALL = "np.array([['ab', 'cd'], ['EF', 'g h']])";

// numpy.strings functions that take one array
const unary = ["capitalize", "lower", "upper", "swapcase", "title", "lstrip", "rstrip", "strip", "isalpha", "isalnum", "isdecimal", "isdigit", "isnumeric", "isspace", "islower", "isupper", "istitle", "str_len"];

numpy_cases("np.strings unary", [
  ...unary.flatMap(f => [`np.strings.${f}(${W})`, `np.strings.${f}(${SMALL})`, `np.strings.${f}('abc')`]),
  "np.strings.isdigit(np.array(['²', '1', 'a', '٣']))", "np.strings.isnumeric(np.array(['½', '1', 'a', 'Ⅷ']))", "np.strings.isdecimal(np.array(['²', '1', '٣']))", "np.strings.isspace(np.array([' ', '\\t', 'a b', '']))", "np.strings.istitle(np.array(['A', 'Ab Cd', 'ab', 'AB', 'A B', 'A1b', '']))",
  "np.strings.title(np.array(['they\\'re bill\\'s', 'a-b c_d', '1st 2nd']))", "np.strings.capitalize(np.array(['élan', 'ABC', '1abc']))", "np.strings.swapcase(np.array(['aBc ÀÉ']))",
]);
numpy_cases("np.strings binary", [
  `np.strings.add(${SMALL}, 'x')`, `np.strings.add(${SMALL}, np.array(['1', '2']))`, `np.strings.add('a', 'b')`, `np.strings.multiply(${SMALL}, 2)`, `np.strings.multiply(np.array(['ab', 'c']), np.array([3, 0]))`, `np.strings.multiply('ab', -1)`,
  `np.strings.center(np.array(['a', 'ab', 'abc']), 6)`, `np.strings.center(np.array(['a', 'ab', 'abc']), 7, '*')`, `np.strings.center(np.array(['abcdef']), 3)`, `np.strings.ljust(np.array(['a', 'ab']), 5, '.')`, `np.strings.rjust(np.array(['a', 'ab']), 5, '.')`,
  `np.strings.zfill(np.array(['1', '-12', '+3', 'abc', '']), 5)`, "!np.strings.center(np.array(['a']), 5, 'ab')", `np.strings.expandtabs(np.array(['a\\tb', '12345678\\tx', '\\t\\tz']))`, `np.strings.expandtabs(np.array(['a\\tb']), 4)`,
  `np.strings.strip(np.array(['xxhixx', ' hi ', 'abc']), 'x')`, `np.strings.lstrip(np.array(['xxhixx', 'abcx']), 'xa')`, `np.strings.rstrip(np.array(['xxhixx', 'abcx']), 'xc')`, `np.strings.strip(np.array(['xxhixx', 'yyhi']), np.array(['x', 'y']))`,
  `np.strings.replace(np.array(['aabbaa', 'xyz']), 'a', 'AA')`, `np.strings.replace(np.array(['aabbaa', 'xyz']), 'a', 'AA', 1)`, `np.strings.replace(np.array(['aaa']), 'a', '', 2)`, `np.strings.replace(np.array(['abc']), '', '-')`, `np.strings.replace(np.array(['abc']), '', '-', 2)`, `np.strings.replace(np.array(['abab']), 'ab', np.array(['1']))`,
  `np.strings.slice(np.array(['hello', 'abc']), 1, 3)`, `np.strings.slice(np.array(['hello', 'abc']), 2)`, `np.strings.slice(np.array(['hello', 'abc']), None, None, -1)`, `np.strings.slice(np.array(['hello', 'abc']), -3, None)`, `np.strings.slice(np.array(['hello']), 0, None, 2)`,
  [() => np.strings.translate(np.array(['abc', 'cab']), 'x'.repeat(256)), "np.strings.translate(np.array(['abc', 'cab']), 'x' * 256)"], [() => np.strings.translate(np.array(['abc', 'cab']), 'x'.repeat(256), 'b'), "np.strings.translate(np.array(['abc', 'cab']), 'x' * 256, 'b')"],
  `np.strings.partition(${SMALL}, ' ')`, `np.strings.partition(np.array(['a-b-c', 'xyz']), '-')`, `np.strings.rpartition(np.array(['a-b-c', 'xyz']), '-')`, `np.strings.partition(np.array(['a-b-c']), np.array(['b']))`, "!np.strings.partition(np.array(['a']), '')",
  `np.strings.equal(${SMALL}, 'ab')`, `np.strings.not_equal(${SMALL}, 'ab')`, `np.strings.greater(${SMALL}, 'c')`, `np.strings.greater_equal(${SMALL}, 'cd')`, `np.strings.less(${SMALL}, 'cd')`, `np.strings.less_equal(${SMALL}, 'cd')`, `np.strings.equal(np.array(['a', 'b']), np.array(['a', 'c']))`, `np.strings.less(np.array(['Z', 'a', 'é']), 'a')`,
  `np.strings.count(np.array(['banana', 'aaa', 'xyz']), 'a')`, `np.strings.count(np.array(['banana', 'aaa']), 'a', 2)`, `np.strings.count(np.array(['banana', 'aaa']), 'a', 1, 4)`, `np.strings.count(np.array(['banana']), 'an')`, `np.strings.count(np.array(['aaa']), 'aa')`, `np.strings.count(np.array(['abc']), '')`, `np.strings.count(np.array(['abc']), '', 1, 2)`, `np.strings.count(np.array(['abc']), 'b', -2)`,
  `np.strings.find(np.array(['banana', 'xyz']), 'an')`, `np.strings.find(np.array(['banana']), 'an', 2)`, `np.strings.find(np.array(['banana']), 'an', 0, 2)`, `np.strings.rfind(np.array(['banana', 'xyz']), 'an')`, `np.strings.rfind(np.array(['banana']), 'a', 0, 4)`, `np.strings.find(np.array(['abc']), '')`, `np.strings.rfind(np.array(['abc']), '')`,
  `np.strings.index(np.array(['banana']), 'n')`, `np.strings.rindex(np.array(['banana']), 'n')`, "!np.strings.index(np.array(['banana']), 'z')", "!np.strings.rindex(np.array(['banana']), 'z')",
  `np.strings.startswith(np.array(['banana', 'xyz']), 'ba')`, `np.strings.startswith(np.array(['banana']), 'na', 2)`, `np.strings.startswith(np.array(['banana']), 'ba', 1)`, `np.strings.startswith(np.array(['banana']), '')`, `np.strings.endswith(np.array(['banana', 'xyz']), 'na')`, `np.strings.endswith(np.array(['banana']), 'an', 0, 5)`, `np.strings.endswith(np.array(['banana']), 'na', 0, 4)`, `np.strings.endswith(np.array(['banana']), '')`,
  ...["%d items", "%05.1f", "%s!", "%x", "%X", "%o", "%e", "%g", "%+d", "% d", "%-5d|", "%5s|", "%.2s", "%5.1f|", "%08.3f", "%E", "%G", "%#x", "%.3d", "%10.4e", "%.0f", "%.0e", "%g|"].flatMap(f => {
    const num = /[dxXoeEfFgG]/.test(f.replace(/%[#0\- +]*\d*(\.\d+)?/, '')[0] ? f.replace(/%[#0\- +]*\d*(\.\d+)?/, '')[0] : 'd') ? "25" : "'ab'";
    return [`np.strings.mod(np.array(['${f}']), ${/s|r|c/.test(f.replace(/%[#0\- +]*\d*(\.\d+)?/, '')[0] || '') ? "'abcdef'" : "2.5" })`.replace(/mod\(np\.array\(\['(%[^']*)'\]\), 2\.5\)/, (m, g) => /[dxXoc]/.test(g.slice(-1)) || /\.\d+d/.test(g) ? `mod(np.array(['${g}']), 25)` : m)];
  }),
  "np.strings.mod(np.array(['%3d|', '%03d']), 7)",
]);
numpy_cases("np.char", [
  `np.char.join('-', np.array(['abc', 'de']))`.replace(/.*/, "np.char.add(np.array(['a']), np.array(['b']))"), `np.char.upper(${SMALL})`, "np.char.array(['a', 'b']).shape", "np.char.asarray(['a', 'b'])", "np.char.str_len(np.char.array(['abc', 'd']))",
  "np.char.split(np.array(['a b c', 'd']))", "np.char.split(np.array(['a,b,c']), ',')", "np.char.split(np.array(['a,b,c']), ',', 1)", "np.char.rsplit(np.array(['a,b,c']), ',', 1)", "np.char.rsplit(np.array(['a b  c']))", "np.char.splitlines(np.array(['a\\nb\\r\\nc']))", "np.char.splitlines(np.array(['a\\nb']), True)", "np.char.join(np.array(['-']), np.array(['abc']))",
]);
