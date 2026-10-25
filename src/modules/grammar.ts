//@ts-check
import * as ohm from 'ohm-js';
import { isarray, asarray, NDArray } from '../NDArray';
import { new_buffer, float64, complex128 } from '../dtypes';
import { Complex } from '../complex';
import * as jsInterface from '../array/js-interface';
import * as operators from '../array/operators';
import * as elementwise from '../array/elementwise';

import { np as np } from '../np_circular_import';

export const grammar = String.raw`
ArrayGrammar {
  Instruction
  = Variable "[" Slice "]" AssignSymbol ArithmeticLogicExp -- sliceAssignment
  | ArithmeticLogicExp                       -- expression
  | Constant
  
  Variable
   = "#" digit+ "#"
  
  AssignSymbol
  ="="|"+="|"-="|"*="|"/="|"%="|"&="|"|="|"^="|"@="|"**="|"<<="|">>="|"//="|"or="|"and="|"xor="|"max="|"min="
  
  /* Declaration in precedence order (weakest first) */
  ArithmeticLogicExp = Precedence11

  /* https://docs.python.org/3/reference/expressions.html */
  Operator11 = "<=" | ">=" | "!=" | "==" | "<" | ">"
  Operator10 = "|"
  Operator09 = "^"
  Operator08 = "&"
  Operator07 = "<<" | ">>"
  Operator06 = "+" | "-"
  Operator05 = "*" | "@" | "//" | "/" | "%"
  Operator04 = "~" /* Unary */
  Operator03 = "+" | "-" /* Unary. Special treatment to prevent "-1.3" to be "-(array of 1.3)" */
  Operator02 = "**"
  /* Operator01 = "x[index]" | "x[index:index]" | "x(arguments...)" | "x.attribute" */
  /* Operator00 = "(expressions...)" */

  Precedence11 = Precedence11 Operator11 Precedence10 | "" "" Precedence10
  Precedence10 = Precedence10 Operator10 Precedence09 | "" "" Precedence09
  Precedence09 = Precedence09 Operator09 Precedence08 | "" "" Precedence08
  Precedence08 = Precedence08 Operator08 Precedence07 | "" "" Precedence07
  Precedence07 = Precedence07 Operator07 Precedence06 | "" "" Precedence06
  Precedence06 = Precedence06 Operator06 Precedence05 | "" "" Precedence05
  Precedence05 = Precedence05 Operator05 Precedence04 | "" "" Precedence04
  Precedence04 = ""           Operator04 Precedence03 | "" "" Precedence03 /* Unary */
  Precedence03 = ""           Operator03 Precedence02 | "" "" Precedence02 /* Special */
  Precedence02 = Precedence02 Operator02 Precedence03 | "" "" Precedence01
  Precedence01 = Arr
  
  Parenthesis = "(" ArithmeticLogicExp ")"
  Arr
    = Arr "." Name CallArgs     -- method
    | Arr "." Name              -- attribute
    | Arr "[" Slice "]"         -- slice
    | Arr CallArgs              -- invoke
    | Parenthesis
    | Name ("." Name)* CallArgs -- call
    | Name ("." Name)+ -- path
    | number
    | Variable
    | Constant -- constant

  Name  (an identifier)
    = (letter|"_") (letter|"_"|digit)*

  number  (a number)
    = ("+"|"-")? numberBody exponent? ("j"|"J")?
  numberBody
    = digit+ "." digit* -- a
    | "." digit+        -- b
    | digit+            -- c
  exponent = ("e"|"E") ("+"|"-")? digit+

  int (an integer) = ""  digit+ | "-" digit+ | "+" digit+

  CallArgs // Using empty strings instead of separate rules
   = "(" Args ","  KwArgs ","? ")"
   | "(" Args ","? ""      ""  ")"
   | "(" ""   ","? KwArgs ","? ")"
   | "(" ""    ""  ""      ""  ")"
   
  Args = NonemptyListOf<ArgValue, ",">
  KwArgs = NonemptyListOf<KwArg, ",">
  KwArg = Name "=" ArgValue

  ArgValue = Constant | PyType | String | ArithmeticLogicExp | JsArray
  PyType = ("int" | "float" | "bool" | "complex" | "str" | "object") ~(alnum | "_")
  Constant = constantWord
  constantWord = (numericConstant | "True" | "False" | "None" | "np.newaxis") ~(alnum | "_")
  numericConstant
    = ("+"|"-")? "np.nan"
    | ("+"|"-")? "np.inf"
    | ("+"|"-")? "np.euler_gamma"
    | ("+"|"-")? "np.pi"
    | ("+"|"-")? "np.e"

  JsArray
    = "[" ListOf<ArgValue, ","> ","? "]"
    | "(" ListOf<ArgValue, ","> ","? ")"

  String = stringLiteral
  stringLiteral = "\'" sqChar* "\'" | "\"" dqChar* "\""
  escChar = "\\" any
  sqPlain = ~"\'" any
  dqPlain = ~"\"" any
  sqChar = escChar | sqPlain
  dqChar = escChar | dqPlain
   
  Slice = NonemptyListOf<SliceTerm, ",">
  SliceTerm
    = SliceRange
    | (":" | "..." | "None") -- constant
    | ("True" | "False") -- boolean_constant
    | ArithmeticLogicExp
    | JsArray
  
  SliceRange
    = SliceBound ":" SliceBound ":" SliceBound
    | SliceBound ":" SliceBound ""  ""
    | SliceBound ":" ""  ":" SliceBound
    | SliceBound ":" ""  ""  ""
    | ""  ":" SliceBound ":" SliceBound
    | ""  ":" SliceBound ""  ""
    | ""  ":" ""  ":" SliceBound
    | ""  ":" ""  ""  ""
  SliceBound = ArithmeticLogicExp
}
`;

export const ohmGrammar = ohm.grammar(grammar);

/** Python string escapes */
function unescape_string(s: string): string {
  return s.replace(/\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)/g, (_m, c: string) => {
    switch (c[0]) {
      case 'n': return '\n'; case 't': return '\t'; case 'r': return '\r'; case '0': return '\0'; case 'b': return '\b'; case 'f': return '\f'; case 'v': return '\v'; case 'a': return '\x07';
      case 'x': case 'u': return c.length > 1 ? String.fromCharCode(parseInt(c.slice(1), 16)) : c;
      default: return c;
    }
  });
}

const FLOAT_LITERAL = /^[+-]?(\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d*\.\d+[eE][+-]?\d+)$/;

/** 0-d arrays that appear as arguments of calls (e.g. from unary minus) are passed as plain JS values */
function unwrap_scalar(v: any) {
  // Integer-valued float literals (2.0) stay weak float arrays: they would become int32 otherwise
  if (isarray(v) && v.shape.length == 0 && !(v._weak && v.dtype === float64 && Number.isInteger(v.item()))) return v.item();
  return v;
}

function weak_complex(im: number) {
  const out = new NDArray(new_buffer([new Complex(0, im)], complex128), [], complex128);
  out._weak = true;
  return out;
}

function weak_float(value: number) {
  const out = new NDArray(new_buffer([value], float64), [], float64);
  out._weak = true;
  return out;
}


export const __makeSemantics = () => {

  const semanticVariables: any[] = [];
  const semantics = {
    Instruction_sliceAssignment($tgt, _open, $where, _close, $symbol, $src) {
      const tgt = $tgt.parse();
      const src = $src.parse();
      const symbol = $symbol.sourceString;
      const where = $where.parse();
      const assign = operators.op_assign[symbol];
      if (!assign) throw new Error(`Unsupported assignment operator ${symbol}`);
      if (isarray(tgt)) assign(tgt, asarray(src), where);
      else {
        // Plain JS arrays are converted, modified and written back
        const arr = asarray(tgt);
        assign(arr, asarray(src), where);
        const result = jsInterface.tolist(arr);
        tgt.length = 0;
        tgt.push(...result);
      }
      return null;
    },
    Instruction_expression($arr) {
      let arr = $arr.parse();
      if (typeof arr === "number") return arr;
      if (typeof arr === "boolean") return arr;
      if (Array.isArray(arr)) return arr;
      if (isarray(arr)) arr = arr.item_if_scalar();
      return arr;
    },
    Precedence11: BinaryOperation,
    Precedence10: BinaryOperation,
    Precedence09: BinaryOperation,
    Precedence08: BinaryOperation,
    Precedence07: BinaryOperation,
    Precedence06: BinaryOperation,
    Precedence05: BinaryOperation,
    Precedence04: UnaryOperation,
    Precedence03: UnaryOperation,
    Precedence02: BinaryOperation,
    number: function (_sign, _body, _exp, _imag) {
      const source = this.sourceString;
      const value = parseFloat(source);
      if (/[jJ]$/.test(source)) return weak_complex(value);
      // Float literals (1.0, 2., 1e3) are python floats: weak float64 scalars, so that np.array([1]) * 2.0 is float
      if (/[.eE]/.test(source)) return weak_float(value);
      return value;
    },
    Arr_slice($arr, _open, $where, _close) {
      const arr = $arr.parse();
      const where = $where.parse();
      if (Array.isArray(arr)) {
        // Plain JS lists/tuples (e.g. the result of np.linalg.svd) can be indexed with a single integer
        if (where.length != 1) throw new Error(`Lists can only be indexed with one index`);
        const i = unwrap_scalar(where[0]);
        if (typeof i !== 'number') throw new Error(`Lists can only be indexed with integers`);
        return arr[i < 0 ? arr.length + i : i];
      }
      return arr.index(...where);
    },
    SliceTerm_constant($x) {
      return $x.sourceString;
    },
    SliceTerm_boolean_constant($x) {
      return $x.sourceString === "True";
    },
    Arr_call($name, $names, _, $callArgs) {
      let name = $name.sourceString + $names.sourceString;
      if (name.slice(0, 3) == "np.") name = name.slice(3);
      const path = name.split('.');
      if (path.some(part => part === 'constructor' || part === '__proto__' || part === 'prototype')) throw new Error(`Unrecognized function ${name}`);
      const func = path.reduce((obj, part) => (obj === undefined || obj === null) ? undefined : obj[part], np);
      if (typeof func !== 'function') throw new Error(`Unrecognized function ${name}`);
      const { args, kwArgs } = $callArgs.parse();
      const all = Object.keys(kwArgs).length == 0 ? args : [...args, kwArgs];
      // classes (np.poly1d, np.polynomial.Polynomial, ...) are called like in python, without `new`
      if (/^class[\s{]/.test(Function.prototype.toString.call(func))) return new (func as any)(...all);
      return func(...all);
      // return func.bind(kwArgs)(...args); // Old version
    },
    Arr_invoke($arr, $callArgs) {
      // f(x)(y): calling the result (e.g. a poly1d object)
      const fn = $arr.parse();
      const { args, kwArgs } = $callArgs.parse();
      const all = Object.keys(kwArgs).length == 0 ? args : [...args, kwArgs];
      if (typeof fn === 'function') return fn(...all);
      if (fn && typeof fn.call === 'function') return fn.call(...all);
      throw new Error(`object is not callable`);
    },
    Arr_method($arr, _dot, $name, $callArgs) {
      let arr = $arr.parse();
      let name = $name.sourceString;
      const { args, kwArgs } = $callArgs.parse();
      if (arr[name] === undefined) throw new Error(`Unrecognized method ${name}`);
      // return arr.withKwArgs(kwArgs)[name](...args); // Old version
      return Object.keys(kwArgs).length == 0 ? arr[name](...args) : arr[name](...args, kwArgs);
    },
    Arr_constant($c) { return $c.parse(); },
    Arr_path($name, _dots, $names) {
      const path = ($name.sourceString + $names.sourceString.replace(/\s+/g, '')).split('.');
      if (path.some(part => part === 'constructor' || part === '__proto__' || part === 'prototype')) throw new Error(`Unrecognized name ${path.join('.')}`);
      if (path[0] === 'np') path.shift();
      const value = path.reduce((obj: any, part) => (obj === undefined || obj === null) ? undefined : obj[part], np);
      if (value === undefined) throw new Error(`Unrecognized name ${path.join('.')}`);
      return value;
    },
    Parenthesis(_, $arr, __) { return $arr.parse(); },
    Arr_attribute($arr, _, $name) { return $arr.parse()[$name.sourceString]; },
    Variable(_, $i, __) {
      const i = parseInt($i.sourceString);
      let value = semanticVariables[i];
      const isListOfArrays = Array.isArray(value) && value.length && isarray(value[0]);
      if (Array.isArray(value) && !isListOfArrays) value = jsInterface.array(value);
      return value;
    },
    int($sign, $value) {
      const value = parseInt($value.sourceString);
      if ($sign.sourceString == '-') return -value;
      else return value
    },
    SliceRange($start, _, $stop, sep, $step) {
      // Bounds can be any integer expression (numbers, parentheses, ${variables}); rebuild "start:stop:step"
      const bound = (n: any) => {
        const src = n.sourceString.trim();
        if (src === '') return '';
        if (/^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?[jJ]?$/.test(src)) return src; // plain literal (also floats and complex steps for r_ and mgrid)
        let v = n.parse();
        if (isarray(v)) v = v.item();
        if (typeof v !== 'number' || !Number.isInteger(v)) throw new Error(`Slice indices must be integers. Found ${n.sourceString}`);
        return String(v);
      };
      return bound($start) + ':' + bound($stop) + (sep.sourceString.trim() === '' ? '' : ':' + bound($step));
    },
    SliceBound($e) { return $e.parse(); },
    PyType($t) { return $t.sourceString; }, // python builtin types used as dtypes: astype(int)
    Constant($x) {
      switch ($x.sourceString) {
        case "True": return true;
        case "False": return false;
        case "None": return null;
        case "np.nan": return Number.NaN;
        case "+np.nan": return Number.NaN;
        case "-np.nan": return Number.NaN;
        case "np.inf": return Number.POSITIVE_INFINITY;
        case "+np.inf": return Number.POSITIVE_INFINITY;
        case "-np.inf": return Number.NEGATIVE_INFINITY;
        case "np.e": return Math.E;
        case "+np.e": return Math.E;
        case "-np.e": return -Math.E;
        case "np.pi": return Math.PI;
        case "+np.pi": return Math.PI;
        case "-np.pi": return -Math.PI;
        case "np.euler_gamma": return 0.5772156649015329; // Euler's constant
        case "+np.euler_gamma": return 0.5772156649015329; // Euler's constant
        case "-np.euler_gamma": return -0.5772156649015329; // Euler's constant
        case "np.newaxis": return null;
      }
      throw new Error(`Unrecognized constant ${$x.sourceString}`)
    },
    String($lit) {
      return unescape_string($lit.sourceString.slice(1, -1));
    },
    CallArgs(_open, $args, _comma, $kwArgs, _trailing, _close) {
      const args = ($args.parse() || []).map(unwrap_scalar);
      let entries = $kwArgs.parse() || [];
      let kwArgs = Object.fromEntries(entries.map(([k, v]) => {
        // The following is needed because minus integer gets parsed as array.
        return [k, unwrap_scalar(v)];
      }));
      return { args, kwArgs };
    },
    KwArg($key, _equals, $value) {
      const key = $key.sourceString;
      const value = $value.parse();
      return [key, value];
    },
    EmptyListOf() { return []; },
    NonemptyListOf(first, _, more) {
      return [first, ...more.children].map(c => c.parse());
    },
    JsArray(_open, $list, _trailing, _close) {
      const list = $list.parse();
      // Remember whether a float literal appears (so that [1., 2.] is a float array, as in python)
      const children = $list.asIteration().children;
      const isFloat = children.some((c, i) => FLOAT_LITERAL.test(c.sourceString.trim()) || (Array.isArray(list[i]) && (list[i] as any).__float));
      // Downcast arrays (needed because, e.g., for [-1, 3, -2], -1 and -2 are interpreted as MyArray rather than int)
      const { tolist } = jsInterface;
      // Only scalars (e.g. -1 is parsed as an array) are downcast. Real arrays stay arrays
      for (let i in list) if (isarray(list[i]) && list[i].shape.length == 0) list[i] = tolist(list[i]);
      if (isFloat) (list as any).__float = true;
      return list;
    },
    _terminal() { return null; },
  };

  function BinaryOperation($A, $symbol, $B) {
    const A = $A.parse();
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "" && A === null) return B;
    const { op_binary: op } = operators;
    // objects that implement the operator as a method (np.poly1d, np.polynomial.Polynomial, ...)
    const methods: { [s: string]: string } = { '+': 'add', '-': 'sub', '*': 'mul', '**': 'pow', '//': 'floordiv', '%': 'mod' };
    const isObj = (v: any) => v !== null && typeof v === 'object' && !isarray(v) && !Array.isArray(v);
    const name = methods[symbol];
    if (name && isObj(A) && typeof A[name] === 'function') return A[name](B);
    if (name && isObj(B) && typeof B[name] === 'function' && (symbol === '+' || symbol === '*')) return B[name](A);
    return op[symbol](A, B);
  }

  function UnaryOperation(_, $symbol, $B) {
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "") return B;
    const { ops } = elementwise;
    const keepWeak = (res: NDArray) => { if ((typeof B === 'number' || (isarray(B) && B._weak)) && isarray(res)) res._weak = true; return res; }; // python scalars stay weak
    switch (symbol) {
      case "+": return keepWeak(ops["+"](B));
      case "-": return keepWeak(ops["-"](B));
      case "~": return ops["~"](B);
      case "!": return ops["not"](B);
    }
    throw new Error(`Programming Error: ${symbol}`);
  }

  const ohmSemantics = ohmGrammar.createSemantics();
  ohmSemantics.addOperation('parse', semantics);
  /**
   * @param {TemplateStringsArray} template
   * @param {any[]} variables
   */
  function parse(template, ...variables) {
    // Replace variables in template with #0# #1# #2#, ...
    let idx = 0;
    const template_with_placeholders = template.join('###').replace(/###/g, () => `#${idx++}#`);
    semanticVariables.length = 0;
    semanticVariables.push(...variables);
    const match = ohmGrammar.match(template_with_placeholders);
    if (!match.succeeded()) throw new Error(match["message"]);
    return ohmSemantics(match).parse();
  }

  return { parse, ohmSemantics, semantics, semanticVariables, busy: 0 };
}


export const __parser_pool = [__makeSemantics()];

/**
 * @param {TemplateStringsArray} template
 * @param {any[]} variables
 */
export const parse = function (template, ...variables) {
  // Thread control, because the parser depends on semanticVariables,
  // but we don't want to waste CPU time recreating the parser on each call
  // No cleaning is done (we assume that the number of threads is negligible compared to the memory size)
  const pool = __parser_pool;
  for (let i = 0; i < pool.length; i++) {
    const parser = pool[i];
    if (parser.busy++ == 0) {
      try { return parser.parse(template, ...variables); }
      finally { parser.busy = 0; }
    }
    if (i == pool.length) pool.push(__makeSemantics());
  }
}
