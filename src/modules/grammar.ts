//@ts-check
import { np, nd_modules } from "./_globals";
import * as ohm from 'ohm-js';


export const grammar = String.raw`
ArrayGrammar {
  Instruction
  = Variable "[" Slice "]" AssignSymbol ArithmeticLogicExp -- sliceAssignment
  | ArithmeticLogicExp                       -- expression
  
  Variable
   = "#" digit+ "#"
  
  AssignSymbol
  ="="|"+="|"-="|"*="|"/="|"%="|"&="|"|="|"^="|"@="|"**="|"<<="|">>="|"//="|"or="|"and="|"xor="|"max="|"min="
  
  /* Declaration in precedence order (weakest first) */
  ArithmeticLogicExp = Precedence11

  /* https://docs.python.org/3/reference/expressions.html */
  Operator11 = "<" | "<=" | ">" | ">=" | "!=" | "=="
  Operator10 = "|"
  Operator09 = "^"
  Operator08 = "&"
  Operator07 = "<<" | ">>"
  Operator06 = "+" | "-"
  Operator05 = "*" | "@" | "/" | "//" | "%"
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
    | Parenthesis
    | Name ("." Name)* CallArgs -- call
    | number
    | Variable

  Name  (an identifier)
    = (letter|"_") (letter|"_"|digit)*

  number  (a number)
    = ("+"|"-")? digit* "." digit+ "E" ("+"|"-")? "digit+"
    | ("+"|"-")? digit* "." digit+ "e" ("+"|"-")? "digit+"
    | ("+"|"-")? digit* "." digit+  ""  ""        ""
    | ("+"|"-")? digit+ ""  ""      ""  ""        ""
  
  int (an integer) = ""  digit+ | "-" digit+ | "+" digit+

  CallArgs // Using empty strings instead of separate rules
   = "(" Args ","  KwArgs ","? ")"
   | "(" Args ","? ""      ""  ")"
   | "(" ""   ","? KwArgs ","? ")"
   | "(" ""    ""  ""      ""  ")"
   
  Args = NonemptyListOf<ArgValue, ",">
  KwArgs = NonemptyListOf<KwArg, ",">
  KwArg = Name "=" ArgValue

  ArgValue = Constant | JsArray | ArithmeticLogicExp | String
  Constant = "True" | "False" | "None" | "np.nan" | "np.inf"
  JsArray
    = "[" ListOf<ArgValue, ","> ","? "]"
    | "(" ListOf<ArgValue, ","> ","? ")"

  String = "\'" any* "\'" | "\"" any* "\""
   
  Slice = NonemptyListOf<SliceTerm, ",">
  SliceTerm
    = SliceRange
    | (":" | "..." | "None") -- constant
    | JsArray
    | ArithmeticLogicExp
  
  SliceRange
    = int ":" int ":" int
    | int ":" int ""  ""
    | int ":" ""  ":" int
    | int ":" ""  ""  ""
    | ""  ":" int ":" int
    | ""  ":" int ""  ""
    | ""  ":" ""  ":" int
    | ""  ":" ""  ""  ""
}
`;

export const ohmGrammar = ohm.grammar(grammar);


export const __makeSemantics = () => {

  const semanticVariables: any[] = [];
  const semantics = {
    Instruction_sliceAssignment($tgt, _open, $where, _close, $symbol, $src) {
      const _tgt = $tgt.parse();
      const _src = $src.parse();
      const symbol = $symbol.sourceString;
      const where = $where.parse();
      let tgt = nd_modules.basic.asarray(_tgt);
      nd_modules.operators.op_assign[symbol](_tgt, where, _src);
      if (tgt !== _tgt) {
        // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
        tgt = nd_modules.jsInterface.tolist(tgt);
        while (_tgt.length) _tgt.pop();
        // @ts-ignore
        _tgt.push(..._tgt);
      }
      return null;
    },
    Instruction_expression($arr) {
      let arr = $arr.parse();
      if (typeof arr === "number") return arr;
      if (typeof arr === "boolean") return arr;
      if (Array.isArray(arr)) return arr;
      if (nd_modules.basic.isarray(arr)) arr = nd_modules.basic.number_collapse(arr);
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
    number: function (arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
      return parseFloat(this.sourceString)
    },
    Arr_slice($arr, _open, $where, _close) {
      const arr = $arr.parse();
      const where = $where.parse();
      return arr.index(...where);
    },
    SliceTerm_constant($x) {
      return $x.sourceString;
    },
    Arr_call($name, $names, _, $callArgs) {
      let name = $name.sourceString + $names.sourceString;
      if (name.slice(0, 3) == "np.") name = name.slice(3);
      const func = name.split('.').reduce((obj, name) => obj[name], np);
      if (func === undefined) throw new Error(`Unrecognized function ${name}`)
      const { args, kwArgs } = $callArgs.parse();
      return func.bind(kwArgs)(...args);
    },
    Arr_method($arr, _dot, $name, $callArgs) {
      let arr = $arr.parse();
      let name = $name.sourceString;
      const { args, kwArgs } = $callArgs.parse();
      if (arr[name] === undefined) throw new Error(`Unrecognized method ${name}`);
      return arr.withKwArgs(kwArgs)[name](...args);
    },
    Parenthesis(_, $arr, __) { return $arr.parse(); },
    Arr_attribute($arr, _, $name) { return $arr.parse()[$name.sourceString]; },
    Variable(_, $i, __) {
      const i = parseInt($i.sourceString);
      let value = semanticVariables[i];
      const isListOfArrays = Array.isArray(value) && value.length && nd_modules.basic.isarray(value[0]);
      if (Array.isArray(value) && !isListOfArrays) value = nd_modules.basic.array(value);
      return value;
    },
    int($sign, $value) {
      const value = parseInt($value.sourceString);
      if ($sign.sourceString == '-') return -value;
      else return value
    },
    SliceRange($start, _, $stop, __, $step) {
      return this.sourceString;
    },
    Constant($x) {
      switch ($x.sourceString) {
        case "True": return true;
        case "False": return false;
        case "None": return null;
        case "np.nan": return Number.NaN;
        case "np.inf": return Number.POSITIVE_INFINITY;
      }
      throw new Error(`Unrecognized constant ${$x.sourceString}`)
    },
    String(_open, $str, _close) {
      return $str.sourceString;
    },
    CallArgs(_open, $args, _comma, $kwArgs, _trailing, _close) {
      const args = $args.parse() || [];
      let entries = $kwArgs.parse() || [];
      let kwArgs = Object.fromEntries(entries.map(([k, v]) => {
        // The following is needed because minus integer gets parsed as array.
        if (nd_modules.basic.isarray(v)) v = nd_modules.basic.number_collapse(v);
        return [k, v];
      }));
      return { args, kwArgs };
    },
    KwArg($key, _equals, $value) {
      const key = $key.sourceString;
      const value = $value.parse();
      return [key, value];
    },
    NonemptyListOf(first, _, more) {
      return [first, ...more.children].map(c => c.parse());
    },
    JsArray(_open, $list, _trailing, _close) {
      const list = $list.parse();
      // Downcast arrays (needed because, e.g., for [-1, 3, -2], -1 and -2 are interpreted as MyArray rather than int)
      const { tolist } = nd_modules.jsInterface;
      for (let i in list) if (nd_modules.basic.isarray(list[i])) list[i] = tolist(list[i]);
      return list;
    },
    _terminal() { return null; },
  };

  function BinaryOperation($A, $symbol, $B) {
    const A = $A.parse();
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "" && A === null) return B;
    const { op_binary: op } = nd_modules.operators;
    return op[symbol](A, B);
  }

  function UnaryOperation(_, $symbol, $B) {
    const B = $B.parse();
    const symbol = $symbol.sourceString;
    if (symbol == "") return B;
    const { op_unary } = nd_modules.operators;
    switch (symbol) {
      case "+": return op_unary["+"](B);
      case "-": return op_unary["-"](B);
      case "~": return op_unary["~"](B);
      case "!": return op_unary["not"](B);
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
    if (!match.succeeded()) throw new Error(match.message);
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
