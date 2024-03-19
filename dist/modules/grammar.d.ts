import * as ohm from 'ohm-js';
export declare const grammar: string;
export declare const ohmGrammar: ohm.Grammar;
export declare const __makeSemantics: () => {
    parse: (template: any, ...variables: any[]) => any;
    ohmSemantics: ohm.Semantics;
    semantics: {
        Instruction_sliceAssignment($tgt: any, _open: any, $where: any, _close: any, $symbol: any, $src: any): any;
        Instruction_expression($arr: any): any;
        Precedence11: ($A: any, $symbol: any, $B: any) => any;
        Precedence10: ($A: any, $symbol: any, $B: any) => any;
        Precedence09: ($A: any, $symbol: any, $B: any) => any;
        Precedence08: ($A: any, $symbol: any, $B: any) => any;
        Precedence07: ($A: any, $symbol: any, $B: any) => any;
        Precedence06: ($A: any, $symbol: any, $B: any) => any;
        Precedence05: ($A: any, $symbol: any, $B: any) => any;
        Precedence04: (_: any, $symbol: any, $B: any) => any;
        Precedence03: (_: any, $symbol: any, $B: any) => any;
        Precedence02: ($A: any, $symbol: any, $B: any) => any;
        number: (arg1: any, arg2: any, arg3: any, arg4: any, arg5: any, arg6: any, arg7: any) => number;
        Arr_slice($arr: any, _open: any, $where: any, _close: any): any;
        SliceTerm_constant($x: any): any;
        Arr_call($name: any, $names: any, _: any, $callArgs: any): any;
        Arr_method($arr: any, _dot: any, $name: any, $callArgs: any): any;
        Parenthesis(_: any, $arr: any, __: any): any;
        Arr_attribute($arr: any, _: any, $name: any): any;
        Variable(_: any, $i: any, __: any): any;
        int($sign: any, $value: any): number;
        SliceRange($start: any, _: any, $stop: any, __: any, $step: any): any;
        Constant($x: any): number | boolean;
        String(_open: any, $str: any, _close: any): any;
        CallArgs(_open: any, $args: any, _comma: any, $kwArgs: any, _trailing: any, _close: any): {
            args: any;
            kwArgs: {
                [k: string]: any;
            };
        };
        KwArg($key: any, _equals: any, $value: any): any[];
        NonemptyListOf(first: any, _: any, more: any): any[];
        JsArray(_open: any, $list: any, _trailing: any, _close: any): any;
        _terminal(): any;
    };
    semanticVariables: any[];
    busy: number;
};
export declare const __parser_pool: {
    parse: (template: any, ...variables: any[]) => any;
    ohmSemantics: ohm.Semantics;
    semantics: {
        Instruction_sliceAssignment($tgt: any, _open: any, $where: any, _close: any, $symbol: any, $src: any): any;
        Instruction_expression($arr: any): any;
        Precedence11: ($A: any, $symbol: any, $B: any) => any;
        Precedence10: ($A: any, $symbol: any, $B: any) => any;
        Precedence09: ($A: any, $symbol: any, $B: any) => any;
        Precedence08: ($A: any, $symbol: any, $B: any) => any;
        Precedence07: ($A: any, $symbol: any, $B: any) => any;
        Precedence06: ($A: any, $symbol: any, $B: any) => any;
        Precedence05: ($A: any, $symbol: any, $B: any) => any;
        Precedence04: (_: any, $symbol: any, $B: any) => any;
        Precedence03: (_: any, $symbol: any, $B: any) => any;
        Precedence02: ($A: any, $symbol: any, $B: any) => any;
        number: (arg1: any, arg2: any, arg3: any, arg4: any, arg5: any, arg6: any, arg7: any) => number;
        Arr_slice($arr: any, _open: any, $where: any, _close: any): any;
        SliceTerm_constant($x: any): any;
        Arr_call($name: any, $names: any, _: any, $callArgs: any): any;
        Arr_method($arr: any, _dot: any, $name: any, $callArgs: any): any;
        Parenthesis(_: any, $arr: any, __: any): any;
        Arr_attribute($arr: any, _: any, $name: any): any;
        Variable(_: any, $i: any, __: any): any;
        int($sign: any, $value: any): number;
        SliceRange($start: any, _: any, $stop: any, __: any, $step: any): any;
        Constant($x: any): number | boolean;
        String(_open: any, $str: any, _close: any): any;
        CallArgs(_open: any, $args: any, _comma: any, $kwArgs: any, _trailing: any, _close: any): {
            args: any;
            kwArgs: {
                [k: string]: any;
            };
        };
        KwArg($key: any, _equals: any, $value: any): any[];
        NonemptyListOf(first: any, _: any, more: any): any[];
        JsArray(_open: any, $list: any, _trailing: any, _close: any): any;
        _terminal(): any;
    };
    semanticVariables: any[];
    busy: number;
}[];
/**
 * @param {TemplateStringsArray} template
 * @param {any[]} variables
 */
export declare const parse: (template: any, ...variables: any[]) => any;
//# sourceMappingURL=grammar.d.ts.map