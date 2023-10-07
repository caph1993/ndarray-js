//@ts-check
/** @typedef {NumberConstructor|BooleanConstructor} DType */
/** @typedef {NDArray|number|boolean} ArrayOrConstant */

import { GLOBALS } from './globals';
const { np } = GLOBALS;


class NDArray {
  _flat: number[];
  shape: number[];
  dtype: any;
  modules: {
    casting: typeof casting; basic: typeof basic; indexes: typeof
    indexes;
    //elementwise: typeof elementwise; print: typeof print; reduce: typeof reduce; operators: typeof operators; transform: typeof transform; jsInterface: typeof jsInterface;
  };

  constructor(flat: number[], shape: number[], dtype: any = Number) {
    this.shape = shape; // invariant: immutable
    this._flat = flat;
    this.dtype = dtype;
    this._simpleIndexes = null;
  }
  // /** @type {import("./core-indexes").AxesIndex|null} */ _simpleIndexes: import("./core-indexes").AxesIndex | null;

  _simpleIndexes: any;

  get size() {
    return this._simpleIndexes == null ? this._flat.length : this._simpleIndexes.size;
  }
  get flat() {
    if (this._simpleIndexes == null) return this._flat;
    const indices = this._simpleIndexes.indices;
    return indices.map(i => this._flat[i]);
  }
  set flat(list) {
    if (list.length != this.size)
      throw new Error(`Length mismatch. Can't write ${list.length} values into ${this.size} available positions.`);
    const n = this.size;
    if (this._simpleIndexes == null) {
      for (let i = 0; i < n; i++) this._flat[i] = list[i];
    } else {
      const indices = this._simpleIndexes.indices;
      for (let i = 0; i < n; i++) this._flat[indices[i]] = list[i];
    }
  }
  get T() {
    throw new Error(`NOT IMPLEMENTED`)
    //return np.transpose(this);
  }
  __popKwArgs() {
    let out = this['__kwArgs'];
    if (out === undefined) return {};
    delete this['__kwArgs'];
    return out;
  }
  /** @param {Object<string, any>} kwArgs */
  withKwArgs(kwArgs: { [s: string]: any; }) {
    this['__kwArgs'] = kwArgs;
    return this;
  }
  *[Symbol.iterator]() {
    throw new Error(`NOT IMPLEMENTED`)
    // for (let i = 0; i < this.shape[0]; i++) yield this.index(i);
  }
  get length() {
    return this.shape[0] || 0;
  }
  copy: () => NDArray;
}

GLOBALS.NDArray = NDArray;

import * as casting from './casting';
import * as basic from './core-basic';
import * as indexes from './core-indexes';
// import * as elementwise from './core-elementwise';
// import * as print from './core-print';
// import * as reduce from './core-reduce';
// import * as operators from './core-operators';
// import * as transform from './core-transform';
// import * as jsInterface from './core-js-interface';

const modules = {
  casting,
  basic,
  indexes,
  // elementwise,
  // print,
  // reduce,
  // operators,
  // transform,
  // jsInterface,
};

//@ts-ignore
NDArray.prototype.modules = modules; // NEEDED before loading the modules!





// // ==============================
// //    Basic methods
// // ==============================

// const basic = modules.basic;

// NDArray.prototype.reshape = function (shape, ...more_shape) {
//   return basic.reshape(this, shape, ...more_shape);
// };

// NDArray.prototype.ravel = function () {
//   return basic.ravel(this);
// };

// NDArray.prototype.copy = function () {
//   return basic.copy(this);
// };


// // ==============================
// //    Indexing
// // ==============================

// /**
//  * @param {import("./core-indexes").GeneralIndexSpec[]} where
//  */
// NDArray.prototype.index = function (...where: import("./core-indexes").GeneralIndexSpec[]) {
//   return modules.indexes.index(this, where);
// }

// // ==============================
// //    Printing
// // ==============================

// NDArray.prototype.toString = function () {
//   return modules.print.humanReadable(this);
// }


// // ==============================
// //    Reduce
// // ==============================






// function reduceDecorator(func) {
//   /** @param {import("./core-reduce").AxisArg} axis  @param {boolean} keepdims */
//   return function (axis: import("./core-reduce").AxisArg = null, keepdims: boolean = false) {
//     ({ axis, keepdims } = Object.assign({ axis, keepdims }, this.__popKwArgs()));
//     return func(this, axis, keepdims);
//   }
// }

// NDArray.prototype.sum = reduceDecorator(modules.reduce.reducers.sum);
// NDArray.prototype.product = reduceDecorator(modules.reduce.reducers.product);
// NDArray.prototype.any = reduceDecorator(modules.reduce.reducers.any);
// NDArray.prototype.all = reduceDecorator(modules.reduce.reducers.all);
// NDArray.prototype.max = reduceDecorator(modules.reduce.reducers.max);
// NDArray.prototype.min = reduceDecorator(modules.reduce.reducers.min);
// NDArray.prototype.argmax = reduceDecorator(modules.reduce.reducers.argmax);
// NDArray.prototype.argmin = reduceDecorator(modules.reduce.reducers.argmin);
// NDArray.prototype.mean = reduceDecorator(modules.reduce.reducers.mean);
// NDArray.prototype.var = reduceDecorator(modules.reduce.reducers.var);
// NDArray.prototype.std = reduceDecorator(modules.reduce.reducers.std);



// // ==============================
// //       Operators: Binary operations, assignment operations and unary boolean_not
// // ==============================

// NDArray.prototype._binary_operation = modules.operators.binary_operation;


// /**
//  * @param {import("./core-operators").BinaryOperator} func
//  * @returns {import("./core-operators").SelfBinaryOperator}
//  */
// function binaryOpDecorator(func: import("./core-operators").BinaryOperator): import("./core-operators").SelfBinaryOperator {
//   return function (other, out = null) {
//     ({ out } = Object.assign({ out }, this.__popKwArgs()));
//     return func(this, other, out);
//   }
// }

// NDArray.prototype.add = binaryOpDecorator(modules.operators.op_binary["+"]);
// NDArray.prototype.subtract = binaryOpDecorator(modules.operators.op_binary["-"]);
// NDArray.prototype.multiply = binaryOpDecorator(modules.operators.op_binary["*"]);
// NDArray.prototype.divide = binaryOpDecorator(modules.operators.op_binary["/"]);
// NDArray.prototype.mod = binaryOpDecorator(modules.operators.op_binary["%"]);
// NDArray.prototype.divide_int = binaryOpDecorator(modules.operators.op_binary["//"]);
// NDArray.prototype.pow = binaryOpDecorator(modules.operators.op_binary["**"]);

// NDArray.prototype.maximum = binaryOpDecorator(modules.operators.op_binary["↑"]);
// NDArray.prototype.minimum = binaryOpDecorator(modules.operators.op_binary["↓"]);

// NDArray.prototype.bitwise_or = binaryOpDecorator(modules.operators.op_binary["|"]);
// NDArray.prototype.bitwise_and = binaryOpDecorator(modules.operators.op_binary["&"]);
// NDArray.prototype.bitwise_xor = binaryOpDecorator(modules.operators.op_binary["^"]);
// NDArray.prototype.bitwise_shift_left = binaryOpDecorator(modules.operators.op_binary["<<"]);
// NDArray.prototype.bitwise_shift_right = binaryOpDecorator(modules.operators.op_binary[">>"]);

// NDArray.prototype.logical_or = binaryOpDecorator(modules.operators.op_binary["or"]);
// NDArray.prototype.logical_and = binaryOpDecorator(modules.operators.op_binary["and"]);
// NDArray.prototype.logical_xor = binaryOpDecorator(modules.operators.op_binary["xor"]);

// NDArray.prototype.greater = binaryOpDecorator(modules.operators.op_binary[">"]);
// NDArray.prototype.less = binaryOpDecorator(modules.operators.op_binary["<"]);
// NDArray.prototype.greater_equal = binaryOpDecorator(modules.operators.op_binary[">="]);
// NDArray.prototype.less_equal = binaryOpDecorator(modules.operators.op_binary["<="]);
// NDArray.prototype.equal = binaryOpDecorator(modules.operators.op_binary["=="]);
// NDArray.prototype.not_equal = binaryOpDecorator(modules.operators.op_binary["!="]);



// /**
//  * @param {import("./core-operators").UnaryOperator} func
//  * @returns {import("./core-operators").SelfUnaryOperator}
//  */
// function unaryOpDecorator(func: import("./core-operators").UnaryOperator): import("./core-operators").SelfUnaryOperator {
//   return function (out = null) {
//     ({ out } = Object.assign({ out }, this.__popKwArgs()));
//     return func(this, out);
//   }
// }
// // Unary operations: only boolean_not. Positive is useless and negative is almost useless
// NDArray.prototype.bitwise_not = unaryOpDecorator(modules.operators.op_unary["~"]);
// NDArray.prototype.logical_not = unaryOpDecorator(modules.operators.op_unary["not"]);


// NDArray.prototype.isclose = modules.operators.isclose;
// NDArray.prototype.allclose = modules.operators.allclose;


// /**
//  * @param {import("./core-operators").AssignmentOperator} func
//  * @returns {import("./core-operators").SelfAssignmentOperator}
//  */
// function assignOpDecorator(func: import("./core-operators").AssignmentOperator): import("./core-operators").SelfAssignmentOperator {
//   //@ts-ignore
//   return function (...args) { return func(this, ...args); }
// }
// NDArray.prototype.assign = assignOpDecorator(modules.operators.op_assign["="]);
// NDArray.prototype.add_assign = assignOpDecorator(modules.operators.op_assign["+="]);
// NDArray.prototype.subtract_assign = assignOpDecorator(modules.operators.op_assign["-="]);
// NDArray.prototype.multiply_assign = assignOpDecorator(modules.operators.op_assign["*="]);
// NDArray.prototype.divide_assign = assignOpDecorator(modules.operators.op_assign["/="]);
// NDArray.prototype.mod_assign = assignOpDecorator(modules.operators.op_assign["%="]);
// NDArray.prototype.divide_int_assign = assignOpDecorator(modules.operators.op_assign["//="]);
// NDArray.prototype.pow_assign = assignOpDecorator(modules.operators.op_assign["**="]);

// NDArray.prototype.maximum_assign = assignOpDecorator(modules.operators.op_assign["↑="]);
// NDArray.prototype.minimum_assign = assignOpDecorator(modules.operators.op_assign["↓="]);

// NDArray.prototype.bitwise_or_assign = assignOpDecorator(modules.operators.op_assign["|="]);
// NDArray.prototype.bitwise_and_assign = assignOpDecorator(modules.operators.op_assign["&="]);
// NDArray.prototype.bitwise_shift_left_assign = assignOpDecorator(modules.operators.op_assign["<<="]);
// NDArray.prototype.bitwise_shift_right_assign = assignOpDecorator(modules.operators.op_assign[">>="]);

// NDArray.prototype.logical_or_assign = assignOpDecorator(modules.operators.op_assign["or="]);
// NDArray.prototype.logical_and_assign = assignOpDecorator(modules.operators.op_assign["and="]);



// // ==============================
// //    array instantiation and reshaping
// // ==============================

// NDArray.prototype.JS = function () {
//   return modules.jsInterface.toJS(this);
// }
// NDArray.prototype.fromJS = function (A) {
//   return modules.jsInterface.fromJS(A);
// }

// // ==============================
// //    elementwise methods
// // ==============================

// NDArray.prototype.round = function (decimals = 0) {
//   ({ decimals } = Object.assign({ decimals }, this.__popKwArgs()));
//   return modules.elementwise.round(this, decimals);
// };

// // ==============================
// //    transform methods
// // ==============================

// /** @param {null|number[]} axes */
// NDArray.prototype.transpose = function (axes: null | number[] = null) {
//   ({ axes } = Object.assign({ axes }, this.__popKwArgs()));
//   return modules.transform.transpose(this, axes);
// };

// NDArray.prototype.sort = function (axis = -1) {
//   ({ axis } = Object.assign({ axis }, this.__popKwArgs()));
//   modules.transform.sort(this, axis);
//   return null;
// };


// //=============================





// // /** @typedef {'+'|'-'|'*'} BinaryOperatorSymbol */
// // /** @typedef {'~'|'!'} UnaryOperatorSymbol */
// // /** @typedef {'='|'+='|'*='} AssignmentOperatorSymbol */

// // /**
// //  * @callback BinaryOp
// //  * @param  {BinaryOperatorSymbol} symbol
// //  * @param  {NDArray} other
// //  * @param  {NDArray?} out
// //  * @returns {NDArray}
// //  */
// // /**
// //  * @callback AssignmentOp
// //  * @param  {AssignmentOperatorSymbol} symbol
// //  * @param  {NDArray} other
// //  * @param  {...import("./core-indexes").GeneralIndexSpec} indexSpec
// //  * @returns {NDArray}
// //  */
// // /**
// //  * @callback UnaryOp
// //  * @param  {UnaryOperatorSymbol} symbol
// //  * @returns {NDArray}
// //  */

// // // /**@type {BinaryOp|AssignmentOp}*/

// // /**
// //  * @overload
// //  * @param  {BinaryOperatorSymbol} symbol
// //  * @param  {NDArray} other
// //  * @param  {NDArray?} out
// //  * @returns {NDArray}
// // */
// // /**
// //  * @overload
// //  * @param  {AssignmentOperatorSymbol} symbol
// //  * @param  {NDArray} other
// //  * @param  {...import("./core-indexes").GeneralIndexSpec} indexSpec
// //  * @returns {NDArray}
// // */
// // /**
// //  * @param  {UnaryOperatorSymbol} symbol
// //  * @returns {NDArray}
// // */
// // NDArray.prototype.op = function (...arguments) {

// //   return this;
// // }




// /** @typedef {"+" | "-" | "*" | "/" | "%" | "//" | "**" | "<" | ">" | ">=" | "<=" | "==" | "!=" | " | " | "&" | "^" | "<<" | ">>" | "or" | "and" | "xor" | "max" | "min"} BinaryOpSymbol */
// /** @typedef {"=" | "+=" | "-=" | "*=" | "/=" | "%=" | "//=" | "**=" | "|=" | "&=" | "^=" | "<<=" | ">>=" | "max=" | "min=" | "or=" | "and="} AssignmentOpSymbol */
// /** @typedef {"~" | "not" | "-"} UnaryOpSymbol */
// /** @typedef {import("./core-indexes").Where} Where */

// /**
//  * @type {{():NDArray; (where:Where):ArrayOrConstant; (where:Where, op:AssignmentOpSymbol, B:ArrayOrConstant):NDArray; (op:AssignmentOpSymbol, B:ArrayOrConstant):NDArray; ( op:BinaryOpSymbol, B:ArrayOrConstant):NDArray; (UnaryOpSymbol):NDArray; }}  */
// NDArray.prototype.op = function (...arguments): NDArray {
//   if (!arguments.length) return this;
//   if (typeof arguments[0] == "string") {
//     const symbol = arguments[0];
//     if (arguments.length == 1) {
//       let func = modules.operators.op_unary[symbol];
//       if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.operators.op_unary)]}`);
//       return func(this, symbol);
//     }
//     if (arguments.length > 2) throw new Error(`Too many arguments provided: ${[...arguments]}`);
//     const other = arguments[1];
//     let func = modules.operators.op_binary[symbol];
//     if (func) return func(this, other);
//     func = modules.operators.op_assign[symbol];
//     if (func) return func(this, other);
//     if (symbol.includes(':')) throw new Error(`Expected index or operator symbol. Found "${symbol}". Did you mean ${[symbol]}?`);
//     throw new Error(`Expected index or operator symbol. Found "${symbol}"`);
//   }
//   const where = arguments[0];
//   if (where instanceof NDArray) throw new Error(`Expected operator or index. Found numpy array`);
//   //@ts-ignore
//   if (arguments.length == 1) return this.index(where);
//   const symbol = arguments[1];
//   let func = modules.operators.op_assign[symbol];
//   if (!func) throw new Error(`Unknown unary operator "${symbol}". Options:${[...Object.keys(modules.operators.op_unary)]}`);
//   if (arguments.length > 2) throw new Error(`Too many arguments provided: ${[...arguments]}`);
//   const other = arguments[2];
//   return func(this, where, other);
// }





export default NDArray;