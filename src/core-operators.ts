//@ts-check
import * as indexes from './core-indexes';
import * as elementwise from './core-elementwise';
import { number_collapse, asarray, NDArray } from './casting';
type NDArray = import("./core").default;

type ArrayOrConstant = NDArray | number | boolean;
type Index = indexes.Where;

const shape_shifts = indexes.shape_shifts;

/**
 * 
 * @param {ArrayOrConstant} A 
 * @param {ArrayOrConstant} B 
 * @param {*} func
 * @param {*} dtype
 * @param {NDArray?} out
 * @returns {ArrayOrConstant}
 */
function binary_operation(A: ArrayOrConstant, B: ArrayOrConstant, func: any, dtype: any, out: NDArray | null = null): ArrayOrConstant {
  if (this instanceof NDArray) return func.bind(NDArray.prototype)(this, ...arguments);
  // Find output shape and input broadcast shapes
  A = asarray(A);
  B = asarray(B);
  const [shape, shapeA, shapeB] = _broadcast_shapes(A.shape, B.shape);
  if (out == null) out = NDArray.prototype._new(shape, (_) => undefined, dtype);
  else if (!(out instanceof NDArray)) throw new Error(`Out must be of type ${NDArray}. Got ${typeof out}`);
  // Iterate with broadcasted indices
  const flatOut = [];
  const shiftsA = shape_shifts(shapeA);
  const shiftsB = shape_shifts(shapeB);
  const flatA = A.flat;
  const flatB = B.flat;
  for (let i = 0; i < out.size; i++) {
    let idxA = 0, idxB = 0, idx = i;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      idxA += shiftsA[axis] * (idx % shapeA[axis]);
      idxB += shiftsB[axis] * (idx % shapeB[axis]);
      idx = Math.floor(idx / shape[axis]);
    }
    flatOut.push(func(flatA[idxA], flatB[idxB]));
  };
  out.flat = flatOut;
  return number_collapse(out);
}

function _broadcast_shapes(shapeA, shapeB) {
  const shape = [];
  const maxDim = Math.max(shapeA.length, shapeB.length);
  shapeA = [...Array.from({ length: maxDim - shapeA.length }, () => 1), ...shapeA];
  shapeB = [...Array.from({ length: maxDim - shapeB.length }, () => 1), ...shapeB];
  for (let axis = maxDim - 1; axis >= 0; axis--) {
    const dim1 = shapeA[axis];
    const dim2 = shapeB[axis];
    if (dim1 !== 1 && dim2 !== 1 && dim1 !== dim2)
      throw new Error(`Can not broadcast axis ${axis} with sizes ${dim1} and ${dim2}`);
    shape.unshift(Math.max(dim1, dim2));
  }
  return [shape, shapeA, shapeB];
}

/** @typedef {(A:ArrayOrConstant, B:ArrayOrConstant, out?:NDArray|null)=>NDArray} BinaryOperator */
/** @typedef {(other:ArrayOrConstant, out:NDArray|null)=>NDArray} SelfBinaryOperator */

function __make_operator(dtype, func) {
  function operator(A, B, out = null) {
    if (this instanceof NDArray) return operator.bind(NDArray.prototype)(this, ...arguments);
    return binary_operation(A, B, func, dtype, out);
  };
  return /**@type {BinaryOperator}*/(operator);
}

function __make_operator_special(funcNum, funcBool) {
  /** @param {NDArray?} out */
  function operator(A, B, out: NDArray | null = null) {
    if (this instanceof NDArray) return operator.bind(NDArray.prototype)(this, ...arguments);
    A = asarray(A);
    B = asarray(B);
    let dtype = A.dtype, func;
    if (A.dtype != B.dtype) console.warn(`Warning: operating arrays of different dtypes. Using ${dtype}`);
    if (dtype == Boolean) func = funcBool;
    else func = funcNum;
    return binary_operation(A, B, func, dtype, out);
  };
  return /**@type {BinaryOperator}*/(operator);
}

const op_binary = {
  "+": __make_operator(Number, (a, b) => a + b),
  "-": __make_operator(Number, (a, b) => a - b),
  "*": __make_operator(Number, (a, b) => a * b),
  "/": __make_operator(Number, (a, b) => a / b),
  "%": __make_operator(Number, (a, b) => (a % b)),
  "//": __make_operator(Number, (a, b) => Math.floor(a / b)),
  "**": __make_operator(Number, (a, b) => Math.pow(a, b)),
  "<": __make_operator(Boolean, (a, b) => a < b),
  ">": __make_operator(Boolean, (a, b) => a > b),
  ">=": __make_operator(Boolean, (a, b) => a >= b),
  "<=": __make_operator(Boolean, (a, b) => a <= b),
  "==": __make_operator(Boolean, (a, b) => a == b),
  "!=": __make_operator(Boolean, (a, b) => a != b),
  "|": __make_operator_special((a, b) => a | b, (a, b) => a || b),
  "&": __make_operator_special((a, b) => a & b, (a, b) => a && b),
  "^": __make_operator(Number, (a, b) => a ^ b),
  "<<": __make_operator(Number, (a, b) => a << b),
  ">>": __make_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "or": __make_operator(Boolean, (a, b) => a || b),
  "and": __make_operator(Boolean, (a, b) => a && b),
  "xor": __make_operator((a, b) => !(a) != !(b)),

  "max": __make_operator(Number, (a, b) => Math.max(a, b)),
  "min": __make_operator(Number, (a, b) => Math.min(a, b)),

  // "isclose": ,
}

op_binary["↑"] = op_binary["max"];
op_binary["↓"] = op_binary["min"];
op_binary["≤"] = op_binary["leq"];
op_binary["≥"] = op_binary["geq"];
op_binary["≠"] = op_binary["neq"];


/** @typedef {(A:ArrayOrConstant, out?:NDArray|null)=>NDArray} UnaryOperator */
/** @typedef {(out?:NDArray|null)=>NDArray} SelfUnaryOperator */

const op_unary = {
  // Unary operators:
  "~": elementwise.bitwise_not,
  "not": elementwise.logical_not,
  "+": elementwise.__make_elementwise(x => x),
  "-": elementwise.__make_elementwise(x => -x, Number),
};


/**
 * @param {NDArray} tgt
 * @param {ArrayOrConstant} src
 * @param {Index} where
 * */
function assign_operation(tgt: NDArray, src: ArrayOrConstant, where: Index, func, dtype) {

  if (tgt['__warnAssignment']) {
    console.warn(`Warning: You are assigning on a copy that resulted from an advanced index on a source array.\nIf this is intentional, use yourArray = source.withKwArgs({copy:true}).index(...yourIndex) to make explicit your awareness of the copy operation.\nInstead, if you want to assign to the source array, use source.op('=', other) or source.op(['::3', -1, '...', [5,4]], '*=', other).\n`);
    delete tgt['__warnAssignment'];
  }
  const { _binary_operation } = NDArray.prototype;
  if (!(tgt instanceof NDArray)) return _assign_operation_toJS(/**@type {*}*/(tgt), src, where, func, dtype)
  if (!where) {
    _binary_operation(tgt, src, func, dtype, tgt);
  } else {
    src = asarray(src);
    let { indices } = indexes.AxesIndex.prototype.parse(tgt.shape, where);
    let tmpTgt;
    if (func == null) {
      // Small optimization: unlike "+=", "*=", etc., for "=", we don't need to reed the target
      func = (a, b) => b;
      tmpTgt = NDArray.prototype._new(indices.length, () => undefined, tgt.dtype);
    } else {
      tmpTgt = asarray(indices.map(i => tgt._flat[i]));
    }
    _binary_operation(tmpTgt, ravel(src), func, dtype, tmpTgt);
    for (let i of indices) tgt._flat[i] = tmpTgt._flat[i];
  }
};


/**
 * @param {any[]} tgtJS
 * @param {any} src
 * @param {any} func
 * @param {any} dtype
 * @param {Index} where
 */
function _assign_operation_toJS(tgtJS: any[], src: any, where: Index, func: any, dtype: any) {
  if (!Array.isArray(tgtJS)) throw new Error(`Can not assign to a non-array. Found ${typeof tgtJS}: ${tgtJS}`);
  console.warn('Assignment to JS array is experimental and slow.')
  // Parse the whole array
  const cpy = asarray(tgtJS);
  assign_operation(cpy, src, where, func, dtype);
  // WARNING: Creates a copy. This is terrible for arr[2, 4, 3] = 5
  const outJS = NDArray.prototype.modules.jsInterface.toJS(cpy);
  while (tgtJS.length) tgtJS.pop();
  // @ts-ignore
  tgtJS.push(...outJS);
}


/** @typedef {{(tgt:NDArray, src:ArrayOrConstant):NDArray; (tgt:NDArray, where:Index, src:ArrayOrConstant):NDArray; }} AssignmentOperator */
/** @typedef {{(other:ArrayOrConstant):NDArray; (where:Index, other:ArrayOrConstant):NDArray; }} SelfAssignmentOperator */


/**@returns {AssignmentOperator} */
function __make_assignment_operator(dtype, func): AssignmentOperator {
  function operator(...arguments) {
    if (this instanceof NDArray) return operator.bind(NDArray.prototype)(this, ...arguments);
    if (arguments.length < 2) throw new Error(`Not enough arguments for assignment operator`);
    if (arguments.length > 3) throw new Error(`Too many arguments for assignment operator`);
    let tgt = arguments[0];
    let src = arguments[arguments.length == 3 ? 2 : 1];
    let where = arguments.length == 2 ? arguments[1] : null;
    return assign_operation(tgt, src, where, func, dtype);
  }
  return operator;
}

const op_assign = {
  "=": __make_assignment_operator(Number, (a, b) => b),
  "+=": __make_assignment_operator(Number, (a, b) => a + b),
  "-=": __make_assignment_operator(Number, (a, b) => a - b),
  "*=": __make_assignment_operator(Number, (a, b) => a * b),
  "/=": __make_assignment_operator(Number, (a, b) => a / b),
  "%=": __make_assignment_operator(Number, (a, b) => (a % b)),
  "//=": __make_assignment_operator(Number, (a, b) => Math.floor(a / b)),
  "**=": __make_assignment_operator(Number, (a, b) => Math.pow(a, b)),
  "|=": __make_assignment_operator(Number, (a, b) => a | b),
  "&=": __make_assignment_operator(Number, (a, b) => a & b),
  "^=": __make_assignment_operator(Number, (a, b) => a ^ b),
  "<<=": __make_assignment_operator(Number, (a, b) => a << b),
  ">>=": __make_assignment_operator(Number, (a, b) => a >> b),
  // Operators with custom ascii identifiers:
  "max=": __make_assignment_operator(Number, (a, b) => Math.max(a, b)),
  "min=": __make_assignment_operator(Number, (a, b) => Math.min(a, b)),
  "or=": __make_assignment_operator(Boolean, (a, b) => a || b),
  "and=": __make_assignment_operator(Boolean, (a, b) => a && b),
};


op_assign["↑="] = op_assign["max="];
op_assign["↓="] = op_assign["min="];
// op["≈≈"] = op[MyArray.prototype.isclose,


// ====================================

function isclose(A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  const func = (a, b) => {
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  return binary_operation(A, B, func, Boolean)
}

function allclose(A, B, rtol = 1.e-5, atol = 1.e-8, equal_nan = false) {
  ({ rtol, atol, equal_nan } = Object.assign({ rtol, atol, equal_nan }, this));
  // Equivalent to all(isclose(A, B, rtol, atol, equal_nan)), but shortcutting if false 
  const func = (a, b) => { //copied from isclose
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * b;
    return (a == b) || (equal_nan && Number.isNaN(a) && Number.isNaN(b));
  }
  const different = new Error('');
  const wrapper = (a, b) => {
    if (!func(a, b)) throw different;
    return 0;
  }
  try { binary_operation(A, B, wrapper, Number) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
}


export default {
  op_binary, op_assign, op_unary,
  binary_operation, assign_operation,
  isclose, allclose,
} 