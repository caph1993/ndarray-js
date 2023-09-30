//@ts-check

/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./core-globals").GLOBALS;


function fromJS(arr) {
  if (typeof arr === "number") return new NDArray([arr], [], Number);
  if (typeof arr === "boolean") return new NDArray([arr ? 1 : 0], [], Boolean);
  if (!Array.isArray(arr)) throw new Error(`Can't parse as array: ${arr}`);
  const shape = [];
  let root = arr;
  while (Array.isArray(root)) {
    shape.push(root.length);
    root = root[0];
    if (shape.length > 256) throw new Error(`Circular reference or excessive array depth`);
  }
  let dtype = typeof root === "boolean" ? Boolean : Number;
  const flat = [];
  const pushToFlat = (arr, axis) => {
    // Check consistency
    if (axis == shape.length - 1) {
      for (let elem of arr) {
        if (Array.isArray(elem)) throw new Error(`Inconsistent shape`);
        flat.push(elem);
        // Update dtype
      }
    } else {
      if (!Array.isArray(arr)) throw new Error(`Inconsistent shape`);
      for (let sub of arr) {
        if (sub.length != shape[axis + 1])
          throw new Error(`Inconsistent shape: found sibling arrays of lengths ${sub.length} and ${shape[axis + 1]}`);
        pushToFlat(sub, axis + 1);
      }
    }
  }
  pushToFlat(arr, 0);
  return new NDArray(flat, shape, dtype)
}

function toJS(arr) {
  if (typeof arr == "number" || typeof arr == "boolean") return arr;
  if (Array.isArray(arr)) return arr.map(toJS);
  if (!(arr instanceof NDArray)) throw new Error(`Expected MyArray. Got ${typeof arr}: ${arr}`);
  arr = NDArray.prototype.__number_collapse(arr);
  if (!(arr instanceof NDArray)) return arr;

  // let out = [], top;
  // let q = /**@type {[MyArray, any][]}*/([[arr, out]])
  // while (top = q.pop()) {
  //   let [arr, out] = top;
  //   if (arr.shape.length <= 1) {
  //     out.push(...arr.flat);
  //   } else {
  //     for (let i = 0; i < arr.shape[0]; i++) {
  //       let l = []
  //       out.push(l);
  //       q.push([arr.slice(i), l]);
  //     }
  //   }
  // }
  // return out;
  function recursiveReshape(flatArr, shapeArr) {
    if (shapeArr.length === 0) {
      return flatArr.shift();
    }
    const innerShape = shapeArr.slice(1);
    const outerSize = shapeArr[0];
    const innerArray = [];
    for (let i = 0; i < outerSize; i++) {
      innerArray.push(recursiveReshape(flatArr, innerShape));
    }
    return innerArray;
  }
  return recursiveReshape([...arr.flat], arr.shape);
}




const nested = {
  _binary_operation(A, B, func) {
    // Pointwise check for equality for arbitrary nested js arrays (without broadcasting)
    const C = [];
    const q = [[A, B, C, 0]];
    let seen;
    while (true) {
      const _next = q.pop();
      if (!_next) return true;
      const [a, b, c, depth] = _next
      if (Array.isArray(a) && Array.isArray(b) && a.length == b.length) {
        for (let i in a) {
          const c_i = [];
          c.push(c_i);
          q.push([a[i], b[i], c_i, depth + 1]);
        }
      }
      else c.push(func(a, b));
      if (depth > 10000 && Array.isArray(a)) { // Activate circular reference detection
        // Checking only A suffices (the other will exhaust otherwise)
        seen = /**@type {any[]}*/(seen || []);
        if (seen.includes(a)) throw new Error(`Circular reference found. ${a}`)
        seen[depth - 10000] = a;
      }
    }
  },
  ravel(A) {
    // Flatten js array
    const q = [[A, 0]], flat = [];
    let seen;
    while (true) {
      const _next = q.pop();
      if (!_next) break;
      const [a, depth] = _next
      if (depth > 10000 && Array.isArray(a)) { // Activate circular reference detection
        seen = /**@type {any[]}*/(seen || []);
        if (seen.includes(a)) throw new Error(`Circular reference found. ${a}`)
        seen[depth - 10000] = a;
      }
      if (Array.isArray(a)) {
        q.push(...a.map(v => [v, depth + 1]));
        continue;
      }
      flat.push(a);
    }
    return flat;
  },
  allEq(A, B, nan_equal = false) {
    const different = new Error('');
    const func = (a, b) => {
      if (a !== b && !(nan_equal && Number.isNaN(a) && Number.isNaN(b))) throw different;
      return 0;
    }
    try { nested._binary_operation(A, B, func) }
    catch (err) {
      if (err === different) return false;
      else throw err;
    }
    return true;
  },
  allClose(A, B, rtol = 1.e-5, atol = 1.e-8, nan_equal = false,) {
    const func = (a, b) => { //copied from isclose
      if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * Math.abs(b);
      return (a === b) || (nan_equal && Number.isNaN(a) && Number.isNaN(b));
    }
    const different = new Error('');
    const wrapper = (a, b) => {
      if (!func(a, b)) throw different;
      return 0;
    }
    try { nested._binary_operation(A, B, wrapper) }
    catch (err) {
      if (err === different) return false;
      else throw err;
    }
    return true;
  },
}


module.exports = {
  fromJS, toJS, nested
} 