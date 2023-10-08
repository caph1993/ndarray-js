//@ts-check

export function binary_operation(A, B, func) {
  // Pointwise check for equality for arbitrary js arrays (without broadcasting)
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
};

export function ravel(A) {
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
};

export function allEq(A, B, nan_equal = false) {
  const different = new Error('');
  const func = (a, b) => {
    if (a !== b && !(nan_equal && Number.isNaN(a) && Number.isNaN(b))) throw different;
    return 0;
  }
  try { binary_operation(A, B, func) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
};

export function allClose(A, B, rtol = 1.e-5, atol = 1.e-8, nan_equal = false,) {
  const func = (a, b) => { //copied from isclose
    if (Number.isFinite(a) && Number.isFinite(b)) return Math.abs(a - b) <= atol + rtol * Math.abs(b);
    return (a === b) || (nan_equal && Number.isNaN(a) && Number.isNaN(b));
  }
  const different = new Error('');
  const wrapper = (a, b) => {
    if (!func(a, b)) throw different;
    return 0;
  }
  try { binary_operation(A, B, wrapper) }
  catch (err) {
    if (err === different) return false;
    else throw err;
  }
  return true;
}