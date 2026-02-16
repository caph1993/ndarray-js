//@ts-check
import { spawnSync } from 'child_process'; // For testing
import { np } from '../src';
import { allClose } from '../src/utils-js';
import { parse as JSON5_parse } from 'json5';

console.log('Start')


function npTest(template: TemplateStringsArray, ...variables: any[]) {
  let idx = 0;
  const str = template.join('###').replace(/###/g, () => {
    let value = variables[idx++];
    if (value instanceof np.NDArray) value = value.tolist();
    let isListOfArrays = false;
    if (Array.isArray(value) && value[0] instanceof np.NDArray) isListOfArrays = true;
    if (isListOfArrays) value = value.map(np.tolist);
    if (Number.isNaN(value)) return "np.nan";
    if (value === Infinity) return "np.inf";
    if (value === -Infinity) return "-np.inf";
    let out = JSON.stringify(value);
    if (Array.isArray(value) && !isListOfArrays) out = `np.array(${out})`;
    return out;
  });
  console.log("========", str, "============");
  const program = `
import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

out = ${str}
if isinstance(out, np.ndarray):
    out = out.tolist()
print(json.dumps(out, cls=NpEncoder), flush=True)
`
  const process = spawnSync('python3', ['-c', program]);
  const stdout = process.stdout.toString().trim();
  if (!stdout.length) throw new Error(`No output produced by python program. STDERR:\n${process.stderr.toString()}`);

  const expected = JSON5_parse(stdout);
  let obtained;
  try {
    obtained = np.tolist(template, ...variables);
  } catch (err) {
    console.error('EXPECTED (from Python):');
    console.error(expected && np.array(expected));
    console.error('OBTAINED (from ndarray-js):');
    console.error(`(ERROR)`);
    throw err;
  }
  if (!allClose(obtained, expected, undefined, undefined, true)) {
    console.error('EXPECTED (from Python):');
    console.error(expected && np.array(expected));
    console.error('OBTAINED (from ndarray-js):');
    console.error(obtained && np.array(obtained));
    // console.error(np.nonzero(np.isclose(obtained, expected).logical_not()));
    // console.error(np.array(obtained).index(313));
    // console.error(np.array(expected).index(313));
    throw new Error(`Mismatch for ${str}`);
  }
  console.log(obtained && np.array(obtained).toString());
  return obtained;
}

npTest`np.floor_divide([10, 20], [3, 7])`
npTest`np.power([2, 3], [3, 2])`
npTest`np.pow([2, 3], [3, 2])`
npTest`np.float_power([2.5, 3.5], [2, 1.5])`
npTest`np.positive([1, -2, 3])`
npTest`np.angle([1, 2, 3])`
npTest`np.real([1, 2, 3])`
npTest`np.imag([1, 2, 3])`
npTest`np.conj([1, 2, 3])`
npTest`np.conjugate([1, 2, 3])`
npTest`np.maximum([1, 5, 3], [2, 3, 4])`
npTest`np.minimum([1, 5, 3], [2, 3, 4])`
npTest`np.amax([1, 5, 3, 2])`
npTest`np.amin([1, 5, 3, 2])`

// New functions from next batch
npTest`np.sqrt([1, 4, 9, 16])`
npTest`np.cbrt([1, 8, 27])`
npTest`np.square([1, 2, 3, 4])`
npTest`np.absolute([-1, -2, 3])`
npTest`np.fabs([-1, -2, 3])`
npTest`np.sign([-3, 0, 2])`


npTest`np.all([True, True, False])`
npTest`np.any([False, False, True])`
npTest`np.array([0, np.inf, np.nan])`
npTest`np.array(np.array([0, np.inf, np.nan]))`
npTest`np.isfinite(np.array([0, np.inf, np.nan]))`
npTest`np.isfinite([0, np.inf, np.nan])`
// npTest`np.isinf([0, np.inf, -np.inf])`
npTest`np.isinf([0, ${np.inf}, -${np.inf}])`
npTest`np.isposinf([0, np.inf, -np.inf])`
npTest`np.isneginf([0, np.inf, -np.inf])`
npTest`np.isnan([0, np.nan])`
npTest`np.iscomplex([1, 2])`
npTest`np.isreal([1, np.nan, np.inf])`
npTest`np.isscalar(3)`
npTest`np.logical_and([1, 0], [1, 1])`
npTest`np.logical_or([0, 0], [1, 0])`
npTest`np.logical_not([1, 0])`
npTest`np.logical_xor([1, 0, 1], [1, 1, 0])`
npTest`np.allclose([1, 2], [1, 2.000001])`
npTest`np.isclose([1, 2], [1, 2.000001])`
npTest`np.array_equal([1, 2], [1, 2])`
npTest`np.array_equiv([[1], [2]], [1, 2])`
npTest`np.greater([1, 2], [2, 1])`
npTest`np.greater_equal([1, 2], [1, 3])`
npTest`np.less([1, 2], [2, 1])`
npTest`np.less_equal([1, 2], [1, 1])`
npTest`np.equal([1, 2], [1, 3])`
npTest`np.not_equal([1, 2], [1, 3])`
npTest`np.sin([0, 1.5707963267948966])`
npTest`np.cos([0, 1.5707963267948966])`
npTest`np.tan([0, 0.7853981633974483])`
npTest`np.arcsin([0, 1])`
npTest`np.arccos([1, 0])`
npTest`np.arctan([0, 1])`
npTest`np.hypot([3, 5], [4, 12])`
npTest`np.arctan2([1, 2], [3, 4])`
npTest`np.atan2([1, 2], [3, 4])`

// New arithmetic and utility functions
npTest`np.add([1, 2], [3, 4])`
npTest`np.subtract([5, 6], [1, 2])`
npTest`np.multiply([2, 3], [4, 5])`
npTest`np.divide([10, 20], [2, 4])`
npTest`np.true_divide([10, 20], [2, 4])`
npTest`np.mod([10, 20], [3, 7])`


// (!)
// npTest`np.sin(np.linspace(9,10,100))`

// npTest`np.sum(np.ones(${[10, 2]}), axis=0, keepdims=True)`
// npTest`np.sum(np.ones(${[10, 2]}), axis=0, keepdims=True) + np.arange(2)`
npTest`np.array([[10, 10]]) + np.arange(2)`


npTest`np.arange(5)`

npTest`np.array(3).transpose()`
npTest`np.arange(120).reshape(3,-1).transpose()`

var a = npTest`np.array(([1,2], [3,6], [9, 10]))`
npTest`${a}.sum(axis=0)`

npTest`np.zeros(${[3, 5, 4]})`
npTest`np.sum(np.ones(${[10, 2]}), axis=0, keepdims=True) + np.arange(2)`

var a = npTest`np.array(([1,2], [3,6], [9, 10]))`
npTest`${a}.sum(axis=1)`
var a = npTest`np.array([[1,2], [2,4], [3,6]])`
npTest`${a}.sum(axis=0)`

var a = npTest`np.arange(12).reshape([2, 3, 2])`
npTest`${a}.sum(axis=-1)`



npTest`np.sin(np.arange(100))`
npTest`np.argsort(np.sin(np.arange(100).reshape(10, 10)), axis=0)`

npTest`np.clip([-10, 5, 15, 20], a_min=0, a_max=10)`
npTest`np.clip(np.linspace(-10, 10, 200).reshape(10, 10, 2), a_min=0, a_max=None)`
npTest`np.nan_to_num([1, np.nan, np.inf, -np.inf])`
npTest`np.real_if_close([1, 2, 3])`
npTest`np.heaviside([-1, 0, 1, 2], [1, 2, 3, 4])`
// npTest`np.convolve([1, 2, 3], [0, 1])`
// npTest`np.interp([1.5, 2.5], [1, 2, 3], [10, 20, 30])`
// npTest`np.unique([3, 1, 4, 1, 5, 9, 2, 6, 5])`
// npTest`np.intersect1d([1, 2, 3, 4], [3, 4, 5, 6])`
// npTest`np.union1d([1, 2, 3], [3, 4, 5])`
// npTest`np.setdiff1d([1, 2, 3, 4], [2, 4])`
// npTest`np.setxor1d([1, 2, 3, 4], [3, 4, 5, 6])`
// npTest`np.isin([1, 2, 3, 4], [2, 4])`
npTest`np.argsort([3, 1, 4, 1, 5])`




npTest`np.array([0, np.e, np.pi, np.nan, np.inf])`
npTest`np.array([${1}, ${2}, ${3}])`
npTest`np.array([0, ${np.inf}, ${np.nan}])`
npTest`np.array([0, ${-np.inf}, -${np.inf}, -${np.nan}])`

npTest`np.isposinf([0, np.inf, -np.inf])`

// (!)
// npTest`np.sin(np.linspace(0,10,1000))`


var a = npTest`np.arange(6).reshape([2, 3])`
npTest`-(${a} * ${a}).sum(axis=0, keepdims=True)`

npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],None,:].shape`

npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],None,:]`




console.log('Success')