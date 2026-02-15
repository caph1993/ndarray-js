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

npTest`np.array([0, np.e, np.pi, np.nan, np.inf])`
npTest`np.array([${1}, ${2}, ${3}])`
npTest`np.array([0, ${np.inf}, ${np.nan}])`
npTest`np.array([0, ${-np.inf}, -${np.inf}, -${np.nan}])`

npTest`np.isposinf([0, np.inf, -np.inf])`


npTest`np.sin(np.linspace(0,10,1000))`
npTest`np.arange(120).reshape(3,-1).transpose()`
npTest`np.array(3).transpose()`

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

var a = npTest`np.arange(6).reshape([2, 3])`
npTest`-(${a} * ${a}).sum(axis=0, keepdims=True)`

npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],None,:].shape`

npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],None,:]`

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


console.log('Success')