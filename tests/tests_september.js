//@ts-check
console.log('Start')
var { spawnSync } = require('child_process'); // For testing
var np = require('../index');
// console.log(np`np.arange(120).reshape([-1,3])`)
// console.log(np`np.arange(120).reshape([-1, 3])[2:-3:6]`)



function npTest(template, ...variables) {
  let idx = 0;
  const str = template.join('###').replace(/###/g, () => {
    let value = variables[idx++];
    if (value instanceof np.NDArray) value = value.JS();
    let isListOfArrays = false;
    if (Array.isArray(value) && value[0] instanceof np.NDArray) isListOfArrays = true;
    if (isListOfArrays) value = value.map(np.JS);
    let out = JSON.stringify(value)
    if (Array.isArray(value) && !isListOfArrays) out = `np.array(${out})`;
    return out;
  });
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
        return super(NpEncoder, self).default(obj)

out = ${str}
if isinstance(out, np.ndarray):
    out = out.tolist()
print(json.dumps(out, cls=NpEncoder), flush=True)
`
  console.log("========", str, "============");
  const process = spawnSync('python3', ['-c', program]);
  const stdout = process.stdout.toString();
  if (!stdout.length) throw new Error(`No output produced by python program. STDERR:\n${process.stderr.toString()}`);
  const expected = JSON.parse(stdout);
  let obtained;
  try {
    obtained = np.JS(template, ...variables);
  } catch (err) {
    console.error('EXPECTED:');
    console.error(expected && np.fromJS(expected));
    console.error('OBTAINED:');
    console.error(`(ERROR)`);
    throw err;
  }
  if (!np.modules.jsUtils.allClose(obtained, expected)) {
    console.error('EXPECTED:');
    console.error(expected && np.fromJS(expected));
    console.error('OBTAINED:');
    console.error(obtained && np.fromJS(obtained));
    throw new Error(`Mismatch for ${str}`);
  }
  console.log(obtained && np.fromJS(obtained).toString());
  return obtained;
}


// Unit tests:

npTest`np.concatenate(${Array.from({ length: 5 }, () => np`np.arange(100)`)}, axis=0)`
npTest`np.concatenate(${Array.from({ length: 5 }, () => np`np.arange(100)`)}, axis=-1)`
npTest`np.concatenate(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=1)`
npTest`np.concatenate(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=2)`
npTest`np.concatenate(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=-1)`
npTest`np.concatenate(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=-2)`


npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(100)`)}, axis=0)`
npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(100)`)}, axis=-1)`
npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=-1)`
npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=1)`
npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=2)`
npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=-1)`
npTest`np.stack(${Array.from({ length: 5 }, () => np`np.arange(120).reshape(6,4,5)`)}, axis=-2)`
npTest`np.transpose(np.arange(120).reshape(2,5,4,3))`

npTest`np.transpose(np.arange(120).reshape(2,5,4,3))`
npTest`np.transpose(np.arange(120).reshape(2,5,4,3), axes=[1,3,2,0])`

npTest`np.arange(120).reshape(2,5,4,3).transpose()`
npTest`np.arange(120).transpose()`
npTest`np.arange(120).reshape(3,-1).transpose()`
npTest`np.array(3).transpose()`



npTest`np.sort(-np.arange(120).reshape(2,5,4,3))`
npTest`np.sort(-np.arange(120).reshape(2,5,4,3), axis=0)`

npTest`np.sum(-np.arange(120).reshape(2,5,4,3), axis=0)`
npTest`np.sum(-np.arange(120).reshape(2,5,4,3), axis=1)`
npTest`np.sum(-np.arange(120).reshape(2,5,4,3), axis=2)`
npTest`np.sum(-np.arange(120).reshape(2,5,4,3), axis=3)`


npTest`np.sin(np.linspace(0,10,1000))`
npTest`np.sort(np.sin(np.linspace(0,10,1000)))`
npTest`np.sin(np.linspace(0,10,1000)).sort()`




npTest`np.reshape( np.arange(120), [2, 3, 4, 5] )`
npTest`np.reshape( np.arange(120), [2, 3, 4, 5] )[ :, 0, [1, 2], : ]`
npTest`np.ravel(np.reshape( np.arange(120), [2, 3, 4, 5] )[ :, 0, [1, 2], : ])`
npTest`np.ravel(np.reshape( np.arange(120), ${[2, 3, 4, 5]} )[ :, 0, ${[1, 2]}, : ])`
npTest`np.linspace(200, -45.3, 100)`

var a = npTest`np.array(([1,2], [3,6], [9, 10]))`
npTest`${a}.sum(axis=0)`
npTest`${a}.sum(axis=1)`

npTest`np.ones(${[10, 2]})`
npTest`np.sum(np.ones(${[10, 2]}), axis=0, keepdims=True)`
npTest`np.sum(np.ones(${[10, 2]}), axis=0, keepdims=True) + np.arange(2)`

npTest`np.arange(10000).reshape(100,2,50)`
npTest`np.arange(10000).reshape(100,10,10,1)`
npTest`np.arange(10000).reshape(100,2,50,1)`

npTest`np.arange(120).reshape(2,3,4,5)[1][1]`


npTest`np.arange(120).reshape(2,3,4,5)[1][:,:][2]`
npTest`np.arange(120).reshape(2,3,4,5)[1][:,:][2][:]`


npTest`np.arange(120).reshape(2,3,4,5)[0][:]`
npTest`np.arange(120).reshape(2,3,4,5)[:]`
npTest`np.arange(120).reshape(2,3,4,5)[:,:]`
npTest`np.arange(120).reshape(2,3,4,5)[:,:][0]`


npTest`np.arange(120).reshape(2,3,4,5)[0]`
npTest`np.arange(120).reshape(2,3,4,5)[0][:,:]`
npTest`np.arange(120).reshape(2,3,4,5)[0][:,:][2]`
npTest`np.arange(120).reshape(2,3,4,5)[0][:,:][2][:]`
npTest`np.arange(120).reshape(2,3,4,5)[0][:,:][2][:][1]`
npTest`np.arange(120).reshape(2,3,4,5)[0][:,:][2][:][1,3]`

npTest`np.arange(120).reshape(2,3,4,5)[0, np.arange(12).reshape((3,4))<5]`
npTest`np.arange(120).reshape(2,3,4,5)[0,2,1,3]`
npTest`np.arange(120).reshape(2,3,4,5)[0][:,:][2][:][1,3]`


npTest`np.linspace(0, 1, 10)`
npTest`np.linspace(0.5, 1.3, 10)`
npTest`np.geomspace(32, 45, 13)`
npTest`np.exp(${np`np.linspace(3, 6.5, 10)`})`

npTest`np.arange(120).reshape([-1,3])`
npTest`np.arange(120).reshape((-1,3))`
npTest`np.arange(120).reshape(-1,3)`

npTest`np.arange(120).reshape(2,3,4,5)[..., np.arange(12).reshape((3,4))<5, None, :]`
npTest`np.linspace(0,1,120, endpoint=True).reshape(2,3,4,5)[..., np.arange(12).reshape((3,4))<3, 0]`
npTest`np.linspace(0,1,120, endpoint=True).reshape(2,3,20)[..., 1]`

npTest`-1**2`
npTest`2**-0.5`
npTest`2**-0.5**2`
npTest`2**-0.5*2`
npTest`2**-(0.5)`

npTest`-1**2`
npTest`2**-0.5`
npTest`2**-0.5**2`
npTest`2**-0.5*2`
npTest`2**-(0.5)`

npTest`np.arange(120).reshape([-1,3])`
npTest`np.arange(120).reshape((-1,3))`
npTest`np.arange(120).reshape(-1,3)`
npTest`np.zeros(3)`
npTest`np.zeros(${[3, 5, 4]})`
npTest`np.sum(np.ones(${[10, 2]}), axis=0, keepdims=True) + np.arange(2)`
npTest`np.ravel(np.reshape( np.arange(120), ${[2, 3, 4, 5]} ))`
npTest`np.ravel(np.reshape( np.arange(120), ${[2, 3, 4, 5]} )[ :, 0, ${[1, 2]}, : ])`
npTest`np.ravel([[1,2], [3, 4]])`
npTest`np.arange(120).reshape((2,3,4,5))[:,0,[1,2],None,:].shape`
npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],:].shape`
npTest`np.arange(120).std()`

var a = npTest`np.array(([1,2], [3,6], [9, 10]))`
npTest`${a}.sum(axis=0)`
npTest`${a}.sum(axis=1)`
var a = npTest`np.array([[1,2], [2,4], [3,6]])`
npTest`${a}.sum(axis=0)`
npTest`${a}.sum(axis=1)`
npTest`${a}.sum(axis=0, keepdims=True)`
npTest`${a}.sum(axis=1, keepdims=True)`
npTest`${a}.sum(axis=0, keepdims=False)`

var a = npTest`np.arange(12).reshape([2, 3, 2])`
npTest`${a}.sum(axis=0)`
npTest`${a}.sum(axis=1)`
npTest`${a}.sum(axis=-1)`

var a = npTest`np.arange(6).reshape([2, 3])`
npTest`${a} * ${a}`
npTest`2 * ${a}`
npTest`-(${a} * ${a})`
npTest`-(${a} * ${a}).sum(axis=0, keepdims=True)`
npTest`-${a} * ${a}.sum(axis=0, keepdims=True)`
npTest`-(${a} * ${a}).sum(axis=-1, keepdims=True)`
npTest`np.sum(np.ones(np.array([10,2])), axis=0, keepdims=True) + np.arange(2)`

npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],None,:].shape`

npTest`np.arange(120).reshape([2,3,4,5])[:,0,[1,2],None,:]`


console.log('Success')