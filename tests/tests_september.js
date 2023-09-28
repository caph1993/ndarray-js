
console.log('Start')
var { spawnSync } = require('child_process'); // For testing
var MyArray = require('../index');
// console.log(np`np.arange(120).reshape([-1,3])`)
// console.log(np`np.arange(120).reshape([-1, 3])[2:-3:6]`)

/**
 * @param {TemplateStringsArray} template
 * @param {any[]} variables
 * */
function np(template, ...variables) {
  return MyArray.prototype.grammar.parse(template, ...variables);
}


function npTest(template, ...variables) {
  let idx = 0;
  const str = template.join('###').replace(/###/g, () => {
    let value = variables[idx++];
    let out = JSON.stringify(value)
    if (Array.isArray(value)) out = `np.array(${out})`;
    return out;
  });
  const program = `
import numpy as np
import json
out = ${str}
if isinstance(out, np.ndarray):
    out = out.tolist()
print(json.dumps(out), flush=True)
`
  const process = spawnSync('python3', ['-c', program]);
  const stdout = process.stdout.toString();
  if (!stdout.length) throw new Error(process.stderr.toString());
  const expected = JSON.parse(stdout);
  let obtained;
  try {
    obtained = np(template, ...variables);
  } catch (err) {
    console.error(JSON.stringify(expected));
    console.error(`numpy-js failed for ${str}`);
    throw err;
  }
  if (!MyArray.prototype.nested.allClose(obtained, expected)) {
    console.error(JSON.stringify(expected));
    console.error(JSON.stringify(obtained));
    throw new Error(`Mismatch for ${str}`);
  }
  return obtained;
}

// Unit tests:

npTest`np.linspace(0, 1, 10)`
npTest`np.linspace(0.5, 1.3, 10)`
npTest`np.geomspace(32, 45, 13)`
npTest`np.linspace(200, -45.3, 100)`
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