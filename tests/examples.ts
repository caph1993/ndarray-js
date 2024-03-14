//@ts-check
import { spawnSync } from 'child_process'; // For testing
import { np } from '../src';
import { allClose } from '../src/utils-js';

console.log('Start')


function npTest(template, ...variables) {
  let idx = 0;
  const str = template.join('###').replace(/###/g, () => {
    let value = variables[idx++];
    if (value instanceof np.NDArray) value = value.tolist();
    let isListOfArrays = false;
    if (Array.isArray(value) && value[0] instanceof np.NDArray) isListOfArrays = true;
    if (isListOfArrays) value = value.map(np.tolist);
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
    obtained = np.tolist(template, ...variables);
  } catch (err) {
    console.error('EXPECTED:');
    console.error(expected && np.array(expected));
    console.error('OBTAINED:');
    console.error(`(ERROR)`);
    throw err;
  }
  if (!allClose(obtained, expected)) {
    console.error('EXPECTED:');
    console.error(expected && np.array(expected));
    console.error('OBTAINED:');
    console.error(obtained && np.array(obtained));
    throw new Error(`Mismatch for ${str}`);
  }
  console.log(obtained && np.array(obtained).toString());
  return obtained;
}


// Unit tests:
let x: any = np.array([1, 2, 3]);
x = np.arange(120).reshape(20, 3, 2).mean(0);
console.log(x.shape);
console.log(x.mean());
console.log(x);
x.pow(2);
console.log(x);
x.index(0, ':').op('=', -1);
x.index('...', ':2').op('+=', 5);

console.log(x.min());
console.log(x.argmin());

console.log('First row:', x.index(0));
console.log('Second row:', x.index(1));

x = np.random.randn([100, 2])
var norm = x.pow(2).sum({ axis: -1, keepdims: true }).pow(0.5)
console.log(x.op('/', norm).index([0, 1, 2]))
var norm = x.pow(2).sum(-1, { keepdims: false }).pow(0.5)
var norm = norm.index(':', null);
console.log(norm);
console.log(x.op('/', norm).index([0, 1, 2]))
var norm = x.pow(2).sum(-1, true).pow(0.5)
console.log(x.op('/', norm).index([0, 1, 2]))

// Invalid operation (prints a warning)



