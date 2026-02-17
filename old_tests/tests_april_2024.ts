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
  const stdout = process.stdout.toString().trim();
  if (!stdout.length) throw new Error(`No output produced by python program. STDERR:\n${process.stderr.toString()}`);
  // Parse "Infinity"
  const expected = JSON.parse(stdout.replace(/Infinity/g, '1e500'));
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
    // console.error(obtained);
    throw new Error(`Mismatch for ${str}`);
  }
  console.log(obtained && np.array(obtained).toString());
  return obtained;
}


// console.log(np.apply_along_axis(np.ones([2, 5]), 0, (arr) => {
//   return np.ones(10).tolist();
// }));



// Unit tests:
npTest`np.linspace(0,1,10).reshape(5, 2)`;

for (let q of np.linspace(0, 1, 11).tolist()) {
  npTest`np.quantile(np.linspace(0,1,100).reshape(5, 10, 2), ${q}, axis=1)`;
  npTest`np.quantile(np.linspace(0,1,100).reshape(5, 2, 10), ${q}, axis=1)`;
  npTest`np.quantile(np.linspace(0,1,100).reshape(5, 4, 5), ${q}, axis=1)`;
  // npTest`np.quantile((np.linspace(0,1,100)**2).reshape(5, 4, 5), ${q}, axis=1)`; // Syntax not yet working
}

npTest`np.array(3)[False]`;
npTest`np.array(3)[True]`;

npTest`1/np.linspace(0,1,100)`;
// npTest`1/np.linspace(0,1,100)[::-1]`;
npTest`np.sum(np.linspace(0,1,100))`;
npTest`np.quantile(1/np.linspace(0,1,100), 1)`;

npTest`np.quantile(np.sin(np.linspace(0,10,100)), np.linspace(0, 1, 12))`;

for (let q of np.linspace(0, 1, 11).tolist()) {
  npTest`np.quantile(1/np.linspace(0,1,100)**2, ${q})`;
  npTest`np.nanquantile(1/np.linspace(0,1,100)**2, ${q})`;
  npTest`np.quantile(1/np.linspace(0,1,100).reshape(10, 10), ${q})`;
}
for (let q of np.linspace(0, 1, 11).tolist()) {
  npTest`np.quantile(np.linspace(0,1,100).reshape(5, 4, 5), ${q}, axis=1)`;
  // npTest`np.quantile((np.linspace(0,1,100)**2).reshape(5, 4, 5), ${q}, axis=1)`; // Syntax not yet working
}


npTest`np.array(10)`;
npTest`np.array(10)[True]`;
npTest`np.array(10)[False]`;
npTest`np.ones([10, 10])[True]`;
npTest`np.ones([10, 10])[False]`;

npTest`np.arange(120).reshape(2,3,4,5)[1][1]`
npTest`np.arange(120).reshape(2,3,4,5)[np.arange(120).reshape(2,3,4,5)<10]`


console.log('Success')