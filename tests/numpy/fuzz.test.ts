import { numpy_cases } from "../helpers/cases";
import * as fs from "fs";
import * as path from "path";

// Random compositions (1 to 4 steps) of elementwise functions, reductions, shape functions and indexing on small random
// arrays of several dtypes, generated once with numpy and compared against it. Found: weak/strong scalar bugs, view bugs...
const cases: string[] = JSON.parse(fs.readFileSync(path.join(__dirname, "fuzz_cases.json"), "utf8"));
numpy_cases("fuzz: random expression chains", cases, { rtol: 1e-5, atol: 1e-8, auto_errors: true });

// Second set: also NaN, complex, float32 and int8 arrays, and nan-functions, padding, statistics, fft, polynomial helpers...
// (fft of arrays with NaN and NaN -> int casts are undefined behaviour in numpy and are left out)
const cases2: string[] = JSON.parse(fs.readFileSync(path.join(__dirname, "fuzz_cases2.json"), "utf8"));
numpy_cases("fuzz: random expression chains (wider function set)", cases2, { rtol: 1e-5, atol: 1e-8, auto_errors: true });
