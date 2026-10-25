import { numpy_cases, Case } from "../helpers/cases";
import { np } from "../../src/index";
import * as fs from "fs";
import * as path from "path";

// Random indexed assignments / in-place operators through views (transposes, slices, reshapes, ...) compared with numpy.
const N: any = np;
const data: { base: string, view: string, idx: string, op: string, val: string, py: string }[] = JSON.parse(fs.readFileSync(path.join(__dirname, "fuzz_assign_cases.json"), "utf8"));
const tpl = (s: string) => Object.assign(s.split("\u0000"), { raw: s.split("\u0000") });
const cases: Case[] = data.map(c => [() => {
  const x = N.copy(N(tpl(c.base)));
  const v = c.view ? N(tpl(`\u0000${c.view}`), x) : x;
  N(tpl(`\u0000[${c.idx}] ${c.op} ${c.val}`), v);
  return x;
}, c.py] as Case);
numpy_cases("fuzz: indexed assignment through views", cases, { rtol: 1e-6, atol: 1e-9, auto_errors: true });
