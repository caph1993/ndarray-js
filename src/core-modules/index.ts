//@ts-check

// Import core-globals first!
import './-core-globals';

import * as basic from './core-basic';
import * as indexes from './core-indexes';
import * as jsInterface from './core-js-interface';
import * as elementwise from './core-elementwise';
import * as print from './core-print';
import * as reduce from './core-reduce';
import * as operators from './core-operators';
import * as transform from './core-transform';

export const modules = {
  basic,
  jsInterface,
  indexes,
  elementwise,
  print,
  reduce,
  operators,
  transform,
};