//@ts-check

// Import core-globals first!
import './_globals';

import * as basic from './basic';
import * as indexes from './indexes';
import * as jsInterface from './js-interface';
import * as elementwise from './elementwise';
import * as print from './print';
import * as reduce from './reduce';
import * as operators from './operators';
import * as transform from './transform';

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