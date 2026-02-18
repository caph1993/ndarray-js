//@ts-check

import NDArray from '../NDArray';
import * as indexes from './indexes';
import * as jsInterface from './js-interface';
import * as elementwise from './elementwise';
import * as print from './print';
import * as shape from './shape_operations';
import * as reduce from './reduce';
import * as operators from './operators';
import * as transform from './transform';

export const modules = {
  jsInterface,
  indexes,
  elementwise,
  print,
  reduce,
  shape,
  operators,
  transform,
};

NDArray.prototype.modules = modules;