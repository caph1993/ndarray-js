//@ts-check


/** @typedef {import("./core")} NDArray*/

const { NDArray } = require("./core-globals").GLOBALS;

/**
 * @param {NDArray} arr 
 * @returns {string}
 */
function humanReadable(arr) {
  if (arr.shape.length == 0) return arr.flat[0].toString();
  let budgets = arr.shape.map(_ => 1);
  let lBudget = 30;
  for (let i = 0; i < arr.shape.length; i++) {
    let before = budgets[i];
    budgets[i] = Math.min(arr.shape[i], lBudget);
    if (budgets[i] > before) lBudget = Math.floor(lBudget / (budgets[i] - before));
  }
  let rBudget = 30;
  for (let i = arr.shape.length - 1; i >= 0; i--) {
    let before = budgets[i];
    budgets[i] = Math.min(arr.shape[i], rBudget);
    if (budgets[i] > before) rBudget = Math.floor(rBudget / (budgets[i] - before));
  }
  function simplify(list, depth = 0) {
    if (depth == arr.shape.length) return list;
    if (2 * budgets[depth] >= list.length) {
      return list.map(l => simplify(l, depth + 1));
    }
    const left = list.slice(0, budgets[depth]).map(l => simplify(l, depth + 1));
    const right = list.slice(-budgets[depth]).map(l => simplify(l, depth + 1));
    return [...left, '...', ...right];
  }
  let rLimit = arr.shape.length - 1;
  while (rLimit > 0 && arr.shape[rLimit] == 1) {
    rLimit--;
  }
  if (arr.dtype == Number) arr = NDArray.prototype.round(arr, 2);
  let list = NDArray.prototype.toJS(arr);

  function str(list, indent = 0, depth = 0) {
    if (list == '...' || depth >= arr.shape.length) return list;
    if (depth == arr.shape.length - 1) return `[${list.join(', ')}]`;
    let sep = depth >= rLimit ? ' ' : '\n' + ' '.repeat(indent + 1);
    const out = [];
    for (let i = 0; i < list.length; i++) {
      let s = str(list[i], indent + 1, depth + 1) + ',';
      out.push(i < list.length - 1 ? s : s.slice(0, -1));
    }
    return `[${out.join(sep)}]`;
  }

  let prefix = 'Arr';
  let suffix = `, shape=(${arr.shape}), dtype=${arr.dtype.name}`;
  let out = str(simplify(list), 1 + prefix.length);
  function alignColumns(inputString, delimiter = ',') {
    // Split the input string into rows
    const rows = inputString.split('\n');
    // Initialize an array to store the maximum width of each column
    const columnWidths = Array(rows[0].split(delimiter).length).fill(0);
    // Find the maximum width for each column
    for (const row of rows) {
      let columns = row.split(delimiter);
      for (let i = 0; i < columns.length; i++) {
        columnWidths[i] = Math.max(columnWidths[i], columns[i].trim().length);
      }
    }
    // Build the formatted outputs
    let formattedString = '';
    for (const row of rows) {
      let columns = row.split(delimiter);
      columns = columns.map((s, i) => i == columns.length - 1 ? s : s + delimiter);
      for (let i = 0; i < columns.length; i++) {
        const column = columns[i].trim();
        formattedString += column.padStart(columnWidths[i] + 1).padEnd(columnWidths[i] + 2); // Add 1 for padding
      }
      formattedString += '\n';
    }
    return formattedString;
  }
  out = out.replace(/.*?(\n|$)/g, (match) => {
    // Split with a newline every 0 characters, but only after a comma,
    return match.replace(/(.{60,}?,)/g, '$1\n');
  }).replace(/\n+/g, '\n');
  out = alignColumns(`${prefix}(${out}`).trim();
  out = `${out}${suffix})`;
  return out;

}


module.exports = {
  humanReadable
} 