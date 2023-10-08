//@ts-check
/** @type {import('typedoc').TypeDocOptions} */
module.exports = {
  //"$schema": "https://typedoc.org/schema.json",
  "entryPoints": [
    //"./src/index.ts",
    "./src/np.ts",
    "./src/core.ts",
    "./src/core-modules/core-basic.ts",
    "./src/core-modules/core-indexes.ts",
    "./src/core-modules/core-reduce.ts",
    "./src/core-modules/core-operators.ts",
    "./src/core-modules/core-transform.ts",
    "./src/core-modules/core-elementwise.ts",
    "./src/core-modules/core-js-interface.ts",
    "./src/core-modules/core-print.ts",
    "./src/np-modules/np-random.ts",
    "./src/np-modules/np-constructors.ts",
    "./src/np-modules/np-grammar.ts",
  ],
  "out": "docs",
}