//@ts-check
/** @type {import('typedoc').TypeDocOptions} */
module.exports = {
  //"$schema": "https://typedoc.org/schema.json",
  entryPoints: [
    "./src/index.ts",
    "./src/NDArray-class.ts",
    "./src/modules/random.ts",
    "./src/modules/constructors.ts",
    "./src/modules/grammar.ts",
  ],
  out: "docs",
  plugin: [
    "@mxssfd/typedoc-theme",
  // "typedoc-default-themes",
  // "typedoc-theme-hierarchy",
  // "typedoc-plugin-merge-modules",
  ],
  // theme: "hierarchy",
  theme: "my-theme",
  basePath: "./src",
  // entryPointStrategy: "Expand",
  // mergeModulesRenameDefaults: true,
  // mergeModulesMergeMode: "project",
  readme: "docs-readme.md",
}