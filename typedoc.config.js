//@ts-check
/** @type {import('typedoc').TypeDocOptions} */
module.exports = {
  "$schema": "https://typedoc.org/schema.json",
  entryPoints: [
    "./src/docs-index.ts",
    // "./src/NDArray-class.ts",
    // "./src/modules/random.ts",
    // "./src/modules/constructors.ts",
    // "./src/modules/grammar.ts",
  ],
  out: "docs",
  plugin: [
    "@mxssfd/typedoc-theme",
    "typedoc-plugin-missing-exports",
    // "./typedoc-plugin-module-theme.js",
    // "./typedoc-plugin-expand.js",
    // "typedoc-github-wiki-theme",
    // "typedoc-default-themes",
    // "typedoc-theme-hierarchy",
    // "typedoc-plugin-merge-modules",
  ],
  // "placeInternalsInOwningModule": true,
  // theme: "module",
  internalModule: "(internals)",
  theme: "my-theme",
  includes: "./docs-src",
  // version: "1.0.0",
  // basePath: "./src",
  // mergeModulesRenameDefaults: true,
  // mergeModulesMergeMode: "project",
  // readme: "./docs-readme.md",
  readme: "none",
}