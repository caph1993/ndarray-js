// typedoc --plugin ./path/to/module-theme.js --theme module
// @ts-check
const td = require("typedoc");

// @ts-ignore getMapping is marked private, probably shouldn't be
// but I've been putting off making updates to that since I want to change
// how URLs are generated completely...
class ModuleTheme extends td.DefaultTheme {
  constructor(renderer) {
    super(renderer);

    // Unrelated to links, just a small style tweak to push members over so that they look less
    // like root exports on pages with classes.
    renderer.hooks.on("head.end", () => {
      return {
        tag: "style",
        props: null,
        children: [
          {
            tag: td.JSX.Raw,
            props: { html: ".tsd-member .tsd-member { margin-left: 2em; }" },
            children: [],
          },
        ],
      };
    });
  }

  /**
   * @param {td.DeclarationReflection} reflection
   */
  getMapping(reflection) {
    if (reflection.kindOf(td.ReflectionKind.Module)) {
      return {
        kind: [td.ReflectionKind.Module],
        directory: "modules",
        template: this.reflectionTemplate,
      };
    }
  }
}

/** @param {td.Application} app */
exports.load = function load(app) {
  app.renderer.defineTheme("module", ModuleTheme);
};