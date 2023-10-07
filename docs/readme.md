


October 7 2023:

I decided to change the whole project to TS because JSdoc looks very immature.

I was having these issues because JSDoc uses a different syntax for importing modules:
  https://github.com/jsdoc/jsdoc/issues/1961
  https://github.com/onury/jsdoc-x/issues/12

Then (after long time) I found a solution, which is to use a plugin:
  https://github.com/polyforest/jsdoc-tsimport-plugin/

By creating a `jsdoc.conf.json` file with content:
```
{
  "plugins": [
    "../node_modules/jsdoc-tsimport-plugin/index.js"
  ]
}
```
and running:
  jsdoc --verbose -c ./jsdoc.conf.json ../src/index.js

But then, I couldn't get the plugin to be used by sphinx-js:
  https://github.com/mozilla/sphinx-js

Sphinx-js does not provide a "plugins" option, and it does not teach how to add them either.
Moreover, they say they use a different syntax to JSDoc for path names.

Then I found this second issue, JSDOC does not support tuples, for my range objects `[start, stop, step]`:
  https://github.com/jsdoc/jsdoc/issues/1703

It also complains about the type `{1|-1}` of readDir.

For certain objects I can't write `{foo:Bar}[]`, it must be `Array<{foo:Bar}>`.

Overall, it does not satisfy my needs. TSDOC looks more promising...
  https://tsdoc.org/
  