



## Tasks:

### np and np_

Maintain two namespaces:
- `np` is the exposed module, which handles Kwargs
- `np_` is the internal module, without Kwargs handler.

The library can use internally either `np` or `np_`, though the latter is preferred for speed.

Isn't this premature optimization?

### Put kwargs handlers closer to the main index

Right now, the kwargs are defined in each file.

What about NDArray methods?

Isn't this merely turning this around?

### Find a project structure that can simplify the agent automatization





