


TO DO:

- (Done) comparisons
- (Done) reducers: any, all
- (Done) Boolean masks (requires dtypes. would be only number and bool)
- (Done) logical operations
- (Done) numbers in -1.4e-3 notation
- (Done) isclose
- (Done) Concurrency: the variables are being replaced with a global variable
- (Done) First integration to D3JS inside the notebook
- (DONE) shortcut reduce of any and all and allclose
- (DONE) linspace and geomspace
- (DONE) sign, abs, exp, log, sin, cos, sinh, cosh, asin, asin, ...
- (DONE) Round
- (DONE) arguments
- (DONE) ellipsis in slices
- (DONE) Better support for kwargs using this 
- (DONE) npm and license
- (DONE) (can be improved) improve assignment: options have add(A,B, out=..., where=...) they have `__iadd__ = add(A, B, A) ` and define array views. I should do the same, but views take time, I think. They also say `.reshape() creates a view` and `advanced indexing creates a copy` and `It must be noted here that during the assignment of x[[1, 2]] = [[10, 11, 12], [13, 14, 15]], no view or copy is created as the assignment happens in-place.`. So, `y = x[[1, 2]]` creates a copy but `y = x[1:3]` creates a view. According to my experiments, "advanced indexing" occurs when the sliceSpec contains boolean masks or integer indices. For fixed integer, range slices, ellipsis, 'None', and ':'.


- List all the functions that access array.flat

- speed up tests by running several python tests at once. Maybe printing to file?
- Get click to work in Guake

- parser should not return to_list, should it?
- Print in different format
- Handle the or operator "|" properly: for booleans gives booleans. For numbers gives numbers. 
- Support for 128 bits dtype? (I don't use that much)

- What about np`statement; \n statement;\n ...;\n return blah, blah, blah`

- readme and docs
- np.where
- maximum and minimum with ↑ and ↓. What priority wr to + *, ** and the logical operators?


- np.unique, np.sort, np.random.shuffle
- np.stack, np.concatenate, np.dstack, etc.

- group_by function
- np.split

- Write concurrency tests
- Random module
- Linalg module



