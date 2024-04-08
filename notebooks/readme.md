


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
- (DONE) (can be improved) improve assignment: options have add(A,B, out=..., where=...) they have `__iadd__ = add(A, B, A) ` and define array views. I should do the same, but views take time, I think. They also say `.reshape() creates a view` and `advanced indexing creates a copy` and `It must be noted here that during the assignment of x[[1, 2]] = [[10, 11, 12], [13, 14, 15]], no view or copy is created as the assignment happens in-place.`. So, `y = x[[1, 2]]` creates a copy but `y = x[1:3]` creates a view. According to my experiments, "advanced indexing" occurs when the indexSpec contains boolean masks or integer indices. For fixed integer, range slices, ellipsis, 'None', and ':'.
- (DONE) List all the functions that access array.flat
- (DONE) compose simple indexes
- (DONE) Print in different format
- (DONE) Handle the or operator "|" properly: for booleans gives booleans. For numbers gives numbers. 
- (DONE) parser should not return to_list, should it?
- (DONE) Support for 128 bits dtype? (Nope. I don't use that much, and js does not support that natively)
- (DONE) make the module callable
- (DONE) provide two calls: if template string... otherwise, asarray np`...` np(...)
- (DONE) don't put everything into the array class. E.g., np.array.linspace makes no sense.
- (DONE) separate these as core: slice, _binary_operator, _assign_operator, _reduce, _apply, "_transform" (e.g. sort)
- (DONE) I can't find a "apply_pointwise" function. Instead they have "vectorize" for functions. call it apply_pointwise
- (DONE) enable GeneralIndexSpec to accept string '::-1' instead of already parsed range object.
- (DONE) fix the JSDoc signature of methods that replace this.
- (DONE) is there a way to copy the jsdoc of another function, but replacing the first argument? No.
- (DONE) np.sort
- (DONE) np.random.normal
- (DONE) implement "transform". Take into account that reduce is called "apply_along_axis" in numpy jargon, and it supports nd output, so it's my "transform"
- (DONE) implement transposition
- (DONE) decide a notation for slice and assignment
- (DONE) implement index notation
- (DONE) np.stack, np.concatenate
- (DONE) array iterator
- (DONE) np.random.shuffle, np.random."shuffled", 
- (DONE) Optimize with webpack
- (DONE) purposeful readme, with plots and good examples




  nonzero(a) Return the indices of the elements that are non-zero.
  where(condition, [x, y], /) Return elements chosen from x or y depending on condition.
  indices(dimensions[, dtype, sparse])
  diag_indices(n[, ndim]) Return the indices to access the main diagonal of an array.
  diag_indices_from(arr)
  mask_indices(n, mask_func[, k])
  np.ndindex()
  np.ndenumerate(a) Order statistics  - ptp(a[, axis, out, keepdims]) Range of values (maximum - minimum) along an axis.


  - percentile(a, q[, axis, out, ...]) Compute the q-th percentile of the data along the specified axis.
  - nanpercentile(a, q[, axis, out, ...]) Compute the qth percentile of the data along the specified axis, while ignoring nan values.
  - quantile(a, q[, axis, out, overwrite_input, ...]) Compute the q-th quantile of the data along the specified axis.
  - nanquantile(a, q[, axis, out, ...]) Compute the qth quantile of the data along the specified axis, while ignoring nan values.

Averages and variances

  - median(a[, axis, out, overwrite_input, keepdims]) Compute the median along the specified axis.
  - average(a[, axis, weights, returned, keepdims]) Compute the weighted average along the specified axis.
  - mean(a[, axis, dtype, out, keepdims, where]) Compute the arithmetic mean along the specified axis.
  - std(a[, axis, dtype, out, ddof, keepdims, where]) Compute the standard deviation along the specified axis.
  - var(a[, axis, dtype, out, ddof, keepdims, where]) Compute the variance along the specified axis.
  - nanmedian(a[, axis, out, overwrite_input, ...]) Compute the median along the specified axis, while ignoring NaNs.
  - nanmean(a[, axis, dtype, out, keepdims, where]) Compute the arithmetic mean along the specified axis, ignoring NaNs.
  - nanstd(a[, axis, dtype, out, ddof, ...]) Compute the standard deviation along the specified axis, while ignoring NaNs.
  - nanvar(a[, axis, dtype, out, ddof, ...]) Compute the variance along the specified axis, while ignoring NaNs.


Correlating
  - corrcoef(x[, y, rowvar, bias, ddof, dtype]) Return Pearson product-moment correlation coefficients.
  - correlate(a, v[, mode]) Cross-correlation of two 1-dimensional sequences.
  - cov(m[, y, rowvar, bias, ddof, fweights, ...]) Estimate a covariance matrix, given data and weights.

Histograms
 - histogram(a[, bins, range, density, weights]) Compute the histogram of a dataset.
 - histogram2d(x, y[, bins, range, density, ...]) Compute the bi-dimensional histogram of two data samples.
 - histogramdd(sample[, bins, range, density, ...]) Compute the multidimensional histogram of some data.
 - bincount(x, /[, weights, minlength]) Count number of occurrences of each value in array of non-negative ints.
 - histogram_bin_edges(a[, bins, range, weights]) Function to calculate only the edges of the bins used by the histogram function.
 - digitize(x, bins[, right]) Return the indices of the bins to which each value in input array belongs.

Set routines
lib.arraysetops
Set operations for arrays based on sorting.

Making proper sets
 - unique(ar[, return_index, return_inverse, ...]) Find the unique elements of an array.

Boolean operations
 - in1d(ar1, ar2[, assume_unique, invert, kind]) Test whether each element of a 1-D array is also present in a second array.
 - intersect1d(ar1, ar2[, assume_unique, ...]) Find the intersection of two arrays.
 - isin(element, test_elements[, ...]) Calculates element in test_elements, broadcasting over element only.
 - setdiff1d(ar1, ar2[, assume_unique]) Find the set difference of two arrays.
 - setxor1d(ar1, ar2[, assume_unique]) Find the set exclusive-or of two arrays.
union1d(ar1, ar2) Find the union of two arrays.



Sorting
 - sort(a[, axis, kind, order]) Return a sorted copy of an array.
 - lexsort(keys[, axis]) Perform an indirect stable sort using a sequence of keys.
 - argsort(a[, axis, kind, order]) Returns the indices that would sort an array.
 - ndarray.sort([axis, kind, order]) Sort an array in-place.
 - sort_complex(a) Sort a complex array using the real part first, then the imaginary part.
 - partition(a, kth[, axis, kind, order]) Return a partitioned copy of an array.
 - argpartition(a, kth[, axis, kind, order]) Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.

Searching
 - argmax(a[, axis, out, keepdims]) Returns the indices of the maximum values along an axis.
 - nanargmax(a[, axis, out, keepdims]) Return the indices of the maximum values in the specified axis ignoring NaNs.
 - argmin(a[, axis, out, keepdims]) Returns the indices of the minimum values along an axis.
 - nanargmin(a[, axis, out, keepdims]) Return the indices of the minimum values in the specified axis ignoring NaNs.
 - argwhere(a) Find the indices of array elements that are non-zero, grouped by element.
 - nonzero(a) Return the indices of the elements that are non-zero.
 - flatnonzero(a) Return indices that are non-zero in the flattened version of a.
 - where(condition, [x, y], /) Return elements chosen from x or y depending on condition.
 - searchsorted(a, v[, side, sorter]) Find indices where elements should be inserted to maintain order.
 - extract(condition, arr) Return the elements of an array that satisfy some condition.
Counting
 - count_nonzero(a[, axis, keepdims]) Counts the number of non-zero values in the array a.


Padding Arrays
 - pad(array, pad_width[, mode]) Pad an array.

Polynomials

Truth value testing
 - all(a[, axis, out, keepdims, where]) Test whether all array elements along a given axis evaluate to True.
 - any(a[, axis, out, keepdims, where]) Test whether any array element along a given axis evaluates to True.
 - Array contents
 - isfinite(x, /[, out, where, casting, order, ...]) Test element-wise for finiteness (not infinity and not Not a Number).
 - isinf(x, /[, out, where, casting, order, ...]) Test element-wise for positive or negative infinity.
 - isnan(x, /[, out, where, casting, order, ...]) Test element-wise for NaN and return result as a boolean array.
 - isnat(x, /[, out, where, casting, order, ...]) Test element-wise for NaT (not a time) and return result as a boolean array.
 - isneginf(x[, out]) Test element-wise for negative infinity, return result as bool array.
 - isposinf(x[, out]) Test element-wise for positive infinity, return result as bool array.

Array type testing
 - iscomplex(x) Returns a bool array, where True if input element is complex.
 - iscomplexobj(x) Check for a complex type or an array of complex numbers.
 - isfortran(a) Check if the array is Fortran contiguous but not C contiguous.
 - isreal(x) Returns a bool array, where True if input element is real.
 - isrealobj(x) Return True if x is a not complex type or an array of complex numbers.
 - isscalar(element) Returns True if the type of element is a scalar type.

Matrix and vector products
 - dot(a, b[, out]) Dot product of two arrays.
 - linalg.multi_dot(arrays, *[, out]) Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.
 - vdot(a, b, /) Return the dot product of two vectors.
 - inner(a, b, /) Inner product of two arrays.
 - outer(a, b[, out]) Compute the outer product of two vectors.
 - matmul(x1, x2, /[, out, casting, order, ...]) Matrix product of two arrays.
 - tensordot(a, b[, axes]) Compute tensor dot product along specified axes.
 - einsum(subscripts, *operands[, out, dtype, ...]) Evaluates the Einstein summation convention on the operands.
 - einsum_path(subscripts, *operands[, optimize]) Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.
 - linalg.matrix_power(a, n) Raise a square matrix to the (integer) power n.
 - kron(a, b) Kronecker product of two arrays.

Decompositions
 - linalg.cholesky(a) Cholesky decomposition.
 - linalg.qr(a[, mode]) Compute the qr factorization of a matrix.
 - linalg.svd(a[, full_matrices, compute_uv, ...]) Singular Value Decomposition.
 - Matrix eigenvalues
 - linalg.eig(a) Compute the eigenvalues and right eigenvectors of a square array.
 - linalg.eigh(a[, UPLO]) Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
 - linalg.eigvals(a) Compute the eigenvalues of a general matrix.
 - linalg.eigvalsh(a[, UPLO]) Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

Norms and other numbers
 - linalg.norm(x[, ord, axis, keepdims]) Matrix or vector norm.
 - linalg.cond(x[, p]) Compute the condition number of a matrix.
 - linalg.det(a) Compute the determinant of an array.
 - linalg.matrix_rank(A[, tol, hermitian]) Return matrix rank of array using SVD method
 - linalg.slogdet(a) Compute the sign and (natural) logarithm of the determinant of an array.
 - trace(a[, offset, axis1, axis2, dtype, out]) Return the sum along diagonals of the array.

Solving equations and inverting matrices
 - linalg.solve(a, b) Solve a linear matrix equation, or system of linear scalar equations.
 - linalg.tensorsolve(a, b[, axes]) Solve the tensor equation a x = b for x.
 - linalg.lstsq(a, b[, rcond]) Return the least-squares solution to a linear matrix equation.
 - linalg.inv(a) Compute the (multiplicative) inverse of a matrix.
 - linalg.pinv(a[, rcond, hermitian]) Compute the (Moore-Penrose) pseudo-inverse of a matrix.
 - linalg.tensorinv(a[, ind]) Compute the 'inverse' of an N-dimensional array.



- Export map.js as well
- Add license to webpack

- implement vectorize
- np.where
- np.unique

- np.split
- np.cov
- np.hist
- np.bins

- group_by function


- np.d3_snippets: examples



- add correct dtypes to operations

- make asarray global
- allow these syntax forms:  A.op('+', B) ;  A.op('+=', B) ; A.op('~') ; A.op(['::-1', 0], '+=', B)
- What about A([':', 0]), A('+', B([[false, true]])) // I like it. It's very succinct and clear.
- What about np(A, [':', 0], '+=', B, [[false, true]], '+') // NO. The reader has no clue of the precedence. 

- Download mozilla reference



- search for autogenerated docs options


- Get click to work in Guake

- What about np`statement; \n statement;\n ...;\n return blah, blah, blah`? Proposal: let {a, b, c} = np.block`${np.var('a')} = ...`});   Alternative: let {a, b, c} = np.block`VARS {a, b, c}; a = ...; b=...; b[:,a]=3; c = b.sum()`

- speed up tests by running several python tests at once. Maybe printing to file?


- add shortcut rules to operators, so that 0+x or 1*x or false||x returns x without iterating 


- maximum and minimum with ↑ and ↓. What priority wr to + *, ** and the logical operators?


- Write concurrency tests
- Random module
- Linalg module



