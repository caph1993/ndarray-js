import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

const A = "np.arange(6).reshape(2, 3)";
const B = "np.arange(24).reshape(2, 3, 4)";
const F = "np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])";

numpy_cases("constructors", [
  "np.eye(3)", "np.eye(2, 3)", "np.eye(3, k=1)", "np.eye(3, k=-2)", "np.eye(3, 4, 1, dtype='int32')", "np.identity(3)", "np.identity(2, dtype='bool')", "np.tri(3)", "np.tri(3, 4, 1)", "np.tri(3, 2, -1, dtype='int32')",
  `np.tril(${A})`, `np.tril(${A}, 1)`, `np.tril(${A}, -1)`, `np.triu(${A})`, `np.triu(${A}, 1)`, `np.triu(${A}, -1)`, `np.tril(${B})`, `np.triu(${B}, 1)`, "np.tril(np.array([1, 2]))",
  "np.diag(np.array([1, 2, 3]))", "np.diag(np.array([1, 2, 3]), 1)", "np.diag(np.array([1, 2, 3]), -2)", `np.diag(${A})`, `np.diag(${A}, 1)`, `np.diag(${A}, -1)`, `np.diag(${A}, 5)`, `np.diagflat(${A})`, "np.diagflat(np.array([1, 2]), 1)", `!np.diag(${B})`,
  "np.full(3, 7)", "np.full((2, 2), 1.5)", "np.full((2, 3), True)", "np.full((2, 2), 7, dtype='float64')", "np.full((2, 2), np.array([1, 2]))", "np.full((2, 2), 3.0)", "np.full(0, 1)", "np.full((2, 2), 2.5, dtype='int32')",
  `np.full_like(${A}, 9)`, `np.full_like(${A}, 2.5)`, `np.full_like(${F}, 2)`, `np.full_like(${A}, 1, dtype='float64')`, `np.zeros_like(${A})`, `np.ones_like(${F})`, `np.ones_like(${A}, dtype='bool')`, `np.zeros_like(${A}, shape=(3,))`, `np.empty_like(${A}).shape`,
  "np.zeros_like([1, 2, 3])", "np.ones_like(np.array([True, False]))",
  "np.logspace(0, 3, 4)", "np.logspace(0, 3, 4, endpoint=False)", "np.logspace(1, 2, 3, base=2)", "np.logspace(0, 1)", "np.logspace(0, 2, 3, dtype='int32')", "np.logspace(2, 0, 3)",
  "np.meshgrid(np.array([1, 2, 3]), np.array([4, 5]))", "np.meshgrid(np.array([1, 2, 3]), np.array([4, 5]), indexing='ij')", "np.meshgrid(np.array([1, 2]), np.array([3, 4]), np.array([5, 6, 7]))",
  "np.meshgrid(np.array([1, 2]), np.array([3, 4]), np.array([5, 6, 7]), indexing='ij')", "np.meshgrid(np.array([1, 2, 3]), np.array([4, 5]), sparse=True)", "np.meshgrid(np.array([1, 2, 3]), np.array([4, 5]), sparse=True, indexing='ij')",
  "np.meshgrid(np.array([1, 2, 3]))", "np.meshgrid(np.array([[1, 2], [3, 4]]), np.array([5, 6]))",
  [() => np.mgrid('0:3', '0:2'), "np.mgrid[0:3, 0:2]"], [() => np.mgrid('0:4'), "np.mgrid[0:4]"], [() => np.mgrid('0:1:0.25', '1:3'), "np.mgrid[0:1:0.25, 1:3]"], [() => np.mgrid('0:1:5j'), "np.mgrid[0:1:5j]"],
  [() => np.ogrid('0:3', '0:2'), "np.ogrid[0:3, 0:2]"], [() => np.ogrid('0:2', '0:2', '1:4'), "np.ogrid[0:2, 0:2, 1:4]"],
  "np.ndim(5)", `np.ndim(${B})`, `np.shape(${B})`, "np.shape(5)", "np.shape([[1, 2], [3, 4], [5, 6]])", `np.size(${B})`, `np.size(${B}, axis=1)`, `np.size(${B}, -1)`, "np.size(5)", "!np.size(np.arange(3), axis=1)",
  `np.astype(${A}, 'float64')`, `np.astype(${F}, 'int32')`, `np.astype(${F}, 'bool')`, `np.astype(np.array([0, 1, 2]), 'bool')`, `np.astype(${A}, 'int32', copy=False)`, `${F}.astype('int32')`, `np.copy(${A})`, `np.copy(${A}[:, ::-1])`, "np.asanyarray([1, 2])", `np.require(${A}, dtype='float64')`,
]);

numpy_cases("axes and broadcasting", [
  `np.moveaxis(${B}, 0, -1)`, `np.moveaxis(${B}, -1, 0)`, `np.moveaxis(${B}, [0, 1], [-1, -2])`, `np.moveaxis(${B}, 1, 2)`, `np.moveaxis(${B}, [0, 1], [1, 0])`, `!np.moveaxis(${B}, 0, 5)`, `!np.moveaxis(${B}, [0, 0], [1, 2])`,
  `np.rollaxis(${B}, 2)`, `np.rollaxis(${B}, 2, 1)`, `np.rollaxis(${B}, 1, 3)`, `np.rollaxis(${B}, 0, 3)`, `np.rollaxis(${B}, 1, 0)`, `np.rollaxis(${B}, -1)`,
  `np.swapaxes(${B}, 0, 2)`, `np.swapaxes(${B}, 1, -1)`, `${B}.swapaxes(0, 1)`, `np.permute_dims(${B}, (2, 0, 1))`, `np.matrix_transpose(${B})`, `!np.matrix_transpose(np.arange(3))`,
  "np.atleast_1d(5)", "np.atleast_1d(np.array([1, 2]))", "np.atleast_1d(1, np.array([2, 3]))", "np.atleast_2d(5)", "np.atleast_2d(np.array([1, 2]))", `np.atleast_2d(${A})`, "np.atleast_2d(1, np.array([1, 2]))",
  "np.atleast_3d(5)", "np.atleast_3d(np.array([1, 2]))", `np.atleast_3d(${A})`, `np.atleast_3d(${B})`,
  "np.broadcast_to(np.array([1, 2, 3]), (2, 3))", "np.broadcast_to(5, (2, 2))", "np.broadcast_to(np.array([[1], [2]]), (2, 3))", "np.broadcast_to(np.array([1, 2]), (3, 2))", "np.broadcast_to(np.array([1, 2]), 2)",
  "!np.broadcast_to(np.array([1, 2, 3]), (2, 2))", "!np.broadcast_to(np.array([[1, 2], [3, 4]]), (2,))",
  "np.broadcast_arrays(np.array([1, 2, 3]), np.array([[4], [5]]))", "np.broadcast_arrays(np.array([1, 2]), 5)", "np.broadcast_arrays(np.array([[1, 2]]))", "!np.broadcast_arrays(np.array([1, 2]), np.array([1, 2, 3]))",
  [() => np.broadcast(np.array([1, 2, 3]), np.array([[4], [5]])).shape, "np.broadcast(np.array([1, 2, 3]), np.array([[4], [5]])).shape"],
  `np.expand_dims(${A}, 0)`, `np.expand_dims(${A}, 1)`, `np.expand_dims(${A}, -1)`, `np.expand_dims(${A}, (0, 3))`, `np.expand_dims(np.array([1, 2]), [0, 1])`, "np.expand_dims(5, 0)", `!np.expand_dims(${A}, 4)`,
  "np.squeeze(np.zeros((1, 3, 1)))", "np.squeeze(np.zeros((1, 3, 1)), axis=0)", "np.squeeze(np.zeros((1, 3, 1)), axis=(0, 2))", "np.squeeze(np.zeros((1, 1)))", "np.squeeze(np.arange(3))", "!np.squeeze(np.zeros((1, 3, 1)), axis=1)", "np.zeros((1, 3, 1)).squeeze()", "np.zeros((1, 3, 1)).squeeze(2)",
]);

numpy_cases("joining and splitting", [
  `np.vstack([${A}, ${A}])`, "np.vstack([np.array([1, 2]), np.array([3, 4])])", "np.vstack((np.array([1, 2]), np.array([[3, 4], [5, 6]])))", "np.vstack([np.array(1), np.array(2)])",
  "np.hstack([np.array([1, 2]), np.array([3])])", `np.hstack([${A}, ${A}])`, "np.hstack([np.array(1), np.array(2)])", `np.hstack([${A}, np.array([[7], [8]])])`,
  `np.dstack([${A}, ${A}])`, "np.dstack([np.array([1, 2]), np.array([3, 4])])", "np.column_stack([np.array([1, 2]), np.array([3, 4])])", `np.column_stack([${A}, np.array([7, 8])])`,
  "np.concat([np.array([1]), np.array([2])])", "!np.vstack([np.array([1, 2]), np.array([1, 2, 3])])",
  "np.block([np.array([1]), np.array([2])])", `np.block([[${A}, ${A}], [${A}, ${A}]])`, "np.block([[np.array([1, 2])], [np.array([3, 4])]])", "np.block([[1, 2], [3, 4]])", "np.block([1, 2, 3])", `np.block([${A}, ${A}])`, "np.block(np.array([[1, 2]]))",
  "np.block([[np.array([[1, 2], [3, 4]]), np.array([[5], [6]])], [np.array([[7, 8, 9]])]])",
  "np.split(np.arange(6), 3)", "np.split(np.arange(6), [2, 3])", "np.split(np.arange(6), [2, 3, 10])", `np.split(${A}, 2, axis=0)`, `np.split(${A}, 3, axis=1)`, `np.split(${A}, [1], axis=1)`, "np.split(np.arange(6), [4, 2])", "!np.split(np.arange(6), 4)",
  "np.split(np.arange(6), [-2])", `np.split(${B}, [1, 3], axis=-1)`,
  "np.array_split(np.arange(7), 3)", "np.array_split(np.arange(8), 3)", `np.array_split(${A}, 2, axis=1)`, "np.array_split(np.arange(3), 5)", "np.array_split(np.arange(6), [1, 4])",
  `np.hsplit(${A}, 3)`, "np.hsplit(np.arange(6), 2)", `np.hsplit(${A}, [1])`, `np.vsplit(${A}, 2)`, `np.dsplit(${B}, 2)`, `np.dsplit(${B}, [1, 3])`, "!np.vsplit(np.arange(4), 2)", `!np.dsplit(${A}, 2)`,
  `np.unstack(${A})`, `np.unstack(${A}, axis=1)`, `np.unstack(${B}, axis=-1)`, "np.unstack(np.arange(3))",
]);

numpy_cases("repeating, inserting, deleting", [
  "np.tile(np.array([1, 2]), 3)", "np.tile(np.array([1, 2]), (2, 2))", `np.tile(${A}, 2)`, `np.tile(${A}, (2, 1))`, `np.tile(${A}, (2, 1, 2))`, "np.tile(np.array([1, 2]), (2, 1, 1))", "np.tile(5, 3)", "np.tile(np.array([1, 2]), 0)", `np.tile(${A}, (1, 0))`, "np.tile(np.array([True, False]), 2)",
  "np.repeat(np.array([1, 2]), 2)", "np.repeat(np.array([1, 2]), [1, 3])", `np.repeat(${A}, 2)`, `np.repeat(${A}, 2, axis=0)`, `np.repeat(${A}, 2, axis=1)`, `np.repeat(${A}, [1, 2], axis=0)`, `np.repeat(${A}, [1, 0, 2], axis=1)`, "np.repeat(3, 4)", "np.repeat(np.array([1, 2]), 0)",
  `${A}.repeat(2, axis=1)`, "!np.repeat(np.array([1, 2]), -1)", "!np.repeat(np.array([1, 2]), [1, 2, 3])",
  "np.delete(np.arange(6), 2)", "np.delete(np.arange(6), [0, 5])", "np.delete(np.arange(6), -1)", "np.delete(np.arange(6), np.array([True, False, True, False, True, False]))", `np.delete(${A}, 1, axis=1)`, `np.delete(${A}, 0, axis=0)`, `np.delete(${A}, 1)`,
  `np.delete(${A}, [0, 2], axis=1)`, "np.delete(np.arange(6), [])", "np.delete(np.arange(6), 1, None)", "!np.delete(np.arange(6), 6)", "!np.delete(np.arange(3), 5)",
  "np.insert(np.arange(4), 2, 99)", "np.insert(np.arange(4), [1, 3], 99)", "np.insert(np.arange(4), [1, 3], [8, 9])", "np.insert(np.arange(4), 0, [7, 8])", "np.insert(np.arange(4), 4, 1.5)", "np.insert(np.arange(4), -1, 5)", "np.insert(np.arange(4), [2, 2], [5, 6])", "np.insert(np.arange(4), [3, 1], [5, 6])",
  `np.insert(${A}, 1, 99)`, `np.insert(${A}, 1, 99, axis=0)`, `np.insert(${A}, 1, 99, axis=1)`, `np.insert(${A}, 1, np.array([7, 8, 9]), axis=0)`, `np.insert(${A}, 1, np.array([7, 8]), axis=1)`, `np.insert(${A}, [1, 2], 0, axis=1)`, `np.insert(${A}, [0, 1], np.array([[1, 2], [3, 4]]), axis=1)`,
  "!np.insert(np.arange(4), 7, 1)",
  "np.append(np.arange(3), 5)", "np.append(np.arange(3), np.array([4, 5]))", `np.append(${A}, 9)`, `np.append(${A}, ${A}, axis=0)`, `np.append(${A}, ${A}, axis=1)`, `np.append(${A}, np.array([[7], [8]]), axis=1)`, "np.append(np.array([1.5]), np.array([2]))",
  "np.resize(np.arange(4), (2, 3))", "np.resize(np.arange(6), (2, 2))", "np.resize(np.arange(3), 7)", "np.resize(5, (2, 2))", `np.resize(${A}, (3, 3))`, "np.resize(np.array([]), (2, 2))",
  "np.trim_zeros(np.array([0, 0, 1, 2, 0, 3, 0, 0]))", "np.trim_zeros(np.array([0, 0, 1, 2, 0, 3, 0, 0]), 'f')", "np.trim_zeros(np.array([0, 0, 1, 2, 0, 3, 0, 0]), 'b')", "np.trim_zeros(np.array([0, 0, 0]))", "np.trim_zeros(np.array([1, 2]))",
  "np.trim_zeros(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))", "np.trim_zeros(np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0]]), axis=0)", "np.trim_zeros(np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0]]), trim='f')", "np.trim_zeros(np.array([0.0, np.nan, 0.0]))",
]);

numpy_cases("reordering", [
  `np.flip(${B})`, `np.flip(${B}, axis=0)`, `np.flip(${B}, axis=-1)`, `np.flip(${B}, (0, 2))`, "np.flip(np.arange(5))", "np.flip(5)", `np.fliplr(${A})`, `np.flipud(${A})`, "np.flipud(np.arange(4))", "!np.fliplr(np.arange(4))", `np.flip(${A}[:, ::2])`,
  "np.roll(np.arange(5), 2)", "np.roll(np.arange(5), -1)", "np.roll(np.arange(5), 7)", `np.roll(${A}, 1)`, `np.roll(${A}, 1, axis=0)`, `np.roll(${A}, 1, axis=1)`, `np.roll(${A}, -1, axis=1)`, `np.roll(${A}, (1, 2), axis=(0, 1))`, `np.roll(${A}, (1, 1), axis=(1, 1))`, `np.roll(${A}, 1, axis=(0, 1))`, `np.roll(${A}, (1, 2))`,
  "np.roll(np.array([]), 3)", "!np.roll(np.arange(4), 1, axis=2)",
  `np.rot90(${A})`, `np.rot90(${A}, 2)`, `np.rot90(${A}, 3)`, `np.rot90(${A}, -1)`, `np.rot90(${A}, 4)`, `np.rot90(${B})`, `np.rot90(${B}, axes=(1, 2))`, `np.rot90(${B}, 3, (0, 2))`, "!np.rot90(np.arange(3))", `!np.rot90(${A}, axes=(0, 0))`,
  "np.bitwise_count(np.array([0, 1, 7, 255, -5, 2**40 + 3]))", "np.bitwise_count(np.array([[1, 2], [3, 4]]))", "np.bitwise_invert(np.array([0, 5]))", "!np.bitwise_count(np.array([1.5]))",
]);

numpy_cases("pad", [
  ...["constant", "edge", "reflect", "symmetric", "wrap", "mean", "maximum", "minimum", "median", "linear_ramp"].flatMap(m => [
    `np.pad(np.array([1, 2, 3, 4]), 2, mode='${m}')`, `np.pad(np.array([1, 2, 3, 4]), (1, 3), mode='${m}')`, `np.pad(${A}, 1, mode='${m}')`, `np.pad(${A}, ((1, 0), (2, 1)), mode='${m}')`, `np.pad(${F}, ((0, 2), (1, 1)), mode='${m}')`,
  ]),
  "np.pad(np.array([1, 2, 3]), 5, mode='reflect')", "np.pad(np.array([1, 2, 3]), 7, mode='symmetric')", "np.pad(np.array([1, 2, 3]), 7, mode='wrap')", "np.pad(np.array([5]), 3, mode='reflect')", "np.pad(np.array([5]), 3, mode='symmetric')",
  "np.pad(np.array([1, 2, 3, 4]), 2, mode='reflect', reflect_type='odd')", "np.pad(np.array([1, 2, 3, 4]), 2, mode='symmetric', reflect_type='odd')",
  "np.pad(np.array([1, 2, 3]), 2, constant_values=9)", "np.pad(np.array([1, 2, 3]), (1, 2), constant_values=(7, 8))", `np.pad(${A}, ((1, 1), (1, 1)), constant_values=((1, 2), (3, 4)))`, "np.pad(np.array([1.5, 2.5]), 1, constant_values=0.25)",
  "np.pad(np.array([1, 2, 3]), 3, mode='linear_ramp', end_values=9)", "np.pad(np.array([1.0, 2.0, 3.0]), (2, 3), mode='linear_ramp', end_values=(5, -4))", "np.pad(np.array([1, 2, 3, 10]), 2, mode='maximum', stat_length=2)", "np.pad(np.array([1, 2, 3, 10]), 2, mode='mean', stat_length=(1, 3))",
  "np.pad(np.array([1, 2]), 0)", `np.pad(${A}, (1, 2))`, "np.pad(np.array([1, 2]), [[1, 1]])", "np.pad(np.array([]), 2)", "!np.pad(np.array([1, 2]), -1)", "!np.pad(np.array([1, 2]), 1, mode='bogus')", "!np.pad(np.array([]), 2, mode='edge')",
  "np.pad(np.array([True, False]), 1)", "np.pad(np.array([1, 2], dtype='float32'), 1)",
]);

numpy_cases("copyto", [
  [() => { const d = np.zeros(3); np.copyto(d, np.array([1, 2, 3])); return d; }, "np.array([1.0, 2.0, 3.0])"],
  [() => { const d = np.zeros(3); np.copyto(d, 5); return d; }, "np.array([5.0, 5.0, 5.0])"],
  [() => { const d = np.arange(4); np.copyto(d, 9, 'same_kind', np.array([true, false, true, false])); return d; }, "np.array([9, 1, 9, 3])"],
  [() => { const d = np.zeros([2, 2]); np.copyto(d, np.array([1, 2]), 'same_kind', np.array([[true, false], [false, true]])); return d; }, "np.array([[1.0, 0.0], [0.0, 2.0]])"],
]);
