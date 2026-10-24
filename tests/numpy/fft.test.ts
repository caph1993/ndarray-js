import { numpy_cases } from "../helpers/cases";

const X = "np.array([1.0, 2.0, 0.5, -1.0, 3.0, 2.5, 0.0, 1.5])";   // length 8
const Y = "np.array([1.0, 2.0, 0.5, -1.0, 3.0, 2.5, 0.0])";        // length 7
const C = "np.array([1+2j, 2-1j, 0.5j, -1, 3+0.5j])";
const M = "np.arange(24.0).reshape(2, 3, 4) ** 1.5";
const norms = ["", ", norm='ortho'", ", norm='forward'", ", norm='backward'"];

numpy_cases("1-d transforms", [
  ...["fft", "ifft", "rfft", "hfft", "ihfft"].flatMap(f => [X, Y, ...(f == "hfft" ? [] : ["np.array([2.0])"]), "np.array([1.0, 2.0])", "np.arange(12.0)"].flatMap(a => norms.map(n => `np.fft.${f}(${a}${n})`))),
  ...["fft", "ifft"].flatMap(f => [C, "np.array([1j, 2])"].flatMap(a => norms.map(n => `np.fft.${f}(${a}${n})`))),
  ...["irfft"].flatMap(f => [C, "np.array([1+0j, 2+1j, 3])"].flatMap(a => norms.map(n => `np.fft.${f}(${a}${n})`))),
  `np.fft.fft(${X}, n=4)`, `np.fft.fft(${X}, n=11)`, `np.fft.fft(${Y}, 16)`, `np.fft.ifft(${C}, n=8)`, `np.fft.rfft(${X}, n=5)`, `np.fft.rfft(${X}, n=16)`, `np.fft.irfft(${C}, n=7)`, `np.fft.irfft(${C}, n=12)`, `np.fft.irfft(${C}, 3)`,
  `np.fft.hfft(${C}, n=6)`, `np.fft.ihfft(${X}, n=5)`,
  `np.fft.fft(${M}, axis=0)`, `np.fft.fft(${M}, axis=1)`, `np.fft.fft(${M})`, `np.fft.ifft(${M}, axis=-2)`, `np.fft.rfft(${M}, axis=1)`, `np.fft.rfft(${M})`, `np.fft.irfft(np.fft.rfft(${M}, axis=1), n=3, axis=1)`,
  `np.fft.ifft(np.fft.fft(${Y}))`, `np.fft.irfft(np.fft.rfft(${Y}), n=7)`, `np.fft.irfft(np.fft.rfft(${X}))`,
  "np.fft.fft(np.arange(1000.0))", "np.fft.fft(np.arange(997.0))", "np.fft.ifft(np.fft.fft(np.arange(30.0)))", "np.fft.fft([1, 2, 3, 4])", "np.fft.fft(np.array([True, False, True]))", "np.fft.fft(np.array([1, 2, 3, 4], dtype='float32'))",
  "!np.fft.hfft(np.array([2.0]))", "!np.fft.irfft(np.array([4.0]))", "np.fft.irfft(np.array([4.0]), n=2)", "!np.fft.fft(np.array([1.0, 2.0]), n=0)", "!np.fft.fft(np.array([1.0]), norm='bogus')", "!np.fft.fft(5.0)", "!np.fft.rfft(np.array([1j, 2]))", "!np.fft.fft(np.array([]))",
], { rtol: 1e-8, atol: 1e-9 });

numpy_cases("n-d transforms", [
  ...["fft2", "ifft2", "fftn", "ifftn", "rfft2", "rfftn", "irfft2", "irfftn"].flatMap(f => [`np.fft.${f}(${M})`, `np.fft.${f}(${M}, axes=(0, 1))`, `np.fft.${f}(${M}, norm='ortho')`].filter(c => !(f.startsWith('irfft') && !c.includes('axes')))),
  `np.fft.fft2(${M}, s=(4, 2))`, `np.fft.fftn(${M}, s=(3, 5))`, `np.fft.fftn(${M}, s=(2, 2), axes=(0, 2))`, `np.fft.fft2(np.arange(12.0).reshape(3, 4))`, `np.fft.ifft2(np.fft.fft2(np.arange(12.0).reshape(3, 4)))`, `np.fft.rfft2(np.arange(12.0).reshape(3, 4))`,
  `np.fft.irfft2(np.fft.rfft2(np.arange(12.0).reshape(3, 4)), s=(3, 4))`, `np.fft.irfftn(np.fft.rfftn(${M}), s=(2, 3, 4))`, `np.fft.fftn(np.arange(8.0))`, "!np.fft.fft2(np.arange(8.0))",
], { rtol: 1e-8, atol: 1e-9 });

numpy_cases("fft helpers", [
  "np.fft.fftfreq(8)", "np.fft.fftfreq(7)", "np.fft.fftfreq(5, d=0.1)", "np.fft.fftfreq(1)", "np.fft.rfftfreq(8)", "np.fft.rfftfreq(7, 0.5)", "np.fft.rfftfreq(1)", "!np.fft.fftfreq(0)",
  "np.fft.fftshift(np.fft.fftfreq(8))", "np.fft.fftshift(np.fft.fftfreq(7))", "np.fft.ifftshift(np.fft.fftshift(np.arange(7)))", "np.fft.ifftshift(np.arange(8))", "np.fft.ifftshift(np.arange(7))",
  "np.fft.fftshift(np.arange(12).reshape(3, 4))", "np.fft.fftshift(np.arange(12).reshape(3, 4), axes=1)", "np.fft.fftshift(np.arange(12).reshape(3, 4), axes=0)", "np.fft.ifftshift(np.arange(12).reshape(3, 4), axes=(0, 1))", "np.fft.ifftshift(np.arange(30).reshape(5, 6))",
]);
