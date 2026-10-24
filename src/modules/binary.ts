//@ts-check
/** Conversion between arrays and bytes (little endian), used by tobytes, frombuffer, save, load, fromfile... */
import { NDArray } from "../NDArray";
import { DType, TYPE_BY_NAME, new_buffer, fix_buffer, fround16, canonical_dtype } from "../dtypes";
import { Complex } from "../complex";

export function itemsize(dtype: DType): number {
  const t = TYPE_BY_NAME[dtype];
  if (t.kind === 'O') throw new Error("object arrays can not be converted to bytes");
  return t.bits / 8;
}

/** float16 bits to number */
export function half_to_double(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * Math.pow(2, -24);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);
}
/** number to float16 bits */
export function double_to_half(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = (x < 0 || Object.is(x, -0)) ? 0x8000 : 0;
  const v = fround16(Math.abs(x));
  if (v === Infinity) return sign | 0x7c00;
  if (v === 0) return sign;
  let e = Math.floor(Math.log2(v));
  if (e < -14) return sign | Math.round(v / Math.pow(2, -24));
  return sign | ((e + 15) << 10) | Math.round((v / Math.pow(2, e) - 1) * 1024);
}

export function to_bytes(a: NDArray): Uint8Array {
  const dtype = a.dtype;
  const n = a.size, sz = itemsize(dtype);
  const out = new Uint8Array(n * sz);
  const dv = new DataView(out.buffer);
  const f = a.flat as any;
  const t = TYPE_BY_NAME[dtype];
  const big = (v: number, signed: boolean) => {
    const B = (globalThis as any).BigInt;
    return signed ? B.asIntN(64, B(Math.trunc(v))) : B.asUintN(64, B(Math.trunc(v)));
  };
  for (let i = 0; i < n; i++) {
    const o = i * sz;
    switch (dtype) {
      case 'bool': dv.setUint8(o, f[i] ? 1 : 0); break;
      case 'int8': dv.setInt8(o, f[i]); break;
      case 'uint8': dv.setUint8(o, f[i]); break;
      case 'int16': dv.setInt16(o, f[i], true); break;
      case 'uint16': dv.setUint16(o, f[i], true); break;
      case 'int32': dv.setInt32(o, f[i], true); break;
      case 'uint32': dv.setUint32(o, f[i], true); break;
      case 'int64': dv.setBigInt64(o, big(f[i], true), true); break;
      case 'uint64': dv.setBigUint64(o, big(f[i], false), true); break;
      case 'float16': dv.setUint16(o, double_to_half(f[i]), true); break;
      case 'float32': dv.setFloat32(o, f[i], true); break;
      case 'float64': dv.setFloat64(o, f[i], true); break;
      case 'complex64': dv.setFloat32(o, f[i].re, true); dv.setFloat32(o + 4, f[i].im, true); break;
      case 'complex128': dv.setFloat64(o, f[i].re, true); dv.setFloat64(o + 8, f[i].im, true); break;
      default: throw new Error(`cannot convert dtype ${dtype} to bytes`);
    }
  }
  return out;
}

export function from_bytes(bytes: Uint8Array, dtype: DType, count: number = -1, offset: number = 0, little: boolean = true): any {
  const sz = itemsize(dtype);
  const avail = Math.floor((bytes.byteLength - offset) / sz);
  if (offset < 0 || offset > bytes.byteLength) throw new Error(`offset must be non-negative and no greater than buffer length (${bytes.byteLength})`);
  if (count >= 0 && count > avail) throw new Error(`buffer is smaller than requested size`);
  if (count < 0 && (bytes.byteLength - offset) % sz !== 0) throw new Error(`buffer size must be a multiple of element size`);
  const n = count >= 0 ? count : avail;
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, n * sz);
  const out = new_buffer(n, dtype);
  for (let i = 0; i < n; i++) {
    const o = i * sz;
    switch (dtype) {
      case 'bool': out[i] = dv.getUint8(o) ? 1 : 0; break;
      case 'int8': out[i] = dv.getInt8(o); break;
      case 'uint8': out[i] = dv.getUint8(o); break;
      case 'int16': out[i] = dv.getInt16(o, little); break;
      case 'uint16': out[i] = dv.getUint16(o, little); break;
      case 'int32': out[i] = dv.getInt32(o, little); break;
      case 'uint32': out[i] = dv.getUint32(o, little); break;
      case 'int64': out[i] = Number(dv.getBigInt64(o, little)); break;
      case 'uint64': out[i] = Number(dv.getBigUint64(o, little)); break;
      case 'float16': out[i] = half_to_double(dv.getUint16(o, little)); break;
      case 'float32': out[i] = dv.getFloat32(o, little); break;
      case 'float64': out[i] = dv.getFloat64(o, little); break;
      case 'complex64': out[i] = new Complex(dv.getFloat32(o, little), dv.getFloat32(o + 4, little)); break;
      case 'complex128': out[i] = new Complex(dv.getFloat64(o, little), dv.getFloat64(o + 8, little)); break;
      default: throw new Error(`cannot read dtype ${dtype} from bytes`);
    }
  }
  return out;
}

/** Bytes of any buffer-like object */
export function as_bytes(buffer: any): Uint8Array {
  if (buffer instanceof Uint8Array) return buffer;
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  if (ArrayBuffer.isView(buffer)) return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (typeof buffer === "string") return new TextEncoder().encode(buffer);
  throw new TypeError(`a bytes-like object is required, not '${typeof buffer}'`);
}
