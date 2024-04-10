
export namespace Method_other_out {
  export type Kwargs = { other?: NDArray<any>, out?: NDArray<any> | null };
  export type Wrapper = (other: NDArray<any> | Kwargs, out?: NDArray<any> | null | Kwargs) => NDArray<any>;
}