type npPrototype = typeof import(".").np;

export let np: npPrototype;

export function set_np(value) {
  np = value;
}