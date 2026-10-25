# Generates random numpy expression chains for tests/numpy/fuzz.test.ts:  python gen_fuzz.py <seed> <count>  (writes /tmp/probe_fuzz_<seed>.json)
import json, random, sys
import numpy as np
seed=int(sys.argv[1]); N=int(sys.argv[2]); random.seed(seed)
def base():
    nd=random.choice([1,2,2,3,3,4])
    shape=[random.randint(1,4) for _ in range(nd)]
    size=int(np.prod(shape))
    kind=random.choice(["i","i","f","f","b","u8","neg"])
    r=f"np.arange({size}).reshape({tuple(shape) if nd>1 else shape[0]})"
    if nd==1: r=f"np.arange({size})"
    a=random.randint(1,7); m=random.randint(3,11)
    if kind=="i": return f"({r} * {a} % {m})"
    if kind=="neg": return f"({r} * {a} % {m} - {m//2})"
    if kind=="f": return f"(({r} * {a} % {m} - {m//2}) / 3)"
    if kind=="b": return f"({r} % {m} > {m//2})"
    return f"({r} * {a} % {m}).astype('uint8')"
def rax(nd): return random.randint(-nd,nd-1)
def step(X, v):
    nd=v.ndim; sh=v.shape
    c=[]
    c += [f"np.negative({X})" if v.dtype!=bool else f"np.logical_not({X})", f"np.abs({X})", f"({X}) * 2", f"({X}) + 1.5", f"({X}) - 3", f"({X}) ** 2" if v.dtype!=bool else f"({X}) & True", f"np.square({X})", f"({X}) // 3" if v.dtype!=bool else f"({X}) | False", f"({X}) % 5" if v.dtype!=bool else f"({X}) ^ True",
          f"np.sqrt(np.abs({X}))", f"np.exp(({X}) / 10)", f"np.floor(({X}) / 2)", f"np.sign({X})", f"np.maximum({X}, 2)", f"np.minimum({X}, -1)", f"np.clip({X}, 1, 4)", f"({X}).astype('float32')", f"({X}).astype('int64')", f"({X}) > 2", f"np.isfinite({X})", f"np.round(({X}) / 3, 1)", f"np.where(({X}) > 2, {X}, -({X}))" if v.dtype!=bool else f"np.where({X}, 1, 2)"]
    if nd>0 and v.size>0:
        a=rax(nd)
        c += [f"np.sum({X}, axis={a})", f"np.sum({X}, axis={a}, keepdims=True)", f"np.mean({X}, axis={a})", f"np.max({X}, axis={a})", f"np.min({X}, axis={a})", f"np.cumsum({X}, axis={a})", f"np.argmax({X}, axis={a})", f"np.std({X}, axis={a})", f"np.median({X}, axis={a})", f"np.any({X}, axis={a})", f"np.all({X}, axis={a})", f"np.sort({X}, axis={a})", f"np.argsort({X}, axis={a}, kind='stable')", f"np.flip({X}, axis={a})", f"np.roll({X}, 1, axis={a})", f"np.diff({X}, axis={a})", f"np.concatenate([{X}, {X}], axis={a})", f"np.stack([{X}, {X}], axis={a})", f"np.repeat({X}, 2, axis={a})", f"np.take({X}, [0, -1], axis={a})", f"np.cumprod(np.clip({X}, 0, 2), axis={a})", f"np.prod(np.clip({X}, 0, 2), axis={a})", f"np.var({X}, axis={a}, ddof=1)", f"np.expand_dims({X}, {a})", f"np.cummax({X}, axis={a})" if False else f"np.maximum.accumulate({X}, axis={a})", f"np.add.reduce({X}, axis={a})", f"np.percentile({X}, 30, axis={a})", f"np.ptp({X}, axis={a})" if v.dtype!=bool else f"np.sum({X})"]
        c += [f"({X}).T", f"({X}).reshape(-1)", f"({X})[::2]", f"({X})[1:]", f"({X})[..., 0]", f"({X})[None]", f"({X})[..., ::-1]", f"np.squeeze({X})", f"np.tile({X}, 2)", f"np.ravel({X})", f"np.sum({X})", f"np.mean({X})", f"np.max({X})", f"np.unique({X})", f"np.cumsum({X})", f"np.sort({X}, axis=None)", f"np.nonzero({X})[0]" if True else "", f"np.argwhere({X} > 2)"]
        if nd>=2:
            perm=list(range(nd)); random.shuffle(perm)
            c += [f"np.transpose({X}, {tuple(perm)})", f"np.swapaxes({X}, 0, -1)", f"np.moveaxis({X}, 0, -1)", f"({X}).reshape({sh[0]}, -1)", f"np.triu({X})", f"({X}) + ({X}).T.T", f"np.trace({X})", f"np.diagonal({X})", f"np.dot({X}, ({X}).T)" if nd==2 else f"np.sum({X}, axis=(0, 1))", f"np.sum({X}, axis=(0, -1))", f"({X})[0]", f"({X})[:, 0]", f"({X})[-1, ...]"]
        if nd>=1: c += [f"({X}) + np.arange({sh[-1]})", f"({X}) * np.arange({sh[0]}).reshape({(sh[0],)+(1,)*(nd-1)})" if nd>1 else f"({X}) * np.arange({sh[0]})", f"np.maximum({X}, ({X})[::-1])", f"({X}) - ({X}).mean(axis=0)"]
    return c
out=[]
tries=0
while len(out)<N:
    X=base()
    try: v=eval(X.replace("np.","np."), {"np":np})
    except Exception: continue
    depth=random.randint(1,4)
    expr=X; ok=True
    for _ in range(depth):
        cands=step(expr, v); random.shuffle(cands)
        for c in cands:
            try:
                nv=eval(c, {"np":np})
            except Exception: continue
            nv=np.asarray(nv)
            if nv.size>400 or nv.ndim>5: continue
            expr=c; v=nv; break
        else: break
    out.append(expr)
json.dump(out, open(f'/tmp/probe_fuzz_{seed}.json','w')); print(len(out), out[:3])
