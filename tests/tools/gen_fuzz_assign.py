# Generates random indexed-assignment cases (views, slices, ellipsis, one index list, in-place operators) for tests/numpy/fuzz_assign.test.ts:  python gen_fuzz_assign.py <seed> <count>
import json, random, sys
import numpy as np
seed=int(sys.argv[1]); N=int(sys.argv[2]); random.seed(seed)
ops=["=","+=","-=","*=","//=","%=","**=","|=","&=","^=","<<=",">>="]
def idxitem(n):
    k=random.random()
    if k<0.3: return str(random.randint(-n,n-1))
    if k<0.6:
        a=random.choice(["","1","-2","0","2"]); b=random.choice(["","2","-1","n","3"]); c=random.choice(["","","2","-1","-2"])
        b=b.replace("n",str(n)); 
        return f"{a}:{b}" + (f":{c}" if c else "")
    if k<0.7: return ":"
    if k<0.8: return "..."
    if k<0.9: return "[" + ", ".join(str(random.randint(-n,n-1)) for _ in range(random.randint(1,3))) + "]"
    return ":"
VIEWS=["", "", ".T", "[::2]", "[1:]", "[..., ::-1]", ".T[::-1]", ".swapaxes(0, -1)", "[::-1]", ".T[1:]", "[:, ::2]", ".T.T", "[::-1].T", ".reshape(-1)", ".ravel()", "[None]", ".T[..., None]"]
out=[]
while len(out)<N:
    nd=random.choice([1,2,2,3])
    shape=[random.randint(2,5) for _ in range(nd)]
    kind=random.choice(["int","int","float","bool"])
    size=int(np.prod(shape))
    base=f"np.arange({size}).reshape({tuple(shape) if nd>1 else shape[0]})" if nd>1 else f"np.arange({size})"
    if kind=="float": base=f"({base} / 4)"
    if kind=="bool": base=f"({base} % 3 == 0)"
    view=random.choice(VIEWS)
    try:
        vshape=list(eval(f"({base}.copy()){view}",{"np":np}).shape)
    except Exception: continue
    if len(vshape)==0: continue
    shape=vshape; nd=len(shape)
    nidx=random.randint(1,nd)
    items=[idxitem(shape[i]) for i in range(nidx)]
    # at most one advanced index (orthogonal indexing differs from numpy otherwise) and ellipsis at most once
    adv=[i for i,x in enumerate(items) if x.startswith("[")]
    for i in adv[1:]: items[i]=":"
    if items.count("...")>1:
        first=True
        for i,x in enumerate(items):
            if x=="..." :
                if first: first=False
                else: items[i]=":"
    idx=", ".join(items)
    op=random.choice(ops) if kind!="bool" else random.choice(["=","|=","&=","^="])
    if kind=="float" and op in ("|=","&=","^=","<<=",">>="): op="+="
    if op in ("//=","%=") : val=random.choice(["3","2","5"])
    elif op in ("<<=",">>="): val=random.choice(["1","2"])
    elif op=="**=": val=random.choice(["2","3"])
    elif kind=="bool": val=random.choice(["True","False","np.array([True, False])"])
    else: val=random.choice(["7","-3","2.5","np.arange(2)","np.array([10, 20])","np.array(100)","9"])
    if kind=="int" and "." in val and op!="=": val="3"
    if kind=="int" and op=="=" and "." in val: val="4"
    py_idx=f"({idx},)" if nidx>0 else idx
    stmt_py=f"(lambda x: (lambda v: (v.__setitem__(({idx},), {val}) if '{op}'=='=' else v.__setitem__(({idx},), v[({idx},)] {op[:-1]} {val}), x)[1])(x{view}))({base}.copy())"
    try:
        v=eval(stmt_py,{"np":np})
    except Exception as e:
        continue
    out.append({"base":base,"view":view,"idx":idx,"op":op,"val":val,"py":stmt_py})
json.dump(out,open(f'/tmp/fuzz_assign_{seed}.json','w')); print(len(out))
