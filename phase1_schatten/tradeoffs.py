import os,json,sys
import numpy as np
from experiment import *
from metrics import hinfty_grid,h2score
rows=[]
for case,seed,n,r in [('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2),('oscillatory',7,12,3),('slow',13,12,3)]:
  A,B,C,*_=make_system(seed,n,case=case);lb=hsv(A,B,C)[r]
  z0=pack(*baseline(A,B,r,'H'))
  for method,p in [('projection',None),('H2','h2'),('HSH',2),('p32',32)]:
    res=None if p is None else optimize(A,B,r,p,z0,150)[0]
    z=z0 if res is None else res.x
    dg=diagnostics(A,B,z,r)
    rows.append(dict(case=case,seed=seed,n=n,r=r,method=method,hankel_ratio=float(dg['hankel']/lb),H2=h2score(A,B,z,r),Hinf_grid=hinfty_grid(A,B,z,r)))
  print(case,seed,' '.join(str((q['method'],round(q['hankel_ratio'],3),round(q['H2'],3),round(q['Hinf_grid'],3))) for q in rows if q['seed']==seed),flush=True)
with open(os.path.join(os.path.dirname(__file__),'tradeoffs.json'),'w') as f: json.dump(rows,f,indent=2)
