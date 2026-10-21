"""Uniform dissipation floor sensitivity R=LL^T+rho I.
   The floor is a sufficient numerical-robustness constraint; it is NOT needed for passivity.
"""
import os,json
import numpy as np
import experiment as ex
ORIGINAL_UNPACK=ex.unpack
rows=[]
for case,seed,n,r in [('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2)]:
    A,B,C,*_=ex.make_system(seed,n,case=case);lo=ex.hsv(A,B,C)[r];J,R,Br=ex.baseline(A,B,r,'H')
    for rho in [0.0,0.005,0.01,0.02]:
        def unpack_with_floor(z,r,m):
            J,RR,B,L,it,dm=ORIGINAL_UNPACK(z,r,m)
            return J,RR+rho*np.eye(r),B,L,it,dm
        ex.unpack=unpack_with_floor
        z0=ex.pack(J,R-rho*np.eye(r),Br)
        res,meta=ex.optimize(A,B,r,32,z0,160)
        d=ex.diagnostics(A,B,res.x,r)
        rows.append(dict(case=case,seed=seed,rho=rho,ratio=d['hankel']/lo,min_R=d['min_dissipation'],spectral_abscissa=d['spectral_abscissa'],seconds=meta['seconds'],success=bool(res.success),message=str(res.message)))
    print(case,[(z['rho'],round(z['ratio'],6),round(z['min_R'],6),z['success']) for z in rows if z['seed']==seed],flush=True)
ex.unpack=ORIGINAL_UNPACK
with open(os.path.join(os.path.dirname(__file__),'margin_results.json'),'w') as f:json.dump(rows,f,indent=2)
