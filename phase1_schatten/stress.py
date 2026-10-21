"""Tests: high-p warm-start vs direct; perturbation starts, H-infinity/H2 tradeoff and certificate."""
import sys,time,json,os
sys.path.insert(0,os.path.dirname(__file__))
from experiment import *
from scipy.linalg import solve_continuous_lyapunov

def hinfty_grid(A,B,z,r):
    J,R,Br,*_=unpack(z,r,B.shape[1]);Ar=J-R
    w=np.concatenate([[0.],np.geomspace(1.e-3,4e2,900)])
    v=0.
    for x in w:
        E=B.T@np.linalg.solve(1j*x*np.eye(len(A))-A,B)-Br.T@np.linalg.solve(1j*x*np.eye(r)-Ar,Br)
        v=max(v,np.linalg.svd(E,compute_uv=False)[0])
    return float(v)

def h2score(A,B,z,r):
    J,R,Br,*_=unpack(z,r,B.shape[1]);AE,BE,CE=model_error(A,B,J-R,Br)
    P,_=gramian(AE,BE,CE)
    return float(np.sqrt(max(0,np.trace(CE@P@CE.T))))

jobs=[('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2),('oscillatory',7,12,3),('slow',13,12,3)]
rs=[]
for case,seed,n,r in jobs:
    A,B,C,J,R=make_system(seed,n,case=case); bound=hsv(A,B,C)[r]
    init=pack(*baseline(A,B,r,'H'))
    for rep in range(3):
        gen=np.random.default_rng(seed+1000*rep)
        z=init.copy() if rep==0 else init+gen.normal(scale=(.05 if rep==1 else .22),size=len(init))
        out_direct,meta_direct=optimize(A,B,r,32,z,150)
        zc=z.copy();hist=[];tm=0;nit=0
        for p in [2,4,8,16,32]:
            out,info=optimize(A,B,r,p,zc,100)
            zc=out.x.copy();tm+=info['seconds'];nit+=info['iterations'];hist.append(dict(p=p,iterations=info['iterations'],hankel=diagnostics(A,B,zc,r)['hankel']))
        for method,zout,sec,it in [('direct32',out_direct.x,meta_direct['seconds'],meta_direct['iterations']),('homotopy32',zc,tm,nit)]:
            d=diagnostics(A,B,zout,r)
            rs.append(dict(case=case,seed=seed,n=n,r=r,start=rep,method=method,ratio=float(d['hankel']/bound),hankel=d['hankel'],time=float(sec),iter=int(it),h2=h2score(A,B,zout,r),hinf_grid=hinfty_grid(A,B,zout,r),min_R=d['min_dissipation'],max_real_ev=d['spectral_abscissa'],trace=hist if method=='homotopy32' else []))
    print('robust',case,seed,' '.join(f'{z["start"]}:{z["method"]}={z["ratio"]:.5f}/{z["time"]:.2f}s' for z in rs if z['seed']==seed),flush=True)

with open(os.path.join(DIR,'stress_results.json'),'w') as f:json.dump(rs,f,indent=2)
