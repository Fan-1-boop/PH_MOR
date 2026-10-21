"""Reproducible small-scale port-Hamiltonian Hankel/Schatten-p MOR experiments.
   Fully analytical continuous-Lyapunov adjoint gradients.
   SPD coordinate chart (log-Cholesky); optimized by L-BFGS.
   Only near-global behaviour is NOT claimed (nonconvex).
"""
import numpy as np
from numpy.linalg import eigvalsh, norm
from scipy.linalg import solve_continuous_lyapunov, cholesky, qr
from scipy.optimize import minimize, check_grad
import time, json, os, argparse

DIR=os.path.dirname(os.path.abspath(__file__))

def gramian(A,B,C):
    P=solve_continuous_lyapunov(A, -(B@B.T)); Q=solve_continuous_lyapunov(A.T,-(C.T@C))
    return (P+P.T)/2,(Q+Q.T)/2

def hsv(A,B,C):
    P,Q=gramian(A,B,C)
    lam=eigvalsh(cholesky(P+1e-14*np.eye(len(P)), lower=True).T@Q@cholesky(P+1e-14*np.eye(len(P)),lower=True))
    return np.sqrt(np.maximum(0,lam[::-1]))

def struct_rom(J,R,B):
    return J-R,B,B.T

def make_system(seed,n=12,m=2,case='oscillatory'):
    rng=np.random.default_rng(seed)
    # skew coupling includes diffusion-like local interactions and oscillatory modes
    J=np.zeros((n,n));
    for i in range(n//2):
        k=2*i; w=(0.35+0.75*i)/(n/2) if case=='slow' else (0.2+0.5*i)
        J[k,k+1]=w; J[k+1,k]=-w
    for i in range(n-1):
        v=rng.normal(scale=0.06)
        J[i,i+1]+=v;J[i+1,i]-=v
    if case=='slow':
        rr=np.geomspace(0.03,2.2,n)
    elif case=='cluster':
        rr=np.repeat(np.linspace(0.14,0.55,n//2),2)
    else:
        rr=np.geomspace(0.12,1.9,n)
    R=np.diag(rr)
    T=rng.normal(size=(n,m)); T/=np.sqrt(n)
    # ensure broad excitation of paired oscillator modes
    B= T + .23*np.sin(np.arange(n)[:,None]+np.arange(m)[None,:]); B/=1.6
    A=J-R
    return A,B,B.T,J,R

def baseline(A,B,r,kind='P'):
    P,Q=gramian(A,B,B.T)
    if kind=='P':
        _,U=np.linalg.eigh(P)
        V=U[:,-r:]
    elif kind=='H':
        # right singular controllability/observability modal projection projected into Euclidean pH coordinates
        L=cholesky(P+1e-13*np.eye(A.shape[0]),lower=True)
        T=L.T@Q@L
        v,us=np.linalg.eigh(T)
        V=qr(L@us[:,-r:],mode='economic')[0]
    else: raise ValueError(kind)
    Ar=V.T@A@V; Br=V.T@B
    J=(Ar-Ar.T)/2; R=-(Ar+Ar.T)/2
    return J,R,Br

def pack(J,R,B):
    r=len(J); ix=np.triu_indices(r,1)
    L=cholesky(R,lower=True)
    v=np.concatenate([J[ix], L[np.tril_indices(r)],B.flatten()])
    # diagonal Cholesky factors use log to enforce SPD
    d_indices=[i for i,(a,b) in enumerate(zip(*np.tril_indices(r))) if a==b]
    v[len(ix[0])+np.asarray(d_indices)]=np.log(v[len(ix[0])+np.asarray(d_indices)])
    return v

def unpack(z,r,m):
    ix=np.triu_indices(r,1); it=np.tril_indices(r); t=len(ix[0]); l=len(it[0])
    J=np.zeros((r,r));J[ix]=z[:t];J-=J.T
    L=np.zeros((r,r)); vals=z[t:t+l].copy()
    diagmask=(it[0]==it[1]); vals[diagmask]=np.exp(np.clip(vals[diagmask],-12,10))
    L[it]=vals
    R=L@L.T
    B=z[t+l:].reshape(r,m)
    return J,R,B,L,it,diagmask

def model_error(A,B,Ar,Br):
    n=len(A);r=len(Ar); m=B.shape[1]
    AE=np.zeros((n+r,n+r));AE[:n,:n]=A;AE[n:,n:]=Ar
    BE=np.vstack([B,Br]);CE=np.hstack([B.T,-Br.T])
    return AE,BE,CE

def norm_and_grad(A,B,z,r,p=2):
    # p even or 'h2' (H2 objective)
    m=B.shape[1];J,R,Br,L,it,dm=unpack(z,r,m)
    Ar=J-R; AE,BE,CE=model_error(A,B,Ar,Br)
    P,Q=gramian(AE,BE,CE)
    if p=='h2':
        # H2 squared = tr(C P C^T) ; derivatives from P adjoint
        s=float(np.trace(CE@P@CE.T))
        U=solve_continuous_lyapunov(AE.T,-(CE.T@CE)); U=(U+U.T)/2
        GA=2*U@P;GB=2*U@BE;GC=2*CE@P
        f=s
    else:
        k=p//2
        PQ=P@Q; QP=Q@P
        s=float(np.trace(np.linalg.matrix_power(PQ,k)))
        if not np.isfinite(s) or s<=0:
            raise FloatingPointError('Bad trace(power(PQ))')
        Dp=k*Q@np.linalg.matrix_power(PQ,k-1)
        Dq=k*P@np.linalg.matrix_power(QP,k-1)
        Dp=(Dp+Dp.T)/2;Dq=(Dq+Dq.T)/2
        U=solve_continuous_lyapunov(AE.T,-Dp);U=(U+U.T)/2
        W=solve_continuous_lyapunov(AE,-Dq);W=(W+W.T)/2
        GA=2*(U@P+Q@W);GB=2*U@BE;GC=2*CE@W
        f=s**(1./p)
        scale=f/(p*s)
        GA*=scale;GB*=scale;GC*=scale
    gAr=GA[len(A):,len(A):]
    gBr=GB[len(A):,:]-GC[:,len(A):].T
    # dA_r=dJ-dR; on skew J, only skew component counts
    gJ=gAr-gAr.T
    gR=-(gAr+gAr.T)/2
    gL=(gR+gR.T)@L
    ix=np.triu_indices(r,1)
    lgrad=gL[it].copy(); lgrad[dm]*=L.diagonal()
    return f,np.concatenate([gJ[ix],lgrad,gBr.flatten()])

def diagnostics(A,B,z,r):
    J,R,Br,*_=unpack(z,r,B.shape[1]); Ar=J-R
    AE,BE,CE=model_error(A,B,Ar,Br)
    ss=hsv(AE,BE,CE)
    # Numerically PSD of Hermitian part on imaginary axis
    omegas=np.geomspace(1e-3,1e3,60)
    mine=np.inf
    for w in omegas:
        G=Br.T@np.linalg.solve(1j*w*np.eye(r)-Ar,Br)
        mine=min(mine,eigvalsh((G+G.conj().T)/2).min())
    return dict(hankel=float(ss[0]),S2=float(norm(ss)),
        min_dissipation=float(eigvalsh(R)[0]),spectral_abscissa=float(np.max(np.real(np.linalg.eigvals(Ar)))),
        passive_grid_min=float(mine))

def optimize(A,B,r,p,z0,maxiter=120):
    start=time.perf_counter(); count=0
    def fun(z):
        nonlocal count
        count+=1
        return norm_and_grad(A,B,z,r,p)
    res=minimize(fun,z0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':2e-11,'gtol':2e-7,'maxls':28})
    return res,dict(iterations=int(res.nit),calls=count,seconds=float(time.perf_counter()-start),success=bool(res.success),message=str(res.message),stationarity_inf=float(np.max(np.abs(res.jac))))

def run_suite(large=False):
    jobs=[('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2),('oscillatory',7,12,3),('slow',13,12,3)]
    if large: jobs.extend([('oscillatory',17,14,3),('cluster',29,14,3)])
    data=[]
    for case,seed,n,r in jobs:
        A,B,C,J,R=make_system(seed,n,case=case);bound=hsv(A,B,C)[r]
        j0,r0,b0=baseline(A,B,r,'H')
        z0=pack(j0,r0,b0)
        f0=diagnostics(A,B,z0,r)
        tasks=[('projection',None,None,{'iterations':0,'calls':0,'seconds':0.})]
        # Independent HSH, H2, fixed-p and homotopy
        sols={'projection':z0};metas={'projection':{'iterations':0,'calls':0,'seconds':0.}}
        for label,p in [('H2','h2'),('HSH',2),('p16_direct',16)]:
            res,meta=optimize(A,B,r,p,z0,140)
            sols[label]=res.x;metas[label]=meta
        z=z0.copy(); ttotal=0; its=0
        for p in [2,4,8,16,32]:
            res,meta=optimize(A,B,r,p,z,110)
            z=res.x;ttotal+=meta['seconds'];its+=meta['iterations']
            sols['homotopy_p'+str(p)]=z.copy()
            metas['homotopy_p'+str(p)]=dict(meta,cum_seconds=ttotal,cum_iterations=its)
        for label,z in sols.items():
            d=diagnostics(A,B,z,r)
            row=dict(case=case,seed=seed,n=n,r=r,method=label,sigma_r1=float(bound),ratio=float(d['hankel']/bound),**d,**metas[label]);data.append(row)
        print(case,seed,n,r,'lower',round(bound,6), ' '.join(f'{name}:{diagnostics(A,B,sols[name],r)["hankel"]/bound:.5f}' for name in ['projection','H2','HSH','p16_direct','homotopy_p16','homotopy_p32']),flush=True)
    return data

def check():
    A,B,C,J,R=make_system(5,8); jj,rr,bb=baseline(A,B,2);z=pack(jj,rr,bb)
    g=[]
    for p in [2,4,8,16,'h2']:
        a=check_grad(lambda x:norm_and_grad(A,B,x,2,p)[0],lambda x:norm_and_grad(A,B,x,2,p)[1],z,epsilon=1e-6)
        grad=norm_and_grad(A,B,z,2,p)[1]
        a_relative=a/max(1.,norm(grad))
        g.append([str(p),float(a_relative)])
    return g

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--large',action='store_true');ar=ap.parse_args()
    grads=check();print('gradient checks',grads,flush=True)
    rows=run_suite(ar.large)
    with open(os.path.join(DIR,'results.json'),'w') as f:json.dump({'gradient_validation':grads,'experiments':rows},f,indent=2,ensure_ascii=False)
