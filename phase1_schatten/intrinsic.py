"""Actual Riemannian steepest descent on Skew(r) x SPD(r) x R^(r*m).
  Uses induced Frobenius metric on Skew and Euclidean B, affine-invariant metric on SPD.
  Geodesic update for SPD, add update for Skew and B, Armijo backtracking.
  Research proof-of-principle, not a competitive RTR or Riemannian L-BFGS solver.
"""
import os,sys,json,time
sys.path.insert(0,os.path.dirname(__file__))
from experiment import *

def eval_and_grad(A,B,J,R,Br,p):
    r=len(J);AE,BE,CE=model_error(A,B,J-R,Br)
    P,Q=gramian(AE,BE,CE);k=p//2;PQ=P@Q;QP=Q@P
    s=float(np.trace(np.linalg.matrix_power(PQ,k)))
    f=s**(1./p)
    Dp=k*Q@np.linalg.matrix_power(PQ,k-1)
    Dq=k*P@np.linalg.matrix_power(QP,k-1)
    Dp=(Dp+Dp.T)/2;Dq=(Dq+Dq.T)/2
    U=solve_continuous_lyapunov(AE.T,-Dp)
    W=solve_continuous_lyapunov(AE,-Dq)
    U=(U+U.T)/2;W=(W+W.T)/2
    scale=f/(p*s)
    ga=scale*2*(U@P+Q@W)
    gb=scale*2*U@BE
    gc=scale*2*CE@W
    ga=ga[len(A):,len(A):]
    gJ=(ga-ga.T)/2
    gR=-(ga+ga.T)/2
    gB=gb[len(A):,:]-gc[:,len(A):].T
    return f,gJ,gR,gB

def root(R):
    w,U=np.linalg.eigh((R+R.T)/2)
    return (U*np.sqrt(w))@U.T,(U*(1/np.sqrt(w)))@U.T

def riemannian_descent(A,B,J,R,Br,p=16,iters=280):
    T=time.perf_counter();history=[];vals=[]
    for i in range(iters):
        try: f,gJ,gR,gB=eval_and_grad(A,B,J,R,Br,p)
        except Exception:break
        rt,irt=root(R)
        gr=R@gR@R
        # negative Riemannian gradient is dR=-R*gR*R (AI metric)
        dJ=-gJ;dR=-gr;dB=-gB
        s2=norm(gJ)**2+norm(rt@gR@rt)**2+norm(gB)**2
        history.append((i,float(f),float(s2)))
        if s2<1e-13: break
        tangent=irt@dR@irt; lw,lv=np.linalg.eigh((tangent+tangent.T)/2)
        t=1.0;accepted=False
        for line in range(28):
            W=lv@np.diag(np.exp(np.clip(t*lw,-50,50)))@lv.T
            Rn=rt@W@rt;Rn=(Rn+Rn.T)/2
            Jn=J+t*dJ;Bn=Br+t*dB
            try: fn,*_=eval_and_grad(A,B,Jn,Rn,Bn,p)
            except Exception:fn=np.inf
            if fn<=f-1e-4*t*s2:
                J,R,Br=Jn,Rn,Bn;accepted=True;break
            t*=.5
        if not accepted:break
    return (J,R,Br),history,time.perf_counter()-T

if __name__=='__main__':
    case,seed,n,r='oscillatory',5,10,2
    A,B,C,*_=make_system(seed,n,case=case)
    J,R,Br=baseline(A,B,r,'H')
    f0=diagnostics(A,B,pack(J,R,Br),r)
    (Jf,Rf,Bf),hist,secs=riemannian_descent(A,B,J,R,Br,p=16,iters=280)
    f1=diagnostics(A,B,pack(Jf,Rf,Bf),r)
    out=dict(case=case,seed=seed,n=n,r=r,initial=f0,optimized=f1,iterations=len(hist),time=secs,history=hist,riemannian_gradient_final=hist[-1][2])
    print('Genuine Riemannian update:',f0['hankel'],f1['hankel'],'iters',len(hist),'time',secs, 'stationary',hist[-1][2],flush=True)
    with open(os.path.join(DIR,'intrinsic_result.json'),'w') as f:json.dump(out,f,indent=2)
