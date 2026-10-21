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
