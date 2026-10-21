#!/usr/bin/env python3
"""Research audit: a strict feedthrough-preserving passive Hankel gap.

Requires numpy/scipy/matplotlib.  Reproduce with:
  OPENBLAS_NUM_THREADS=1 python reproduce.py

The model is collocated, strictly proper, 4th-order passive port-Hamiltonian.
We compare degree<=1 strictly proper passive (k >= 0), with arbitrary
stable degree<=1 (k in R).  All optimization results are independently
bounded by certified-in-exact-arithmetic spectral estimates.

The spectral result is an application of the standard variational principle,
NOT claimed to be a new mathematical theorem or a complete solver.
"""
import json, os, time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_sylvester, solve_continuous_lyapunov, eigh
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
np.set_printoptions(precision=9,suppress=True)


def make_model(par=None):
    if par is None: par=np.array([.5,2.2,2.4,.09,1.3,1.34])
    a1,w1,b1,a2,w2,b2=par
    J=np.zeros((4,4));J[0,1]=w1;J[1,0]=-w1;J[2,3]=w2;J[3,2]=-w2
    R=np.diag([a1,a1,a2,a2]);B=np.array([[b1],[0.],[b2],[0.]])
    A=J-R
    return A,B,J,R


def hsv_from_gramians(A,B):
    P=solve_continuous_lyapunov(A,-B@B.T)
    Q=solve_continuous_lyapunov(A.T,-B@B.T)
    l,U=eigh((P+P.T)/2)
    X=(U*np.sqrt(np.maximum(l,0.)))@U.T
    sig=np.sqrt(np.maximum(0.,eigh(X@Q@X,eigvals_only=True)))
    return sig[::-1]


def signed_hankel_spectrum(A,B):
    # Nonzero spectrum of the SISO, selfadjoint Hankel operator = spectrum
    # of the classical (possibly non-symmetric) cross Gramian W.
    W=solve_sylvester(A,A,-B@B.T)
    l=np.linalg.eigvals(W)
    if np.max(np.abs(l.imag))>2e-7:raise ArithmeticError('SISO signed Hankel eigenvalues unexpectedly non-real')
    return np.sort(l.real),W


def hnorm_error(A,B,alpha,k):
    n=len(A)
    Ae=np.zeros((n+1,n+1));Ae[:n,:n]=A;Ae[-1,-1]=-alpha
    Be=np.vstack((B,[[1.]]))
    Ce=np.hstack((B.T,[[-k]]))
    P=solve_continuous_lyapunov(Ae,-Be@Be.T)
    Q=solve_continuous_lyapunov(Ae.T,-Ce.T@Ce)
    l,U=eigh((P+P.T)/2)
    X=(U*np.sqrt(np.maximum(l,0)))@U.T
    return float(np.sqrt(max(0.,eigh(X@Q@X,eigvals_only=True)[-1])))


def compressed_hankel(A,B,shifts):
    """Ritz spectrum on span(e^{-z_j t}), z complex, Re(z)>0.
    Gram Sij=1/(conj(zi)+zj).
    Hij=B^T (conj(zi)I-A)^-1 (zj I-A)^-1 B.
    Ritz eigenvalues solve H v = lambda S v.
    Negative Ritz eigenvalues yield lower bounds for negative spectrum.
    """
    shifts=np.asarray(shifts,dtype=complex)
    n=len(A); z=len(shifts)
    S=1/(shifts.conj()[:,None]+shifts[None,:])
    H=np.empty((z,z),complex)
    eye=np.eye(n)
    for i,zi in enumerate(shifts):
        for j,zj in enumerate(shifts):
            H[i,j]=(B.T@np.linalg.solve(zi.conjugate()*eye-A,np.linalg.solve(zj*eye-A,B))).item()
    H=(H+H.conj().T)/2
    S=(S+S.conj().T)/2
    ev=eigh(H,S,eigvals_only=True)
    return np.sort(ev.real),float(np.linalg.cond(S))


def do_opt(A,B,sign):
    kbound=(0.,20.) if sign=='positive' else (-20.,20.)
    def obj(x):return hnorm_error(A,B,np.exp(x[0]),x[1])
    result=differential_evolution(obj,[(-7.,7.),kbound],seed=42,popsize=12,maxiter=350,tol=1e-9,polish=True)
    return dict(err=float(result.fun),alpha=float(np.exp(result.x[0])),k=float(result.x[1]),nfev=int(result.nfev))


def test_perturbations(base):
    rng=np.random.default_rng(321)
    rows=[]
    for amp in [.01,.025,.05,.10,.20]:
        ratio=[]
        for _ in range(400):
            q=base*(1+rng.uniform(-amp,amp,6));A,B,*_=make_model(q)
            val,_=signed_hankel_spectrum(A,B)
            singular=np.sort(abs(val))[::-1]
            ratio.append(-min(val)/singular[1])
        ratio=np.asarray(ratio)
        rows.append(dict(relative_perturbation=amp,strict_gap_frequency=float(np.mean(ratio>1+1e-6)),
                         ratio_median=float(np.median(ratio)),ratio_q05=float(np.quantile(ratio,.05)),
                         ratio_q95=float(np.quantile(ratio,.95))))
    return rows


def main():
    base=np.array([.5,2.2,2.4,.09,1.3,1.34])
    A,B,J,R=make_model(base)
    sigma=hsv_from_gramians(A,B)
    signed,W=signed_hankel_spectrum(A,B)
    neg=float(-signed[0]);hopt=float(sigma[1]);orig=float(sigma[0])
    assert max(abs(sigma-np.sort(abs(signed))[::-1]))<1e-9
    assert np.linalg.eigvalsh(R)[0]>.0
    assert neg>hopt
    shifts_all=np.array([.5+2.2j,.5-2.2j,.09+1.3j,.09-1.3j])
    plans={
        'real_12':np.geomspace(.01,20,12).astype(complex),
        'slow_pair_2':shifts_all[2:],
        'fast_pair_2':shifts_all[:2],
        'both_pairs_4':shifts_all,
    }
    cert=[]
    for name,shifts in plans.items():
        spec,cond=compressed_hankel(A,B,shifts)
        cert.append(dict(name=name,dimension=len(shifts),negative_Ritz=float(max(0,-spec[0])),
                         ratio_to_exact=float(max(0,-spec[0])/neg),gram_condition=cond,
                         Ritz_spectrum=[float(x) for x in spec]))
    uncon=do_opt(A,B,'unconstrained')
    passive=do_opt(A,B,'positive')
    assert abs(uncon['err']-hopt)<2e-5, (uncon,hopt)
    assert abs(passive['err']-neg)<2e-5, (passive,neg)
    direct_D_to_restore_PR=max(0.,-uncon['k']/uncon['alpha'])
    noise=test_perturbations(base)
    out={
      'description':'strictly proper stable passive SISO PH with two decoupled oscillators',
      'parameters':{'a1':base[0],'omega1':base[1],'b1':base[2],'a2':base[3],'omega2':base[4],'b2':base[5]},
      'A':A.tolist(),'B':B[:,0].tolist(),'skew_norm':float(np.linalg.norm(J+J.T)),
      'dissipation_smallest_eigen':float(np.linalg.eigvalsh(R)[0]),
      'signed_Hankel_eigenvalues':signed.tolist(),'HSV':sigma.tolist(),
      'unconstrained_Glover_lower_sigma2':hopt,
      'structure_aware_lower_bound_negative':neg,
      'relative_certified_gap_percent':100*(neg/hopt-1),
      'best_passive_degree_le_1_exact':orig,
      'signed_residue_candidate':uncon,
      'passive_optimization_check':passive,
      'D_needed_if_one_allows_feedthrough_to_repair_positive_realness':direct_D_to_restore_PR,
      'compressed_Ritz_certificates':cert,
      'robustness':noise,
    }
    with open(OUT/'results.json','w') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    # Chart 1: lower bound comparisons
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    names=['Unconstrained\nHNA bound','Signed-Hankel\nstructure bound','Passive r<=1\nexact optimum','Unconstrained\nr=1 witness']
    v=[hopt,neg,orig,uncon['err']]
    ax.bar(names,v)
    ax.set_ylim(0,neg*1.15);ax.set_ylabel('Hankel approximation error')
    ax.set_title('A strict structure-induced optimality gap')
    for i,x in enumerate(v):ax.text(i,x+0.035, f'{x:.6f}',ha='center',fontsize=10)
    ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUT/'certified_gap.png',dpi=160);plt.close(fig)
    # Chart 2: frequency-informed Ritz certificates (using shifted solves)
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    ax.bar([x['name'] for x in cert],[x['negative_Ritz'] for x in cert])
    ax.axhline(neg,label='Full signed-Hankel value',linestyle='--')
    ax.axhline(hopt,label='Unstructured HNA bound',linestyle=':')
    ax.set_ylabel('Numerical Ritz lower bound')
    ax.set_title('Subspace-aware certificates without full Gramians')
    ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUT/'ritz_certificates.png',dpi=160);plt.close(fig)
    # Chart 3 robust perturb
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    pp=[100*x['relative_perturbation'] for x in noise]
    ax.plot(pp,[100*x['strict_gap_frequency'] for x in noise],marker='o')
    ax.set_ylim(0,105);ax.set_xlabel('Independent relative coefficient perturbation (%)')
    ax.set_ylabel('Fraction with certified strict gap (%)')
    ax.set_title('Robustness over 400 samples per perturbation level')
    ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'perturbation_robustness.png',dpi=160);plt.close(fig)
    print(json.dumps({'HSV':sigma.tolist(),'signed':signed.tolist(),'gap_percent':out['relative_certified_gap_percent'],
       'opt_unconstrained':uncon,'opt_passive':passive,'D_repair':direct_D_to_restore_PR,
       'ritz':cert, 'robustness':noise},indent=2))

if __name__=='__main__':main()
