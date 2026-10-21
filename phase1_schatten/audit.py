import os,sys,json
sys.path.insert(0,os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import *
from metrics import hinfty_grid,h2score
from intrinsic import riemannian_descent
from scipy.linalg import cholesky
from scipy.optimize._numdiff import approx_derivative

DIR=os.path.dirname(__file__)
rows=json.load(open(os.path.join(DIR,'results.json')))['experiments']
stress=json.load(open(os.path.join(DIR,'stress_results.json')))
name=lambda z: z['case']+f" (n={z['n']},r={z['r']})"
labels=list(dict.fromkeys(name(x) for x in rows))
methods=['projection','H2','HSH','homotopy_p4','homotopy_p8','homotopy_p16','homotopy_p32']

plt.figure(figsize=(11,5.2))
xx=np.arange(len(labels));wd=.15
for i,m in enumerate(['projection','H2','HSH','homotopy_p16','homotopy_p32']):
 z=[next(v['ratio'] for v in rows if name(v)==lab and v['method']==m) for lab in labels]
 plt.bar(xx+(i-2)*wd,z,width=wd,label=m)
plt.xticks(xx,labels,rotation=16,ha='right');plt.ylabel('Hankel error / unstructured lower bound');plt.axhline(1,linestyle='--',linewidth=1);plt.legend(ncol=5,fontsize=8);plt.tight_layout();plt.savefig(os.path.join(DIR,'hankel_ratio_methods.png'),dpi=190);plt.close()

plt.figure(figsize=(9,5.3))
for lab in labels:
 z=[next(v['ratio'] for v in rows if name(v)==lab and v['method']==m) for m in ['HSH','homotopy_p4','homotopy_p8','homotopy_p16','homotopy_p32']]
 plt.plot([2,4,8,16,32],z,marker='o',label=lab)
plt.xscale('log',base=2);plt.xticks([2,4,8,16,32],['2','4','8','16','32']);plt.ylabel('True Hankel / HSV lower bound');plt.xlabel('Optimized Schatten p');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(os.path.join(DIR,'p_effect.png'),dpi=190);plt.close()

paired=[]
for case,seed,n,r in [('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2),('oscillatory',7,12,3),('slow',13,12,3)]:
 z1=[z for z in stress if z['seed']==seed and z['method']=='direct32']
 z2=[z for z in stress if z['seed']==seed and z['method']=='homotopy32']
 paired.extend([dict(case=case,seed=seed,start=a['start'],ratio_diff=b['ratio']-a['ratio'],time_direct=a['time'],time_homotopy=b['time']) for a,b in zip(z1,z2)])
plt.figure(figsize=(9,5))
plt.scatter([x['time_direct'] for x in paired],[x['time_homotopy'] for x in paired])
m=max(max(x['time_direct'],x['time_homotopy']) for x in paired)
plt.plot([0,m],[0,m],linestyle='--');plt.xlabel('Direct p=32: seconds');plt.ylabel('2→4→8→16→32: seconds');plt.title('Matched initializations; 15 pairs');plt.tight_layout();plt.savefig(os.path.join(DIR,'homotopy_time.png'),dpi=190);plt.close()

# accurate independent central diff check
A,B,*_=make_system(5,8); a=baseline(A,B,2,'H');z=pack(*a)
diffchecks=[]
for p in [2,4,8,16,32,'h2']:
 ref=approx_derivative(lambda v:norm_and_grad(A,B,v,2,p)[0],z,method='3-point',rel_step=2e-6).reshape(-1)
 analytic=norm_and_grad(A,B,z,2,p)[1]
 diffchecks.append({'p':str(p),'rel_grad_err':float(norm(ref-analytic)/max(norm(ref),1e-14))})

# check Schatten powered trace numerics versus gramian eigen-svd
trace_checks=[]
for case,seed,n,r in [('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2),('oscillatory',7,12,3),('slow',13,12,3)]:
 A,B,C,*_=make_system(seed,n,case=case);J,R,Br=baseline(A,B,r,'H')
 z=pack(J,R,Br)
 for p in [2,8,16,32]:
  v=norm_and_grad(A,B,z,r,p)[0]
  AE,BE,CE=model_error(A,B,J-R,Br);ss=hsv(AE,BE,CE)
  v2=float(np.linalg.norm(ss,ord=p));trace_checks.append(float(abs(v-v2)/v2))

# test product manifold intrinsic on 3 cases
intrinsic=[]
for case,seed,n,r in [('oscillatory',5,10,2),('slow',11,10,2),('cluster',23,10,2)]:
 A,B,*_=make_system(seed,n,case=case); J,R,Br=baseline(A,B,r,'H')
 zinit=pack(J,R,Br);h0=diagnostics(A,B,zinit,r)['hankel']
 (Jf,Rf,Bf),hist,secs=riemannian_descent(A,B,J,R,Br,p=16,iters=400)
 hh=diagnostics(A,B,pack(Jf,Rf,Bf),r)
 intrinsic.append(dict(case=case,seed=seed,initial_h=h0,optimized_h=hh['hankel'],iterations=len(hist),final_grad_norm_sq=hist[-1][2],time=secs,stable=hh['spectral_abscissa']<0,passivity_grid=hh['passive_grid_min']))

summary={'paired':paired,'central_gradient_checks':diffchecks,'max_relative_power_trace_mismatch':max(trace_checks),'intrinsic_checks':intrinsic}
with open(os.path.join(DIR,'audits.json'),'w') as f:json.dump(summary,f,indent=2)
print('central gradients',diffchecks,flush=True)
print('max trace_svd mismatch',summary['max_relative_power_trace_mismatch'],flush=True)
print('intrinsic',intrinsic,flush=True)
