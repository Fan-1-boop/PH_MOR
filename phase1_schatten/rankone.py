import os,sys,json,numpy as np
sys.path.insert(0,os.path.dirname(__file__))
from experiment import *
rows=[]
for case in ['oscillatory','slow','cluster']:
 for seed in range(1,9):
  A,B,C,*_=make_system(100+seed,8,2,case)
  lo=hsv(A,B,C)[1];base=baseline(A,B,1,'H');z0=pack(*base)
  f0=diagnostics(A,B,z0,1)['hankel']/lo
  res,meta=optimize(A,B,1,32,z0,140)
  z=res.x
  best=diagnostics(A,B,z,1)['hankel']/lo
  rows.append(dict(case=case,seed=seed,n=8,r=1,project_ratio=f0,optimized_ratio=best,time=meta['seconds'],success=meta['success']))
 print(case,'range',min(x['optimized_ratio'] for x in rows if x['case']==case),max(x['optimized_ratio'] for x in rows if x['case']==case),flush=True)
with open(os.path.join(DIR,'rankone_results.json'),'w') as f:json.dump(rows,f,indent=2)
