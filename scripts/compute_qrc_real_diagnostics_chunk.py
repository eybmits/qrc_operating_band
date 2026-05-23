#!/usr/bin/env python3
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import math, time, sys
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
QRC_PATH=Path(__file__).resolve().with_name('qrc_stateful_minimal_suite.py')
qrc=SourceFileLoader('qrcmod_real_diag_chunk', str(QRC_PATH)).load_module()
DATA=ROOT/'data'
PARTS=DATA/'diag_parts'
PARTS.mkdir(parents=True, exist_ok=True)
qgrid=pd.read_csv(DATA/'qrc_seed_ensemble_grid.csv')
configs=qgrid[['seed','beta_pi','lambda_pi','gamma']].drop_duplicates().sort_values(['seed','beta_pi','lambda_pi','gamma']).reset_index(drop=True)
start=int(sys.argv[1]) if len(sys.argv)>1 else 0
end=int(sys.argv[2]) if len(sys.argv)>2 else len(configs)
end=min(end,len(configs))
configs=configs.iloc[start:end].reset_index(drop=True)
print(f'chunk {start}:{end} n={len(configs)}', flush=True)
T=900; W=150
rng=np.random.default_rng(12345)
u=rng.uniform(-1.0,1.0,size=T).astype(float)
y_dummy=np.zeros(T)
task=qrc.TaskData('diag_iid_uniform',u,y_dummy,W,T-W,0,0)
Dmem_mc=10; Dmem_ipc=20; Dnonlin=10
pairs=[(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(1,5),(2,6),(5,10),(10,15)]
max_delay=max([Dmem_ipc,Dnonlin]+[max(p) for p in pairs])
t_index=np.arange(W,T)
valid=t_index>=max_delay
n_valid=int(valid.sum()); split=int(0.70*n_valid); tr=slice(0,split); te=slice(split,n_valid)

def fit_score(X,y,alpha=1e-8):
    Xtr=X[tr]; Xte=X[te]; ytr=y[tr]; yte=y[te]
    Xa=np.column_stack([Xtr,np.ones(len(Xtr))])
    A=Xa.T@Xa; reg=np.eye(A.shape[0])*alpha; reg[-1,-1]=0.0; b=Xa.T@ytr
    try: w=np.linalg.solve(A+reg,b)
    except np.linalg.LinAlgError: w=np.linalg.pinv(A+reg)@b
    pred=np.column_stack([Xte,np.ones(len(Xte))])@w
    var=np.var(yte)
    if var<1e-12: return 0.0
    return float(max(0.0,1.0-np.mean((yte-pred)**2)/(var+1e-12)))

def effective_rank(X):
    Xc=X-X.mean(axis=0,keepdims=True)
    C=(Xc.T@Xc)/max(1,len(Xc)-1)
    eig=np.clip(np.linalg.eigvalsh(C),0,None)
    s=eig.sum()
    if s<=1e-14: return 0.0
    p=eig/s; p=p[p>1e-15]
    return float(np.exp(-np.sum(p*np.log(p))))

rows=[]; t0=time.time(); uin_cache={}
for local_idx,row in configs.iterrows():
    seed=int(row.seed); beta_pi=float(row.beta_pi); lam_pi=float(row.lambda_pi); gamma=float(row.gamma)
    cfg=qrc.QRCConfig(n=4,layers=2,beta=beta_pi*math.pi,lam=lam_pi*math.pi,gamma=gamma,channel='amplitude',topology='ring',mixer='rx_zz',input_mode='uniform',seed=seed)
    Xz,Xzz=qrc.simulate_features(task,cfg,uin_cache=uin_cache)
    X=np.column_stack([Xz,Xzz])[valid]
    X=np.nan_to_num(np.asarray(X,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    Vfeat=float(np.mean(np.std(X,axis=0)))
    reff=effective_rank(X)
    MC=0.0; IPCmem=0.0; IPCnonlin=0.0
    for d in range(1,Dmem_mc+1):
        MC += fit_score(X,u[t_index[valid]-d])
    for d in range(1,Dmem_ipc+1):
        IPCmem += fit_score(X,np.sqrt(3.0)*u[t_index[valid]-d])
    for d in range(1,Dnonlin+1):
        x=u[t_index[valid]-d]
        IPCnonlin += fit_score(X,np.sqrt(5.0)*0.5*(3.0*x*x-1.0))
    for d1,d2 in pairs:
        y=(np.sqrt(3.0)*u[t_index[valid]-d1])*(np.sqrt(3.0)*u[t_index[valid]-d2])
        IPCnonlin += fit_score(X,y)
    rows.append(dict(seed=seed,beta_pi=beta_pi,lambda_pi=lam_pi,gamma=gamma,MC=MC,IPCmem=IPCmem,IPCnonlin=IPCnonlin,IPCtot=IPCmem+IPCnonlin,Vfeat=Vfeat,reff=reff,n_features=X.shape[1],n_diag_samples=len(X)))
    if (local_idx+1)%100==0:
        print(f' local {local_idx+1}/{len(configs)} elapsed {time.time()-t0:.1f}s', flush=True)
out=pd.DataFrame(rows)
path=PARTS/f'part_{start:04d}_{end:04d}.csv'
out.to_csv(path,index=False)
print('wrote',path,'elapsed',time.time()-t0, flush=True)
