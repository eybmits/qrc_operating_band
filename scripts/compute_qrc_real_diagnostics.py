#!/usr/bin/env python3
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import math, time, json
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

ROOT=Path(__file__).resolve().parents[1]
QRC_PATH=Path(__file__).resolve().with_name('qrc_stateful_minimal_suite.py')
qrc=SourceFileLoader('qrcmod_real_diag', str(QRC_PATH)).load_module()
DATA=ROOT/'data'
OUT=DATA
OUT.mkdir(parents=True, exist_ok=True)
qgrid=pd.read_csv(DATA/'qrc_seed_ensemble_grid.csv')
# Unique current configs actually evaluated in the v6/v5 paper.
configs=qgrid[['seed','beta_pi','lambda_pi','gamma']].drop_duplicates().sort_values(['seed','beta_pi','lambda_pi','gamma']).reset_index(drop=True)
print('configs', len(configs), flush=True)
# Diagnostic driving sequence, same for all configs for comparability.
T=900
W=150
rng=np.random.default_rng(12345)
u=rng.uniform(-1.0,1.0,size=T).astype(float)
dummy_y=np.zeros(T)
task=qrc.TaskData('diag_iid_uniform', u, dummy_y, W, T-W, 0, 0)
Dmem_mc=10
Dmem_ipc=20
Dnonlin=10
pairs=[(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(1,5),(2,6),(5,10),(10,15)]
max_delay=max([Dmem_ipc,Dnonlin]+[max(p) for p in pairs])
# X rows correspond to t=W..T-1.
t_index=np.arange(W,T)
# only use rows where all delay targets exist.
valid=t_index>=max_delay
# train/test split on diagnostic features.
n_valid=int(valid.sum())
split=int(0.70*n_valid)
tr=slice(0,split)
te=slice(split,n_valid)

def fit_score(X, y, alpha=1e-8):
    # Train/test R^2 with intercept; no penalty on intercept.
    Xtr=X[tr]; Xte=X[te]
    ytr=y[tr]; yte=y[te]
    Xa=np.column_stack([Xtr, np.ones(len(Xtr))])
    A=Xa.T@Xa
    reg=np.eye(A.shape[0])*alpha
    reg[-1,-1]=0.0
    b=Xa.T@ytr
    try:
        w=np.linalg.solve(A+reg,b)
    except np.linalg.LinAlgError:
        w=np.linalg.pinv(A+reg)@b
    pred=np.column_stack([Xte, np.ones(len(Xte))])@w
    var=np.var(yte)
    if var < 1e-12:
        return 0.0
    r2=1.0 - np.mean((yte-pred)**2)/(var+1e-12)
    return float(max(0.0, r2))

def effective_rank(X):
    Xc=X - X.mean(axis=0, keepdims=True)
    C=(Xc.T@Xc)/max(1, len(Xc)-1)
    eig=np.linalg.eigvalsh(C)
    eig=np.clip(eig,0,None)
    s=eig.sum()
    if s <= 1e-14:
        return 0.0
    p=eig/s
    p=p[p>1e-15]
    return float(np.exp(-np.sum(p*np.log(p))))

def feature_targets(u, t_index):
    # Return standardized target arrays for capacity calculation.
    uu=u[t_index]
    return uu

rows=[]
t0=time.time()
uin_cache={}
for idx,row in configs.iterrows():
    seed=int(row.seed)
    beta_pi=float(row.beta_pi)
    lam_pi=float(row.lambda_pi)
    gamma=float(row.gamma)
    cfg=qrc.QRCConfig(n=4,layers=2,beta=beta_pi*math.pi,lam=lam_pi*math.pi,gamma=gamma,channel='amplitude',topology='ring',mixer='rx_zz',input_mode='uniform',seed=seed)
    Xz,Xzz=qrc.simulate_features(task,cfg,uin_cache=uin_cache)
    X=np.column_stack([Xz,Xzz])
    X=X[valid]
    # Sanitize numerics
    X=np.asarray(X,dtype=float)
    X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Feature diagnostics
    Vfeat=float(np.mean(np.std(X,axis=0)))
    reff=effective_rank(X)
    # Memory capacity score: linear delayed input, limited to 10 delays.
    MC=0.0
    IPCmem=0.0
    # Legendre-scaled uniform targets; scaling does not alter R^2 but makes interpretation standard.
    for d in range(1,Dmem_mc+1):
        y=u[t_index[valid]-d]
        MC += fit_score(X,y)
    for d in range(1,Dmem_ipc+1):
        y=np.sqrt(3.0)*u[t_index[valid]-d]
        IPCmem += fit_score(X,y)
    IPCnonlin=0.0
    # Second-order Legendre targets for individual delays.
    for d in range(1,Dnonlin+1):
        x=u[t_index[valid]-d]
        y=np.sqrt(5.0)*0.5*(3.0*x*x-1.0)
        IPCnonlin += fit_score(X,y)
    # Cross-delay products.
    for d1,d2 in pairs:
        x1=np.sqrt(3.0)*u[t_index[valid]-d1]
        x2=np.sqrt(3.0)*u[t_index[valid]-d2]
        y=x1*x2
        IPCnonlin += fit_score(X,y)
    IPCtot=IPCmem+IPCnonlin
    rows.append(dict(seed=seed,beta_pi=beta_pi,lambda_pi=lam_pi,gamma=gamma,MC=MC,IPCmem=IPCmem,IPCnonlin=IPCnonlin,IPCtot=IPCtot,Vfeat=Vfeat,reff=reff,n_features=X.shape[1],n_diag_samples=len(X)))
    if (idx+1)%100==0:
        print(f'{idx+1}/{len(configs)} elapsed {time.time()-t0:.1f}s', flush=True)

diag=pd.DataFrame(rows)
diag.to_csv(OUT/'qrc_real_current_intrinsic_diagnostics.csv',index=False)
# Merge with performance ranks.
qgrid['replicate']=qgrid.task+'__seed'+qgrid.seed.astype(str)
qgrid['val_rank_pct']=qgrid.groupby('replicate').val_nmse.rank(method='average',pct=True)
perf=qgrid.groupby(['seed','beta_pi','lambda_pi','gamma']).agg(mean_val_rank=('val_rank_pct','mean'), mean_val_nmse=('val_nmse','mean'), mean_test_nmse=('test_nmse','mean')).reset_index()
merged=diag.merge(perf,on=['seed','beta_pi','lambda_pi','gamma'],how='left')
merged.to_csv(OUT/'qrc_real_current_intrinsic_diagnostics_with_perf.csv',index=False)
# Spearman correlations.
from scipy.stats import spearmanr
metrics=['MC','IPCmem','IPCtot','IPCnonlin','Vfeat','reff']
corr=[]
for m in metrics:
    corr.append(dict(metric=m, spearman_vs_val_rank=float(spearmanr(merged[m], merged.mean_val_rank).correlation)))
pd.DataFrame(corr).to_csv(OUT/'qrc_real_current_diagnostic_spearman.csv',index=False)
print('done', time.time()-t0, flush=True)
print(pd.DataFrame(corr).to_string(index=False), flush=True)
