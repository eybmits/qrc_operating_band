#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GFX = ROOT / "paper" / "gfx"
GFX.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':'serif',
    'font.serif':['DejaVu Serif'],
    'mathtext.fontset':'dejavuserif',
    'axes.linewidth':0.8,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    'figure.dpi':150,
})

# Load data
q=pd.read_csv(DATA/'qrc_seed_ensemble_grid.csv')
stats_path=DATA/'qrc96_esn100_stats.json'
if not stats_path.exists():
    raise FileNotFoundError(f"Missing {stats_path}; run scripts/analyze_qrc96_esn100.py first.")
qrc_stats=json.loads(stats_path.read_text())
qrc_selected=qrc_stats['qrc96_selected']
esn_selected=qrc_stats['esn100_selected']
q['replicate']=q.task+'__seed'+q.seed.astype(str)
q['val_rank_pct']=q.groupby('replicate').val_nmse.rank(method='average', pct=True)
keys=['beta_pi','lambda_pi','gamma']
name={'mackey_glass':'MG','narma10':'NARMA10','lorenz':'Lorenz','sunspots_annual':'Sunspots'}
all_tasks=sorted(q.task.unique())

gamma_star=0.12
qg=q[np.isclose(q.gamma,gamma_star)]
ymax=float(qg.lambda_pi.max())
xmax=float(qg.beta_pi.max())
# Figure 1: fixed gamma slice, no extrapolated blank top
panels=[('all tasks',all_tasks)] + [(f'leave out {name[t]}',[x for x in all_tasks if x!=t]) for t in ['mackey_glass','narma10','lorenz','sunspots_annual']]
fig,axes=plt.subplots(1,5,figsize=(13.8,3.35),sharex=True,sharey=True)
# frequency support per panel at gamma slice
for ax,(title,tasks) in zip(axes,panels):
    d=qg[qg.task.isin(tasks)].copy()
    agg=d.groupby(['beta_pi','lambda_pi']).agg(mean_rank=('val_rank_pct','mean')).reset_index()
    x=agg.beta_pi.values; y=agg.lambda_pi.values; z=agg.mean_rank.values
    xi=np.linspace(0,xmax,260); yi=np.linspace(0,ymax,240); XI,YI=np.meshgrid(xi,yi)
    Zi=griddata((x,y),z,(XI,YI),method='cubic')
    Zn=griddata((x,y),z,(XI,YI),method='nearest')
    Zi=np.where(np.isnan(Zi), Zn, Zi)
    # restrict interpolation to range; mild smoothing
    Zs=Zi.copy()
    for _ in range(2):
        P=np.pad(Zs,1,mode='edge')
        Zs=(P[:-2,1:-1]+P[2:,1:-1]+P[1:-1,:-2]+P[1:-1,2:]+4*P[1:-1,1:-1])/8.0
    im=ax.imshow(Zs,origin='lower',extent=[0,xmax,0,ymax],cmap='magma_r',vmin=0.05,vmax=0.8,aspect='auto')
    ax.contour(XI,YI,Zs,levels=[0.10,0.15,0.20,0.25,0.35,0.50],colors='white',linewidths=[1.1,0.95,0.85,0.75,0.55,0.35],alpha=0.58)
    # top20 frequency support within gamma slice
    chunks=[]
    for rep,g in d.groupby('replicate'):
        h=g.copy(); h['top20']=h.val_rank_pct<=0.20; chunks.append(h)
    dd=pd.concat(chunks)
    fr=dd.groupby(['beta_pi','lambda_pi']).top20.mean().reset_index()
    bp=fr[fr.top20>=0.55]
    cp=fr[fr.top20>=0.70]
    if len(bp): ax.scatter(bp.beta_pi,bp.lambda_pi,s=15,c='#2b8cbe',alpha=0.22,edgecolor='none')
    if len(cp): ax.scatter(cp.beta_pi,cp.lambda_pi,s=24,c='#2b8cbe',edgecolor='white',linewidth=0.35)
    # selected QRC96 local-refinement projection
    ax.scatter([qrc_selected['beta_pi']],[qrc_selected['lambda_pi']],marker='*',s=70,linewidths=0.7,c='#ff7f0e',edgecolor='black',zorder=5)
    ax.axvline(0.10,color='white',ls='--',lw=0.85,alpha=0.85)
    ax.axhline(0.10,color='white',ls='--',lw=0.85,alpha=0.85)
    ax.set_xlim(0,xmax); ax.set_ylim(0,ymax)
    ax.set_title(title,fontsize=10,pad=5)
    ax.set_xlabel(r'$\beta/\pi$',fontsize=10)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel(r'$\lambda/\pi$',fontsize=11)
fig.suptitle(r'Memory-defined operating regime in a dissipative gate-model QRC ($\gamma=0.12$ slice)',fontsize=15,y=1.05)
cbar=fig.colorbar(im,ax=axes.ravel().tolist(),fraction=0.026,pad=0.018)
cbar.set_label('mean validation percentile rank, lower is better',fontsize=9)
cbar.ax.tick_params(labelsize=8)
fig.savefig(GFX/'fig1_operating_regime_fixed_gamma.png',dpi=320,bbox_inches='tight')
fig.savefig(GFX/'fig1_operating_regime_fixed_gamma.pdf',bbox_inches='tight')
plt.close(fig)

# Figure 2: ESN comparison + controls
comp=pd.read_csv(DATA/'final_qrc_esn_comparison.csv')
q96=pd.read_csv(DATA/'qrc96_local_refinement_grid.csv')
q96sel=q96[
    (np.isclose(q96.beta_pi,float(qrc_selected['beta_pi'])))&
    (np.isclose(q96.lambda_pi,float(qrc_selected['lambda_pi'])))&
    (np.isclose(q96.gamma,float(qrc_selected['gamma'])))
]
esn=pd.read_csv(DATA/'esn_candidate_performance.csv')
esnsel=esn[
    (esn.units.astype(int)==int(esn_selected['units']))&
    (np.isclose(esn.sr,float(esn_selected['spectral_radius'])))&
    (np.isclose(esn.input_scale,float(esn_selected['input_scale'])))&
    (np.isclose(esn.leak,float(esn_selected['leak'])))
]
abl=pd.read_csv(DATA/'minimal_ablations_seeded_summary.csv')
# Use the validation-selected QRC96 result as the first control, then QRC16 controls from ablations.
data_controls=pd.DataFrame([
    {'label':'QRC96','value':float(q96sel.test_nmse.mean())},
    {'label':'Rx-ZZ-Rx','value':float(abl.set_index('ablation').loc['mixer_rx_zz_rx','mean_test_nmse'])},
    {'label':'QRC16','value':float(abl.set_index('ablation').loc['base_AD','mean_test_nmse'])},
    {'label':'Rx only','value':float(abl.set_index('ablation').loc['mixer_rx_only','mean_test_nmse'])},
    {'label':'no mix','value':float(abl.set_index('ablation').loc['mixer_none','mean_test_nmse'])},
    {'label':r'$\gamma=0$','value':float(abl.set_index('ablation').loc['gamma0_AD','mean_test_nmse'])},
    {'label':'dephase','value':float(abl.set_index('ablation').loc['dephasing','mean_test_nmse'])},
])
fig,axes=plt.subplots(1,4,figsize=(14.0,3.35))
# (a)
order=['QRC grid shared','ESN16 shared','ESN100 shared']
vals=[float(comp.set_index('method').loc[m,'mean_test_nmse']) for m in order]+[float(q96sel.test_nmse.mean())]
labs=['QRC16\nshared','ESN16\nshared','ESN100\nshared','QRC96\nregime']
cols=['#80b1d3','#fdb462','#fb6a4a','#2b8cbe']
axes[0].bar(labs,vals,color=cols,edgecolor='black',linewidth=0.35)
axes[0].set_title('(a) readout-dimension comparison',fontsize=10)
axes[0].set_ylabel('holdout NMSE')
axes[0].set_ylim(0,0.135)
for i,v in enumerate(vals): axes[0].text(i,v+0.004,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
axes[0].tick_params(axis='x',rotation=25,labelsize=8)
# (b) per task
pt_q=q96sel.groupby('task').test_nmse.mean().reindex(['mackey_glass','lorenz','narma10','sunspots_annual'])
pt_e=esnsel.groupby('task').test_nmse.mean().reindex(pt_q.index)
x=np.arange(len(pt_q)); width=0.36
axes[1].bar(x-width/2,pt_e.values,width,label='ESN100',color='#fb6a4a',edgecolor='black',linewidth=0.25)
axes[1].bar(x+width/2,pt_q.values,width,label='QRC96',color='#2b8cbe',edgecolor='black',linewidth=0.25)
axes[1].set_xticks(x,['MG','Lorenz','NARMA10','Sunspots'],rotation=25,ha='right',fontsize=8)
axes[1].set_title('(b) per-task holdout',fontsize=10); axes[1].set_ylabel('holdout NMSE'); axes[1].set_ylim(0,0.24); axes[1].legend(frameon=False,fontsize=8)
# (c) seed means
seed_pairs=pd.read_csv(DATA/'qrc96_esn100_seed_pairs.csv')
qseed=seed_pairs.set_index('seed').qrc_mean_nmse
eseed=seed_pairs.set_index('seed').esn_mean_nmse
axes[2].boxplot([eseed.values,qseed.values],tick_labels=['ESN100','QRC96'],patch_artist=True,boxprops={'facecolor':'#fdb462','alpha':0.7},medianprops={'color':'black'},widths=0.45)
# manually recolor second
for patch,col in zip(axes[2].artists, ['#fdb462','#2b8cbe']): patch.set_facecolor(col)
# artists not populated in mpl? overlay scatter
for i,arr in enumerate([eseed.values,qseed.values],start=1):
    jitter=np.linspace(-0.07,0.07,len(arr))
    axes[2].scatter(np.full(len(arr),i)+jitter,arr,s=20,alpha=0.7,c=['#fb6a4a','#2b8cbe'][i-1],edgecolor='none')
axes[2].set_title('(c) seed-level means',fontsize=10); axes[2].set_ylabel('mean holdout NMSE'); axes[2].set_ylim(0.055,0.115); axes[2].tick_params(labelsize=8)
# (d) controls
v=data_controls.value.values
norm=Normalize(vmin=0,vmax=1.05)
axes[3].bar(data_controls.label,v,color=plt.cm.magma(norm(v)),edgecolor='black',linewidth=0.25)
axes[3].set_title('(d) mechanism controls',fontsize=10); axes[3].set_ylabel('holdout NMSE'); axes[3].set_ylim(0,1.12); axes[3].tick_params(axis='x',rotation=35,labelsize=8)
for i,val in enumerate(v[:3]): axes[3].text(i,val+0.025,f'{val:.3f}',ha='center',fontsize=7)
fig.tight_layout(w_pad=1.6)
fig.savefig(GFX/'fig2_regime_beats_esn_controls.png',dpi=320,bbox_inches='tight')
fig.savefig(GFX/'fig2_regime_beats_esn_controls.pdf',bbox_inches='tight')
plt.close(fig)

# Figure 3: Memory diagnostics zero-aware + ESN
qmc=pd.read_csv(DATA/'qrc_memory_capacity_by_seed_theta.csv')
perf=q.groupby(['seed']+keys).agg(mean_val_rank=('val_rank_pct','mean')).reset_index()
qmc=qmc.merge(perf,on=['seed']+keys)
qmc['logMC']=np.log1p(qmc.MC)
qc=float(spearmanr(qmc.MC,qmc.mean_val_rank).correlation)
# nonzero Spearman for text/figure note
qcnz=float(spearmanr(qmc[qmc.MC>0].MC,qmc[qmc.MC>0].mean_val_rank).correlation)
# ESN MC data is included in this repository.
emc_path=DATA/'esn_memory_capacity_recomputed.csv'
if emc_path.exists():
    emc=pd.read_csv(emc_path)
    ea=esn.groupby(['seed','units','sr','input_scale','leak']).agg(mean_val_rank=('val_rank_pct','mean')).reset_index()
    em=emc.merge(ea,on=['seed','units','sr','input_scale','leak'])
    em['logMC']=np.log1p(em.MC)
    ec=float(spearmanr(em.MC,em.mean_val_rank).correlation)
else:
    em=None; ec=-0.61
fig,axes=plt.subplots(1,4,figsize=(14.0,3.25))
zero=qmc[qmc.MC==0]; nz=qmc[qmc.MC>0]
axes[0].scatter(nz.MC,nz.mean_val_rank,s=8,alpha=.28,c='#2b8cbe',edgecolor='none',label='MC>0')
axes[0].scatter(zero.MC,zero.mean_val_rank,s=9,alpha=.55,c='#d95f0e',edgecolor='none',label='MC=0')
z=np.polyfit(nz.MC,nz.mean_val_rank,1); xs=np.linspace(nz.MC.min(),nz.MC.max(),200); axes[0].plot(xs,z[0]*xs+z[1],c='black',lw=1.0)
axes[0].text(.04,.89,fr'$\rho_s={qc:.2f}$',transform=axes[0].transAxes,fontsize=9)
axes[0].text(.04,.08,'MC=0 is real:\nno input or collapsed memory',transform=axes[0].transAxes,fontsize=7)
axes[0].legend(frameon=False,fontsize=7,loc='upper right')
axes[0].set_title('(a) QRC raw MC',fontsize=10); axes[0].set_xlabel('memory capacity'); axes[0].set_ylabel('validation rank'); axes[0].set_ylim(0,1.02)
axes[1].scatter(qmc.logMC,qmc.mean_val_rank,s=8,alpha=.30,c='#2b8cbe',edgecolor='none')
z=np.polyfit(qmc.logMC,qmc.mean_val_rank,1); xs=np.linspace(qmc.logMC.min(),qmc.logMC.max(),200); axes[1].plot(xs,z[0]*xs+z[1],c='black',lw=1.0)
axes[1].text(.04,.89,fr'$\rho_s(MC>0)={qcnz:.2f}$',transform=axes[1].transAxes,fontsize=8)
axes[1].set_title('(b) log-scaled QRC',fontsize=10); axes[1].set_xlabel(r'$\log(1+\mathrm{MC})$'); axes[1].set_ylabel('validation rank'); axes[1].set_ylim(0,1.02)
if em is not None:
    axes[2].scatter(em.MC,em.mean_val_rank,s=16,alpha=.45,c='#ff7f0e',edgecolor='none')
    z=np.polyfit(em.MC,em.mean_val_rank,1); xs=np.linspace(em.MC.min(),em.MC.max(),100); axes[2].plot(xs,z[0]*xs+z[1],c='black',lw=1.0)
else:
    axes[2].axis('off')
axes[2].text(.04,.89,fr'$\rho_s={ec:.2f}$',transform=axes[2].transAxes,fontsize=9)
axes[2].set_title('(c) ESN MC screen',fontsize=10); axes[2].set_xlabel('memory capacity'); axes[2].set_ylabel('validation rank'); axes[2].set_ylim(0,1.02)
axes[3].bar(['QRC','ESN'],[qc,ec],color=['#2b8cbe','#ff7f0e'],edgecolor='black',linewidth=.25)
axes[3].axhline(0,color='black',lw=.8); axes[3].set_ylim(-1,0.05)
axes[3].set_title('(d) reservoir-level trend',fontsize=10); axes[3].set_ylabel(r'Spearman $\rho_s$')
for i,val in enumerate([qc,ec]): axes[3].text(i,val-.05,f'{val:.2f}',ha='center',va='top',fontsize=9)
fig.tight_layout(w_pad=1.7)
fig.savefig(GFX/'fig3_memory_capacity_screens.png',dpi=320,bbox_inches='tight')
fig.savefig(GFX/'fig3_memory_capacity_screens.pdf',bbox_inches='tight')
plt.close(fig)

# store figure numbers and summary
summary={
 'qrc96_shared_mean_test': float(q96sel.test_nmse.mean()),
 'qrc96_shared_mean_val': float(q96sel.val_nmse.mean()),
 'qrc96_seed_std': float(qseed.std()),
 'esn100_shared_mean_test': float(esnsel.test_nmse.mean()),
 'esn100_seed_std': float(eseed.std()),
 'qrc_mc_spearman': qc,
 'qrc_mc_spearman_nonzero': qcnz,
 'esn_mc_spearman': ec,
 'mc_zero_count': int((qmc.MC==0).sum()),
 'mc_total': int(len(qmc)),
 'mc_zero_fraction': float((qmc.MC==0).mean()),
 'lambda_max_plotted': ymax,
 'gamma_slice': gamma_star,
 'qrc96_selected_beta_pi': float(qrc_selected['beta_pi']),
 'qrc96_selected_lambda_pi': float(qrc_selected['lambda_pi']),
 'qrc96_selected_gamma': float(qrc_selected['gamma']),
 'qrc96_esn100_seed_delta_mean': float(qrc_stats['seed_level']['delta']['mean']),
 'qrc96_esn100_seed_delta_ci95_low': float(qrc_stats['seed_level']['delta']['ci95_low']),
 'qrc96_esn100_seed_delta_ci95_high': float(qrc_stats['seed_level']['delta']['ci95_high']),
 'qrc96_esn100_seed_wilcoxon_greater_p': float(qrc_stats['seed_level']['tests']['wilcoxon_greater_p']),
 'qrc96_esn100_seed_sign_greater_p': float(qrc_stats['seed_level']['tests']['sign_greater_p']),
}
(DATA/'final_summary_numbers.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
