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

PHASE_CMAP = plt.cm.magma_r
PHASE_GOLD = PHASE_CMAP(0.08)
PHASE_AMBER = PHASE_CMAP(0.18)
PHASE_CORAL = PHASE_CMAP(0.32)
PHASE_ROSE = PHASE_CMAP(0.46)
PHASE_PLUM = PHASE_CMAP(0.66)
PHASE_VIOLET = PHASE_CMAP(0.82)
PHASE_DARK = PHASE_CMAP(0.94)
QRC_BLUE = PHASE_ROSE
ESN_ORANGE = PHASE_AMBER
INK = '#1f2933'
MUTED = '#6b7280'
GRID = '#d9dee7'


def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def savefig_dual(fig, stem, aliases=()):
    fig.savefig(GFX / f'{stem}.png', dpi=360, bbox_inches='tight')
    fig.savefig(GFX / f'{stem}.pdf', bbox_inches='tight')
    for alias in aliases:
        fig.savefig(GFX / f'{alias}.png', dpi=360, bbox_inches='tight')
        fig.savefig(GFX / f'{alias}.pdf', bbox_inches='tight')
    plt.close(fig)

# Load data
q=pd.read_csv(DATA/'qrc_seed_ensemble_grid.csv')
stats_path=DATA/'qrc96_esn100_stats.json'
if not stats_path.exists():
    raise FileNotFoundError(f"Missing {stats_path}; run scripts/analyze_qrc96_esn100.py first.")
qrc_stats=json.loads(stats_path.read_text())
taskwise_stats_path=DATA/'qrc96_esn100_taskwise_stats.json'
if not taskwise_stats_path.exists():
    raise FileNotFoundError(f"Missing {taskwise_stats_path}; run scripts/analyze_qrc96_esn100_taskwise.py first.")
taskwise_stats=json.loads(taskwise_stats_path.read_text())
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

# Short-paper Figure 1: compact, polished phase maps.
short_panels = [
    ('all tasks', all_tasks),
    ('leave out MG', [x for x in all_tasks if x != 'mackey_glass']),
    ('leave out NARMA10', [x for x in all_tasks if x != 'narma10']),
    ('leave out Lorenz', [x for x in all_tasks if x != 'lorenz']),
    ('leave out Sunspots', [x for x in all_tasks if x != 'sunspots_annual']),
]
fig, axes = plt.subplots(1, 5, figsize=(7.25, 1.72), sharex=True, sharey=True)
for ax, (title, tasks) in zip(axes, short_panels):
    d = qg[qg.task.isin(tasks)].copy()
    agg = d.groupby(['beta_pi', 'lambda_pi']).agg(mean_rank=('val_rank_pct', 'mean')).reset_index()
    x = agg.beta_pi.values
    y = agg.lambda_pi.values
    z = agg.mean_rank.values
    xi = np.linspace(0, xmax, 220)
    yi = np.linspace(0, ymax, 180)
    XI, YI = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (XI, YI), method='cubic')
    Zn = griddata((x, y), z, (XI, YI), method='nearest')
    Zs = np.where(np.isnan(Zi), Zn, Zi)
    for _ in range(2):
        P = np.pad(Zs, 1, mode='edge')
        Zs = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] + 4 * P[1:-1, 1:-1]) / 8.0
    im = ax.imshow(
        Zs,
        origin='lower',
        extent=[0, xmax, 0, ymax],
        cmap='magma_r',
        vmin=0.05,
        vmax=0.8,
        aspect='auto',
    )
    ax.contour(XI, YI, Zs, levels=[0.15, 0.25, 0.40], colors='white', linewidths=[0.55, 0.45, 0.35], alpha=0.78)
    chunks = []
    for _, g in d.groupby('replicate'):
        h = g.copy()
        h['top20'] = h.val_rank_pct <= 0.20
        chunks.append(h)
    fr = pd.concat(chunks).groupby(['beta_pi', 'lambda_pi']).top20.mean().reset_index()
    core = fr[fr.top20 >= 0.70]
    if len(core):
        ax.scatter(core.beta_pi, core.lambda_pi, s=10, color='#2b8cbe', edgecolor='white', linewidth=0.25, zorder=4)
    ax.scatter(
        [qrc_selected['beta_pi']],
        [qrc_selected['lambda_pi']],
        marker='*',
        s=42,
        linewidths=0.35,
        color='#ff7f0e',
        edgecolor='black',
        zorder=5,
    )
    ax.set_title(title, fontsize=7.5, pad=2.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlabel(r'$\beta/\pi$', fontsize=7)
axes[0].set_ylabel(r'$\lambda/\pi$', fontsize=7.5)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.012)
cbar.set_label('validation rank percentile', fontsize=7)
cbar.ax.tick_params(labelsize=6, length=2)
fig.suptitle(r'Phase maps reveal a transferable low-input, nonzero-damping operating regime ($\gamma=0.12$)', fontsize=9.5, y=1.08, color=INK)
savefig_dual(fig, 'fig1_short_phase_maps')

# Figure 2: ESN comparison + controls
comp=pd.read_csv(DATA/'final_qrc_esn_comparison.csv')
q96=pd.read_csv(DATA/'qrc96_same_arch_expanded_grid.csv')
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
cols=[PHASE_ROSE, PHASE_AMBER, PHASE_CORAL, PHASE_GOLD]
axes[0].bar(labs,vals,color=cols,edgecolor='black',linewidth=0.35)
axes[0].set_title('(a) readout-dimension comparison',fontsize=10)
axes[0].set_ylabel('holdout NMSE')
axes[0].set_ylim(0,0.135)
for i,v in enumerate(vals): axes[0].text(i,v+0.004,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
axes[0].tick_params(axis='x',rotation=25,labelsize=8)
# (b) task-wise selected per task
taskwise_pt=pd.read_csv(DATA/'qrc96_esn100_taskwise_per_task.csv').set_index('task').reindex(['mackey_glass','lorenz','narma10','sunspots_annual'])
pt_q=taskwise_pt.qrc96_mean_nmse
pt_e=taskwise_pt.esn100_mean_nmse
x=np.arange(len(pt_q)); width=0.36
axes[1].bar(x-width/2,pt_e.values,width,label='ESN100',color=PHASE_AMBER,edgecolor='black',linewidth=0.25)
axes[1].bar(x+width/2,pt_q.values,width,label='QRC96',color=PHASE_ROSE,edgecolor='black',linewidth=0.25)
axes[1].set_xticks(x,['MG','Lorenz','NARMA10','Sunspots'],rotation=25,ha='right',fontsize=8)
axes[1].set_title('(b) task-wise plateau holdout',fontsize=10); axes[1].set_ylabel('holdout NMSE'); axes[1].set_ylim(0,0.24); axes[1].legend(frameon=False,fontsize=8)
# (c) seed means
seed_pairs=pd.read_csv(DATA/'qrc96_esn100_seed_pairs.csv')
qseed=seed_pairs.set_index('seed').qrc_mean_nmse
eseed=seed_pairs.set_index('seed').esn_mean_nmse
axes[2].boxplot([eseed.values,qseed.values],tick_labels=['ESN100','QRC96'],patch_artist=True,boxprops={'facecolor':'#fdb462','alpha':0.7},medianprops={'color':'black'},widths=0.45)
# manually recolor second
for patch,col in zip(axes[2].artists, [PHASE_AMBER, PHASE_ROSE]): patch.set_facecolor(col)
# artists not populated in mpl? overlay scatter
for i,arr in enumerate([eseed.values,qseed.values],start=1):
    jitter=np.linspace(-0.07,0.07,len(arr))
    axes[2].scatter(np.full(len(arr),i)+jitter,arr,s=20,alpha=0.7,color=[PHASE_AMBER, PHASE_ROSE][i-1],edgecolor='none')
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
axes[0].scatter(nz.MC,nz.mean_val_rank,s=8,alpha=.28,color=PHASE_ROSE,edgecolor='none',label='MC>0')
axes[0].scatter(zero.MC,zero.mean_val_rank,s=9,alpha=.55,color=PHASE_AMBER,edgecolor='none',label='MC=0')
z=np.polyfit(nz.MC,nz.mean_val_rank,1); xs=np.linspace(nz.MC.min(),nz.MC.max(),200); axes[0].plot(xs,z[0]*xs+z[1],c='black',lw=1.0)
axes[0].text(.04,.89,fr'$\rho_s={qc:.2f}$',transform=axes[0].transAxes,fontsize=9)
axes[0].text(.04,.08,'MC=0 is real:\nno input or collapsed memory',transform=axes[0].transAxes,fontsize=7)
axes[0].legend(frameon=False,fontsize=7,loc='upper right')
axes[0].set_title('(a) QRC raw MC',fontsize=10); axes[0].set_xlabel('memory capacity'); axes[0].set_ylabel('validation rank'); axes[0].set_ylim(0,1.02)
axes[1].scatter(qmc.logMC,qmc.mean_val_rank,s=8,alpha=.30,color=PHASE_ROSE,edgecolor='none')
z=np.polyfit(qmc.logMC,qmc.mean_val_rank,1); xs=np.linspace(qmc.logMC.min(),qmc.logMC.max(),200); axes[1].plot(xs,z[0]*xs+z[1],c='black',lw=1.0)
axes[1].text(.04,.89,fr'$\rho_s(MC>0)={qcnz:.2f}$',transform=axes[1].transAxes,fontsize=8)
axes[1].set_title('(b) log-scaled QRC',fontsize=10); axes[1].set_xlabel(r'$\log(1+\mathrm{MC})$'); axes[1].set_ylabel('validation rank'); axes[1].set_ylim(0,1.02)
if em is not None:
    axes[2].scatter(em.MC,em.mean_val_rank,s=16,alpha=.45,color=PHASE_AMBER,edgecolor='none')
    z=np.polyfit(em.MC,em.mean_val_rank,1); xs=np.linspace(em.MC.min(),em.MC.max(),100); axes[2].plot(xs,z[0]*xs+z[1],c='black',lw=1.0)
else:
    axes[2].axis('off')
axes[2].text(.04,.89,fr'$\rho_s={ec:.2f}$',transform=axes[2].transAxes,fontsize=9)
axes[2].set_title('(c) ESN MC screen',fontsize=10); axes[2].set_xlabel('memory capacity'); axes[2].set_ylabel('validation rank'); axes[2].set_ylim(0,1.02)
axes[3].bar(['QRC','ESN'],[qc,ec],color=[PHASE_ROSE, PHASE_AMBER],edgecolor='black',linewidth=.25)
axes[3].axhline(0,color='black',lw=.8); axes[3].set_ylim(-1,0.05)
axes[3].set_title('(d) reservoir-level trend',fontsize=10); axes[3].set_ylabel(r'Spearman $\rho_s$')
for i,val in enumerate([qc,ec]): axes[3].text(i,val-.05,f'{val:.2f}',ha='center',va='top',fontsize=9)
fig.tight_layout(w_pad=1.7)
fig.savefig(GFX/'fig3_memory_capacity_screens.png',dpi=320,bbox_inches='tight')
fig.savefig(GFX/'fig3_memory_capacity_screens.pdf',bbox_inches='tight')
plt.close(fig)

# Short-paper Figure 2: paired comparisons, deltas, controls, diagnostics.
fig = plt.figure(figsize=(7.25, 4.85))
gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.40)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

# (a) Seed-paired shared comparison as a slope plot.
ax = axes[0]
seed_pairs = pd.read_csv(DATA / 'qrc96_esn100_seed_pairs.csv').sort_values('seed')
for _, row in seed_pairs.iterrows():
    ax.plot([0, 1], [row.esn_mean_nmse, row.qrc_mean_nmse], color=PHASE_CMAP(0.74), lw=0.75, alpha=0.58, zorder=1)
    ax.scatter([0, 1], [row.esn_mean_nmse, row.qrc_mean_nmse], s=10, color=[PHASE_AMBER, PHASE_ROSE], edgecolor='white', linewidth=0.25, zorder=2)
mean_esn = float(seed_pairs.esn_mean_nmse.mean())
mean_qrc = float(seed_pairs.qrc_mean_nmse.mean())
ax.plot([0, 1], [mean_esn, mean_qrc], color=PHASE_DARK, lw=2.0, zorder=4)
ax.scatter([0, 1], [mean_esn, mean_qrc], s=38, color=[ESN_ORANGE, QRC_BLUE], edgecolor='white', linewidth=0.7, zorder=5)
for x0, val, label in [(0, mean_esn, '0.090'), (1, mean_qrc, '0.077')]:
    ax.text(x0, val + 0.0042, label, ha='center', va='bottom', fontsize=7.0, color=INK)
ax.text(
    0.98,
    0.94,
    r'10/10 seeds' + '\n' + r'$\Delta=0.0134$' + '\nCI [0.008, 0.019]',
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=6.3,
    color=INK,
    bbox=dict(boxstyle='round,pad=0.25', facecolor=PHASE_CMAP(0.02), edgecolor=PHASE_CMAP(0.58), linewidth=0.35, alpha=0.18),
)
ax.set_xlim(-0.22, 1.22)
ax.set_ylim(0.058, 0.108)
ax.set_xticks([0, 1], ['ESN100', 'QRC96'])
ax.set_ylabel('seed-mean holdout NMSE', fontsize=7.4)
ax.set_title('(a) Shared selector, paired by seed', fontsize=8.2, color=INK, pad=4)
ax.grid(axis='y', color=GRID, linewidth=0.45, alpha=0.78)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=7)

# (b) Task-wise non-floor deltas as a compact strip plot.
ax = axes[1]
tw_pairs = pd.read_csv(DATA / 'qrc96_esn100_taskwise_task_seed_pairs.csv')
tasks_nf = ['narma10', 'sunspots_annual']
labels_nf = ['NARMA10', 'Sunspots']
ypos = np.array([1, 0])
for y, task, label in zip(ypos, tasks_nf, labels_nf):
    vals = tw_pairs[tw_pairs.task == task].delta_esn_minus_qrc.to_numpy(dtype=float)
    jitter = np.linspace(-0.13, 0.13, len(vals))
    colors = [
        PHASE_CMAP(0.44 - 0.36 * np.clip(v / 0.12, 0, 1)) if v > 0 else PHASE_VIOLET
        for v in vals
    ]
    ax.scatter(vals, y + jitter, s=22, color=colors, edgecolor='white', linewidth=0.35, zorder=3)
    mean = float(vals.mean())
    ci = taskwise_stats['by_task'][task]['delta']
    ax.plot([ci['ci95_low'], ci['ci95_high']], [y, y], color=PHASE_DARK, lw=1.25, zorder=2)
    ax.scatter([mean], [y], marker='D', s=34, color=PHASE_GOLD, edgecolor=INK, linewidth=0.45, zorder=4)
    wins = taskwise_stats['by_task'][task]['tests']['wins']
    ax.text(mean + 0.008, y + 0.17, f'{mean:.3f}, {wins}/10', fontsize=6.6, color=INK, ha='left', va='center')
ax.axvline(0, color=INK, lw=0.8)
ax.axvspan(0, 0.125, color=PHASE_GOLD, alpha=0.075, zorder=0)
ax.set_yticks(ypos, labels_nf)
ax.set_xlim(-0.03, 0.125)
ax.set_xlabel(r'$\Delta$ holdout NMSE (ESN100 - QRC96)', fontsize=7.4)
ax.set_title('(b) Non-floor task-wise deltas', fontsize=8.2, color=INK, pad=4)
ax.grid(axis='x', color=GRID, linewidth=0.45, alpha=0.78)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=7)

# (c) Mechanism controls as a log-scale lollipop plot.
ax = axes[2]
ctrl = data_controls.copy()
ctrl['display'] = ['QRC96', 'Rx-ZZ-Rx', 'QRC16', 'Rx only', 'no mix', r'$\gamma=0$', 'dephase']
ctrl = ctrl.sort_values('value', ascending=False).reset_index(drop=True)
y = np.arange(len(ctrl))
ctrl_norm = Normalize(vmin=np.log10(ctrl.value.min()), vmax=np.log10(ctrl.value.max()))
ctrl_cols = [PHASE_CMAP(ctrl_norm(np.log10(v))) for v in ctrl.value]
ax.hlines(y, 0.05, ctrl.value, color=PHASE_CMAP(0.80), linewidth=1.2, alpha=0.28, zorder=1)
ax.scatter(ctrl.value, y, s=42, color=ctrl_cols[: len(ctrl)], edgecolor=INK, linewidth=0.35, zorder=3)
ax.axvline(float(q96sel.test_nmse.mean()), color=PHASE_GOLD, lw=1.05, ls='--', alpha=0.90)
ax.set_xscale('log')
ax.set_xlim(0.045, 1.6)
ax.set_yticks(y, ctrl.display)
ax.invert_yaxis()
ax.set_xlabel('holdout NMSE (log scale)', fontsize=7.4)
ax.set_title('(c) Mechanism controls', fontsize=8.2, color=INK, pad=4)
for yi, val in zip(y, ctrl.value):
    if val < 0.22 or val > 0.85:
        ax.text(val * 1.08, yi, f'{val:.3f}', va='center', fontsize=6.3, color=INK)
ax.grid(axis='x', color=GRID, linewidth=0.45, alpha=0.75)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=7)

# (d) Memory diagnostics as a diverging coefficient plot.
ax = axes[3]
diag = pd.read_csv(DATA / 'qrc_real_current_diagnostic_spearman_named.csv')
diag = diag.set_index('metric').reindex(['MC', 'IPCmem', 'IPCtot', 'IPCnonlin', 'Vfeat', 'reff']).reset_index()
diag['display'] = ['MC', r'IPC$_m$', r'IPC$_t$', r'IPC$_n$', r'$V_f$', r'$r_e$']
diag_values = diag['spearman_vs_val_rank'].to_numpy(dtype=float)
y = np.arange(len(diag))
colors = [PHASE_CMAP(0.18 + 0.24 * (1 - min(abs(v), 1.0))) if v < 0 else PHASE_VIOLET for v in diag_values]
ax.barh(y, diag_values, color=colors, edgecolor='white', linewidth=0.7, height=0.66)
ax.axvline(0, color=INK, linewidth=0.75)
ax.set_xlim(-1.0, 0.25)
ax.set_yticks(y, diag.display)
ax.invert_yaxis()
ax.set_xlabel(r'Spearman $\rho_s$ vs. validation rank', fontsize=7.4)
ax.set_title('(d) Memory diagnostics transfer', fontsize=8.2, color=INK, pad=4)
for yi, val in zip(y, diag_values):
    if val < 0:
        ax.text(-0.035, yi, f'{val:.2f}', ha='right', va='center', fontsize=6.4, color=INK)
    else:
        ax.text(val + 0.025, yi, f'{val:.2f}', ha='left', va='center', fontsize=6.4, color=INK)
ax.grid(axis='x', color=GRID, linewidth=0.45, alpha=0.75)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=7)

fig.suptitle('Supporting evidence: paired gains, task-wise deltas, and mechanisms', fontsize=9.0, y=1.005, color=INK)
savefig_dual(fig, 'fig2_short_evidence', aliases=('fig2_regime_beats_esn_controls',))

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
 'qrc96_taskwise_non_floor_delta_mean': float(taskwise_stats['non_floor_summary']['delta']['mean']),
 'qrc96_taskwise_non_floor_delta_ci95_low': float(taskwise_stats['non_floor_summary']['delta']['ci95_low']),
 'qrc96_taskwise_non_floor_delta_ci95_high': float(taskwise_stats['non_floor_summary']['delta']['ci95_high']),
 'qrc96_taskwise_sunspots_delta_mean': float(taskwise_stats['by_task']['sunspots_annual']['delta']['mean']),
 'qrc96_taskwise_sunspots_wins': int(taskwise_stats['by_task']['sunspots_annual']['tests']['wins']),
 'qrc96_taskwise_non_floor_claim_allowed': bool(taskwise_stats['gates']['non_floor_claim_allowed']),
}
summary_path = DATA / 'final_summary_numbers.json'
if summary_path.exists():
    existing_summary = json.loads(summary_path.read_text())
    for key, value in existing_summary.items():
        if key.startswith('current_intrinsic_') and key not in summary:
            summary[key] = value
summary_path.write_text(json.dumps(summary,indent=2) + '\n')
print(json.dumps(summary,indent=2))
