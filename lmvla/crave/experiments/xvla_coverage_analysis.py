import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); rng=np.random.RandomState(0)
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
DS=[('kai','kai_dinov3base',1000),('vis','vis_dinov3base',10000),('xvla','xvla_dinov3base_full',1000),('coffee','coffee_dinov3base',10000)]
for a in ['cups_open','candy','screw_driver','vinh_cup','ziploc_slide','coffee_new','pro_pencil','vinh_cup_left']:
    DS.append((a,f'aloha_static_{a}_dinov3base',10000))
def load_bank(bank):
    d=REPO/'lmvla/crave/data'/bank; z=np.load(d/'index.npz'); s=np.load(d/'shard_0.npz'); return z['E'],z['FR'],s['feat']
sub=[]; XV=None
for name,bank,cap in DS:
    E,FR,feat=load_bank(bank); eps=sorted(np.unique(E).tolist())
    if len(eps)>cap: eps=[eps[i] for i in sorted(rng.choice(len(eps),cap,replace=False))]
    keep=np.isin(E,eps); E=E[keep];FR=FR[keep];feat=feat[keep]
    ss=rng.choice(len(feat),min(20000,len(feat)),replace=False); sub.append(l2(feat[ss].astype(np.float32)))
    if name=='xvla': XV=(E,FR,feat)
pca=PCA(128,random_state=0).fit(np.concatenate(sub)); pm=pca.mean_.astype(np.float32); pc=pca.components_.astype(np.float32)
E,FR,feat=XV; F768=l2(feat.astype(np.float32)); N=len(F768); eps=sorted(np.unique(E).tolist()); NC=len(eps)
T=np.zeros(N,np.float32)
for e in eps: m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
F128=l2((F768-pm)@pc.T)
bg=BayesianGaussianMixture(n_components=40,covariance_type='diag',weight_concentration_prior=1e-2,max_iter=120,random_state=0).fit(F128[rng.choice(N,80000,replace=False)]); labs=bg.predict(F128)
rows=[]  # (medianT, cov, tstd, size)
for k in range(40):
    m=labs==k
    if m.sum()<20: continue
    Tc=T[m]; cov=len(set(E[m].tolist()))/NC; rows.append((float(np.median(Tc)),cov,float(Tc.std()),int(m.sum())))
rows.sort()
print(f'xvla {N}帧 {NC}ep · 有效簇 {len(rows)} · T全域分布 min/med/max={T.min():.2f}/{np.median(T):.2f}/{T.max():.2f}',flush=True)
print(f"{'medT':>5}{'cov':>6}{'Tstd':>6}{'size':>7}{'  milestone?':>12}")
for mt,cov,ts,sz in rows:
    print(f'{mt:>5.2f}{cov:>6.2f}{ts:>6.2f}{sz:>7}{"  ✓cov≥.5" if cov>=0.5 else "":>12}',flush=True)
# 图: coverage vs medianT, 点大小=size, 颜色=Tstd
fig,axs=plt.subplots(1,3,figsize=(18,4.5))
mt=np.array([r[0] for r in rows]); cv=np.array([r[1] for r in rows]); ts=np.array([r[2] for r in rows]); sz=np.array([r[3] for r in rows])
sc=axs[0].scatter(mt,cv,s=sz/30,c=ts,cmap='viridis'); axs[0].axhline(0.5,color='r',ls='--',label='cov=0.5'); plt.colorbar(sc,ax=axs[0],label='T-std')
axs[0].set_xlabel('median T (progress)'); axs[0].set_ylabel('coverage'); axs[0].set_title('cluster coverage vs progress (color=T-std, size=n)'); axs[0].legend(); axs[0].grid(alpha=.3)
# 全域 T 直方 + 高覆盖簇的 T
axs[1].hist(T,bins=40,color='.7'); axs[1].set_title('all-frame T distribution (should be uniform)'); axs[1].set_xlabel('T')
# 每个 T 区间的最大 coverage
bins=np.linspace(0,1,11); maxcov=[]
for a,b in zip(bins[:-1],bins[1:]):
    cc=[r[1] for r in rows if a<=r[0]<b]; maxcov.append(max(cc) if cc else 0)
axs[2].bar((bins[:-1]+bins[1:])/2,maxcov,width=0.08,color='steelblue'); axs[2].axhline(0.5,color='r',ls='--'); axs[2].set_title('max cluster coverage per progress-bin'); axs[2].set_xlabel('T bin'); axs[2].set_ylabel('max coverage'); axs[2].grid(alpha=.3)
fig.suptitle(f'xvla milestone coverage analysis · {NC}ep across 21 variant-batches',fontsize=13)
fig.tight_layout(); fig.savefig(REPO/'temp/xvla_coverage_analysis.png',dpi=110,bbox_inches='tight'); print('SAVED',flush=True)
