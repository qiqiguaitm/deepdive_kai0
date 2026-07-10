import numpy as np, time
from pathlib import Path
from sklearn.decomposition import PCA
R=Path('/home/tim/workspace/deepdive_kai0/temp'); G=Path('/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/recurrence_graphs')
OUT=R/'multitask_cache'; OUT.mkdir(exist_ok=True)
DSETS=[('kai','crave_full_dinov3h','kai0base_dinov3h',3.0),('vis','vis_dinov3h','vis_dinov3h',30.0),
       ('coffee','coffee_dinov3h','coffee_dinov3h',50.0),('xvla','xvla_dinov3h','xvla_dinov3h',30.0)]
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
def load_bank(bank):
    d=R/bank; idx=np.load(d/'index.npz'); E=idx['E']; FR=idx['FR']; T=idx['T']; N=len(E)
    feat=np.zeros((N,1280),np.float16)
    for sh in sorted(d.glob('shard_*.npz')):
        s=np.load(sh); g=s['gidx']; v=s['valid'] if 'valid' in s else np.ones(len(g),bool); feat[g[v]]=s['feat'][v]
    return E,FR,T,feat
def dawviterbi(F,Ct,pord,lam):  # 双锚 Viterbi(1280 L2, img-only) → 逐帧 value
    Fn=l2(F); Ctn=l2(Ct); vals=np.asarray(pord,float)
    sC=l2(Fn[:3].mean(0)[None])[0]; eC=l2(Fn[-3:].mean(0)[None])[0]
    C2=np.vstack([Ctn,sC,eC]); P=np.concatenate([vals,[0.],[1.]])
    bins=np.unique(np.concatenate([[0.],P,[1.]])); nb=len(bins); cb=[int(np.searchsorted(bins,v)) for v in P]; pen=lam*np.abs(bins[:,None]-bins[None])
    de=np.linalg.norm(Fn[:,None]-C2[None],axis=2); em=np.full((len(Fn),nb),1e3)
    for ti in range(len(P)): em[:,cb[ti]]=np.minimum(em[:,cb[ti]],de[:,ti])
    cost=np.full(nb,1e9); cost[0]=em[0,0]; BP=np.zeros((len(Fn),nb),int)
    for j in range(1,len(Fn)):
        tr=cost[None,:]+pen; kk=tr.argmin(1); cost=em[j]+tr[np.arange(nb),kk]; BP[j]=kk
    si=nb-1; path=np.zeros(len(Fn),int); path[-1]=si
    for j in range(len(Fn)-2,-1,-1): si=BP[j+1][si]; path[j]=si
    step=bins[path]
    # polyline 平滑
    segs=[]; a=0
    for t in range(1,len(step)):
        if step[t]!=step[t-1]: segs.append((a,t-1,step[t-1])); a=t
    segs.append((a,len(step)-1,step[-1])); reps=[]
    for i0,i1,val in segs:
        cand=[ti for ti in range(len(P)) if abs(P[ti]-val)<1e-9]; fr=np.arange(i0,i1+1); bd=1e18; bf=i0
        for ti in cand:
            dd=np.linalg.norm(Fn[fr]-C2[ti],axis=1); k=int(dd.argmin())
            if dd[k]<bd: bd=dd[k]; bf=fr[k]
        reps.append((bf,float(val)))
    if reps[0][0]!=0: reps=[(0,float(step[0]))]+reps
    if reps[-1][0]!=len(step)-1: reps=reps+[(len(step)-1,float(step[-1]))]
    rf=np.array([r[0] for r in reps]); rv=np.array([r[1] for r in reps]); keep=np.concatenate([[True],np.diff(rf)>0])
    return np.interp(np.arange(len(step)),rf[keep],rv[keep]).astype(np.float32)
def cc(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else np.nan
# 先收集所有特征拟合共享 PCA(子采样)
allsub=[]; BANKS={}
for ds,bank,g,fps in DSETS:
    E,FR,T,feat=load_bank(bank); BANKS[ds]=(E,FR,T,feat); 
    sub=np.random.RandomState(0).choice(len(feat),min(30000,len(feat)),replace=False); allsub.append(l2(feat[sub].astype(np.float32)))
    print(f'{ds}: feat {feat.shape} eps {len(np.unique(E))}',flush=True)
pca=PCA(128,random_state=0).fit(np.concatenate(allsub)); pm=pca.mean_.astype(np.float32); pcmp=pca.components_.astype(np.float32)
np.savez(OUT/'shared_pca.npz',mean=pm,components=pcmp); print('shared PCA128 fitted',flush=True)
# per-ep: pca128 feat + teacher value + T
for di,(ds,bank,g,fps) in enumerate(DSETS):
    E,FR,T,feat=BANKS[ds]; z=np.load(G/g/'recurrence_graph.npz',allow_pickle=True); Ct=z['prototype_table'].astype(np.float32); pord=z['pord']
    lam=16.0*fps/3.0; eps=sorted(np.unique(E).tolist()); recs=[]; corrs=[]
    for e in eps:
        loc=np.where(E==e)[0]; o=np.argsort(FR[loc]); gi=loc[o]; F=feat[gi].astype(np.float32); t=T[gi].astype(np.float32); n=len(F)
        if n<8: continue
        val=dawviterbi(F,Ct,pord,lam); pf=l2((l2(F)-pm)@pcmp.T).astype(np.float32)
        recs.append((int(e),pf,val,t)); corrs.append(cc(val,t))
    np.savez(OUT/f'{ds}.npz', ep=np.array([r[0] for r in recs]), task=di,
             **{f'f{r[0]}':r[1] for r in recs},**{f'v{r[0]}':r[2] for r in recs},**{f't{r[0]}':r[3] for r in recs})
    print(f'{ds}: {len(recs)} eps · teacher-vs-T corr mean={np.nanmean(corrs):.3f} median={np.nanmedian(corrs):.3f}',flush=True)
print('DONE prep',flush=True)
