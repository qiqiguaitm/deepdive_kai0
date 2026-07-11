import json, numpy as np, cv2, h5py, torch, torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); DEV='cuda:0'; rng=np.random.RandomState(0); RES=128
XROOT=REPO/'xvla/data/xvla_soft_fold'; WORK=REPO/'temp/xvla_extract_base'
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
DS=[('kai','kai_dinov3base',1000),('vis','vis_dinov3base',10000),('xvla','xvla_dinov3base_full',1000),('coffee','coffee_dinov3base',10000)]
for a in ['cups_open','candy','screw_driver','vinh_cup','ziploc_slide','coffee_new','pro_pencil','vinh_cup_left']:
    DS.append((a,f'aloha_static_{a}_dinov3base',10000))
def load_bank(bank):
    d=REPO/'lmvla/crave/data'/bank; z=np.load(d/'index.npz'); s=np.load(d/'shard_0.npz'); return z['E'],z['FR'],s['feat']
# 共享 PCA: 各数据集 cap 后子采样 20k(同 multitask 顺序/seed)
sub=[]; XV=None
for name,bank,cap in DS:
    E,FR,feat=load_bank(bank); eps=sorted(np.unique(E).tolist())
    if len(eps)>cap: eps=[eps[i] for i in sorted(rng.choice(len(eps),cap,replace=False))]
    keep=np.isin(E,eps); E=E[keep];FR=FR[keep];feat=feat[keep]
    ss=rng.choice(len(feat),min(20000,len(feat)),replace=False); sub.append(l2(feat[ss].astype(np.float32)))
    if name=='xvla': XV=(E,FR,feat)
pca=PCA(128,random_state=0).fit(np.concatenate(sub)); pm=pca.mean_.astype(np.float32); pc=pca.components_.astype(np.float32)
E,FR,feat=XV; feat768=l2(feat.astype(np.float32)); N=len(feat768); eps=sorted(np.unique(E).tolist()); NC=len(eps)
T=np.zeros(N,np.float32)
for e in eps: m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
F128=l2((feat768-pm)@pc.T)
bg=BayesianGaussianMixture(n_components=40,covariance_type='diag',weight_concentration_prior=1e-2,max_iter=120,random_state=0).fit(F128[rng.choice(N,min(80000,N),replace=False)]); labs=bg.predict(F128)
ms=[]
for k in range(40):
    m=labs==k
    if m.sum()>=20 and len(set(E[m].tolist()))/NC>=0.5: ms.append((float(np.median(T[m])), l2(feat768[m].mean(0)[None])[0]))
ms.sort(); print(f'multitask scheme (1000samp+sharedPCA): {len(ms)} milestones',flush=True)
gidmap={}
for g in range(8):
    for bn,lep,gid in json.load(open(WORK/f'chunk_{g}.json')): gidmap[int(gid)]=(bn,int(lep))
net=nn.Module()  # placeholder
class Dec(nn.Module):
    def __init__(s,d=768):
        super().__init__(); s.fc=nn.Linear(d,512*4*4)
        def blk(i,o): return nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.SiLU())
        s.up=nn.Sequential(blk(512,256),blk(256,128),blk(128,64),blk(64,32),blk(32,32)); s.out=nn.Conv2d(32,3,3,1,1)
    def forward(s,x): return torch.sigmoid(s.out(s.up(s.fc(x).view(-1,512,4,4))))
net=Dec().to(DEV); net.load_state_dict(torch.load(REPO/'lmvla/crave/data/base_decoder_xvla.pt')); net.eval()
n=len(ms); fig,ax=plt.subplots(2,max(n,2),figsize=(2.3*max(n,2),5))
for j,(pv,cen) in enumerate(ms):
    with torch.no_grad(): dec=net(torch.tensor(cen[None],device=DEV))[0].permute(1,2,0).cpu().numpy()
    i=int(np.linalg.norm(feat768-cen,axis=1).argmin()); gid=int(E[i]); fr=int(FR[i]); bn,lep=gidmap[gid]
    f=h5py.File(XROOT/bn/f'episode_{lep}.hdf5','r'); raw=f['observations/images/cam_high'][fr]; f.close()
    rimg=cv2.resize(np.ascontiguousarray(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]),(RES,RES))
    ax[0,j].imshow(np.clip(dec,0,1)); ax[0,j].axis('off'); ax[0,j].set_title(('decoded centroid  ' if j==0 else '')+f'p={pv:.2f}',fontsize=9)
    ax[1,j].imshow(rimg); ax[1,j].axis('off'); ax[1,j].set_title('retrieved nearest real frame' if j==0 else '',fontsize=9)
for j in range(n,max(n,2)):
    ax[0,j].axis('off'); ax[1,j].axis('off')
fig.suptitle(f'xvla milestones - multitask scheme (1000 sampled + shared PCA + BGMM + coverage>=0.5) = {n} milestones - top=decoder / bottom=retrieval',fontsize=11)
fig.tight_layout(); fig.savefig(REPO/'temp/centroid_decode_xvla.png',dpi=105,bbox_inches='tight'); print('SAVED',flush=True)
