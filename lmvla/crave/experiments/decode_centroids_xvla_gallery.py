import json, numpy as np, cv2, h5py, torch, torch.nn as nn
from pathlib import Path
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO=Path('/vePFS-North-E/vis_robot/workspace/deepdive_kai0'); DEV='cuda:0'; rng=np.random.RandomState(1); RES=128
XROOT=REPO/'xvla/data/xvla_soft_fold'; WORK=REPO/'temp/xvla_extract_base'
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
class Dec(nn.Module):
    def __init__(s,d=768):
        super().__init__(); s.fc=nn.Linear(d,512*4*4)
        def blk(i,o): return nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.SiLU())
        s.up=nn.Sequential(blk(512,256),blk(256,128),blk(128,64),blk(64,32),blk(32,32)); s.out=nn.Conv2d(32,3,3,1,1)
    def forward(s,x): return torch.sigmoid(s.out(s.up(s.fc(x).view(-1,512,4,4))))
gidmap={}
for g in range(8):
    for bn,lep,gid in json.load(open(WORK/f'chunk_{g}.json')): gidmap[int(gid)]=(bn,int(lep))
z=np.load(REPO/'lmvla/crave/data/xvla_dinov3base_full/index.npz'); s=np.load(REPO/'lmvla/crave/data/xvla_dinov3base_full/shard_0.npz')
E=z['E'];FR=z['FR']; feat=l2(s['feat'].astype(np.float32)); N=len(feat); eps=sorted(np.unique(E).tolist()); NC=len(eps)
T=np.zeros(N,np.float32)
for e in eps: m=np.where(E==e)[0]; o=m[np.argsort(FR[m])]; T[o]=np.linspace(0,1,len(o))
# 60 components, coverage>=0.3, 取10个跨进度
from sklearn.cluster import KMeans
km=KMeans(15,n_init=3,random_state=0).fit(feat[rng.choice(N,50000,replace=False)]); labs=km.predict(feat)
ms=[]
for k in range(15):
    m=labs==k
    if m.sum()>=30: ms.append((float(np.median(T[m])), l2(feat[m].mean(0)[None])[0]))
ms.sort(); print(f'{len(ms)} milestones (cov>=0.3)',flush=True)
if len(ms)>10: ms=[ms[i] for i in np.linspace(0,len(ms)-1,10).astype(int)]
net=Dec().to(DEV); net.load_state_dict(torch.load(REPO/'lmvla/crave/data/base_decoder_xvla.pt')); net.eval()
n=len(ms); fig,ax=plt.subplots(2,n,figsize=(2.2*n,5))
for j,(pv,cen) in enumerate(ms):
    with torch.no_grad(): dec=net(torch.tensor(cen[None],device=DEV))[0].permute(1,2,0).cpu().numpy()
    i=int(np.linalg.norm(feat-cen,axis=1).argmin()); gid=int(E[i]); fr=int(FR[i]); bn,lep=gidmap[gid]
    f=h5py.File(XROOT/bn/f'episode_{lep}.hdf5','r'); raw=f['observations/images/cam_high'][fr]; f.close()
    rimg=cv2.resize(np.ascontiguousarray(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]),(RES,RES))
    ax[0,j].imshow(np.clip(dec,0,1)); ax[0,j].axis('off'); ax[0,j].set_title(('decoded centroid  ' if j==0 else '')+f'p={pv:.2f}',fontsize=9)
    ax[1,j].imshow(rimg); ax[1,j].axis('off'); ax[1,j].set_title('retrieved nearest real frame' if j==0 else '',fontsize=9)
fig.suptitle(f'xvla milestone centroids ({n} shown) - top=decoder synthesis / bottom=retrieval, by progress p',fontsize=12)
fig.tight_layout(); fig.savefig(REPO/'temp/centroid_decode_xvla.png',dpi=105,bbox_inches='tight'); print('SAVED',flush=True)
