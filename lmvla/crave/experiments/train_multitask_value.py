import numpy as np, torch, torch.nn as nn
from pathlib import Path
OUT=Path('/home/tim/workspace/deepdive_kai0/temp/multitask_cache'); DEV='cuda:0'; rng=np.random.RandomState(0); torch.manual_seed(0)
DS=['kai','vis','coffee','xvla']; NT=len(DS)
def cc(a,b): return np.corrcoef(a,b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else np.nan
# 载入缓存 per-ep
DATA={d:[] for d in DS}
for di,d in enumerate(DS):
    z=np.load(OUT/f'{d}.npz'); eps=z['ep']
    for e in eps: DATA[d].append((z[f'f{e}'], z[f'v{e}'], z[f't{e}'], di))   # feat128, teacher, T, task
    rng.shuffle(DATA[d])
split={d:int(len(DATA[d])*0.85) for d in DS}
TR={d:DATA[d][:split[d]] for d in DS}; EV={d:DATA[d][split[d]:] for d in DS}
print('train/eval eps:',{d:(len(TR[d]),len(EV[d])) for d in DS},flush=True)

class GRUv(nn.Module):
    def __init__(s,h=256,L=2,emb=0):
        super().__init__(); s.emb=nn.Embedding(NT,emb) if emb>0 else None; din=128+(emb if emb>0 else 0)
        s.g=nn.GRU(din,h,L,batch_first=True); s.head=nn.Sequential(nn.Linear(h,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x,task,lens):
        if s.emb is not None:
            e=s.emb(task)[:,None,:].expand(-1,x.shape[1],-1); x=torch.cat([x,e],-1)
        p=nn.utils.rnn.pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=False)
        o,_=s.g(p); o,_=nn.utils.rnn.pad_packed_sequence(o,batch_first=True); return torch.sigmoid(s.head(o)).squeeze(-1)

def make_batches(pool,bs=24):  # pool = list of (feat,teacher,T,task)
    order=sorted(range(len(pool)),key=lambda i:len(pool[i][0])); rng.shuffle(order)
    order=sorted(order,key=lambda i:len(pool[i][0]))
    for k in range(0,len(order),bs):
        gp=[pool[i] for i in order[k:k+bs]]; L=max(len(a[0]) for a in gp); B=len(gp)
        X=np.zeros((B,L,128),np.float32); Y=np.zeros((B,L),np.float32); M=np.zeros((B,L),np.float32); tk=np.zeros(B,int); ln=np.zeros(B,int)
        for b,(f,v,t,ti) in enumerate(gp):
            n=len(f); X[b,:n]=f; Y[b,:n]=v; M[b,:n]=1; tk[b]=ti; ln[b]=n
        yield (torch.tensor(X,device=DEV),torch.tensor(Y,device=DEV),torch.tensor(M,device=DEV),torch.tensor(tk,device=DEV),torch.tensor(ln))

@torch.no_grad()
def evalnet(net,tasks):
    net.eval(); res={}
    for d in tasks:
        ct=[]; cT=[]
        for f,v,t,ti in EV[d]:
            pr=net(torch.tensor(f[None],device=DEV),torch.tensor([ti],device=DEV),torch.tensor([len(f)]))[0].cpu().numpy()
            ct.append(cc(pr,v)); cT.append(cc(pr,t))
        res[d]=(np.nanmean(ct),np.nanmean(cT))
    return res

def train(pool,emb,h,ep=40,tasks=DS,cap=250):
    net=GRUv(h=h,emb=emb).to(DEV); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep)
    npar=sum(p.numel() for p in net.parameters())
    for e in range(ep):
        net.train()
        # 平衡采样: 每任务 cap 条
        epool=[]
        for d in tasks:
            src=TR[d]; idx=rng.choice(len(src),min(cap,len(src)),replace=False); epool+=[src[i] for i in idx]
        for X,Y,M,tk,ln in make_batches(epool):
            pr=net(X,tk,ln); loss=(((pr-Y)**2)*M).sum()/M.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net,npar

print('\n=== 配置对比 (per-task 留出 corr vs teacher) ===',flush=True)
rows=[]
# A) 单任务: 每任务独立小模型
sing={}
for d in DS:
    net,npar=train(TR[d],emb=0,h=256,tasks=[d],cap=100000)
    sing[d]=evalnet(net,[d])[d][0]
print(f"{'单任务×4 (各~0.68M)':<26} " + " ".join(f'{d}={sing[d]:.3f}' for d in DS) + f"  mean={np.mean(list(sing.values())):.3f}",flush=True)
# B) 共享 无task条件
netB,nB=train(None,emb=0,h=256); rB=evalnet(netB,DS)
print(f"{'共享 无task ('+f'{nB/1e6:.2f}M)':<26} " + " ".join(f'{d}={rB[d][0]:.3f}' for d in DS) + f"  mean={np.mean([rB[d][0] for d in DS]):.3f}",flush=True)
# C) 共享 +task-embed
netC,nC=train(None,emb=32,h=256); rC=evalnet(netC,DS)
print(f"{'共享 +task32 ('+f'{nC/1e6:.2f}M)':<26} " + " ".join(f'{d}={rC[d][0]:.3f}' for d in DS) + f"  mean={np.mean([rC[d][0] for d in DS]):.3f}",flush=True)
# D) 共享 +task 大
netD,nD=train(None,emb=32,h=384); rD=evalnet(netD,DS)
print(f"{'共享 +task32 大 ('+f'{nD/1e6:.2f}M)':<26} " + " ".join(f'{d}={rD[d][0]:.3f}' for d in DS) + f"  mean={np.mean([rD[d][0] for d in DS]):.3f}",flush=True)
torch.save(netC.state_dict(),Path('/home/tim/workspace/deepdive_kai0/temp/crave_multitask_gru.pt'))
print('DONE',flush=True)
