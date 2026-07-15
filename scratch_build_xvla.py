import sys, numpy as np, cv2, h5py
from pathlib import Path
sys.path.insert(0, "crave/src")

REPO = Path("/home/tim/workspace/deepdive_kai0")
d = np.load(REPO / "temp/xreb_cache_xvla.npz")
F, E, Pord, order, cen = d["F"], d["E"], d["Pord"], d["order"], d["cen"]

feat = F[:, :1280].astype(np.float16)
n = len(E)

# FR per contiguous episode block
FR = np.zeros(n, np.int64)
i = 0
while i < n:
    j = i
    while j < n and E[j] == E[i]:
        j += 1
    FR[i:j] = np.arange(j - i, dtype=np.int64)
    i = j
T = (FR / 30.0).astype(np.float32)
E = E.astype(np.int64)

proto = cen[order][:, :1280].astype(np.float32)
pord = Pord.astype(np.float32)

out = REPO / "temp/xvla_dinov3h"; out.mkdir(parents=True, exist_ok=True)
np.savez(out / "index.npz", E=E, FR=FR, T=T, n=np.int64(n))
np.savez(out / "shard_0.npz", gidx=np.arange(n, dtype=np.int64), feat=feat, valid=np.ones(n, bool))
rg = REPO / "lmwm/data/recurrence_graphs/xvla_dinov3h"; rg.mkdir(parents=True, exist_ok=True)
np.savez(rg / "recurrence_graph.npz", prototype_table=proto, pord=pord)
print("wrote xvla: n=%d feat=%s proto=%s pord[%.3f,%.3f]" % (n, feat.shape, proto.shape, pord.min(), pord.max()))

# --- cosine cross-check: fresh-encode a couple frames ---
from crave.encoders import load_encoder
enc = load_encoder("dinov3-h", device="cuda")
xdir = REPO / "xvla/data/xvla_soft_fold/0707_11pm_stage_1_stage2new_new_cam_very_slow"
checks = [(0, 0), (0, 100), (2, 50)]  # (episode, frame within ep)
imgs = []; gidxs = []
for ep, fr in checks:
    f = h5py.File(xdir / f"episode_{ep}.hdf5", "r")
    raw = f["observations/images/cam_high"][fr]
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1]
    imgs.append(cv2.resize(rgb, (256, 256)))
    # global index = start of ep block + fr
    gi = np.where(E == ep)[0][0] + fr
    gidxs.append(gi)
    f.close()
fresh = enc.encode_pooled(np.stack(imgs))
for (ep, fr), gi, fe in zip(checks, gidxs, fresh):
    cache = F[gi, :1280].astype(np.float32)
    cos = float(fe @ cache / (np.linalg.norm(fe) * np.linalg.norm(cache) + 1e-8))
    print(f"ep{ep} fr{fr} gidx{gi} cosine={cos:.5f}")
