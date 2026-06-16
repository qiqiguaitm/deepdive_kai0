"""可视化 CRAVE 三档(pos/normal/neg)打标 vs AWBC-AE 二值打标 —— ep808(dagger, 末帧叠好)。
静态图(整段)+ 同步视频(相机 + 两条 value 按各自标签着色 + 当前帧标签读数)。
CRAVE: 推进=pos(绿)/平台=normal(灰)/回落=neg(红); AE: task_index pos(绿)/neg(红)。
"""
import glob, json, os
import numpy as np, pandas as pd, matplotlib, cv2, av
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path

_sh = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/SimHei.ttf")
if os.path.exists(_sh): fm.fontManager.addfont(_sh)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
MV = REPO / "temp/mv_value_full"; AW = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all_awbc"
DS = REPO / "kai0/data/Task_A/self_built/A_smooth800_dagger_all"
csAW = json.load(open(AW / "meta/info.json"))["chunks_size"]; csDS = json.load(open(DS / "meta/info.json"))["chunks_size"]
EP = 808; W = 50; EPS = 0.02
COL3 = {1: "#2ca02c", 0: "#999999", -1: "#d62728"}; NAME3 = {1: "POSITIVE", 0: "NORMAL", -1: "NEGATIVE"}

cv = np.load(MV / f"ep{EP}.npy").astype(float)
d = pd.read_parquet(AW / "data" / f"chunk-{EP//csAW:03d}" / f"episode_{EP:06d}.parquet", columns=["absolute_value", "task_index"])
ae_v = d["absolute_value"].to_numpy().astype(float); ti = d["task_index"].to_numpy()
n = min(len(cv), len(ae_v)); cv, ae_v, ti = cv[:n], ae_v[:n], ti[:n]
cadv = np.zeros(n)
for i in range(n): cadv[i] = np.clip(cv[min(i + W, n - 1)] - cv[i], -1, 1)
cls = np.where(cadv > EPS, 1, np.where(cadv < -EPS, -1, 0))   # CRAVE 三档
fp = {1: (cls == 1).mean(), 0: (cls == 0).mean(), -1: (cls == -1).mean()}
print(f"ep{EP} {n}帧 CRAVE: pos {fp[1]:.0%}/normal {fp[0]:.0%}/neg {fp[-1]:.0%}; AE: pos {(ti==1).mean():.0%}/neg {(ti==0).mean():.0%}", flush=True)
x = np.arange(n) / 30.0

# ---- 静态图 ----
fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
ax[0].plot(x, cv, color="#333", lw=0.8, alpha=.5)
for c in (-1, 0, 1):
    m = cls == c; ax[0].scatter(x[m], cv[m], s=6, c=COL3[c], label=f"{NAME3[c]} {fp[c]:.0%}")
ax[0].set_ylabel("CRAVE value"); ax[0].set_ylim(-.05, 1.05); ax[0].legend(fontsize=8, loc="upper left", ncol=3)
ax[0].set_title(f"CRAVE 三档打标 (pos 推进 / normal 平台执行 / neg 回落)", fontsize=10); ax[0].grid(alpha=.2)
ax[1].plot(x, ae_v, color="#333", lw=0.8, alpha=.5)
for c, nm, col in [(1, "POSITIVE", "#2ca02c"), (0, "NEGATIVE", "#d62728")]:
    m = ti == c; ax[1].scatter(x[m], ae_v[m], s=6, c=col, label=f"{nm} {(ti==c).mean():.0%}")
ax[1].set_ylabel("AE absolute_value"); ax[1].set_xlabel("秒"); ax[1].legend(fontsize=8, loc="upper left", ncol=2)
ax[1].set_title(f"AWBC-AE 二值打标 (task_index pos/neg; 专家数据 neg {(ti==0).mean():.0%} 偏高=噪声)", fontsize=10); ax[1].grid(alpha=.2)
fig.suptitle(f"ep{EP}(dagger, 末帧叠好): CRAVE 三档 vs AWBC-AE 二值打标", fontsize=12)
fig.tight_layout(); out = REPO / "docs/visualization/cross_episode_recurrence_value/crave_3level_vs_ae_labels_ep808.png"
fig.savefig(out, dpi=120); print("SAVED", out, flush=True); plt.close(fig)

# ---- 同步视频 ----
vid = DS / "videos" / f"chunk-{EP//csDS:03d}" / "observation.images.top_head" / f"episode_{EP:06d}.mp4"
PFIG = plt.figure(figsize=(9, 5.2), dpi=100); gs = PFIG.add_gridspec(2, 1, hspace=0.5)
a0 = PFIG.add_subplot(gs[0]); a0.plot(x, cv, color="#333", lw=0.7, alpha=.4)
for c in (-1, 0, 1): m = cls == c; a0.scatter(x[m], cv[m], s=4, c=COL3[c])
a0.set_ylim(-.05, 1.05); a0.set_xlim(0, n / 30); a0.set_ylabel("CRAVE", fontsize=8); a0.tick_params(labelsize=7)
a0.set_title("CRAVE 三档: 绿=pos 灰=normal 红=neg", fontsize=9); a0.grid(alpha=.2)
a1 = PFIG.add_subplot(gs[1], sharex=a0); a1.plot(x, ae_v, color="#333", lw=0.7, alpha=.4)
a1.scatter(x[ti == 1], ae_v[ti == 1], s=4, c="#2ca02c"); a1.scatter(x[ti == 0], ae_v[ti == 0], s=4, c="#d62728")
a1.set_ylim(-.2, .8); a1.set_ylabel("AE", fontsize=8); a1.set_xlabel("秒", fontsize=8); a1.tick_params(labelsize=7)
a1.set_title("AE 二值: 绿=pos 红=neg(专家数据 neg 偏多=噪声)", fontsize=9); a1.grid(alpha=.2)
PFIG.suptitle(f"ep{EP} — CRAVE 三档 vs AWBC-AE 二值 打标", fontsize=11)
PFIG.canvas.draw(); PANEL = np.asarray(PFIG.canvas.buffer_rgba())[..., :3].copy(); Hp, Wp = PANEL.shape[:2]


def pm(a):
    bb = a.get_position(); xl, xh = a.get_xlim(); yl, yh = a.get_ylim(); return bb.x0, bb.x1, bb.y0, bb.y1, xl, xh, yl, yh


def pxy(m, sec, val):
    x0, x1, y0, y1, xl, xh, yl, yh = m
    return (int(round((x0 + (sec - xl) / (xh - xl) * (x1 - x0)) * Wp)), int(round((1 - (y0 + (val - yl) / (yh - yl) * (y1 - y0))) * Hp)))


def ysp(m): _, _, y0, y1, *_ = m; return int(round((1 - y1) * Hp)), int(round((1 - y0) * Hp))
M0, M1 = pm(a0), pm(a1); S0, S1 = ysp(M0), ysp(M1); plt.close(PFIG)
c0 = av.open(str(vid)); f0 = next(c0.decode(video=0)).to_ndarray(format="rgb24"); c0.close()
csc = Hp / f0.shape[0]; cw2 = int(round(f0.shape[1] * csc)) // 2 * 2; Wt = (cw2 + Wp) // 2 * 2; Ht = Hp // 2 * 2
omp4 = str(REPO / f"temp/crave_3level_ep{EP}.mp4")
oc = av.open(omp4, mode="w"); stv = oc.add_stream("libx264", rate=30); stv.width, stv.height, stv.pix_fmt = Wt, Ht, "yuv420p"; stv.options = {"preset": "veryfast", "crf": "23"}
cobj = av.open(str(vid)); i = 0; mid = n // 2
for fr in cobj.decode(video=0):
    if i >= n: break
    panel = PANEL.copy(); sec = i / 30.0
    for m, (yt, yb) in ((M0, S0), (M1, S1)):
        px = pxy(m, sec, 0)[0]; cv2.line(panel, (px, yt), (px, yb), (110, 110, 110), 1)
    gx, gy = pxy(M0, sec, float(cv[i])); col = tuple(int(COL3[cls[i]].lstrip("#")[k:k+2], 16) for k in (0, 2, 4))
    cv2.circle(panel, (gx, gy), 7, col, -1); cv2.circle(panel, (gx, gy), 7, (0, 0, 0), 1)
    rx, ry = pxy(M1, sec, float(ae_v[i])); rc = (44, 160, 44) if ti[i] == 1 else (214, 39, 40)
    cv2.circle(panel, (rx, ry), 7, rc, -1); cv2.circle(panel, (rx, ry), 7, (0, 0, 0), 1)
    cam2 = cv2.resize(np.ascontiguousarray(fr.to_ndarray(format="rgb24")), (cw2, Hp))
    cv2.rectangle(cam2, (6, 6), (270, 78), (0, 0, 0), -1)
    cv2.putText(cam2, f"ep{EP} {i}/{n}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cam2, f"CRAVE: {NAME3[cls[i]]}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col[::-1], 2, cv2.LINE_AA)
    cv2.putText(cam2, f"AE: {'POSITIVE' if ti[i]==1 else 'NEGATIVE'}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc[::-1], 2, cv2.LINE_AA)
    canv = np.zeros((Hp, cw2 + Wp, 3), np.uint8); canv[:, :cw2] = cam2; canv[:, cw2:] = panel; frame = np.ascontiguousarray(canv[:Ht, :Wt])
    if i == mid: cv2.imwrite(omp4.replace(".mp4", "_preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for pkt in stv.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")): oc.mux(pkt)
    i += 1
    if i % 1500 == 0: print(f"  vid {i}/{n}", flush=True)
cobj.close()
for pkt in stv.encode(): oc.mux(pkt)
oc.close()
print(f"SAVED {omp4} {i}f", flush=True); print("THREELEVEL_VIZ_DONE", flush=True)
