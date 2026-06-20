"""离线评测 kai0 π₀.₅(A_smooth800_dagger_full)在 visrobot01_val,同 GWP/FastWAM episode_report 协议:
exec coverage(stride=exec_horizon=16)、action_chunk=48、首 200 episode、raw mae@{1,10,24,48}。
复用 serve_policy_v1 的 V1Policy/load_v1_inference(tim 部署栈),abs 输出直接对 GT(use_delta=False)。
逐 episode 均值再跨 ep 均值(匹配 episode_report)。

本机适配版(原 jpsz /data1/tim/eval_kai0_pi05.py):路径改为本机 deepdive_kai0 布局。
"""
import sys, os, json, glob, argparse, time

_WS = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0"
sys.path.insert(0, f"{_WS}/kai0/scripts")
sys.path.insert(0, f"{_WS}/optimize/v1_triton")
import numpy as np, torch, av
import pyarrow.parquet as pq
import serve_policy_v1 as SP
import pi05_infer_tuned as _PIT
SP.Pi05InferenceTuned = _PIT.Pi05InferenceTuned  # 注入懒加载全局
from serve_policy_v1 import V1Policy, load_v1_inference, load_norm_stats, SentencepieceStateEncoder

CK = os.environ.get("EVAL_CK", f"{_WS}/checkpoints/ckpt_v1/A_smooth800_dagger_full_step49999")
VAL = os.environ.get("EVAL_VAL_ROOT", f"{_WS}/kai0/data/wam_fold_v1/visrobot01_val")
TOK = os.environ.get("EVAL_TOK", f"{_WS}/openpi_cache/big_vision/paligemma_tokenizer.model")
_vk_env = os.environ.get("EVAL_VIEW_KEYS")
VK = _vk_env.split(",") if _vk_env else ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
HOR = [1, 10, 24, 48]; AC = 48; EXEC = 16

ap = argparse.ArgumentParser()
ap.add_argument("--n_eps", type=int, default=200)
ap.add_argument("--sanity", type=int, default=0, help=">0: 只跑前 N 窗口,打印 pred vs GT vs state 量级")
a = ap.parse_args()

# ---- 构建 policy(镜像 serve_policy_v1.main,abs)----
v1_infer, emb_w = load_v1_inference(CK + "/v1_p200.pkl", num_views=3, chunk_size=50)
norm = load_norm_stats(glob.glob(CK + "/assets/*/norm_stats.json")[0])
a_stats, s_stats = norm["actions"], norm["state"]
se = SentencepieceStateEncoder(v1_infer, tokenizer_model_path=TOK, embedding_weight=emb_w,
                               state_norm=s_stats, model_state_dim=len(a_stats["mean"]))
policy = V1Policy(v1_infer, action_norm=a_stats, action_dim=14, state_encoder=se,
                  default_prompt="Flatten and fold the cloth.", image_keys=tuple(VK))

# ---- 窗口枚举(同 build_window_indices, exec stride=16, 首 n_eps 集)----
eps = sorted([json.loads(l) for l in open(f"{VAL}/meta/episodes.jsonl") if l.strip()],
             key=lambda e: int(e["episode_index"]))[:a.n_eps]

def decode_ep(ei):
    fr = {}
    for cam in VK:
        p = glob.glob(f"{VAL}/videos/*/{cam}/episode_{ei:06d}.mp4")[0]
        c = av.open(p); c.streams.video[0].thread_type = "AUTO"
        fr[cam] = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]); c.close()
    return fr

ep_maes = {h: [] for h in HOR}; ep_cmaes = {h: [] for h in HOR}
t0 = time.time(); nwin = 0
for e in eps:
    ei, L = int(e["episode_index"]), int(e["length"])
    pq_t = pq.read_table(f"{VAL}/data/chunk-000/episode_{ei:06d}.parquet")
    acts = np.array(pq_t.column("action").to_pylist(), dtype=np.float32)      # (L,14)
    states = np.array(pq_t.column("observation.state").to_pylist(), dtype=np.float32)
    fr = decode_ep(ei); Lf = fr[VK[0]].shape[0]
    starts = list(range(0, max(1, L - AC), EXEC))
    win_mae = {h: [] for h in HOR}; win_cmae = {h: [] for h in HOR}
    for s in starts:
        obs = {"images": {cam: fr[cam][min(s, Lf - 1)] for cam in VK},
               "state": states[s], "prompt": "Flatten and fold the cloth."}
        pred = policy.infer(obs)["actions"][:AC]                              # (48,14) abs denorm
        gt = acts[s:s + AC]; Lm = min(len(pred), len(gt))
        ae = np.abs(pred[:Lm] - gt[:Lm])
        for h in HOR:
            if h <= Lm:
                win_mae[h].append(float(ae[h - 1].mean()))   # single-step
                win_cmae[h].append(float(ae[:h].mean()))     # cumulative
        nwin += 1
        if a.sanity and nwin <= a.sanity:
            print(f"  win ep{ei} s{s}: mae@1={np.abs(pred[0]-gt[0]).mean():.4f} "
                  f"pred[0][:4]={np.round(pred[0][:4],3)} gt[0][:4]={np.round(gt[0][:4],3)} "
                  f"state[:4]={np.round(states[s][:4],3)}", flush=True)
    for h in HOR:
        if win_mae[h]: ep_maes[h].append(np.mean(win_mae[h]))
        if win_cmae[h]: ep_cmaes[h].append(np.mean(win_cmae[h]))
    if a.sanity and nwin >= a.sanity: break
    if ei % 20 == 0: print(f"[{ei}] {nwin} win, {time.time()-t0:.0f}s", flush=True)

if not a.sanity:
    res = {str(h): float(np.mean(ep_maes[h])) for h in HOR}
    cres = {str(h): float(np.mean(ep_cmaes[h])) for h in HOR}
    print("RESULT", json.dumps({"n_eps": len(eps), "n_win": nwin, "raw_mae": res, "cum_mae": cres}), flush=True)
    print("=== kai0 π₀.₅ A_smooth800_dagger_full | %s %dep ===" % (os.path.basename(VAL.rstrip("/")), len(eps)))
    print("  single-step: " + " ".join(f"@{h}={res[str(h)]:.4f}" for h in HOR))
    print("  cumulative : " + " ".join(f"@{h}={cres[str(h)]:.4f}" for h in HOR))
