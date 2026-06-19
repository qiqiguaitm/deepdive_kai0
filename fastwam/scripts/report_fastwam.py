"""FastWAM 完整 report.html —— 版式/各 part 完全参考 gwp episode_report.py:
h2 标题 → held-out 说明 → 视频说明 → 推理性能 → 聚合指标表 → 逐 episode(traj 图 + A 全集长视频 + B 代表窗短视频)。
- action 曲线:纯内联 SVG(fastwam .venv 无 matplotlib),y 轴只由 GT 决定(与 gwp 同公式 → 坐标轴逐格一致)。
- 视频:model.infer 想象 5 关键帧(delta[0,12,24,36,48]),GT/pred 垂直 2 行。

用法(单卡):
  CUDA_VISIBLE_DEVICES=0 EVAL_* ... python scripts/report_fastwam.py --weights <ck.pt> --stats <stats.json> \
    --summary <dual/summary.json> --out_dir <dir> --n_viz_eps 20 --n_metric_eps 100 --max_win_per_ep 6 \
    --nfe 10 --exec_horizon 16 --n_vid_per_ep 3
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR)); sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))
from eval_offline_fold import build_model, prep_image, VAL, VK, HOR, _REPO_DIR  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402

DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]
PI05 = {1: 0.0219, 10: 0.0425, 24: 0.0743, 48: 0.1155}
CSS = """<style>
body{font-family:-apple-system,Segoe UI,monospace;max-width:1180px;margin:24px auto;padding:0 18px;color:#222;line-height:1.5}
h2{border-bottom:2px solid #3a6;padding-bottom:6px} h3{margin-top:26px;color:#3a6}
table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ccc;padding:6px 14px;text-align:center}
th{background:#eef5f0} tr:nth-child(even) td{background:#fafafa}
details{margin:12px 0;border:1px solid #ddd;border-radius:8px;padding:10px 14px;background:#fcfcfc}
summary{cursor:pointer;font-weight:600;padding:4px 0;font-size:15px} summary:hover{color:#3a6}
video{margin:6px 8px 6px 0;border:1px solid #ccc;border-radius:6px;vertical-align:top} img{border:1px solid #eee;border-radius:6px}
.vids{display:flex;flex-wrap:wrap;gap:8px} .note{color:#666;font-size:13px}
ul{line-height:1.7} code{background:#f0f0f0;padding:1px 5px;border-radius:3px}
</style>"""


def svg_series(gt, pred, dim, w=545, h=200, pad=26):
    """单维子图:GT 黑虚线 / pred 红实线,标题=关节名。y 轴只由 GT 决定(与 gwp 同公式 → 两份报告坐标轴完全一致)。"""
    r = float(gt.max() - gt.min()); py = 0.05 * r + 1e-6
    ymin = float(gt.min()) - py; ymax = float(gt.max()) + py
    n = len(gt)
    X = lambda i: pad + i / max(1, n - 1) * (w - 2 * pad)
    def Y(v):
        yy = h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
        return max(float(pad), min(float(h - pad), yy))
    def poly(arr, color, dash=""):
        pts = " ".join(f"{X(i):.1f},{Y(arr[i]):.1f}" for i in range(n))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.3"{da}/>'
    axis = (f'<line x1="{pad}" y1="{h-pad:.0f}" x2="{w-pad}" y2="{h-pad:.0f}" stroke="#ccc" stroke-width="0.6"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad:.0f}" stroke="#ccc" stroke-width="0.6"/>'
            f'<text x="{pad}" y="{pad-6}" font-size="9" fill="#444">[{ymin:.2f},{ymax:.2f}]</text>')
    leg = ('<text x="62" y="14" font-size="9" fill="#222">- - GT</text>'
           '<text x="122" y="14" font-size="9" fill="#d62728">— pred(raw)</text>') if dim == 0 else ""
    return (f'<svg width="{w}" height="{h}" style="margin:2px">'
            f'<text x="3" y="13" font-size="11" font-weight="600" fill="#333">{DIM[dim]}</text>{leg}'
            f'{axis}{poly(gt, "#222", "4,3")}{poly(pred, "#d62728")}</svg>')


def _chw_to_pil(t):  # [3,H,W] in [-1,1] -> PIL RGB
    a = ((t.clamp(-1, 1) + 1) * 0.5 * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def _label(img, txt):
    img = img.copy(); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 86, 15], fill=(0, 0, 0)); d.text((3, 2), txt, fill=(255, 255, 0))
    return img


def _save_2row(gt_pils, pred_pils, path, fps=2):
    """GT(上) / pred(下)垂直 2 行,帧上标行名,save_mp4。"""
    T = min(len(gt_pils), len(pred_pils))
    frames = []
    for i in range(T):
        g = _label(gt_pils[i], "GT"); p = _label(pred_pils[i], "pred(raw)")
        W = max(g.width, p.width); c = Image.new("RGB", (W, g.height + p.height), (0, 0, 0))
        c.paste(g, (0, 0)); c.paste(p, (0, g.height)); frames.append(c)
    save_mp4(frames, path, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--summary", default="")
    ap.add_argument("--n_viz_eps", type=int, default=20); ap.add_argument("--n_metric_eps", type=int, default=100)
    ap.add_argument("--max_win_per_ep", type=int, default=6); ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--exec_horizon", type=int, default=16); ap.add_argument("--n_vid_per_ep", type=int, default=3)
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats = json.load(open(args.stats))
    a_mean = np.array(stats["action"]["default"]["global_mean"]); a_std = np.array(stats["action"]["default"]["global_std"])
    s_mean = np.array(stats["state"]["default"]["global_mean"]); s_std = np.array(stats["state"]["default"]["global_std"])
    _te = os.environ.get("EVAL_TEXT_EMB", "visrobot01_fold")
    t5 = torch.load(list((_REPO_DIR / "data" / "text_embeds_cache" / _te).glob("*.pt"))[0], map_location="cpu", weights_only=False)
    ctx = t5["context"].clone(); cmask = t5["mask"].bool(); ctx[~cmask] = 0.0; cmask = torch.ones_like(cmask)
    if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
    if cmask.ndim == 1: cmask = cmask.unsqueeze(0)
    model = build_model(args.weights)
    from torchcodec.decoders import VideoDecoder

    meta = [json.loads(l) for l in open(f"{VAL}/meta/episodes.jsonl")]
    all_eps = sorted(int(m["episode_index"]) for m in meta)[: args.n_metric_eps]
    viz_eps = [all_eps[i] for i in np.unique(np.linspace(0, len(all_eps) - 1, min(args.n_viz_eps, len(all_eps))).astype(int))]
    print(f"[report] viz {len(viz_eps)} eps", flush=True)

    H = args.exec_horizon; deltas = np.linspace(0, 48, 5).astype(int)
    ctxd = ctx.to(model.device, model.torch_dtype); cmaskd = cmask.to(model.device)
    act_ms, vid_ms = [], []
    blocks = []
    for ep in viz_eps:
        df = pd.read_parquet(f"{VAL}/data/chunk-000/episode_{ep:06d}.parquet")
        gt_all = np.stack(df["action"].to_numpy())[:, :14]; st_all = np.stack(df["observation.state"].to_numpy())[:, :14]
        decs = {k: VideoDecoder(f"{VAL}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4") for k in VK}
        L = len(df)
        wins = list(range(0, max(1, L - 48), args.exec_horizon))  # 对齐 gwp build_window_indices("exec")
        if args.max_win_per_ep and len(wins) > args.max_win_per_ep:
            wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
        def stitch_t(ts):
            fr = {k: decs[k].get_frames_at([min(ts, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK}
            return prep_image(fr)  # [3,384,320] in [-1,1]
        vid_wins = set(wins[:: max(1, len(wins) // max(1, args.n_vid_per_ep))][: args.n_vid_per_ep])
        tp, tg, aes = [], [], []
        gt_full, pred_full = [], []; bvids = []
        for f in wins:
            img = stitch_t(f)
            prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
            gt_act = torch.from_numpy((gt_all[f:f + 48] - a_mean) / (a_std + 1e-8)).float()
            # 动作轨迹(action-serving 路径,与 dual 指标一致)
            t0 = time.time()
            with torch.no_grad():
                oa = model.infer_action(prompt=None, input_image=img, action_horizon=48, proprio=prop,
                                        context=ctxd, context_mask=cmaskd, num_inference_steps=args.nfe, seed=0)
            act_ms.append((time.time() - t0) * 1000)
            pa = oa["action"].float().cpu().numpy() * (a_std + 1e-8) + a_mean
            gt = gt_all[f:f + 48]; n = min(len(pa), len(gt)); aes.append(np.abs(pa[:n] - gt[:n]))
            tp.append(pa[:min(H, n)]); tg.append(gt[:min(H, n)])
            # 想象视频(joint 路径)
            t1 = time.time()
            with torch.no_grad():
                ov = model.infer(prompt=None, input_image=img, num_frames=5, action=gt_act, action_horizon=48,
                                 proprio=prop, text_cfg_scale=1.0, action_cfg_scale=1.0,
                                 num_inference_steps=args.nfe, seed=42, tiled=False, context=ctxd, context_mask=cmaskd)
            vid_ms.append((time.time() - t1) * 1000)
            pv = ov["video"]  # 5 PIL
            gv = [_chw_to_pil(stitch_t(min(f + int(d), L - 1))) for d in deltas]
            gt_full += gv; pred_full += pv
            if f in vid_wins:
                bp = f"ep{ep}_w{f}.mp4"; _save_2row(gv, pv, os.path.join(args.out_dir, bp), args.fps); bvids.append(bp)
        ae = np.concatenate(aes, 0)
        cum = {h: float(ae[:h].mean()) for h in HOR if h <= len(ae)}
        full = f"ep{ep}_full.mp4"; _save_2row(gt_full, pred_full, os.path.join(args.out_dir, full), args.fps)
        P = np.concatenate(tp, 0); G = np.concatenate(tg, 0)
        svgs = "".join(svg_series(G[:, d], P[:, d], d) for d in range(14))
        fullh = f'<p class=note><b>A 全 episode 长视频</b>(沿所有 window 拼):</p><video src="{full}" controls width="320"></video>'
        bvs = "".join(f'<video src="{v}" controls width="220"></video>' for v in bvids)
        bsec = f'<p class=note><b>B 代表窗短视频</b>(各 1s):</p><div class=vids>{bvs}</div>' if bvs else ""
        blocks.append(
            f'<details><summary>ep {ep} &nbsp;|&nbsp; n_win={len(wins)} &nbsp; mae@1={cum.get(1,0):.4f} &nbsp; mae@48={cum.get(48,0):.4f}</summary>'
            f'<p class=note>action 曲线(raw vs GT,沿 exec_horizon 拼接):</p>'
            f'<div style="display:flex;flex-wrap:wrap;max-width:1130px">{svgs}</div>{fullh}{bsec}</details>')
        print(f"[report] ep{ep} cum@48={cum.get(48):.4f}", flush=True)

    # 聚合指标表(读 dual summary 的 cumulative,= gwp 的 action MAE)
    rows_html = ""
    if args.summary and os.path.isfile(args.summary):
        sm = json.load(open(args.summary)); cmm = sm.get("cum_mae", {}); pi = sm.get("pi05", PI05)
        rows_html = "".join(f"<tr><td>{h}</td><td><b>{float(cmm.get(str(h),0)):.4f}</b></td><td>{float(pi.get(str(h),PI05[h])):.4f}</td></tr>" for h in HOR)
        n_metric = sm.get("n_metric_eps", "?")
    else:
        n_metric = "?"
    la = f"action-only <b>{np.mean(act_ms):.0f} ms</b> · with-video {np.mean(vid_ms):.0f} ms · 去噪 {args.nfe} 步"
    ckname = os.path.basename(os.path.dirname(os.path.dirname(args.weights))) + "/" + Path(args.weights).stem
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Episode Report — {ckname}</title>{CSS}</head><body>
<h2>Episode 测试报告 — FastWAM-v6 ({ckname})</h2>
<p class=note>held-out:指标 {n_metric} ep / 可视化 {len(viz_eps)} ep · exec_horizon={H} · <b>开环(非闭环 SR)</b> · 与 gwp_abs_v5 报告同协议/同 episode/同坐标轴</p>
<h3>视频说明</h3><ul>
<li>每视频 <b>2 行</b>(原分辨率,帧上有行名):<b>行1 GT(真值) / 行2 pred(模型 raw 权重想象)</b></li>
<li>每行内为 FastWAM 拼接画面:<b>top_head(俯视,上)/ hand_left | hand_right(左右腕,下)</b></li>
<li><b>A 全 episode 长视频</b>(<code>ep*_full.mp4</code>):沿 exec_horizon 取该集所有 window 拼接,时间轴=window 序列;每 window 5 帧=action chunk 的 delta[0,12,24,36,48]</li>
<li><b>B 代表窗短视频</b>(<code>ep*_w*.mp4</code>):该集 {args.n_vid_per_ep} 个代表 window,各 5 帧 1s(模型 num_frames=5 固定的稀疏长跨度关键帧)</li></ul>
<h3>推理性能</h3><p>{la}</p>
<h3>聚合指标(raw,全 {n_metric} ep · cumulative mae@h)</h3>
<table><tr><th>horizon</th><th>action MAE</th><th>π0.5 参考</th></tr>{rows_html}</table>
<h3>逐 episode(抽样可视化)</h3>{''.join(blocks)}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    print(f"[report] DONE -> {args.out_dir}/report.html", flush=True)


if __name__ == "__main__":
    main()
