"""FastWAM 完整 report.html(对齐 gwp episode_report 风格):
- 指标表:single-step + cumulative mae@{1,10,24,48} vs π0.5 基线(读已算好的 summary.json)
- 逐 viz-episode:action 曲线(raw pred vs GT,沿 exec_horizon 拼接)PNG,14 维网格

用法(单卡):
  CUDA_VISIBLE_DEVICES=0 EVAL_VAL_ROOT=.. EVAL_VIEW_KEYS=.. EVAL_DATA=.. EVAL_TASK=.. EVAL_TEXT_EMB=.. \
  python scripts/report_fastwam.py --weights <ck.pt> --stats <stats.json> --summary <dual/summary.json> \
    --out_dir <report_dir> --n_viz_eps 20 --max_win_per_ep 6 --nfe 10 --exec_horizon 16 --n_metric_eps 100
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))
from eval_offline_fold import build_model, prep_image, VAL, VK, HOR, _REPO_DIR  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402


def _chw_to_pil(t):  # [3,H,W] in [-1,1] -> PIL RGB
    a = ((t.clamp(-1, 1) + 1) * 0.5 * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def _vstack2(top, bot):  # GT 行(上) / pred 行(下)
    W = max(top.width, bot.width)
    c = Image.new("RGB", (W, top.height + bot.height), (0, 0, 0))
    c.paste(top, (0, 0)); c.paste(bot, (0, top.height)); return c


# 关节名,与 gwp episode_report.py 完全一致
DIM = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]


def svg_series(gt, pred, dim, w=545, h=200, pad=26):
    """单维子图,复刻 gwp matplotlib 风格:GT 黑虚线(k--)+ pred 红实线(r-),标题=关节名。
    y 轴范围**只由 GT 决定**(两份报告 GT 相同 → 坐标轴完全一致);pred 超界裁到视口边缘。"""
    r = float(gt.max() - gt.min()); py = 0.05 * r + 1e-6
    ymin = float(gt.min()) - py; ymax = float(gt.max()) + py
    n = len(gt)
    X = lambda i: pad + i / max(1, n - 1) * (w - 2 * pad)
    def Y(v):
        yy = h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)
        return max(float(pad), min(float(h - pad), yy))  # clamp 到视口(超界 pred 不跑飞)
    def poly(arr, color, dash=""):
        pts = " ".join(f"{X(i):.1f},{Y(arr[i]):.1f}" for i in range(n))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.3"{da}/>'
    axis = (f'<line x1="{pad}" y1="{h-pad:.0f}" x2="{w-pad}" y2="{h-pad:.0f}" stroke="#ccc" stroke-width="0.6"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad:.0f}" stroke="#ccc" stroke-width="0.6"/>'
            f'<text x="{pad}" y="{pad-6}" font-size="9" fill="#444">[{ymin:.2f},{ymax:.2f}]</text>')
    leg = ('<text x="60" y="14" font-size="9" fill="#222">- - GT</text>'
           '<text x="120" y="14" font-size="9" fill="#d62728">— pred(raw)</text>') if dim == 0 else ""
    return (f'<svg width="{w}" height="{h}" style="margin:2px">'
            f'<text x="3" y="13" font-size="11" font-weight="600" fill="#333">{DIM[dim]}</text>{leg}'
            f'{axis}{poly(gt, "#222", "4,3")}{poly(pred, "#d62728")}</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--summary", default="")
    ap.add_argument("--n_viz_eps", type=int, default=20); ap.add_argument("--n_metric_eps", type=int, default=100)
    ap.add_argument("--max_win_per_ep", type=int, default=6); ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--exec_horizon", type=int, default=16)
    ap.add_argument("--n_vid_per_ep", type=int, default=0, help=">0 则每集生成 N 个 GT-vs-pred 想象视频(model.infer)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats = json.load(open(args.stats))
    a_mean = np.array(stats["action"]["default"]["global_mean"]); a_std = np.array(stats["action"]["default"]["global_std"])
    s_mean = np.array(stats["state"]["default"]["global_mean"]); s_std = np.array(stats["state"]["default"]["global_std"])

    _te = os.environ.get("EVAL_TEXT_EMB", "visrobot01_fold")
    cache = list((_REPO_DIR / "data" / "text_embeds_cache" / _te).glob("*.pt"))[0]
    t5 = torch.load(cache, map_location="cpu", weights_only=False)
    ctx = t5["context"].clone(); cmask = t5["mask"].bool()
    ctx[~cmask] = 0.0; cmask = torch.ones_like(cmask)
    if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
    if cmask.ndim == 1: cmask = cmask.unsqueeze(0)

    model = build_model(args.weights)
    from torchcodec.decoders import VideoDecoder

    meta = [json.loads(l) for l in open(f"{VAL}/meta/episodes.jsonl")]
    all_eps = sorted(int(m["episode_index"]) for m in meta)[: args.n_metric_eps]
    viz_eps = [all_eps[i] for i in np.unique(np.linspace(0, len(all_eps) - 1, min(args.n_viz_eps, len(all_eps))).astype(int))]
    print(f"[report] viz {len(viz_eps)} eps, exec_horizon={args.exec_horizon} nfe={args.nfe}", flush=True)

    H = args.exec_horizon
    blocks = []
    for ep in viz_eps:
        df = pd.read_parquet(f"{VAL}/data/chunk-000/episode_{ep:06d}.parquet")
        gt_all = np.stack(df["action"].to_numpy())[:, :14]
        st_all = np.stack(df["observation.state"].to_numpy())[:, :14]
        decs = {k: VideoDecoder(f"{VAL}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4") for k in VK}
        L = len(df)
        # 对齐 gwp build_window_indices("exec"):stride=exec_horizon、上界 L-action_chunk(保证未来有完整 48 步 GT)
        wins = list(range(0, max(1, L - 48), args.exec_horizon))
        if args.max_win_per_ep and len(wins) > args.max_win_per_ep:
            wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.max_win_per_ep).astype(int))]
        pred_cat, gt_cat, aes = [], [], []
        for f in wins:
            frames = {k: decs[k].get_frames_at([min(f, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK}
            img = prep_image(frames)
            prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
            with torch.no_grad():
                out = model.infer_action(prompt=None, input_image=img, action_horizon=48, proprio=prop,
                                         context=ctx.to(model.device, model.torch_dtype),
                                         context_mask=cmask.to(model.device), num_inference_steps=args.nfe, seed=0)
            pa = out["action"].float().cpu().numpy() * (a_std + 1e-8) + a_mean
            gt = gt_all[f:f + 48]; n = min(len(pa), len(gt))
            aes.append(np.abs(pa[:n] - gt[:n]))
            pred_cat.append(pa[:min(H, n)]); gt_cat.append(gt[:min(H, n)])
        ae = np.concatenate(aes, 0)
        ss = {h: float(ae[h - 1].mean()) for h in HOR if h <= len(ae)}
        cum = {h: float(ae[:h].mean()) for h in HOR if h <= len(ae)}
        P = np.concatenate(pred_cat, 0); G = np.concatenate(gt_cat, 0)  # [W*H, 14]
        svgs = "".join(svg_series(G[:, d], P[:, d], d) for d in range(14))

        # 可选:GT-vs-pred 想象视频(model.infer,5 关键帧 delta[0,12,24,36,48])
        vids_html = ""
        if args.n_vid_per_ep > 0:
            deltas = np.linspace(0, 48, 5).astype(int)
            vid_wins = [wins[i] for i in np.unique(np.linspace(0, len(wins) - 1, args.n_vid_per_ep).astype(int))]
            def stitch_pil(ts):
                fr = {k: decs[k].get_frames_at([min(ts, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK}
                return _chw_to_pil(prep_image(fr))
            for f in vid_wins:
                try:
                    img = stitch_pil(f); img_t = prep_image({k: decs[k].get_frames_at([min(f, decs[k].metadata.num_frames - 1)]).data[0].permute(1, 2, 0).numpy() for k in VK})
                    prop = torch.from_numpy((st_all[f] - s_mean) / (s_std + 1e-8)).float()
                    gt_act = torch.from_numpy((gt_all[f:f + 48] - a_mean) / (a_std + 1e-8)).float()
                    with torch.no_grad():
                        pred = model.infer(prompt=None, input_image=img_t, num_frames=5, action=gt_act, action_horizon=48,
                                           proprio=prop, text_cfg_scale=1.0, action_cfg_scale=1.0,
                                           num_inference_steps=args.nfe, seed=42, tiled=False,
                                           context=ctx.to(model.device, model.torch_dtype), context_mask=cmask.to(model.device))
                    pv = pred["video"]  # 5 PIL (W,H)=(320,384)
                    gt_pils = [stitch_pil(min(f + int(d), len(gt_all) - 1)) for d in deltas]
                    comb = [_vstack2(g, p) for g, p in zip(gt_pils, pv)]
                    vp = f"ep{ep}_w{f}.mp4"; save_mp4(comb, os.path.join(args.out_dir, vp), fps=2)
                    vids_html += f'<video src="{vp}" controls width="300"></video>'
                except Exception as e:
                    print(f"[report] ep{ep} w{f} video FAIL: {repr(e)[:120]}", flush=True)
            if vids_html:
                vids_html = (f'<p class=note>GT-vs-pred 想象视频(上=GT,下=pred,5 关键帧 delta[0,12,24,36,48]):</p>{vids_html}')

        ssl = " ".join(f"@{h}={ss.get(h, float('nan')):.4f}" for h in HOR)
        cml = " ".join(f"@{h}={cum.get(h, float('nan')):.4f}" for h in HOR)
        blocks.append(f'<details open><summary>ep {ep} &nbsp;|&nbsp; ss[{ssl}] &nbsp; cum[{cml}]</summary>'
                      f'<p class=note>episode {ep} — deploy-style action traj (raw, exec_h={H}) · GT 黑虚线 / pred 红实线</p>'
                      f'<div style="display:flex;flex-wrap:wrap;max-width:1130px">{svgs}</div>{vids_html}</details>')
        print(f"[report] ep{ep} ss@48={ss.get(48):.4f} cum@48={cum.get(48):.4f}", flush=True)

    # 指标表(读已算好的 summary.json;无则留空)
    table = ""
    if args.summary and os.path.isfile(args.summary):
        sm = json.load(open(args.summary))
        ssm = sm.get("raw_mae", {}); cmm = sm.get("cum_mae", {}); pi = sm.get("pi05", {})
        rows = "".join(
            f"<tr><td>mae@{h}</td><td><b>{float(ssm.get(str(h),0)):.4f}</b></td>"
            f"<td><b>{float(cmm.get(str(h),0)):.4f}</b></td><td>{float(pi.get(str(h),0)):.4f}</td></tr>"
            for h in HOR)
        table = (f'<table><tr><th>horizon</th><th>single-step</th><th>cumulative</th><th>π0.5 baseline</th></tr>{rows}</table>'
                 f'<p class=note>n_metric_eps={sm.get("n_metric_eps")} · act-lat={sm.get("latency",{}).get("action_ms",0):.0f}ms</p>')

    html = f"""<!doctype html><html><head><meta charset=utf-8><title>FastWAM-v6 step50000 report</title>
<style>body{{font-family:-apple-system,Arial,sans-serif;margin:24px;max-width:1200px}}
table{{border-collapse:collapse;margin:8px 0}} th,td{{border:1px solid #ccc;padding:5px 12px;text-align:center;font-size:13px}}
th{{background:#f6f6f6}} td:first-child{{text-align:left;font-weight:600}} .note{{font-size:12px;color:#888}}
details{{margin:10px 0;border:1px solid #eee;border-radius:6px;padding:6px}} summary{{cursor:pointer;font-size:13px}}</style></head><body>
<h2>FastWAM-v6 (独立 ActionDiT) — step 50000 完整报告</h2>
<p class=note>held-out visrobot01_v3_val · 同 gwp 协议(n_eps={args.n_metric_eps}, max_win={args.max_win_per_ep}, denoise={args.nfe}) · 开环(非闭环 SR)</p>
<h3>聚合指标(double metric)</h3>{table}
<p class=note>说明:single-step mae@h = 第 h 步那一步误差;cumulative mae@h = 前 h 步平均误差;@1 处两者相等。</p>
<h3>逐 episode action 曲线(raw pred vs GT,抽样 {len(viz_eps)} ep)</h3>{''.join(blocks)}
</body></html>"""
    open(os.path.join(args.out_dir, "report.html"), "w").write(html)
    print(f"[report] DONE -> {args.out_dir}/report.html", flush=True)


if __name__ == "__main__":
    main()
