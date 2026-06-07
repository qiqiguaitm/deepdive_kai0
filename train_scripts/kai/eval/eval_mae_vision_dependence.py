#!/usr/bin/env python3
"""Why is offline MAE good but the model unusable on the robot?
Measure action-chunk MAE vs GT with (a) REAL images and (b) BLANKED images.
If MAE_blank ~= MAE_real, the model reaches its MAE WITHOUT using vision => it has
learned a proprioceptive open-loop shortcut; MAE cannot see this, the robot can.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

H = (1, 10, 25, 50)
GRIP = [6, 13]


def read_video(path, n):
    import av
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = "AUTO"; out = []
    for fr in c.decode(s):
        out.append(fr.to_ndarray(format="rgb24"))
        if len(out) >= n: break
    c.close()
    a = np.stack(out[:n], 0)
    if a.shape[0] < n: a = np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], 0)], 0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--n-frames", type=int, default=20); ap.add_argument("--n-episodes", type=int, default=5)
    args = ap.parse_args()
    from openpi.policies import policy_config as pc
    from openpi.training import config as tc, checkpoints as ck
    ckpt = Path(args.ckpt).resolve(); val = Path(args.val).resolve()
    pol = pc.create_trained_policy(tc.get_config(args.config), ckpt,
                                   norm_stats=ck.load_norm_stats(ckpt / "assets", args.asset_id))
    eps = [__import__("json").loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()][:args.n_episodes]
    cams = ("top_head", "hand_left", "hand_right")
    acc = {m: {h: [] for h in H} for m in ("real", "blank")}
    accg = {m: {h: [] for h in H} for m in ("real", "blank")}
    for ep in eps:
        ei, L = ep["episode_index"], ep["length"]
        df = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
        state = np.stack([np.asarray(x) for x in df["observation.state"]]).astype(np.float32)
        action = np.stack([np.asarray(x) for x in df["action"]]).astype(np.float32)
        vid = {c: read_video(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", L) for c in cams}
        for k in np.linspace(0, L - max(H) - 1, args.n_frames).astype(int):
            real = {c: vid[c][k] for c in cams}
            blank = {c: np.zeros_like(vid[c][k]) for c in cams}
            for tag, imgs in (("real", real), ("blank", blank)):
                pred = np.asarray(pol.infer({"images": imgs, "state": state[k], "prompt": args.prompt})["actions"])
                ch = min(len(pred), max(H))
                for h in H:
                    if h > ch: continue
                    gt = action[k + 1:k + 1 + h]; ph = pred[:h]
                    acc[tag][h].append(float(np.mean(np.abs(gt - ph))))
                    accg[tag][h].append(float(np.mean(np.abs(gt[:, GRIP] - ph[:, GRIP]))))
    print(f"\nckpt: {ckpt.name}")
    print(f"{'h':>4s} {'MAE_real':>9s} {'MAE_blank':>9s} {'blank/real':>10s} | {'gripMAE_real':>12s} {'gripMAE_blank':>13s} {'g_ratio':>8s}")
    for h in H:
        mr, mb = np.mean(acc['real'][h]), np.mean(acc['blank'][h])
        gr, gb = np.mean(accg['real'][h]), np.mean(accg['blank'][h])
        print(f"{h:4d} {mr:9.4f} {mb:9.4f} {mb/max(mr,1e-9):10.2f}x | {gr:12.4f} {gb:13.4f} {gb/max(gr,1e-9):7.2f}x")
    print("  (blank/real ~1 => MAE achieved WITHOUT vision = open-loop shortcut; >>1 => vision-dependent)")


if __name__ == "__main__":
    main()
