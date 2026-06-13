"""逐 step 对齐对比两个 run 的 raw mae@{1,10,24,48}(Δ=A−B,负=A 更好)。
用于 ANS vs naive-lookahead 等 same-step A/B。
用法:
  python scripts/wam_pipeline/cmp_step_by_step.py \
    --run_a runs/visrobot01_fold_abs_ans --run_b runs/visrobot01_fold_abs_lookahead [--from_step 8000]
"""
import argparse
import glob
import json
import os

H = ("1", "10", "24", "48")


def curve(run):
    o = {}
    for f in glob.glob(f"{run}/report_step*/summary.json"):
        s = int(os.path.basename(os.path.dirname(f)).replace("report_step", ""))
        try:
            r = json.load(open(f))["raw_mae"]
            v = {h: r[h] for h in H}
            if all(x == x for x in v.values()):
                o[s] = v
        except Exception:
            pass
    return dict(sorted(o.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", default="runs/visrobot01_fold_abs_ans")
    ap.add_argument("--run_b", default="runs/visrobot01_fold_abs_lookahead")
    ap.add_argument("--from_step", type=int, default=0)
    args = ap.parse_args()
    a, b = curve(args.run_a), curve(args.run_b)
    common = [s for s in a if s in b and s >= args.from_step]
    na, nb = os.path.basename(args.run_a).replace("visrobot01_fold_", ""), os.path.basename(args.run_b).replace("visrobot01_fold_", "")
    print(f"=== {na} vs {nb}  (Δ = {na}−{nb}, 负=A更好) ===")
    print("       |        @1        |        @10       |        @24       |        @48")
    print(" step  |  A     B      Δ  |  A     B      Δ  |  A     B      Δ  |  A     B      Δ")
    for s in common:
        cells = " | ".join(f"{a[s][h]:.4f} {b[s][h]:.4f} {a[s][h]-b[s][h]:+.4f}" for h in H)
        print(f"{s:6d} | {cells}")
    if common:
        import statistics as st
        hi = [s for s in common if s >= max(common) - 6000]  # 末段 ~6k 平滑
        print(f"\n末段({hi[0]//1000}-{hi[-1]//1000}k)平滑 Δ: " +
              " ".join(f"@{h}={st.mean(a[s][h]-b[s][h] for s in hi):+.4f}" for h in H))
    print(f"\nA 已评 {len(a)} 点 / B {len(b)} 点 / 共同(≥{args.from_step}) {len(common)} 点"
          + ("  ⚠️ step≤5k cold-start 无意义" if common and min(common) <= 5000 else ""))


if __name__ == "__main__":
    main()
