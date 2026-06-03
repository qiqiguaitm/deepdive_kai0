"""生成可复现的 held-out 验证集清单(visrobot01)。

可复现性保证:
  - 选择 = 固定 seed 的随机抽样,但**最终落盘的是显式 id 列表**(不是让别人重跑 RNG)→
    复现只需读 manifest 里的 id,与 numpy/RNG 版本无关。
  - 记录 episodes.jsonl 的 sha256 指纹 + total_episodes → 数据一旦变动即可检测(hash 不符则报警)。
  - manifest 入 git,训练/评估都读它:train 用 TRAIN_IDS(排除 held-out),eval 用 HELDOUT_IDS,
    构造上零重叠、零数据复制(靠 LeRobotDataset 的 episodes= 子集)。

用法:
  python -m scripts.wam_pipeline.make_heldout --root ../kai0/data/wam_fold_v1/visrobot01 \
      --n 200 --seed 42 --out assets_visrobot01/heldout_visrobot01.json
  # 校验现有 manifest 是否仍与数据一致:
  python -m scripts.wam_pipeline.make_heldout --root ... --out ... --verify
"""
import argparse
import hashlib
import json
import os

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_indices(root):
    eps = []
    with open(os.path.join(root, "meta", "episodes.jsonl")) as f:
        for l in f:
            l = l.strip()
            if l:
                eps.append(int(json.loads(l)["episode_index"]))
    return sorted(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="visrobot01 数据集根目录")
    ap.add_argument("--n", type=int, default=200, help="held-out 集数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="manifest 输出 json")
    ap.add_argument("--verify", action="store_true", help="只校验现有 manifest 与数据一致(不写)")
    args = ap.parse_args()

    all_ids = episode_indices(args.root)
    epsha = sha256_file(os.path.join(args.root, "meta", "episodes.jsonl"))

    if args.verify:
        m = json.load(open(args.out))
        ok = (m["total_episodes"] == len(all_ids) and m["episodes_jsonl_sha256"] == epsha
              and set(m["heldout_episode_indices"]).issubset(set(all_ids)))
        print("VERIFY", "OK" if ok else "MISMATCH",
              f"(manifest n={len(m['heldout_episode_indices'])}, data N={len(all_ids)}, "
              f"sha {'==' if m['episodes_jsonl_sha256']==epsha else '!='})")
        raise SystemExit(0 if ok else 1)

    assert args.n < len(all_ids), f"n={args.n} >= total {len(all_ids)}"
    rng = np.random.default_rng(args.seed)
    heldout = sorted(int(x) for x in rng.choice(all_ids, size=args.n, replace=False))
    train = sorted(set(all_ids) - set(heldout))

    manifest = {
        "dataset": os.path.basename(args.root.rstrip("/")),
        "rule": f"seeded_random(seed={args.seed})",
        "seed": args.seed,
        "total_episodes": len(all_ids),
        "n_heldout": len(heldout),
        "n_train": len(train),
        "episodes_jsonl_sha256": epsha,
        "heldout_episode_indices": heldout,
        "train_episode_indices": train,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    # 互斥 + 全覆盖自检
    assert set(heldout) & set(train) == set() and len(heldout) + len(train) == len(all_ids)
    print(f"wrote {args.out}: total={len(all_ids)} heldout={len(heldout)} train={len(train)} "
          f"sha={epsha[:12]} first5_heldout={heldout[:5]}")


if __name__ == "__main__":
    main()
