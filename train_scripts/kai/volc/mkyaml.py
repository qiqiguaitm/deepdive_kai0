#!/usr/bin/env python
"""从**一份可移植 body** 生成 cnsh / northe 两个集群的 volc yaml。

背景(2026-07-20):同一个实验要在两个集群跑时, 过去是手工复制 yaml 再改路径 ——
  实测 entrypoint 里有 684 处集群相关硬编码, 漏改一处就是一次失败提交(今日已发生多次)。

分工:
  · **头部字段**(ImageUrl / ResourceQueueName / Flavor / VepfsId / MountPath / SubPath)
    物理上不可能统一 → 由本脚本按 profile 注入。
  · **Entrypoint** 必须可移植 → body 里一律 `source $VOLC_DIR/_cluster_env.sh` 后使用
    $REPO / $PYTHON / $CRAVE_REPO / $LMVLA_LIBERO_ROOT / $ROBOTWIN_PATH / $ROBOTWIN_PYTHON,
    **不得出现 /vePFS/tim、/vePFS-North-E、/home/tim、/vePFS/HuanQian 等字面量**(本脚本会检查并拒绝)。

用法:
  python mkyaml.py body.yaml --cluster northe --gpus 8 -o out.yaml
  python mkyaml.py body.yaml --cluster both  --gpus 8          # 生成 <body>_{cnsh,northe}.yaml
"""
import argparse, os, re, sys
import yaml

PROFILES = {
    "cnsh": {
        "ImageUrl": "visincept-cn-shanghai.cr.volces.com/grasp/kai:kai0-gf0",
        "ResourceQueueName": "robot-task",
        "VepfsId": "vepfs-cnsh075262e1f815",
        "SubPath": "",
        "MountPath": "/vePFS",
        "flavors": {8: "ml.hpcpni2.28xlarge"},          # 8×A100-80G 整节点
        "repo": "/vePFS/tim/workspace/deepdive_kai0",
    },
    "northe": {
        "ImageUrl": "dvs-cr-cn-beijing.cr.volces.com/vis_robot/kai:kai0-gf1",
        "ResourceQueueName": "Robot-North-H20",
        "VepfsId": "vepfs-cnbj875793a96d6b",
        "SubPath": "/vis_robot",
        "MountPath": "/vePFS-North-E/vis_robot",
        # ⚠️ 该队列**无 4 卡规格**(实测 ml.pni3ln.20xlarge 报 InvalidParameter: flavorId)
        "flavors": {1: "ml.pni3ln.5xlarge", 8: "ml.hpcpni3ln.45xlarge"},
        "repo": "/vePFS-North-E/vis_robot/workspace/deepdive_kai0",
    },
}

# entrypoint 里禁止出现的集群字面量
FORBIDDEN = [r"/vePFS/tim\b", r"/vePFS-North-E", r"/home/tim\b", r"/vePFS/HuanQian"]


def check_portable(entrypoint: str) -> list[str]:
    bad = []
    for pat in FORBIDDEN:
        for m in re.finditer(pat, entrypoint):
            line = entrypoint[:m.start()].count("\n") + 1
            bad.append(f"  第{line}行: {pat} → 应改用 $REPO / $ROBOTWIN_PATH 等变量")
    return bad


def build(body: dict, cluster: str, gpus: int) -> dict:
    p = PROFILES[cluster]
    d = dict(body)
    d["ImageUrl"] = p["ImageUrl"]
    d["ResourceQueueName"] = p["ResourceQueueName"]
    if gpus not in p["flavors"]:
        raise SystemExit(f"❌ {cluster} 无 {gpus} 卡规格;可用: {sorted(p['flavors'])}")
    for role in d.get("TaskRoleSpecs", []):
        role["Flavor"] = p["flavors"][gpus]
    d["Storages"] = [{"Type": "Vepfs", "VepfsId": p["VepfsId"],
                      "SubPath": p["SubPath"], "MountPath": p["MountPath"]}]
    # entrypoint 顶部注入 bootstrap(若 body 未自行 source)
    ep = d.get("Entrypoint", "")
    if "_cluster_env.sh" not in ep:
        boot = f'source {p["repo"]}/train_scripts/kai/volc/_cluster_env.sh\n'
        # 在首行 `set -...`(可带 pipefail 等后续词)之后注入; 匹配不到则置于开头。
        new_ep, n = re.subn(r"(^\s*set [^\n]*\n)", lambda m: m.group(1) + boot, ep, count=1, flags=re.M)
        ep = new_ep if n else (boot + ep)
        d["Entrypoint"] = ep
        # ★ 硬校验: 注入必须真的发生(此处曾因正则不匹配而静默失败, 生成出 $REPO 未定义的 yaml)
        assert "_cluster_env.sh" in d["Entrypoint"], "bootstrap 注入失败"
    name = d.get("TaskName", "job")
    d["TaskName"] = f"{name}-{cluster}" if not name.endswith(cluster) else name
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument("--cluster", choices=["cnsh", "northe", "both"], required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("-o", "--out")
    ap.add_argument("--allow-hardcode", action="store_true", help="跳过可移植性检查(不建议)")
    a = ap.parse_args()

    body = yaml.safe_load(open(a.body, encoding="utf-8"))
    bad = check_portable(body.get("Entrypoint", ""))
    if bad and not a.allow_hardcode:
        print(f"❌ {a.body} 的 Entrypoint 含 {len(bad)} 处集群硬编码, 无法跨集群复用:")
        print("\n".join(bad[:12]))
        if len(bad) > 12: print(f"  ... 另有 {len(bad)-12} 处")
        sys.exit(1)

    targets = ["cnsh", "northe"] if a.cluster == "both" else [a.cluster]
    for c in targets:
        d = build(body, c, a.gpus)
        out = a.out if (a.out and a.cluster != "both") else \
            os.path.splitext(a.body)[0].replace("_body", "") + f"_{c}.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False, width=4096)
        print(f"✅ {c}: {out}  (queue={PROFILES[c]['ResourceQueueName']}, "
              f"flavor={PROFILES[c]['flavors'][a.gpus]})")


if __name__ == "__main__":
    main()
