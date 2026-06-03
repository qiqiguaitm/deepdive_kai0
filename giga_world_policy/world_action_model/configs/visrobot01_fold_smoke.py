"""本机 b2 单节点 8卡 冒烟 —— 验证 训练→checkpoint→eval 全链路(非生产,步数极小)。

继承 visrobot01_fold(双 embodiment、修正 norm_stats、held-out 排除即 episodes=),仅缩小步数/
频繁落 ckpt,跑通后再上 AIHC 16卡(visrobot01_fold_16gpu)。单节点 DEEPSPEED 经 accelerate
本地起,不需 IB/pdsh。
"""
import copy

from world_action_model.configs.visrobot01_fold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/visrobot01_fold_smoke"
config["models"]["view_dir"] = "runs/visrobot01_fold_smoke"
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6, 7]
config["dataloaders"]["train"]["batch_size_per_gpu"] = 1   # 冒烟求快/省显存
config["dataloaders"]["train"]["num_workers"] = 4
config["train"].update(
    dict(
        max_steps=30,
        checkpoint_interval=15,          # 第 15、30 步各落一个 → 验证 ckpt + eval 触发
        checkpoint_total_limit=2,
        checkpoint_safe_serialization=True,
        with_ema=True,
        resume=False,
        activation_checkpointing=True,
        mixed_precision="bf16",
        log_interval=1,
    )
)
