"""本机 b2 持续验证 —— 1000 步,7 卡训练(GPU0-6)+ 解耦 eval-agent 占 GPU7。

验证长跑稳定性(loss 趋势、显存、周期 checkpoint)与"触发即评估"的指标-step 曲线。
继承 visrobot01_fold(visrobot01_train + 修正 norm_stats + 双 embodiment)。生产 bs=2。
"""
import copy

from world_action_model.configs.visrobot01_fold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/visrobot01_fold_1k"
config["models"]["view_dir"] = "runs/visrobot01_fold_1k"
config["launch"]["gpu_ids"] = [0, 1, 2, 3, 4, 5, 6]   # 留 GPU7 给并发 eval-agent
config["dataloaders"]["train"]["batch_size_per_gpu"] = 2
config["dataloaders"]["train"]["num_workers"] = 4
config["train"].update(
    dict(
        max_steps=1000,
        checkpoint_interval=200,          # 200/400/600/800/1000 → 5 个 ckpt,eval 逐个评
        checkpoint_total_limit=10,
        checkpoint_safe_serialization=True,
        with_ema=True,
        resume=False,
        activation_checkpointing=True,
        mixed_precision="bf16",
        log_interval=10,
    )
)
