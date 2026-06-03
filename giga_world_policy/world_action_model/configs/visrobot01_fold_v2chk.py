"""本机 b2 验证 #2:patched dataloader(num_workers=12 + prefetch_factor + persistent_workers)。
跑 40 步看稳态单步时间 vs 之前 ~8.6s/step 基线。用 GPU1-4(GPU0 留给 eval-agent)。
"""
import copy
from world_action_model.configs.visrobot01_fold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/v2chk"
config["models"]["view_dir"] = "runs/v2chk"
config["launch"]["gpu_ids"] = [1, 2, 3, 4]
config["dataloaders"]["train"]["batch_size_per_gpu"] = 2
config["dataloaders"]["train"]["num_workers"] = 12
config["dataloaders"]["train"]["prefetch_factor"] = 6
config["dataloaders"]["train"]["persistent_workers"] = True
config["train"].update(dict(max_steps=40, checkpoint_interval=10000, checkpoint_total_limit=1,
                            with_ema=False, resume=False, activation_checkpointing=True,
                            mixed_precision="bf16", log_interval=1))
