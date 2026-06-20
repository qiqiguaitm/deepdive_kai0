"""1-step smoke 配置:验证 ACWMTrainer.forward_step 改写无误(单卡, 2 步, batch=1, 关 eval)。"""
import copy

from world_action_model.configs.acwm_clothfold import config as _base

config = copy.deepcopy(_base)

config["project_dir"] = "runs/acwm_clothfold_smoke"
config["models"]["view_dir"] = "runs/acwm_clothfold_smoke"

# 单卡
config["launch"]["gpu_ids"] = [0]

# 极短训练
config["train"]["resume"] = False
config["train"]["max_steps"] = 2
config["train"]["max_epochs"] = 0
config["train"]["checkpoint_interval"] = 100000
config["train"]["log_interval"] = 1

# 最小 batch / worker
config["dataloaders"]["train"]["batch_size_per_gpu"] = 1
config["dataloaders"]["train"]["num_workers"] = 2

# 关掉内联 fold 评测(smoke 只验前反向)
config["models"]["eval_fold"]["enabled"] = False
