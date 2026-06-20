"""混训 2-step smoke:验证跨本体(visrobot+kairobot)dataloading + 两套 norm + K=2,再开全量。"""
import copy
from world_action_model.configs.acwm_clothfold_mix import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/acwm_clothfold_mix_smoke"
config["models"]["view_dir"] = "runs/acwm_clothfold_mix_smoke"
config["launch"]["gpu_ids"] = [0]
config["train"]["resume"] = False
config["train"]["max_steps"] = 4          # 跑几步覆盖两库采样
config["train"]["max_epochs"] = 0
config["train"]["checkpoint_interval"] = 100000
config["train"]["log_interval"] = 1
config["dataloaders"]["train"]["batch_size_per_gpu"] = 1
config["dataloaders"]["train"]["num_workers"] = 2
config["models"]["eval_fold"]["enabled"] = False
