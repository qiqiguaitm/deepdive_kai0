"""acwm_clothfold_mix —— AC-WM 混训:visrobot01_v3_train + kairobot01_v3,验证 visrobot01_v3_val。

前置:必须先运行 scripts/wam_pipeline/run_unify_latents.sh,把两库重抽成统一布局(vae_latent_uni,
      12×48, T_lat=4)。否则两库 latent 布局不一致会视角错位(见 ACWM_README.md)。
"""
import copy

from world_action_model.configs.acwm_clothfold import config as _base

config = copy.deepcopy(_base)
config["project_dir"] = "runs/acwm_clothfold_mix"
config["models"]["view_dir"] = "runs/acwm_clothfold_mix"

DATA = "../kai0/data/wam_fold_v3"
VIS = "observation.images.top_head,observation.images.hand_left,observation.images.hand_right".split(",")
KAI = "observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist".split(",")

# 两库混训(统一 vae_latent_uni);kairobot 视角名不同,故为它单独指定 view_keys。
NUM_FRAMES = 48
OFFS = list(range(0, NUM_FRAMES + 1, 4))
_vis = dict(_class_name="LeRobotDataset", data_path=f"{DATA}/visrobot01_v3_train",
            latent_dir=f"{DATA}/visrobot01_v3_train/vae_latent_uni",
            delta_info={"action": NUM_FRAMES}, delta_frames={k: OFFS for k in VIS}, embodiment="visrobot01")
_kai = dict(_class_name="LeRobotDataset", data_path=f"{DATA}/kairobot01_v3",
            latent_dir=f"{DATA}/kairobot01_v3/vae_latent_uni",
            delta_info={"action": NUM_FRAMES}, delta_frames={k: OFFS for k in KAI}, embodiment="kairobot01")
config["dataloaders"]["train"]["data_or_config"] = [_vis, _kai]

# transform 默认 view_keys 用 visrobot;kairobot 的 view_keys 由其 data entry 的 delta_frames 决定。
config["dataloaders"]["train"]["transform"]["view_keys"] = VIS

# ===== 跨本体(不同机械臂)处理:每本体独立归一化 + embed_id 映射 =====
# kairobot01 与 visrobot01 是不同机械臂:关节运动学/取值范围不同 → 必须各用自己的 norm stats,
# 否则动作条件被错误归一化。giga 用 robotype→embed_id 选 norm_paths[embed_id]:
#   visrobot01→0→norm_stats_vis_abs;  kairobot01→1→norm_stats_kai_abs
# (模型无显式 robot embedding;本体区分另靠视觉 ref/history 帧 + 一致的相机语义序。)
config["dataloaders"]["train"]["transform"]["robotype_to_embed_id"] = {"visrobot01": 0, "kairobot01": 1}
config["dataloaders"]["train"]["transform"]["norm_path"] = [
    "./assets_visrobot01_v3/norm_stats_vis_abs.json",   # embed_id 0: visrobot01
    "./assets_visrobot01/norm_stats_kai_abs.json",       # embed_id 1: kairobot01
]

# 混训按数据量加权(visrobot 2353 : kairobot 6512)
config["dataloaders"]["train"]["sampler"] = dict(type="LatentEpisodeSampler", stride=4, num_frames=50)

# 统一窗 T_lat=4 → 两库都支持 K≤3;混训取 2
config["models"]["num_history"] = 2

# 验证集 latent 也用 uni
config["models"]["eval_fold"]["enabled"] = True
config["models"]["eval_fold"]["val_root"] = f"{DATA}/visrobot01_v3_val"
