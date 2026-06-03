"""visrobot01 叠衣服 full-FT —— 百度 AIHC 2节点×8 A100 = 16 卡变体。

继承 visrobot01_fold(已含:双 embodiment、修正版 norm_stats、held-out 排除即 episodes=、
tolerance_s 修复),仅覆盖 full-FT 生产旋钮。

注意:AIHC 下用 `accelerate launch`(scripts/aihc/run_train_aihc.sh)拉起,分布式拓扑
(num_machines/num_processes/machine_rank/master)由 accelerate CLI + AIHC 注入的
WORLD_SIZE/RANK/MASTER_ADDR 决定,**不读 config.launch**。故这里 launch 段仅占位。
真正生效的是 dataloaders / models / optimizers / train。
"""
import copy

from world_action_model.configs.visrobot01_fold import config as _base

config = copy.deepcopy(_base)

# 每卡 micro-batch:5B 全参 + 视频 latent 很重(dreamzero 同 backbone 全参用 bs=1+zero2_offload)。
# zero2(不 offload)更省时但更吃显存 → 保守 bs=2 + activation_checkpointing;OOM 则降 1 或开 offload。
config["dataloaders"]["train"]["batch_size_per_gpu"] = 2

config["train"].update(
    dict(
        max_steps=100000,
        checkpoint_interval=5000,        # 每 5000 步落一个 → eval-agent 触发一次评估
        checkpoint_total_limit=10,       # 只留最近 10 个(5B ckpt 很大),自动删旧
        checkpoint_safe_serialization=True,   # safetensors(与 inference_server 一致、更安全)
        with_ema=True,                   # 存 EMA 副本(transformer_ema/,推理首选)
        resume=True,                     # 断点续训(多机长跑必开;从 project_dir 最近 ckpt 恢复)
        activation_checkpointing=True,
        mixed_precision="bf16",
        log_interval=10,
    )
)
