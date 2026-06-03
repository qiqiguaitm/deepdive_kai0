"""b2 验证 latent 训练全链路:dataset 读缓存 latent + transform 透传 + episode-grouped sampler +
trainer skip-VAE。只用 visrobot01_train(已缓存部分集),GPU1-4(GPU0 留 eval)。跑 30 步看
单步时间(应 ~1-2s,远快于 mp4 的 ~9s)+ loss 正常。
"""
dst_size = (256, 192); num_frames = 48; action_dim = 14
DATA = "../kai0/data/wam_fold_v1"
view_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
norm_path = ["./assets_visrobot01/norm_stats_vis.json", "./assets_visrobot01/norm_stats_kai.json"]


def _entry(emb, data_path):
    return dict(_class_name="LeRobotDataset", data_path=data_path, data_size=None, embodiment=emb,
                delta_info={"action": num_frames}, tolerance_s=1e-3,
                skip_video_decoding=True,                          # latent 模式不解视频
                latent_dir=f"{data_path}/vae_latent", latent_cache_size=8)


config = dict(
    project_dir="runs/latent_chk",
    runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
    launch=dict(gpu_ids=[1, 2, 3, 4], distributed_type='DEEPSPEED',
                deepspeed_config=dict(deepspeed_config_file='accelerate_configs/zero2.json'), until_completion=True),
    dataloaders=dict(train=dict(
        data_or_config=[_entry("visrobot01", f"{DATA}/visrobot01_train")],
        batch_size_per_gpu=2, num_workers=8, prefetch_factor=6, persistent_workers=True,
        transform=dict(type='WATransformsLerobot', robotype_to_embed_id={"visrobot01": 0, "kairobot01": 1},
                       dst_size=dst_size, num_frames=num_frames, is_train=True, norm_path=norm_path,
                       model_action_dim=action_dim, num_views=3, t5_len=64, view_keys=view_keys,
                       image_cfg=dict(mask_generator=dict(max_ref_frames=1, start=1, factor=4))),
        sampler=dict(type='LatentEpisodeSampler', stride=4),
        collator=dict(is_equal=True)), test=dict()),
    models=dict(pretrained=f"../checkpoints/Wan2.2-TI2V-5B-Diffusers", strict=False, action_dim=action_dim,
                flow_shift=5.0, expand_timesteps=True, view_dir="runs/latent_chk", state_repeats=1),
    optimizers=dict(type='CAME8Bit', lr=2 ** (-14.5), weight_decay=1e-2),
    schedulers=dict(type='ConstantScheduler'),
    train=dict(resume=False, max_epochs=0, max_steps=30, gradient_accumulation_steps=1, mixed_precision='bf16',
               checkpoint_interval=10000, checkpoint_total_limit=1, with_ema=False, activation_checkpointing=True,
               log_with='tensorboard', log_interval=1),
    test=dict(),
)
