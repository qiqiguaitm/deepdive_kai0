# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``wam_fold_wm_nano`` — Cosmos3-Nano FORWARD-DYNAMICS world-model posttrain on wam_fold_v1.

World-model counterpart of ``wam_fold_nano`` (which trains the POLICY mode from
Policy-DROID): here the BASE ``nvidia/Cosmos3-Nano`` (mid-trained) is fine-tuned in
``forward_dynamics`` mode — action tokens are CLEAN conditioning, the diffusion loss
is on VIDEO tokens — to predict future video of the dual-arm cloth-fold scene given
a 14-D action chunk. Plan: docs/training/future_plans/plans/cosmos3_wam_fold_world_model_plan.md

Differences vs ``wam_fold_nano`` (everything else mirrors it):

  * dataset ``mode="forward_dynamics"``, ``chunk_length=32`` (≈1.07 s @30 FPS;
    obs window 33 frames = 4k+1, exact for the Wan VAE temporal stride);
  * init from BASE Cosmos3-Nano (not Policy-DROID): same fresh-init of the action
    I/O modules via ``keys_to_skip_loading`` (our domain ids 16/17 + 14-D joint
    space are new), per the official action recipe;
  * lr 2e-4 (official action_policy_droid recipe) with 5x multiplier on the
    fresh action modules (tech report §4.2.5). Fallback if smoke diverges: 5e-5;
  * ``max_batch_size=8`` (32-frame samples are ~2x the tokens of 16-frame ones;
    the 45056 token budget is the real cap);
  * separate latent cache env ``WAM_WM_LATENT_CACHE`` (keys carry ``L32`` so they
    can share a dir with the L16 policy cache, but keep them apart for hygiene).

Usage::

    BASE_CKPT_DCP=<base_nano_dcp> torchrun --nproc_per_node=8 \\
        -m cosmos_framework.scripts.train \\
        --sft-toml=wam_fold_wm/train/recipe_wm_nano.toml
"""

from __future__ import annotations

import copy
import os

import torch.utils.data
from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.action.posttrain_config.wam_fold_nano import (
    _MAX_ACTION_DIM,
    ActionDataPacker,
)
from cosmos_framework.data.vfm.action.datasets.wam_fold_dataset import WamFoldLeRobotDataset
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader

cs = ConfigStore.instance()

_NANO_CFG = copy.deepcopy(NANO_MODEL_CONFIG)
# Same rationale as wam_fold_nano: "language" region compiles the heavy MoT GEMMs and
# starts up faster than "all" (measured no-op for speed there).
_NANO_CFG["compile"]["compiled_region"] = "language"


def build_wm_data_source(
    chunk_length: int = 32,
    fps: float = 30.0,
    mode: str = "forward_dynamics",
) -> torch.utils.data.ConcatDataset:
    """Cross-rig FD data source: visrobot01_train x3 + kairobot01 (same mix as policy).

    Mode is forced to ``forward_dynamics`` per sample — post-trained Cosmos3 models
    specialize to a single mode (tech report §6.3.1), so no ``joint`` mixing here.
    """
    vis = WamFoldLeRobotDataset(
        rig="visrobot01", split="train", mode=mode, chunk_length=chunk_length, fps=fps
    )
    kai = WamFoldLeRobotDataset(rig="kairobot01", mode=mode, chunk_length=chunk_length, fps=fps)
    return torch.utils.data.ConcatDataset([vis, vis, vis, kai])


wam_fold_wm_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action",
            name="wam_fold_wm_nano",
            wandb_mode="offline",
        ),
        model=dict(
            config=_NANO_CFG,
        ),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[],  # full fine-tune
            # Official action recipe lr (fresh action I/O modules + 5x multiplier on them,
            # tech report §4.2.5). Substring-matched against param names (optimizer.py:144).
            lr=2.0e-04,
            lr_multipliers={
                "action2llm.": 5.0,
                "llm2action.": 5.0,
                "action_modality_embed": 5.0,
                "action_pos_embed.": 5.0,
            },
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            # One cosine cycle must span the FULL run (sum < max_iter crashes the
            # LambdaCosine cycle lookup — see recipe_nano.toml CRASH FIX note).
            cycle_lengths=[10000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[30],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=300,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                iter_speed=dict(every_n=1, hit_thres=100000, save_s3=False, save_s3_every_log_n=500),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                skip_nan_step=dict(max_consecutive_nan=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Fresh-init the action I/O modules: new domain ids (16/17) + 14-D joint
            # space. llm2action gets no video-loss gradient in FD mode but resetting
            # it is harmless and keeps parity with the policy config.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm.",
                "llm2action.",
                "action_modality_embed",
                "action_pos_embed.",
            ],
            load_ema_to_reg=False,
            load_path="${oc.env:BASE_CKPT_DCP}",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(DataPackerDataLoader)(
            data_source=L(build_wm_data_source)(
                chunk_length=32,
                fps=30.0,
                mode="forward_dynamics",
            ),
            data_packer=L(ActionDataPacker)(
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                resolution="480",
                max_action_dim=_MAX_ACTION_DIM,
                # Action CFG dropout — the lever for inference-time action-branch
                # guidance (plan §2). Keep 0.1; bump to 0.15 only if the Δaction
                # perturbation gate fails (plan Phase 2).
                cfg_dropout_rate=0.1,
                append_idle_frames=True,
                # Separate cache env from the policy run (keys carry L32 anyway).
                latent_cache_dir=(os.environ.get("WAM_WM_LATENT_CACHE") or None),
            ),
            max_tokens=45056,
            # 32-frame samples ≈ 3.7k tokens each (~2x the 16-frame policy samples);
            # token budget is the real cap, this bounds pack-size variance (the
            # policy run's batch-24 OOM lesson).
            max_batch_size=8,
            pool_size=8,
            shuffle=True,
            seed=42,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [wam_fold_wm_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
