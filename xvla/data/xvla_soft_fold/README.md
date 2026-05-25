# XVLA-Soft-Fold Dataset (Facebear)

> **物理存储**: js01:`/DATA/disk0/tim/datasets/xvla_soft_fold/` (14TB NVMe, 9.9TB free)
> **本地路径**: 此目录仅含 manifest, 不存储 hdf5 — 数据集 476GB 远超本地空间
> **下载时间**: 2026-05-20 启动

## 数据集来源
- HF: <https://huggingface.co/datasets/Facebear/XVLA-Soft-Fold>
- License: MIT
- Size: ~476 GB, 1542 HDF5 episodes
- Task: Soft cloth folding (X-VLA 论文使用的数据集)

## 子目录结构 (21 个 subset, 按采集日期+任务变体)

| Subset | Episodes | 备注 |
|---|---:|---|
| 0707_11pm_stage_1_stage2new_new_cam_very_slow | 218 | 主线 stage1+2 慢速 |
| 0929_11am_new | 212 | 9 月新采集 |
| 0714_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve | 204 | 无袖变体 |
| 0711_10am_stage_1_stage2new_new_cam_very_slow | 114 | 主线 |
| 0930_10am_new | 108 | 9 月新采集 |
| 0712_8pm_stage_1_stage2new_new_cam_very_slow_grasp_corner | 104 | 抓边角变体 |
| 0709_10am_stage_1_stage2new_new_cam_very_slow | 101 | 主线 |
| 0928_10am_new | 87 | 9 月新采集 |
| 0702_21pm_stage_1_stage2new_new_cam_very_slow | 82 | 早期主线 |
| 0705_13pm_stage_1_stage2new_new_cam_very_slow | 76 | 主线 |
| 0731_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve | 63 | 无袖 |
| 0804_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve | 61 | 无袖 |
| 0801_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve | 59 | 无袖 |
| 0710_11am_stage_1_stage2new_new_cam_very_slow | 41 | 主线 |
| 0712_8pm_stage_1_stage2new_new_cam_very_slow | 40 | 主线 |
| 0708_11am_stage_1_stage2new_new_cam_very_slow | 36 | 主线 |
| 0805_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve | 30 | 无袖 |
| 0713_8pm_stage_1_stage2new_new_cam_very_slow_no_sleeve | 29 | 无袖 |
| 0808_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve | 26 | 无袖 |
| 0706_17pm_stage_1_stage2new_new_cam_very_slow | 24 | 早期主线 |
| **总计** | **1542 HDF5** | + 9 root metadata |

## 任务变体类型
- **主线** (`stage_1_stage2new_new_cam_very_slow`): 标准 stage 1 (展平) + stage 2 (折叠) 慢速演示
- **无袖** (`_no_sleeve`): 无袖衣物变体, 难度可能稍低
- **抓边角** (`_grasp_corner`): 专门练抓边角技能
- **新版** (`_new`): 9 月新采集 (cam 校准 + 新协议?)

## 在 js01 上访问

```bash
ssh js01
ls /DATA/disk0/tim/datasets/xvla_soft_fold/

# 加载示例 episode
python3 -c "
import h5py
f = h5py.File('/DATA/disk0/tim/datasets/xvla_soft_fold/0702_21pm_stage_1_stage2new_new_cam_very_slow/episode_100.hdf5', 'r')
print(list(f.keys()))
for k, v in f.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: {v.shape} {v.dtype}')
"
```

## 后续利用思路 (与 xvla 项目对齐)

参考 `xvla/README.md`:
- **Soft prompt 实验** (`exp2_soft_prompt_mixed`): XVLA-Soft-Fold 作为 domain 2 (除当前 kai + vis 外的第 3 个 domain)
- **跨本体 SSL pretrain** (战略 `docs/deployment/strategy/cross_embodiment_strategy.md` §7 Tri-track + 详细 `docs/training/future_plans/plans/ssl_phase_pretrain_pipeline.md`):  
  - 加入 V-JEPA / point-track / flow pretraining 数据池
  - 1542 ep × ~308 MB = 475 GB 多样性显著
- **作为 Layer 1 (visual representation) 数据**: 与 A=kai0 (6512 ep) + B=vis (837 ep) 合并训 SSL

## 下载状态

启动: 2026-05-20  
工具: `huggingface-cli download` + `HF_HUB_ENABLE_HF_TRANSFER=1` (Rust 多流加速)  
日志: js01:`/DATA/disk0/tim/logs/hf_xvla_softfold.log`  
预计完成: ~2-4 小时 (取决于 HF Xet CDN 速度)

监控命令:
```bash
ssh js01 'du -sh /DATA/disk0/tim/datasets/xvla_soft_fold/; \
  find /DATA/disk0/tim/datasets/xvla_soft_fold -name "*.hdf5" | wc -l; \
  ps aux | grep huggingface-cli | grep -v grep'
```
