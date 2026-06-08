# Cosmos3 三模型 I2V 世界预测评测 — 结果

> **日期**: 2026-06-06 ｜ **plan**: [`../../future_plans/plans/cosmos3_three_model_i2v_eval_plan.md`](../../future_plans/plans/cosmos3_three_model_i2v_eval_plan.md)
> **产物目录**: `deepdive_kai0/cosmos_eval/` (report.html + report_<model>_<cam>.html ×9 + report_assets/merged/*.mp4 ×81 + results/*.jsonl)
> **环境**: cosmos3-venv (diffusers Cosmos3OmniPipeline, torch2.6/cu124), 本机 8×A100 80G。

## 设置
- 数据: `wam_fold_v1/visrobot01_val`, 3 episodes (102/126/168) × 3 cameras (cam_high/left_wrist/right_wrist)。
- 任务: Image→Video 世界预测, teacher-forced 滑窗 (每窗条件帧=该锚点 GT 帧), horizon ∈ {1s,3s,7s}。
- 锚点/(ep,cam): {1s:4, 3s:2, 7s:2} → 72 windows/model, 共 216 次生成。
- 生成: 480×832, 24fps, 50 步, guidance 6, flow_shift 5, 固定 seed。指标在 4:3 中心裁→256² / 24fps 对齐后算。

## 结果 (全 72/model 均值)

| model | hz | n | PSNR↑ | SSIM↑ | Temporal→1 | gen_s/clip |
|---|---|---|---|---|---|---|
| **Cosmos3-Nano** (16B) | 1s | 36 | **12.88** | **0.594** | 43.6 | 26 |
| | 3s | 18 | **12.60** | **0.610** | 4.21 | 79 |
| | 7s | 18 | **12.22** | **0.560** | 1.51 | 263 |
| **Cosmos3-Super** (64B) | 1s | 36 | 11.79 | 0.557 | 41.6 | 92 |
| | 3s | 18 | 11.01 | 0.539 | 5.74 | 283 |
| | 7s | 18 | 10.77 | 0.494 | 1.71 | 957 |
| **Cosmos3-Super-Image2Video** (64B) | 1s | 36 | 11.36 | 0.534 | 43.3 | 92 |
| | 3s | 18 | 10.36 | 0.504 | 5.44 | 283 |
| | 7s | 18 | 10.34 | 0.484 | 2.21 | 957 |

## 主要发现
1. **像素指标上 Nano(16B) > Super(64B) ≈ Super-I2V(64B)** —— 反直觉:小模型在 PSNR/SSIM 上反而最高。原因:更大/I2V-专精的模型从单条件帧动画化更激进、内容更丰富,在 teacher-forced 轨迹下偏离 GT 更多,被像素指标惩罚。**PSNR/SSIM 只衡量"是否贴近真实那条轨迹",不衡量画面真实感/物理合理性**(后者需 LPIPS/FVD/人评,本轮 LPIPS 因 alexnet 下载慢禁用)。须看合并视频定性判断。
2. **所有模型短 horizon 严重 over-animate**: 1s 的 temporal_absdiff_ratio ≈ 40+(GT 1s 内几乎不动,模型却生成大量运动),随 horizon 增大收敛到 ~1.5–2.2(7s)。
3. **PSNR 随 horizon 单调下降**(内容随时间发散),3 模型一致。
4. **算力**: 64B ≈ 3.6× Nano/clip (7s: 957s vs 263s)。64B 单卡放不下(120G bf16),用自定义 device_map 跨 2 卡分片(glue 全在 cuda:0,仅 layers.* 拆分)。

## 复现/查看
- 查看: 浏览器开 `cosmos_eval/report.html` (总表+链接) → 各 `report_<model>_<cam>.html` (episode×horizon 网格, 每格 GT|预测 并排合并视频)。
- 重跑: `cd cosmos_eval && python gen_worklist.py --n_anchor "1s:4,3s:2,7s:2" && python run_all.py --gpus 0,1,2,3,4,5,6,7`。

## 关键工程坑 (已解, 见 memory cosmos3-i2v-eval-setup)
- transformers 5.x import 引用 `torch.float8_e8m0fnu`(torch≥2.7)→ torch2.6 下加一行 shim。
- 流水线默认下载 gated guardrail(HF 401)→ `from_pretrained(enable_safety_checker=False)`。
- 64B 跨卡: 不能用 pipeline `device_map="balanced"`(只分组件,不拆 transformer→CPU offload);需 `Cosmos3OmniTransformer.from_pretrained(device_map=自定义)` 且 patch `transformer_cosmos3.py` 一处 device 对齐。

## 局限
- 覆盖是**抽样锚点**(非全 episode 平铺,因 64B 7s=16min/clip 全平铺不可行);合并视频按锚点拼接,非连续整段。
- 无 LPIPS/FVD(下载受限);像素指标对"更动态但合理"的预测不友好——定性以合并视频为准。
