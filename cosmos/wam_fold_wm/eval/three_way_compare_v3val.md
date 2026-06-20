# 三方对比:GWP_ABS_v5 vs FASTWAM-v6 vs kai0 π₀.₅ (visrobot01_v3_val, 100ep)

评测日期: 2026-06-19。协议统一: **visrobot01_v3_val 全 100 episode**, exec stride=16, action_chunk=48,
全窗口(无 window cap), 10 去噪步, 逐 episode 均值再跨 ep 均值。三模型均同时输出
**cumulative** `ae[:h].mean()`(团队 2026-06-15 修复后的规范指标) 与 **single-step** `ae[h-1].mean()`(历史指标)。

> ⚠️ 关键:cumulative 与 single-step 在 @48 差近 2×。此前各模型 inline 指标不一致
> (gwp 用 cumulative、fastwam 用 single-step),不可直接比。本表全部重算两套指标,可比。

## Cumulative MAE(规范指标,↓越小越好)

| 模型 | 参数/架构 | @1 | @10 | @24 | @48 |
|------|-----------|------|------|------|------|
| **kai0 π₀.₅** | ~3B 前馈 VLM(无视频分支) | 0.0066 | **0.0170** | **0.0304** | **0.0483** |
| **FASTWAM-v6** (step45000) | 5B MoT(专用 ActionDiT) | **0.0037** | 0.0166 | 0.0337 | 0.0546 |
| **GWP_ABS_v5** (step50000) | 5B 共享 backbone(abs) | 0.0075 | 0.0248 | 0.0520 | 0.0883 |

(注:@10 kai0 0.0170 vs fastwam 0.0166 实质打平)

## Single-step MAE(历史指标,↓越小越好)

| 模型 | @1 | @10 | @24 | @48 |
|------|------|------|------|------|
| **kai0 π₀.₅** | 0.0066 | **0.0272** | **0.0502** | **0.0795** |
| **FASTWAM-v6** | **0.0037** | 0.0294 | 0.0583 | 0.0895 |
| **GWP_ABS_v5** | 0.0075 | 0.0441 | 0.0923 | 0.1508 |

## 评测时延(stock 引擎, 10 步, 共享 A100 — 非部署优化值)

| 模型 | action latency |
|------|----------------|
| kai0 π₀.₅ | (本轮未测;部署优化 RTX5090 ≈43ms) |
| FASTWAM-v6 | 730 ms |
| GWP_ABS_v5 | 539 ms |

> 部署优化值(历史): kai0 ≈43ms · fastwam opt NFE5 exact ≈85ms · gwp_ans fp8 ≈87ms。

## 结论(惊喜点)

1. **前馈 VLM 策略 kai0 π₀.₅ 在 @10/@24/@48 全面领先两个 5B 世界模型**(cumulative 与 single-step 两套指标一致),
   且部署时延最低。再次印证:视频分支对离线 MAE 的长程精度无正贡献,feed-forward 动作头才是杠杆。
2. **FASTWAM-v6 的 @1 最锐**(0.0037,远超 kai0/gwp),@10 与 kai0 打平;但长程(@24/@48)被 kai0 反超。
   专用 MoT ActionDiT 在近期步精度上确有优势。
3. **GWP_ABS_v5(共享 backbone, abs)长程明显最弱**(single-step@48=0.151),与"severed mask / 共享容量被视频任务摊薄"的既有诊断一致。
4. 排名(长程 @48):kai0 < FASTWAM-v6 < GWP_ABS_v5。

## 重要 caveat

- **离线开环 MAE ≠ 闭环成功率(SR)**;@1 含 stay-baseline 假象。最终选型需 best-of-N / 闭环 SR。
- **数据划分**:v5/v6 在 v3_train 训练、v3_val 严格 held-out;kai0(A_smooth800)在 v1 数据训练。
  若 v3_val 与 kai0 训练集有重叠,则 kai0 占便宜——此公平性需另行确认。
- 时延为评测期 stock 引擎(共享 A100),非各模型部署优化引擎的真实值。

## 复现

- kai0: `optimize/v1_triton/eval_kai0_pi05_local.py`(EVAL_VAL_ROOT=v3_val, EVAL_VIEW_KEYS=observation.images.{top_head,hand_left,hand_right})
- fastwam: `/tmp/run_fw6_eval.sh`(eval_offline_fold.py, EVAL_*v3, nfe=10, 全窗口)
- gwp: `/tmp/run_gwp_v5_eval.sh`(episode_report.py, --max_win_per_ep 9999 --n_viz_eps 0 --steps_inf 10)
- 三脚本均已 patch 为同时输出 cumulative + single-step。
- 产物: `fastwam/runs/.../aihc_5n8g_v6/report_step_045000_full_v3/summary.json`,
  `giga_world_policy/runs/gwp_abs_v5/report_step50000_full_v3/summary.json`, `/tmp/kai0_v3_eval_100ep.log`
