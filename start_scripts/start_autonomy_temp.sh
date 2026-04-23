#./start_scripts/start_autonomy.sh --execute config_name:=pi05_stand_box_kai0_allgood_25k
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_stand_box_kai0_allgood_25k/t10_allgood_25k/24999 prompt:='stand up the fallen box'

# ★ 当前激活: Task_P Stage 2 best ckpt @ step 3000 — RTC OFF mode (A/B 对照) ★
# 配置: gf1 bs=128, 8k 配置提前停 (overfit @ step 4000+), ema_decay=0.999, peak_lr=2.5e-5
# Val MAE@1=0.0206 (比 Stage 1 终值 0.0362 好 43%，比 P-T10 baseline 0.0633 好 67%)
#
# 启动后约 10-15 秒，后台子 shell 自动运行 rtc_apply.sh off，禁用 RTC 做 A/B 对照。
# 切换到 RTC ON 模式：注释下面 ( ... ) & 那一行，重启即可。
( source ros2_ws/install/setup.bash 2>/dev/null || true
  echo "[rtc-trigger] waiting for policy_inference node..."
  until ros2 node list 2>/dev/null | grep -q /policy_inference; do sleep 2; done
  echo "[rtc-trigger] node ready; 5s grace then disabling RTC..."; sleep 5
  bash "$(dirname "$0")/rtc_apply.sh" off
  echo "[rtc-trigger] RTC OFF applied (A/B test mode)" ) &

# 选 A: step 3000 (best val MAE, MAE@1=0.0206) — 当前激活
./start_scripts/start_autonomy.sh --execute config_name:=pi05_pick_place_box_kai0_unfreeze_8k \
  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_8k/p_unfreeze_8k_v1/3000 \
  prompt:='pick and place in box'

# 选 B: step 7999 (lowest train loss=0.0009, but overfit: val MAE@1=0.0219, 6% worse than 3000)
# 做 "train loss 最低 vs val MAE 最低" A/B 对照测试。
# 注释掉上面的 A，取消下面 B 的注释即可切换。
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_pick_place_box_kai0_unfreeze_8k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_8k/p_unfreeze_8k_v1/7999 \
#  prompt:='pick and place in box'

# Task_E vision-unfreeze full-param @ step 1999 (gf1 bs=128, 2000 steps, 2026-04-22)
# inline_eval EMA MAE@1=0.0396 (step 1999). Saved params are EMA-weighted (~18% trained + 82% init).
# Note: 保存的 params = EMA params; live params 在 train_state 不单独提取。
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_stand_box_kai0_unfreeze_2k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_stand_box_kai0_unfreeze_2k/unfreeze_2k_v1/1999 \
#  prompt:='stand up the fallen box'

# Task_P vision-unfreeze full-param @ step 1999 (gf0 bs=128, 2000 steps, 2026-04-22)
# inline_eval EMA MAE@1=0.0362 (step 1999). vs P-T10 baseline 0.0633: 43% better (EMA).
#./start_scripts/start_autonomy.sh --execute #config_name:=pi05_pick_place_box_kai0_unfreeze_2k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/#pi05_pick_place_box_kai0_unfreeze_2k/p_unfreeze_2k_v1/1999 \
#  prompt:='pick and place in box'

# awbc_v1 baseline (binary prompt) - joint_1=0.0050 @ step 20,000
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_flatten_fold_awbc
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/#pi05_flatten_fold_awbc/gf0_awbc_baseline_v2/20000 prompt:='Flatten and fold the #cloth. Advantage: positive'

# awbc_v2_vanilla (dagger + mirror, 温和 aug) - joint_1=0.0048 @ step 29,000 (29999 = 实际最优可用点)
# NOTE: prompt is wrapped in YAML-string quotes (outer '…', inner "…") because the ": " in
# "Advantage: positive" otherwise parses as a YAML mapping → launch rejects dict for a str param.
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_flatten_fold_awbc_v2 \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_awbc_v2/gf1_awbc_v2_vanilla/29999 \
#  prompt:='"Flatten and fold the cloth. Advantage: positive"'

# awbc_v2_robust (dagger + mirror, 激进 aug) - joint_1=0.0051 @ step 29,000
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_flatten_fold_awbc_v2_robust \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_awbc_v2_robust/gf0_awbc_v2_robust_v1/29999 \
#  prompt:='"Flatten and fold the cloth. Advantage: positive"'

# ─────────────────────────────────────────────────────────────────────
# RTC (Real-Time Chunking) — 运行时调整推理频率，不改源码
# 默认: inference_rate=3Hz (每 333ms / ≈10 publish-step 重查策略)
# 在第二个终端运行以下命令改变 RTC 粒度（autonomy 启动后）:
#
#   cd /data1/tim/workspace/deepdive_kai0
#   source ros2_ws/install/setup.bash       # ros2 命令可用
#   ./start_scripts/rtc_apply.sh show       # 查看当前值
#   ./start_scripts/rtc_apply.sh rtc5       # 每 5 步 replan (推荐测试)
#   ./start_scripts/rtc_apply.sh rtc3       # 每 3 步 replan (激进)
#   ./start_scripts/rtc_apply.sh default    # 恢复默认
#
# 对 "抓取瞬间偏" 失败模式，rtc5/rtc3 预期改善 15-30%.
# 实时改参无需重启 policy，可 A/B 对比 rollout.
