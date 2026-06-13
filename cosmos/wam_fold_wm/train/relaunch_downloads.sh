#!/usr/bin/env bash
# Relaunch all unfinished cloth downloads, each fully detached (setsid) so they
# survive session/context resets. modelscope download resumes (skips cached files).
S=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train/ms_download.sh
L=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/reports/downloads

launch() {  # launch <logname> <dataset> <subdir> [include]
  local log="$1"; shift
  setsid bash "$S" "$@" >> "$L/$log.log" 2>&1 &
  echo "  launched $log (pid $!)"
}

# --- big single-connection tars (fast) ---
launch agibot_task570    amap_cvlab/AgiBotWorld-Beta_Lerobot_v2 agibot_lerobot_v2 "task_570.tar.gz.part.*"
launch galaxea_fold1     Galaxea/Galaxea-Open-World-Dataset galaxea "lerobot_opensource/Fold_Clothes20250617_001.tar.gz"
launch galaxea_fold2     Galaxea/Galaxea-Open-World-Dataset galaxea "lerobot_opensource/Fold_Clothes_20250807_011.tar.gz"

# --- LeRobot small-file repos ---
launch robocoin_fold_clothes RoboCOIN/Cobot_Magic_fold_clothes robocoin_fold_clothes
launch full_folding      lerobot/full_folding full_folding
launch xvla_soft_fold    lerobot/xvla-soft-fold xvla_soft_fold
launch r1lite_fold       RoboCOIN/R1_Lite_fold_clothes robocoin_r1lite_fold_clothes
launch robomind_fold     X-Humanoid/RoboMIND2.0-Agilex robomind_agilex_fold "data/agilex/fold_clothes/*"

# --- towel series (brown already done) ---
launch towel_blue         RoboCOIN/Agilex_Cobot_Magic_fold_towel_blue        robocoin_fold_towel_blue
launch towel_tray_twice   RoboCOIN/Agilex_Cobot_Magic_fold_towel_tray_twice  robocoin_fold_towel_tray_twice
launch towel_blue_tray    RoboCOIN/Agilex_Cobot_Magic_fold_towel_blue_tray   robocoin_fold_towel_blue_tray
launch short_sleeve_white RoboCOIN/Agilex_Cobot_Magic_fold_short_sleeve_white robocoin_fold_short_sleeve_white

# --- unitree small ---
launch unitree_g1      unitreerobotics/G1_Fold_Towel unitree_g1_fold_towel
launch unitree_z1dex1  unitreerobotics/Z1_Dual_Dex1_FoldClothes_Dataset unitree_z1_dex1_fold
launch unitree_z1      unitreerobotics/Z1_DualArm_FoldClothes_Dataset unitree_z1_fold
launch unitree_h1      lerobot/unitreeh1_fold_clothes unitree_h1_fold

echo "ALL RELAUNCHED $(date '+%F %T')"
