# 左臂标定误差诊断与修复：左臂关节零位漂移

> 日期：2026-06-02 ｜ 数据：`calib/data/recalib/`（head + 左臂 + 右臂同会话采集）
> 状态：已用软件 offset 治标（左臂 9.5px→3.9px）；根治待重标左臂物理关节零位。

## 摘要

下游反馈左臂标定"不准"（未给具体现象）。经逐层排查，根因定位为**左臂关节零位漂移**——大臂(j2)、腕俯仰(j5)、末端(j6)的零点各偏约 +2.4° / +1.9° / +1.6°，使 FK 把左臂末端位姿系统性算偏 ~20mm，hand-eye 标定无法自洽，端到端重投影误差 9.5–12.7px（右臂仅 2.7–3.8px）。已用软件 offset（关节零位修正 δq）修复，左臂端到端降到 3.9px（与右臂持平）。**这是治标**：δq 是等效修正、绑定具体 FK 实现；根治需重标左臂物理关节零位。

## 现象与验证手段

- **现象**：下游只说"不准"，无量化症状。
- **验证手段**（新增 `calib/verify_projection.py`）：用完整标定链把 board 3D 角点重投影回图像，比检测到的 charuco 角点，得像素误差，并分层（L0 内参 / L1 hand-eye 自洽 / L2 全链路重投影 / L3 跨相机）。
- **量化基线**：左臂端到端重投影 9.5–12.7px，右臂 2.7–3.8px。

## 证据链（排除法 → 正面坐实）

| # | 主张 | 证据 | 结论 |
|---|---|---|---|
| 1 | 不是内参 | 左臂单帧 PnP 残差 0.11px | 排除内参/检测 |
| 2 | 不是世界系/两臂拼接 | base 对称 ~0mm；两臂对同一固定板的世界位置分歧 1.1mm；head–臂 0.32mm | 排除世界系定义 |
| 3 | 矛盾在左臂自身 | 左臂位姿离散 13.9mm vs 右臂 3.7mm | 定位到左臂 |
| 4 | 不是采集质量/速度 | 重采 3 次（角点 85→102、慢速），左臂始终 9–13px | 排除采集 |
| 5 | 不是机械重复性/回差 | 同姿态反复定位 RMS 0.3mm；各关节散布 <0.05°；异向≈同向（新增 `test_arm_repeatability.py`） | 排除重复性/回差 |
| 6 | 不是相机松动 | 刚体不变量：相机 vs FK 相对运动，旋转失配 左0.14°≈右0.13°，平移 pitch 左0.40mm<右0.48mm | 排除相机相对 link6 松动 |
| 7 | **是 FK 绝对精度** | 关键矛盾：左臂局部相对运动刚体一致，却全局位姿散 8mm——FK 绝对精度误差的指纹 | 指向 FK 关节零位 |

**关键矛盾解读（第 7 行）**：相邻帧间的相对运动（相机测 vs FK 算）一致，是因为相邻两帧都带同样的零位偏差、差分时抵消；但每帧的绝对位姿因构型不同、偏差传到末端的量不同，无法抵消，全局累积成散布。这恰好排除相机（相机松动会先让相邻相对运动对不上），指向"关节角→末端位姿"的 FK 换算。

## 正面坐实（决定性实验）

| 实验 | 结果 | 说明 |
|---|---|---|
| 拟合左臂关节零位 δq | 位姿散布 4.0mm → 1.4mm；右臂救不动(1.8→1.2) | 误差可由关节零位解释，左臂特有 |
| 逐关节归因 | 主因 j5/j3/j2（腕+大臂） | 定位到具体关节 |
| 留出验证（前半拟合/后半验证） | held-out 帧 4.91mm → 1.75mm | 证明非过拟合 |
| **应用 δq 重新求解** | 左臂端到端 9.5px → 3.9px；hand-eye scatter 2.3mm → 0.9mm | 决定性：修 FK 零位 → 标定就对 |
| δq 的物理量级 | 同一编码器读数，FK(q) vs FK(q+δq) 末端差 ~20mm / 7° | 这就是被修正掉的末端误差 |
| 跨相机验证 | 左右臂角点经世界系投到 head：左 2.3px、右 1.9px | offset 外参下跨相机融合一致 |

## 根因

**左臂关节零位漂移**：大臂(j2)/腕俯仰(j5)/末端(j6)的编码器零点与 FK 标称运动学（`PiperFK(dh_is_offset=0x01)`）假设的基准偏差约 1.6–2.4°。FK 用带偏差的关节角算末端位姿，系统性偏 ~20mm，导致 hand-eye 标定不自洽。属硬件/固件层零点标定问题，非内参/相机/采集/代码。

## 解决：关节零位修正 δq

**δq 计算**（见 `calib/verify_out/build_offset_session.py`）：利用"板和 base 都固定 → `T_base_board(i) = FK(q_i+δq)·T_link6_cam·T_cam_board(i)` 对所有帧应恒定"，用 `scipy.least_squares` 求一组 δq 使各帧 `T_base_board` 平移散布最小（bound ±4°）。再把 δq 喂回完整 hand-eye 求解（`T_link6_cam` 一并重解）得 `calibration_offset.yml`。

**结果**：
```
δq (deg) = [-3.987, 2.414, 0.290, -0.510, 1.914, 1.639]   # 对应 j1..j6
```
- 物理可信：j2/j5/j6（~+2.4°/+1.9°/+1.6°）= 真实大臂+腕部零位漂移。
- j1=-3.987° 不可辨识（基座旋转对目标不敏感、顶到边界），数值无物理意义但无害（标定与部署用同一 δq，自洽抵消）。

**使用约束**（绑定）：
```python
q_corrected = q_left_rad + np.radians([-3.987, 2.414, 0.290, -0.510, 1.914, 1.639])
T_baseL_link6 = fk.fk_homogeneous(q_corrected)   # 仅左臂；右臂/head 不加
```
配 `calibration_offset.yml`。`calibration_offset.yml` + δq 是一套，要么都用要么都不用。

## 局限与根治

- **治标**：δq 是等效修正，吸收了真实零位 + DH 长度误差 + 外参残差，**绑定具体 FK 实现**。下游若用不同 FK（如 MuJoCo），不能直接搬 δq，须用相同观测在该 FK 上重拟合（已验证：搬运到 MuJoCo 仅部分迁移，残留 ~8px 为 FK 模型差异）。
- **根治**：重标左臂物理关节零位（重点 j2/j5/j6）。之后 δq 归零、`T_link6_camL` 重解，跨 FK 通用。
- **流程改进**：标定前增加左臂关节零位/重复性检查，避免再踩。

## 相关文件

新增工具/测试：
- `calib/verify_projection.py` — 投影验证 + 分层诊断（L0–L3）
- `calib/test_verify_projection.py` — 单元测试（6 项，含真实数据 board 编号 sanity）
- `calib/test_arm_repeatability.py` — 机械臂定位重复性测试（纯关节角+FK，不连相机）

改动：
- `calib/capture_handeye.py` — replay 新增 `--speed` 参数（默认不变）

产出（`calib/verify_out/`）：
- `calibration_offset.yml` — offset 修复后的标定
- `left_joint_offset_deg.json` — 左臂 δq（部署 FK 必须加）
- `left_corrected_poses.json` — 20 帧预计算 `fk(q+δq)` 位姿（供下游用不同 FK 时复现/重拟合）
- 分析脚本：`build_offset_session.py`（拟合 δq + 重解）、`fuse_world.py`、`fuse_pointcloud.py`、`reproject_to_head.py`
- 可视化：`verify_report.html`、`fuse_world.html`、`fuse_pointcloud.html`、`reproject_to_head.png`

设计/计划：`calib/docs/specs/2026-06-01-calib-projection-verify-design.md`、`calib/docs/plans/2026-06-01-calib-projection-verify.md`

## 复现命令

```bash
PY=/data1/miniconda3/envs/e3d/bin/python      # cv2 4.11；系统 python3 (cv2 4.6) 会 segfault
cd /data1/tim/workspace/deepdive_kai0/calib

# 验证某份标定（分层诊断 + HTML 报告）
$PY verify_projection.py --session data/recalib

# 测试
$PY -m pytest test_verify_projection.py -v -p no:dash

# 左臂重复性测试（连臂，系统 python3，会动臂）
/usr/bin/python3 test_arm_repeatability.py --arm left --can can_left_slave --speed 20

# 拟合 δq + 重解 offset 标定
$PY verify_out/build_offset_session.py
$PY solve_calibration.py --session verify_out/offset_session --output verify_out/calibration_offset.yml
```
