# 标定投影验证设计 (2026-06-01)

## 背景

下游用户反馈 `data/calib_/`（head + left + right 一起标定）的内外参"不准"，但**未给具体现象**。需要一个能**定位误差在哪段链路**的验证，而非单一 pass/fail。

已排查：
- 部署用的 `config/calibration.yml` 与 `data/calib_/calibration.yml` **逐元素相同** → 排除"用错文件"。
- 板尺寸 `board_9x14.yaml`（20mm 格）与标定元数据一致 → 排除板尺寸填错。
- 每帧单帧 PnP 残差 ~0.12px → **内参与单帧检测良好**，问题大概率在外参链路。

## 目标

复用 `data/calib_/*.npz` 已采集帧做**纯离线**投影验证，输出一个**交互式 web 报告**，定位"不准"在内参 / hand-eye / FK / 世界系 / 跨相机 哪一段。不连任何硬件。

## 几何回顾（标定链路）

board 是唯一锚点：
- 臂（eye-in-hand）：`T_board_cam(i) = T_board_base · T_base_ee(i) · T_link6_cam`
- head（固定）：`T_world_camF` 直接已知
- 世界系由两臂 base 在 board 系的位置定义（origin = 两 base 中点）

## 验证方法

### 核心一招
board 角点 3D 已知（`board_def.get_board().getChessboardCorners()[ids]`，board 系，米）。用完整标定链预测每帧应投到的像素，与实际检测的 charuco 角点比 → 像素误差。

### 分层（每层只暴露一段）

| 层 | 验证 | 计算 | 判据 | 失败→怀疑 |
|---|---|---|---|---|
| L0 内参 | 单帧 PnP 残差 | 读 npz `reproj_err` 汇总 | <0.5px | 内参/检测 |
| L1 hand-eye 自洽 | 链预测 `T_board_cam(i)` vs 该帧 PnP 解 | 比 `T_board_base·T_base_ee·T_link6_cam` 与 `inv(rvec,tvec)` | 平移 std<3mm、旋转 std<0.3° | hand-eye/FK/base-in-board |
| L2 全链路重投影 | 链预测像素 vs 检测（下游真正看到的） | 投影 board 角点→像素 | 均值<2px、p95<5px | 综合 |
| L3 跨相机/世界系 | base 对称性；head 与臂对 board 世界位姿是否一致 | midpoint→origin；每帧反推 `T_world_board` 聚拢度 | 对称<5mm；head-臂 board 位置差<1cm | T_world_camF/世界系定义 |

**参考板位姿** `T_world_board_ref`：取所有帧反推 `T_world_board(i)` 的鲁棒均值（SE3，复用 `solve_calibration._robust_mean_se3`）。L2 全链路像素误差 = 各帧标定链与该参考的不一致。

## 可视化（Plotly 单 HTML，自包含）

生成 `data/calib_/verify/verify_report.html`，浏览器直接打开 / scp 下载，含三块：

1. **3D 世界系一致性**（交互可旋转）：所有 41 帧反推的 board 角点画到世界系（理想重合）+ 三相机位姿 + 双臂 base。聚拢=外参一致，发散=外参不一致。
2. **2D 叠加图**：每帧 RGB + 绿(检测) / 红(预测投影) + 连线，标题写该帧像素误差。base64 内嵌进 HTML。
3. **误差柱状图**：每帧全链路误差，一眼看哪几帧坏 / 整体水平。

报告顶部一句话定位结论（基于 L0–L3 哪层先失败）。同时落 `report.json`（每层数字 + ✓/△/✗）。

## 输出

- `data/calib_/verify/verify_report.html` — 交互式 web 报告（主产出）
- `data/calib_/verify/report.json` — 结构化每层数字

## 范围

- **复用**：`board_def.get_board()`（3D 角点）、`detect_board.py:256` 的 ids→objpts 模式、npz 已存量、`solve_calibration._robust_mean_se3` 与 SE3 工具
- **新写**：`calib/verify_projection.py`（纯 numpy/cv2/plotly，不 import pyrealsense/piper_sdk）
- **不动**：现有 `verify_calibration.py`（实时点云那个）原样保留

## 依赖

- `plotly`（生成 HTML）；若 venv 没有，安装前告知。
- 其余 numpy/opencv/pyyaml 已有。

## 非目标

- 不重新采集、不连硬件、不跑实时点云。
- 不修标定（先定位，修是后续单独任务）。
