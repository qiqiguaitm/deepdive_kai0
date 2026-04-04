# 多相机-双臂标定方案

> 日期: 2026-04-03
> 硬件: D435 (头顶 80cm 30°俯视) + 2×D405 (左右腕) + 双臂 Piper
> 标定板: ChArUco 7×5, sq=38mm, mk=28mm, DICT_5X5_100

---

## 一、目标

为 3D 可视化（点云 + 机器人骨架 + policy 轨迹）提供空间标定，求解 5 组变换：

```
World (两臂基座中点, 自动计算)
├── T_world_camF     ← D435 头顶相机外参
├── T_world_baseL    ← 左臂基座位姿
│   └── FK chain → link6_L
│       └── T_link6_camL  ← 左腕 D405 手眼标定
├── T_world_baseR    ← 右臂基座位姿
│   └── FK chain → link6_R
│       └── T_link6_camR  ← 右腕 D405 手眼标定
└── 3×内参 (fx, fy, cx, cy, dist)  ← RealSense 硬件读取
```

---

## 二、环境与依赖

### 运行环境

系统 python3（不依赖 kai0 venv 或 ROS2），全部硬件直连：

| 依赖 | 版本 | 用途 |
|------|------|------|
| OpenCV + ArUco | 4.13.0 | ChArUco 检测、solvePnP、calibrateHandEye |
| SciPy (Rotation) | — | 旋转矩阵 ↔ 欧拉角 |
| NumPy | 2.2.6 | 矩阵运算 |
| pyrealsense2 | 系统级 | 直连 RealSense 相机 |
| Piper SDK | `/home/tim/workspace/piper_sdk/` | 直连 CAN 读写关节角 |
| PyYAML | — | 输出标定结果 |
| reportlab | 4.4.10 | 标定板 PDF 生成 |

### 硬件配置

| 设备 | 型号 | 序列号 / CAN | 接口 |
|------|------|-------------|------|
| 头顶相机 | D435 | 254622070889 | USB |
| 左腕相机 | D405 | 409122273074 | USB |
| 右腕相机 | D405 | 409122271568 | USB |
| 左臂 (slave) | Piper | can_left_slave | CAN bus |
| 右臂 (slave) | Piper | can_right_slave (can1) | CAN bus |

---

## 三、标定原理

### 3.1 术语说明: Flange / EE / Gripper

在 Piper 机械臂上，以下三个术语指向**同一个坐标系**:

```
joint6 (revolute, 最后一个旋转关节)
  └── link6           ← Flange (法兰面): 机械臂最后关节的输出接口
        └── fixed joint (offset=0,0,0, rpy=0,0,0)
              └── gripper_base  ← Gripper 基座: 夹爪安装基准
                    ├── joint7 → link7 (左指, +135.8mm)
                    └── joint8 → link8 (右指, +135.8mm)
```

| 术语 | 通用含义 | Piper 上对应 |
|------|---------|-------------|
| **Flange (法兰)** | 机械臂末端机械接口面 | link6 |
| **EE (End-Effector)** | 法兰上安装的末端执行器 | gripper_base |
| **Gripper (夹爪)** | 具体的夹持工具 | gripper_base + link7/link8 |

由于 Piper URDF 中 `joint6_to_gripper_base` 是 **fixed joint, offset 全零**, 
link6 = flange = gripper_base **坐标系完全重合**。因此:

- FK 输出 `T_base_ee` = 基座到 link6(法兰) 的变换
- OpenCV `calibrateHandEye` 输出的 `cam2gripper` 中 "gripper" = link6 = 法兰
- 代码中 `ee`, `gripper`, `link6` 可互换使用, 无需 tool offset

> 注意: 如果将来更换末端工具 (如吸盘, 长杆等), 需要增加 `T_flange_tool` 偏移,
> 此时 EE ≠ Flange。当前 Piper 夹爪不需要。

### 3.2 坐标系定义

```
ChArUco 板 (board_frame) ← 临时参考系, 标定后丢弃
  ↓ 标定完成后
World_frame = 两臂基座中点
  - 原点: midpoint(baseL, baseR)
  - X: baseR → baseL 方向
  - Y: 操作方向 (前)
  - Z: 桌面法线 (上)
```

### 3.3 标定流程

```
1. 板放桌面 (不动, 即 board_frame)
2. D435 拍板 → estimatePoseCharucoBoard → T_camF_board → inv → T_board_camF
3. 左臂 D405-L, 15 个姿态:
   每个姿态:
     - FK(q) → T_baseL_link6
     - D405-L 检测板 → T_camL_board
   cv2.calibrateHandEye(DANIILIDIS) → T_link6_camL
   反推: T_board_baseL = inv(T_baseL_link6 · T_link6_camL · inv(T_camL_board))  (多姿态中位数)
4. 右臂同理 → T_link6_camR + T_board_baseR
5. 计算世界系:
   T_board_world = compute_world_frame(T_board_baseL, T_board_baseR)
   T_world_X = inv(T_board_world) · T_board_X  (对所有 X)
```

### 3.4 世界系计算

```python
def compute_world_frame(T_board_baseL, T_board_baseR):
    origin = (T_board_baseL[:3,3] + T_board_baseR[:3,3]) / 2
    x_axis = normalize(T_board_baseR[:3,3] - T_board_baseL[:3,3])  # R→L
    x_axis[2] = 0  # 投影到水平面
    z_axis = [0, 0, 1]  # 桌面法线
    y_axis = cross(z_axis, x_axis)  # 右手系
    T_board_world = [x_axis | y_axis | z_axis | origin]
    return T_board_world
```

---

## 四、文件结构

```
calib/
├── gen_charuco.py            # [已有] 标定板 PDF 生成 (reportlab)
├── board_def.py              # [新建] 板参数常量 + Board 构造
├── piper_fk.py               # [新建] FK 封装 (Piper SDK DH → 4×4)
├── capture_handeye.py        # [新建] 两阶段采集: preview → replay
├── solve_calibration.py      # [新建] 求解标定 + 世界系变换
├── verify_calibration.py     # [新建] 点云对齐验证
├── data/                     # [输出] 采集数据目录
│   └── {session_name}/
│       ├── pose_list.json    # preview 阶段确认的姿态
│       └── pose_*.npz        # replay 阶段采集的数据
├── calibration.yaml          # [输出] 最终标定结果
├── docs/
│   ├── calibration_plan.md   # 本文档
│   └── operation_guide.md    # 操作手册
└── README.md
```

---

## 五、各模块设计

### 5.1 `board_def.py` — 公共常量

提供 ChArUco Board 参数和构造函数，供所有模块 import：

```python
COLS, ROWS = 7, 5
SQUARE_MM = 38.0
MARKER_MM = 28.0
DICT_ID = cv2.aruco.DICT_5X5_100

def get_board() -> cv2.aruco.CharucoBoard
def get_detector_params() -> cv2.aruco.DetectorParameters
```

### 5.2 `piper_fk.py` — FK 封装

封装 Piper SDK 的 `C_PiperForwardKinematics`（DH 参数，含 2° offset 校正）：

```python
class PiperFK:
    def fk_homogeneous(q6_rad) -> np.ndarray [4×4]
        """基座 → 末端 (link6) 的齐次矩阵"""
        
    def fk_all_links(q6_rad) -> list[np.ndarray]
        """基座 → 每个 link 的齐次矩阵 (可视化用)"""
```

- 复用 SDK 内部的变换矩阵链（R01·R12·...·R06），不从 xyz+rpy 反算
- 单位: 输入 rad, 输出 m (SDK 内部 mm, 转换后输出)

### 5.3 `capture_handeye.py` — 两阶段采集（核心）

**硬件直连**：pyrealsense2 + Piper SDK，不依赖 ROS2。

#### 阶段 1: Preview（确认姿态）

```
python3 calib/capture_handeye.py --phase preview --arm left --can can_left_slave --camera-serial 409122273074
```

实时显示面板：

```
┌─────────────────────────────┬──────────────────────┐
│                             │ 姿态可用性           │
│  D405 RGB                   │ ● 角点: 18/24  ✓    │
│  + ChArUco 检测叠加          │ ● marker: 14/17 ✓   │
│  + 坐标轴                    │ ● 重投影误差: 0.3px ✓│
│                             │ ● 板面积占比: 35%  ✓ │
│                             │ ● 图像清晰度: 127  ✓ │
│                             │ ● 运动模糊: 无     ✓ │
│                             │──────────────────────│
│                             │ 综合: ✓ 可用         │
│                             │──────────────────────│
│                             │ [Enter] 确认         │
│                             │ [s] 跳过  [q] 退出   │
│                             │ 已确认: 3/15         │
└─────────────────────────────┴──────────────────────┘
```

**可用性检查项**：

| 检查 | 方法 | 通过条件 |
|------|------|---------|
| 角点数 | `detectBoard()` corners | ≥ 8 个 (24 的 1/3) |
| 重投影误差 | `estimatePoseCharucoBoard` 返回 | < 1.0 px |
| 板面积占比 | corners bbox / image area | 10% - 80% |
| 图像清晰度 | `cv2.Laplacian().var()` | > 50 |
| 运动模糊 | 连续 2 帧角点位移 | < 2 px |

- 全部通过: 绿色 `✓ 可用`
- 任一不通过: 红色 `✗ 不可用`（仍可强制确认，叠加黄色警告）

输出: `calib/data/{session}/pose_list.json`

#### 阶段 2: Replay（自动采集）

```
python3 calib/capture_handeye.py --phase replay --arm left --session my_session
```

遍历 `pose_list.json` 中确认的姿态：

1. 发送关节角指令 → `piper.JointCtrl()`
2. 等待到位: 监控关节角误差 < 0.5°，且关节速度 ≈ 0，超时 5s
3. 到位后额外等待 0.5s 确保稳定
4. 连续采集 5 帧 RGB，取平均（消除噪声）
5. ChArUco 检测 → rvec, tvec
6. 读取关节角 → FK → T_base_ee
7. 读取 rs2_intrinsics → K, dist
8. 保存 `pose_{i}.npz`

每帧保存内容：
```
rgb_image:    uint8 [H, W, 3]
joint_angles: float64 [6]         (rad)
T_base_ee:    float64 [4, 4]
rvec:         float64 [3]         (ChArUco → camera)
tvec:         float64 [3]
intrinsics:   dict {fx, fy, cx, cy, dist}
corners:      float32 [N, 1, 2]   (原始检测)
corner_ids:   int32 [N]
```

#### D435 头顶标定

```
python3 calib/capture_handeye.py --phase head --camera-serial 254622070889
```

单帧采集，不需要机械臂运动。同样显示可用性检查。

### 5.4 `solve_calibration.py` — 求解

```
python3 calib/solve_calibration.py \
  --left calib/data/left_session \
  --right calib/data/right_session \
  --head calib/data/head_session \
  --output calib/calibration.yaml
```

流程：
1. 加载 `pose_*.npz` 数据
2. 左臂: `cv2.calibrateHandEye(method=DANIILIDIS)` → `T_link6_camL`
3. 反推 `T_board_baseL`（多姿态中位数）
4. 右臂: 同上
5. 头顶: `solvePnP` → `T_board_camF`
6. `compute_world_frame()` → 世界系
7. 全部变换到世界系
8. 输出 `calibration.yaml`

输出报告：
- 重投影误差 (px) per 姿态
- T_link6_cam 平移量 (预期 5-15cm)
- 基座对称性检查

### 5.5 `verify_calibration.py` — 验证

```
python3 calib/verify_calibration.py --config calib/calibration.yaml
```

- 三路 RealSense 深度图 → 点云 → 变换到世界系
- 可视化叠加（rerun 或 matplotlib 3D）
- 输出: 基座位置对称性、点云重叠对齐误差

---

## 六、标定板参数

```
ChArUco 7×5 (7 cols × 5 rows)
  字典: DICT_5X5_100
  squareLength: 38mm
  markerLength: 28mm (73.7%)
  板面: 266 × 190mm
  角点: (7-1)×(5-1) = 24 个
  标记: ~17 个
  打印: A4 横版, 100% 实际大小
```

**各相机检测能力**：

| 相机 | 距离 | Marker 像素 | 可见角点 |
|------|------|------------|---------|
| D405 @15cm | 近 | 85px ✓ | ~11/24 (部分遮挡) |
| D405 @20cm | 中 | 64px ✓ | ~20/24 |
| D405 @25-30cm | 远 | 43-51px ✓ | 24/24 全部 |
| D435 @80cm 30° | 头顶 | ~14px △ | 大部分 (ChArUco 降级检测) |

---

## 七、预设姿态策略

每臂 15 组姿态，覆盖 D405 在标定板上方的半球空间：

```
5 个方位角 (0°, 72°, 144°, 216°, 288°) × 3 个俯仰角 (15°, 30°, 45°)
```

要求：
- D405 距板面 20-30cm
- 每组之间旋转 > 15°
- 避开关节极限附近 (留 10% 余量)
- 确保板在 D405 视野中 (≥ 8 角点)

---

## 八、输出格式

### `calibration.yaml`

```yaml
# 世界系: 双臂基座中点, X=L→R, Y=前, Z=上
metadata:
  date: "2026-04-03"
  board: {dict: DICT_5X5_100, size: [7, 5], square_mm: 38, marker_mm: 28}
  method: DANIILIDIS
  reprojection_error_px: {left: 0.35, right: 0.42, head: 0.28}

transforms:
  T_world_camF:  [[...4x4...]]   # 头顶 D435
  T_world_baseL: [[...4x4...]]   # 左臂基座
  T_world_baseR: [[...4x4...]]   # 右臂基座
  T_link6_camL:  [[...4x4...]]   # 左腕 D405 (臂局部)
  T_link6_camR:  [[...4x4...]]   # 右腕 D405 (臂局部)

intrinsics:
  cam_f: {fx: ..., fy: ..., cx: ..., cy: ..., dist: [...]}
  cam_l: {fx: ..., fy: ..., cx: ..., cy: ..., dist: [...]}
  cam_r: {fx: ..., fy: ..., cx: ..., cy: ..., dist: [...]}

hardware:
  cam_f_serial: "254622070889"
  cam_l_serial: "409122273074"
  cam_r_serial: "409122271568"
  left_arm_can: "can_left_slave"
  right_arm_can: "can_right_slave"
```

---

## 九、操作步骤

```
准备:
  □ 打印标定板: python3 calib/gen_charuco.py → A4 横版 100% 打印
  □ 贴在硬纸板上保持平整
  □ 确认 CAN 接口就绪: ip link show can_left_slave can_right_slave
  □ 确认相机连接: rs-enumerate-devices

标定:
  1. 标定板放桌面中央, 平整不动

  2. D435 头顶标定 (~1min):
     python3 calib/capture_handeye.py --phase head --camera-serial 254622070889

  3. 左臂 preview (~5min):
     python3 calib/capture_handeye.py --phase preview --arm left \
       --can can_left_slave --camera-serial 409122273074
     → 确认 15 个姿态

  4. 左臂 replay (~2min):
     python3 calib/capture_handeye.py --phase replay --arm left --session left_calib

  5. 右臂 preview + replay (~7min):
     同上, --arm right --can can_right_slave --camera-serial 409122271568

  6. 求解:
     python3 calib/solve_calibration.py \
       --left calib/data/left_calib \
       --right calib/data/right_calib \
       --head calib/data/head_calib

  7. 验证:
     python3 calib/verify_calibration.py --config calib/calibration.yaml

总耗时: ~20 分钟
```

---

## 十、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 运行环境 | 系统 python3 | 全部依赖可用，不依赖 venv 或 ROS2 |
| 硬件接口 | pyrealsense2 + Piper SDK 直连 | 标定独立于推理环境 |
| FK | Piper SDK DH (含 2° offset) | 比 URDF 理论值更贴合实际硬件 |
| 内参 | rs2_intrinsics 硬件读取 | 工厂标定，精度充足 |
| 采集 | 两阶段 preview→replay | 先确认姿态质量，再自动精确采集 |
| 稳定判断 | 关节误差 < 0.5° + 0.5s 延迟 | 避免运动中采集 |
| 手眼方法 | DANIILIDIS | 适合小噪声、中等姿态数的场景 |
| 世界系 | 两臂基座中点，自动计算 | 不需要手动量测，一块板解决全部 |
| 精度目标 | 平移 ~2mm, 旋转 ~1° | 3D 可视化足够，15 组姿态可达 |
