# 标定流程详解

> 面向操作人员的完整流程文档，含原理图解、命令、检查点和故障处理。

---

## 总览

```
┌─────────────────────────────────────────────────────────────────┐
│                       标定流程总览                                │
│                                                                 │
│  准备 ──→ 头顶D435 ──→ 左臂D405 ──→ 右臂D405 ──→ 求解 ──→ 验证  │
│  (~5min)   (~1min)    (~12min)    (~12min)    (~10s)   (~2min)  │
│                                                                 │
│  总耗时: ~30 分钟                                                │
└─────────────────────────────────────────────────────────────────┘
```

每条臂分 **Preview（确认姿态）→ Replay（自动采集）** 两阶段：

```
Preview                              Replay
┌──────────────┐                    ┌──────────────┐
│ 手动拖动手臂  │                    │ 自动运动到位  │
│ 实时看检测画面│   ──确认15个姿态──→  │ 等待稳定0.5s │
│ 按Enter确认  │                    │ 采集5帧平均   │
│ 保存关节角    │                    │ 保存npz数据   │
└──────────────┘                    └──────────────┘
  ~8min/臂                            ~4min/臂
```

---

## 第 0 步：准备工作 (~5min)

### 0.1 打印标定板

```bash
cd /data1/tim/workspace/deepdive_kai0
python3 calib/gen_charuco.py
# 输出: charuco_7x5_sq38_landscape.pdf
```

打印要求：
- A4 横版，**100% 实际大小**打印（不要 "适合页面"）
- 打印后用尺子量 PDF 右下角 50×50mm 校验方块
- 贴在硬纸板或亚克力板上，**必须绝对平整**

```
标定板规格:
  ChArUco 7×5 (7列 × 5行)
  棋盘格: 38mm × 38mm
  ArUco 标记: 28mm
  字典: DICT_5X5_100
  板面: 266mm × 190mm
```

### 0.2 检查硬件连接

```bash
# 检查三个 RealSense 相机
rs-enumerate-devices | grep "Serial Number"
# 期望:
#   254622070889  (D435, 头顶)
#   409122273074  (D405, 左腕)
#   409122271568  (D405, 右腕)

# 检查 CAN 总线 (使用 symbolic name, 见 config/pipers.yml)
ip link show can_left_slave    # 左臂 slave (can0) → state UP
ip link show can_right_slave   # 右臂 slave (can1) → state UP

# 如果 CAN 未启动:
sudo ip link set can_left_slave up type can bitrate 1000000
sudo ip link set can_right_slave up type can bitrate 1000000
```

### 0.3 人工确认相机画面与机械臂通讯（必做）

标定前**必须**用独立工具人工确认每个设备正常工作，避免标定过程中才发现问题。

#### 相机确认: realsense-viewer

```bash
realsense-viewer
```

逐一打开三个相机，检查：
- [ ] **D435 (头顶)**：RGB 画面正常、能看到桌面、标定板清晰可辨
- [ ] **D405-L (左腕)**：RGB 画面正常、Depth 有数据
- [ ] **D405-R (右腕)**：RGB 画面正常、Depth 有数据
- [ ] 三个相机的**序列号**与 `config/cameras.yml` 一致
- [ ] 无画面花屏、断流、全黑

> 确认后关闭 realsense-viewer（同一时间只能有一个进程占用相机）。

#### 机械臂确认: piper_tools

```bash
cd /data1/tim/workspace/deepdive_kai0

# 确认 CAN 映射关系 (哪条 CAN 对应哪条臂)
python3 piper_tools/calibrate_can_mapping.py --check
# 或手动检查:
#   轻轻推动左臂 → 看 can_left_slave 有数据变化
#   轻轻推动右臂 → 看 can_right_slave 有数据变化
```

逐臂检查：
- [ ] **左臂 (can_left_slave)**：能读到关节角、推动时数值变化
- [ ] **右臂 (can_right_slave)**：能读到关节角、推动时数值变化
- [ ] CAN 接口名与 `config/pipers.yml` 中的 `can_symbolic` 一致
- [ ] 无通讯超时或错误

> 如果 CAN 映射不对（推左臂但右臂数据变），需要重新运行
> `python3 piper_tools/calibrate_can_mapping.py` 校准映射。

### 0.4 放置标定板

```
          D435 (头顶, 80cm, 30°俯视)
            ↓ 拍摄方向
  ┌─────────────────────────────┐
  │                             │
  │    ┌───────────────────┐    │
  │    │                   │    │
  │    │    ChArUco 板     │    │  ← 桌面中央, 平放不动
  │    │    266×190mm      │    │
  │    │                   │    │
  │    └───────────────────┘    │
  │                             │
  │  [左臂基座]     [右臂基座]   │  ← 标定板放在两臂之间
  │                             │
  └─────────────────────────────┘
                桌面
```

**关键**：标定板在整个标定过程中**绝对不能移动**。

---

## 第 1 步：D435 头顶相机标定 (~1min)

### 原理

```
D435 (固定不动)
  ↓ 拍到标定板
ChArUco 板 (桌面, 不动)
  ↓
estimatePoseCharucoBoard → T_cam_board (board 在 cam 系中的位姿)
  ↓ 取逆
T_board_cam = inv(T_cam_board)  (cam 在 board 系中的位姿)
```

单帧即可，不需要机械臂参与。

### 操作

```bash
# camera serial 自动从 config/cameras.yml 读取
python3 calib/capture_handeye.py --phase head --session my_calib
```

屏幕显示：

```
┌─────────────────────────────┬──────────────────────┐
│                             │ 姿态可用性            │
│  D435 实时画面               │ OK Corners: 22/24    │
│  + ChArUco 角点 (绿色)       │ OK Reproj err: 0.3px │
│  + 坐标轴叠加               │ OK Board area: 8%     │
│                             │ OK Sharpness: 230     │
│                             │ OK Motion: 0.0px      │
│                             │                      │
│                             │ USABLE               │
│                             │                      │
│                             │ [Enter] 采集          │
│                             │ [q] 退出              │
└─────────────────────────────┴──────────────────────┘
```

**检查点**：
- [ ] 角点数 ≥ 8（D435 在 80cm 处可能 ~16-22 个，正常）
- [ ] 重投影误差 < 1.0px
- [ ] 画面清晰（不模糊）
- [ ] 绿色 "USABLE" 显示

确认后按 **Enter**，数据保存到 `calib/data/my_calib/head.npz`。

---

## 第 2 步：左臂 D405 手眼标定 (~12min)

### 原理

```
ChArUco 板 (桌面固定, 已知)
  ↑ 被 D405-L 拍到
D405-L (装在左臂 link6/法兰 上, 随臂运动)
  ↑ 位姿由 FK 计算
左臂运动到 N 个不同姿态

每个姿态 i 提供一组约束:
  T_base_ee[i]    ← FK(关节角)
  T_cam_board[i]  ← ChArUco 检测

N 个姿态联立 → cv2.calibrateHandEye → T_cam2gripper (camera→link6)
```

### 2a. Preview — 确认姿态 (~8min)

```bash
# CAN 和 camera serial 自动从 config/ 读取，无需手动指定
python3 calib/capture_handeye.py --phase preview --arm left --session my_calib
# 可选覆盖: --can can_left_slave --camera-serial 409122273074 --num-poses 15
```

**操作过程**：

1. 脚本启动后，左臂进入**被动模式**（电机断电，可手动拖动）
2. 手动拖动左臂，让腕部 D405 对准桌面上的标定板
3. 观察实时画面 + 右侧可用性面板
4. 全部检查通过（绿色 USABLE）后，按 **Enter** 确认
5. 换一个姿态，重复直到确认 15 个

```
┌─────────────────────────────┬──────────────────────┐
│                             │ Pose 5/15            │
│  D405-L 实时画面             │ OK Corners: 18/24    │
│  + ChArUco 角点 (绿色)       │    Markers: 14/17    │
│  + 坐标轴                    │ OK Reproj err: 0.2px │
│                             │ OK Board area: 35%   │
│                             │ OK Sharpness: 127    │
│                             │ OK Motion: 0.3px     │
│                             │                      │
│                             │ USABLE               │
│                             │                      │
│                             │ [Enter] 确认          │
│                             │ [s] 跳过  [q] 退出    │
│                             │ 已确认: 4/15          │
└─────────────────────────────┴──────────────────────┘
```

**姿态多样性要点（关键！）**：

```
好的 15 组姿态: 从不同角度/距离看标定板
                                    
        ①俯视     ②左倾     ③右倾    
         ↓         ↘        ↙        
    ┌─────────┐                       
    │ ChArUco │  板在桌面              
    └─────────┘                       
         ↑         ↗        ↖        
        ④前倾     ⑤后倾     ⑥远距    

  × 3 个不同的腕部旋转角 → 15 组
```

| 要求 | 阈值 | 原因 |
|------|------|------|
| D405 距标定板 | 20-30cm | 太近板出视野，太远 marker 变小 |
| 相邻姿态旋转差 | > 15° | 旋转信息不足会导致标定退化 |
| 覆盖 3 个旋转轴 | 每轴 ±30° | 单轴旋转只能标定部分自由度 |
| 避开关节极限 | 留 10% 余量 | 极限附近 FK 精度下降 |

**常见错误**：
- ❌ 15 个姿态都从正上方往下看 → 旋转多样性不足
- ❌ 只平移不旋转 → 旋转约束缺失
- ❌ 距离太近（<15cm）→ 板只有一小部分可见

确认完成后，数据保存到 `calib/data/my_calib/pose_list.json`。

### 2b. Replay — 自动采集 (~4min)

```bash
python3 calib/capture_handeye.py \
  --phase replay \
  --arm left \
  --session my_calib
```

**自动执行过程**：

```
每个已确认的姿态:
  1. 左臂启动 (电机使能)
  2. 发送关节角指令 → 运动 (30% 速度)
  3. 等待到位: 6个关节角误差全部 < 0.5°
  4. 到位后额外等待 0.5s (确保完全稳定)
  5. 连续采集 5 帧 RGB, 取像素平均 (消除传感器噪声)
  6. ChArUco 检测 → rvec, tvec, 角点
  7. 读取关节角 → FK → T_base_ee
  8. 读取相机内参 (rs2_intrinsics)
  9. 全部保存到 pose_XX.npz
  10. 如果检测失败, 重试 3 次; 仍失败则跳过
  11. 如果到位超时 (5s), 跳过该姿态 (安全保护)
```

屏幕输出示例：

```
[Replay] 15 poses to capture

  [1/15] pose_00: moving... settled -> saved (18 corners, err=0.21px)
  [2/15] pose_01: moving... settled -> saved (22 corners, err=0.18px)
  [3/15] pose_02: moving... settled -> saved (15 corners, err=0.35px)
  ...
  [15/15] pose_14: moving... settled -> saved (20 corners, err=0.25px)

Replay complete. Data saved to calib/data/my_calib/
```

**检查点**：
- [ ] 全部 15 个姿态成功采集（跳过 ≤ 2 个可接受）
- [ ] 重投影误差均 < 1.0px
- [ ] 每帧角点数 ≥ 8

数据保存到 `calib/data/my_calib/pose_00.npz` ~ `pose_14.npz`。

---

## 第 3 步：右臂 D405 手眼标定 (~12min)

流程与第 2 步完全相同，换右臂参数：

### 3a. Preview

```bash
python3 calib/capture_handeye.py \
  --phase preview --arm right --session my_calib
```

### 3b. Replay

```bash
python3 calib/capture_handeye.py --phase replay --arm right --session my_calib
```

---

## 第 4 步：求解标定 (~10s)

### 原理

```
输入:
  head.npz           → T_board_camF    (D435 在 board 系的位姿)
  left/pose_*.npz    → calibrateHandEye → T_cam2gripperL (cam→gripper 变换)
                     → 反推 T_board_baseL (左臂基座在 board 系的位姿)
  right/pose_*.npz   → calibrateHandEye → T_cam2gripperR
                     → 反推 T_board_baseR

变换到世界系:
  T_board_world = 两臂基座中点 (自动计算)
  T_world_* = inv(T_board_world) @ T_board_*

输出:
  calibration.yaml
```

**变换链图解**：

```
                      world (两臂中点)
                     ╱       ╲
              T_world_baseL   T_world_baseR
                  ╱                 ╲
              baseL                 baseR
                |                     |
            FK(q)                  FK(q)
                |                     |
            link6_L                link6_R
                |                     |
          T_cam2gripperL        T_cam2gripperR
                |                     |
            D405-L                D405-R

                      T_world_camF
                          |
                        D435 (头顶)
```

各变换的方向（p_目标 = T · p_源）：

| 变换 | 方向 | 含义 |
|------|------|------|
| `T_base_ee` | ee → base | FK: 将 link6 系的点变到 base 系 |
| `T_cam2gripper` | cam → gripper | 将 cam 系的点变到 link6 系 |
| `T_cam_board` | board → cam | 将 board 系的点变到 cam 系 |
| `T_world_base` | base → world | 将 base 系的点变到 world 系 |
| `T_world_camF` | cam → world | 将 D435 系的点变到 world 系 |

**闭环验证**：`T_board_base · T_base_ee · T_cam2gripper · T_cam_board = I`

### 操作

```bash
python3 calib/solve_calibration.py \
  --left calib/data/my_calib \
  --right calib/data/my_calib \
  --head calib/data/my_calib \
  --output calib/calibration.yaml
```

屏幕输出示例：

```
============================================================
标定求解
============================================================

--- Loading data ---
  Loaded 15 poses from calib/data/my_calib (left)
  Loaded 15 poses from calib/data/my_calib (right)
  Loaded head data

--- Solving hand-eye ---

  [Left] T_cam2gripper (camera→gripper):
    translation: [0.0312, -0.0485, 0.0721] m (norm=9.4 cm)
    rotation:    [178.2, -5.1, 92.3] deg
    reproj errors: mean=0.252, max=0.481 px

  [Right] T_cam2gripper (camera→gripper):
    translation: [-0.0298, -0.0501, 0.0685] m (norm=9.1 cm)
    rotation:    [-175.8, -4.8, -88.7] deg
    reproj errors: mean=0.289, max=0.523 px

  [Head] T_board_camF:
    translation: [0.1523, 0.2845, 0.7821] m
    reproj error: 0.315 px

--- Computing world frame ---
  baseL pos: [ 0.2512  0.0023  0.0015]
  baseR pos: [-0.2508 -0.0019  0.0021]
  midpoint:  [ 0.0002  0.0002  0.0018]

Saved calibration to calib/calibration.yaml
```

**检查点**：

| 指标 | 期望值 | 说明 |
|------|--------|------|
| 重投影误差 (mean) | < 0.5px | > 1.0px 需重新标定 |
| T_cam2gripper 平移 | 5-15cm | 相机到法兰的物理距离 |
| 左右 T_cam2gripper 对称 | 近似镜像 | 两臂安装对称 |
| 基座中点 | ≈ [0, 0, 0] | 偏差 > 5mm 说明有问题 |
| 两基座 x 坐标 | 符号相反 | 左正右负（或反之） |

---

## 第 5 步：验证 (~2min)

```bash
python3 calib/verify_calibration.py --config calib/calibration.yaml
```

### 验证内容

**1. 基座对称性**：
```
baseL: [+0.251, +0.002, +0.002]
baseR: [-0.251, -0.002, +0.002]
对称误差: 0.4mm ✓  (< 5mm)
```

**2. 三路点云对齐**：

三个相机各拍一帧深度图，投影到世界坐标系后叠加：

```
颜色: 红=D435(头顶), 绿=D405-L(左腕), 蓝=D405-R(右腕)

好的结果: 三色点云在重叠区域精确对齐
  ● 桌面边缘三色重合
  ● 标定板区域三色重合
  ● 中位数对齐误差 < 5mm

差的结果: 点云错位
  ● 同一物体出现三个 "影子"
  ● 需要检查标定板是否被移动
```

**3. 定量对齐指标**：
```
head vs left  (overlap 15234+8921 pts): median=3.2mm, p90=7.1mm [GOOD]
head vs right (overlap 14872+9103 pts): median=3.5mm, p90=7.8mm [GOOD]
left vs right (overlap 2341+2189 pts):  median=4.1mm, p90=8.2mm [GOOD]
```

| 评级 | 中位数对齐误差 | 判断 |
|------|-------------|------|
| GOOD | < 5mm | 可以用 |
| OK | 5-10mm | 可接受，可优化 |
| POOR | > 10mm | 需重新标定 |

---

## 输出文件

### `calibration.yaml` 结构

```yaml
metadata:
  date: "2026-04-03 15:30:00"
  board: {dict: DICT_5X5_100, size: [7,5], square_mm: 38, marker_mm: 28}
  method: DANIILIDIS
  reprojection_error_px:
    left_mean: 0.252
    right_mean: 0.289
    head: 0.315
  num_poses: {left: 15, right: 15}

transforms:
  T_world_camF:     [[...4x4...]]   # 头顶 D435 (cam→world)
  T_world_baseL:    [[...4x4...]]   # 左臂基座 (base→world)
  T_world_baseR:    [[...4x4...]]   # 右臂基座 (base→world)
  T_cam2gripperL:   [[...4x4...]]   # 左腕 D405 (cam→gripper)
  T_cam2gripperR:   [[...4x4...]]   # 右腕 D405 (cam→gripper)

intrinsics:
  cam_f: {fx, fy, cx, cy, dist, width, height}
  cam_l: {fx, fy, cx, cy, dist, width, height}
  cam_r: {fx, fy, cx, cy, dist, width, height}

hardware:
  cam_f_serial: "254622070889"
  cam_l_serial: "409122273074"
  cam_r_serial: "409122271568"
  left_arm_can: "can_left_slave"
  right_arm_can: "can_right_slave"
```

### 使用标定结果

```python
import yaml, numpy as np

with open('calib/calibration.yaml') as f:
    calib = yaml.safe_load(f)

# 将 D435 深度图的点云变换到世界系
T_world_cam = np.array(calib['transforms']['T_world_camF'])
pc_world = (T_world_cam[:3,:3] @ pc_cam.T).T + T_world_cam[:3,3]

# 将左腕 D405 深度图的点云变换到世界系 (需要当前关节角)
T_world_base = np.array(calib['transforms']['T_world_baseL'])
T_cam2gripper = np.array(calib['transforms']['T_cam2gripperL'])
T_base_ee = fk.fk_homogeneous(current_joints)  # FK
T_world_cam = T_world_base @ T_base_ee @ T_cam2gripper
pc_world = (T_world_cam[:3,:3] @ pc_cam.T).T + T_world_cam[:3,3]
```

---

## 故障排查

### ChArUco 检测失败

| 症状 | 原因 | 解决 |
|------|------|------|
| Corners: 0/24 | 标定板不在视野中 | 调整手臂角度 |
| Corners: 2/24 | 距离太远或角度太偏 | 拉近到 20-30cm |
| Reproj err > 2px | 标定板弯曲或打印缩放错 | 重新打印，量校验方块 |
| Sharpness < 50 | 图像模糊 | 等待自动曝光稳定 |
| Motion > 2px | 手臂还在动 | 等手臂完全静止再确认 |

### 标定精度差

| 症状 | 原因 | 解决 |
|------|------|------|
| reproj > 1px | 姿态质量差 | 增加姿态多样性，重新 preview |
| T_cam2gripper 距离 > 20cm | FK 不准或检测异常 | 检查 DH offset 参数 |
| 基座中点偏移 > 5mm | 标定板被移动了 | 重新标定 |
| 点云错位 > 10mm | 综合误差 | 增加姿态数到 20，重做 |

### 硬件问题

| 症状 | 原因 | 解决 |
|------|------|------|
| EnablePiper() 失败 | CAN 未启动或断线 | `ip link set canX up type can bitrate 1000000` |
| 到位超时 | 关节卡住或目标超限 | 检查目标关节角是否在限位内 |
| 相机打不开 | USB 断开或被占用 | `rs-enumerate-devices` 检查 |

---

## 重新标定

标定板或相机位置变化后需要重新标定。可以只重做变化的部分：

| 变化 | 需要重做的步骤 |
|------|--------------|
| 头顶 D435 位置变了 | 步骤 1 + 4 + 5 |
| 左腕 D405 松动/拆装 | 步骤 2 + 4 + 5 |
| 右腕 D405 松动/拆装 | 步骤 3 + 4 + 5 |
| 机械臂基座移动了 | 全部重做 |
| 标定板被移动了 | 全部重做 |

旧数据保存在 `calib/data/` 下，按 session 名区分，不会被覆盖。

---

## 完整命令速查

```bash
# 0. 生成标定板 PDF
python3 calib/gen_charuco.py

# 1. 头顶 D435
python3 calib/capture_handeye.py --phase head --camera-serial 254622070889 --session my_calib

# 2a. 左臂 preview (CAN + camera serial 自动从 config/ 读取)
python3 calib/capture_handeye.py --phase preview --arm left --session my_calib

# 2b. 左臂 replay
python3 calib/capture_handeye.py --phase replay --arm left --session my_calib

# 3a. 右臂 preview
python3 calib/capture_handeye.py --phase preview --arm right --session my_calib

# 3b. 右臂 replay
python3 calib/capture_handeye.py --phase replay --arm right --session my_calib

# 4. 求解
python3 calib/solve_calibration.py --left calib/data/my_calib --right calib/data/my_calib --head calib/data/my_calib --output calib/calibration.yaml

# 5. 验证
python3 calib/verify_calibration.py --config calib/calibration.yaml
```
