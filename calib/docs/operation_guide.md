# 标定操作手册

> 面向操作人员的逐步指南

---

## 前置准备

### 1. 打印标定板

```bash
python3 calib/gen_charuco.py
# 输出: charuco_7x5_sq38_landscape.pdf
```

打印要求：
- **A4 横版**，打印机设置 **100% 实际大小**（不要选"适合页面"）
- 打印后用尺子量 PDF 右下角的 50×50mm 校验方块
- 贴在**硬纸板**或**亚克力板**上，保持绝对平整

### 2. 检查硬件

```bash
# 检查相机
rs-enumerate-devices | grep "Serial Number"
# 应看到: 254622070889 (D435), 409122273074 (D405-L), 409122271568 (D405-R)

# 检查 CAN
ip link show can_left_slave    # 左臂 slave
ip link show can_right_slave   # 右臂 slave
# state UP 表示正常
```

### 3. 放置标定板

- 放在**桌面中央**，操作台上
- **平整不动**，整个标定过程中不能移动
- 标定板不能反光（如果表面太亮，用哑光打印）

---

## 标定流程

### 步骤 1: D435 头顶标定 (~1 分钟)

```bash
python3 calib/capture_handeye.py --phase head --camera-serial 254622070889
```

操作：
1. 屏幕显示 D435 画面 + ChArUco 检测叠加
2. 确认绿色 `✓ 可用` 后按 **Enter**
3. 自动保存

### 步骤 2: 左臂 Preview (~5 分钟)

```bash
python3 calib/capture_handeye.py --phase preview --arm left \
  --can can_left_slave --camera-serial 409122273074
```

操作：
1. 手动遥控/拖动左臂，让 D405-L 对准桌上的标定板
2. 观察右侧面板：
   - 角点数 ≥ 8 → ✓
   - 重投影误差 < 1px → ✓
   - 图像清晰 → ✓
   - 无运动模糊 → ✓
3. 全部 ✓ 后按 **Enter** 确认
4. 换下一个姿态，重复直到确认 15 个

**姿态多样性要点**：
- 从不同角度看标定板（正上方、左侧、右侧、前方、倾斜）
- 不同距离（20cm、25cm、30cm）
- 每次旋转手腕 > 15°
- **避免**：多个姿态角度几乎相同

### 步骤 3: 左臂 Replay (~2 分钟)

```bash
python3 calib/capture_handeye.py --phase replay --arm left --session left_calib
```

操作：
1. 脚本自动移动机械臂到每个已确认的姿态
2. 等待稳定后自动采集
3. 每帧显示检测结果和进度
4. 全部完成后自动结束

### 步骤 4: 右臂 Preview + Replay (~7 分钟)

```bash
# Preview
python3 calib/capture_handeye.py --phase preview --arm right \
  --can can0 --camera-serial 409122271568

# Replay
python3 calib/capture_handeye.py --phase replay --arm right --session right_calib
```

操作同步骤 2-3。

### 步骤 5: 求解 (~10 秒)

```bash
python3 calib/solve_calibration.py \
  --left calib/data/left_calib \
  --right calib/data/right_calib \
  --head calib/data/head_calib \
  --output calib/calibration.yaml
```

输出：
- 每组姿态的重投影误差
- 各变换矩阵
- 基座对称性报告

**检查**：
- 重投影误差 < 1.0px → 良好
- T_link6_cam 平移 5-15cm → 合理
- baseL 和 baseR 的 x 坐标符号相反 → 对称

### 步骤 6: 验证 (~1 分钟)

```bash
python3 calib/verify_calibration.py --config calib/calibration.yaml
```

- 三路点云叠加显示
- 红色 = D435, 绿色 = D405-L, 蓝色 = D405-R
- 重叠区域应精确对齐

---

## 故障排查

### 检测不到 ChArUco

| 原因 | 解决 |
|------|------|
| 标定板反光 | 调整灯光角度，或用哑光纸打印 |
| 距离太近 (D405) | 拉远到 20cm+ |
| 距离太远 (D435) | D435 挂高不超过 70cm |
| 标定板弯曲 | 贴在硬纸板上 |
| 打印缩放错误 | 量校验方块，应为 50×50mm |

### 手眼标定精度差

| 原因 | 解决 |
|------|------|
| 姿态多样性不足 | 确保 3 个旋转轴都有覆盖 |
| 采集中机械臂在动 | replay 阶段会自动等稳定 |
| 角点太少 | preview 阶段确保每个姿态 ≥ 8 角点 |
| 姿态数太少 | 增加到 20 组 |

### 点云不对齐

| 原因 | 解决 |
|------|------|
| 标定板在标定中被移动 | 重新标定 |
| 内参错误 | 检查 rs2_intrinsics 是否正确读取 |
| FK 不准 | 检查 DH offset 参数 (应为 0x01) |

---

## 重新标定

如果机械臂被拆卸/重新安装，或相机位置变化，需要重新标定：

```bash
# 只需重新标定变化的部分
# 例如只有左腕相机位置变了:
python3 calib/capture_handeye.py --phase preview --arm left ...
python3 calib/capture_handeye.py --phase replay --arm left ...
python3 calib/solve_calibration.py ...  # 重新求解全部
```

标定数据保存在 `calib/data/` 下，旧数据不会被覆盖（按 session 名区分）。
