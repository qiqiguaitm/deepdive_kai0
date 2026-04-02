# 遥操作指南 (Teleoperation Guide)

本文档覆盖 Piper 双臂 Master-Slave 遥操作和 DAgger 数据采集的完整流程。

## 硬件拓扑

```
                桌面 (120cm × 80cm, height ~75cm)

                ┌─────────────────────────────────┐
                │       操作区域 (60×50cm)          │
                │      T-shirt placement area      │
                └─────────────────────────────────┘

   18cm ←→ ┌─────────────┐  39cm  ┌─────────────┐
           │   左Piper    │ ←────→ │   右Piper    │
           │   slave      │        │   slave      │
           │ can_left_slave│       │ can_right_slave│
           └──────┬───────┘        └──────┬───────┘
                  │                       │
桌前沿 ───────────┼───────────────────────┼─────
                  │        34cm           │
                  │     ┌──┴──┐          │
                  │     │ D435 │          │
                  │     │(top) │          │
                  │     └─────┘          │
                  │    76cm high         │
                  │    30° tilt          │
                  │                      │
           ┌─────────────┐        ┌─────────────┐
           │   左Piper    │        │   右Piper    │
           │   master     │        │   master     │
           │ can_left_mas │        │ can_right_mas│
           └─────────────┘        └─────────────┘
```

## CAN 接口命名机制

**重要**: ROS2 launch 文件使用**符号名** (`can_left_slave`, `can_right_slave`, `can_left_mas`, `can_right_mas`)，
而非系统默认的 `can0`~`can3`。启动前必须用 `can_activate.sh` 脚本根据 USB bus-info 重命名接口。

`can_activate.sh` 工作原理: 根据 USB bus-info 找到对应的物理 `canX` 接口 → 设置 bitrate → 重命名为指定符号名 → 激活。

### sim01 当前 CAN 接口 (2026-04-01 实测)

| 物理接口 | USB Bus-Info | 符号名 (launch 期望) | 角色 |
|----------|-------------|---------------------|------|
| can0 | `1-2.4:1.0` | `can_left_mas` | 左臂 master |
| can1 | `1-2.3:1.0` | `can_left_slave` | 左臂 slave |
| can2 | `1-2.2:1.0` | `can_right_mas` | 右臂 master |
| can3 | `1-2.1:1.0` | `can_right_slave` | 右臂 slave |

4 个 USB-CAN 适配器均已接入，遥操和 DAgger 就绪。

### 查看 / 确认 bus-info

```bash
# 查看所有 CAN 接口的 USB bus-info
for iface in $(ip -br link show type can | awk '{print $1}'); do
  bus=$(sudo ethtool -i "$iface" | grep bus-info | awk '{print $2}')
  echo "$iface → $bus"
done
```

## 相机配置

| 角色       | 型号  | 序列号         | ROS2 Topic                             |
|-----------|-------|---------------|---------------------------------------|
| top/head  | D435  | 254622070889  | `/camera_f/camera_f/color/image_raw`   |
| wrist-L   | D405  | 409122273074  | `/camera_l/camera_l/color/image_rect_raw` |
| wrist-R   | D405  | 409122271568  | `/camera_r/camera_r/color/image_rect_raw` |

## ROS2 Topic 一览

```
# Master 臂 (遥操输入)
/master/joint_left              (JointState) — 左 master 关节状态
/master/joint_right             (JointState) — 右 master 关节状态
/master/linkage_config          (String)     — 在线模式切换 (0xFA=teach, 0xFC=follow)
/master/teach_mode              (Int32)      — 拖拽示教模式
/master/enable                  (Bool)       — 使能信号

# Slave 臂 (执行输出)
/puppet/joint_left              (JointState) — 左 slave 关节状态
/puppet/joint_right             (JointState) — 右 slave 关节状态
/puppet/arm_status              (PiperStatusMsg) — 臂状态 (错误码, 模式等)
/puppet/end_pose                (PoseStamped) — 末端位姿

# 相机
/camera_f/camera_f/color/image_raw       (Image) — D435 俯视 RGB
/camera_l/camera_l/color/image_rect_raw  (Image) — D405 左腕 RGB
/camera_r/camera_r/color/image_rect_raw  (Image) — D405 右腕 RGB

# 策略推理
/policy/actions                 (JointState) — 14-DOF 动作 (7×左 + 7×右)
```

---

## 场景一：纯 Master-Slave 遥操

人操控 master 臂，slave 臂实时跟随。用于初始数据采集或调试。

### 前置条件

- 4 个 USB-CAN 适配器均已连接 (2 master + 2 slave)
- master 臂和 slave 臂均已上电

### 操作步骤

```bash
# 1. 用 can_activate.sh 将物理接口重命名为符号名
#    (bus-info 需根据实际硬件填写, 首次接入时用 ethtool 确认)
cd ~/workspace/deepdive_kai0/kai0/train_deploy_alignment/dagger/agilex

bash ./can_activate.sh can_left_mas    1000000 "1-2.4:1.0"   # can0 → 左 master
bash ./can_activate.sh can_left_slave  1000000 "1-2.3:1.0"   # can1 → 左 slave
bash ./can_activate.sh can_right_mas   1000000 "1-2.2:1.0"   # can2 → 右 master
bash ./can_activate.sh can_right_slave 1000000 "1-2.1:1.0"   # can3 → 右 slave

# 或者一次性执行 (需先更新 activate_can_arms.sh 中的 bus-info):
bash ./activate_can_arms.sh

# 2. 验证符号名接口均已就绪
ip link show type can
# 应看到 can_left_slave, can_right_slave, can_left_mas, can_right_mas 均为 UP

# 3. 验证 CAN 数据流
candump can_left_slave -n 5 &
candump can_right_slave -n 5 &
candump can_left_mas -n 5 &
candump can_right_mas -n 5 &
wait
# 每个接口应收到数据帧, 否则检查对应臂是否上电

# 4. 启动 ROS2 Master-Slave 节点
cd ~/workspace/deepdive_kai0/ros2_ws
source install/setup.bash
ros2 launch piper start_ms_piper_new_launch.py
```

启动后 master 臂进入拖拽示教模式，拖动 master 臂，slave 臂会实时跟随。

### 验证

```bash
# 另开终端, 检查 topic 是否有数据
ros2 topic hz /master/joint_left     # 应 ~200 Hz
ros2 topic hz /puppet/joint_left     # 应 ~200 Hz
ros2 topic echo /puppet/joint_left   # 查看关节角度
```

---

## 场景二：DAgger 数据采集 (策略推理 + 人工纠正)

策略自主执行，操作员观察并在失败时介入纠正。这是 kai0 项目的核心数据采集方式。

### 前置条件

- 场景一的所有条件 (4 个 CAN 接口已用符号名激活)
- 推理模型已部署 (checkpoint 可用)
- 3 个 RealSense 相机已连接

### 操作步骤

共需 3 个终端：

**Terminal 1 — 推理服务**
```bash
cd ~/workspace/deepdive_kai0/kai0
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
  --config pi05_flatten_fold_normal \
  --checkpoint checkpoints/Task_A/mixed_1 \
  --port 8000
```

**Terminal 2 — ROS2 节点 (相机 + 机械臂)**
```bash
cd ~/workspace/deepdive_kai0/ros2_ws
source install/setup.bash
ros2 launch piper start_ms_piper_new_launch.py
```

**Terminal 3 — DAgger 采集脚本**
```bash
cd ~/workspace/deepdive_kai0/kai0/train_deploy_alignment/dagger/agilex
python agilex_openpi_dagger_collect_ros2.py \
  --host localhost --port 8000 \
  --ctrl_type joint --use_temporal_smoothing \
  --chunk_size 50 --dataset_name my_flatfold_dagger_v1
```

### 键盘控制

| 按键    | 功能                                         |
|---------|---------------------------------------------|
| *(自动)* | 策略自主推理执行, 操作员观察                     |
| `d`     | 进入 DAgger 模式 — 暂停策略推理, 启用 master 手动控制 |
| `Space` | 开始录制当前 episode                           |
| `s`     | 保存 episode (写入 HDF5 + MP4)                |
| `r`     | 恢复策略推理                                   |
| `w`     | 删除上一条 episode                             |

### 采集流程

1. 启动后策略自动推理执行动作
2. 观察到策略失败 → 按 `d` 暂停策略, 切换到手动模式
3. 按 `Space` 开始录制
4. 拖动 master 臂示范正确动作
5. 按 `s` 保存 episode
6. 按 `r` 恢复策略推理
7. 重复步骤 2-6, 目标: 50-200 条有效 episode

### 数据输出格式

```
dataset_name/
├── episode_000000.hdf5    # 观测 + 动作数据
├── episode_000001.hdf5
├── ...
└── video/
    ├── cam_high/          # D435 俯视
    │   ├── episode_000000.mp4
    │   └── ...
    ├── cam_left_wrist/    # D405 左腕
    │   └── ...
    └── cam_right_wrist/   # D405 右腕
        └── ...
```

HDF5 格式:
- `observation.state`: [N, 14] — 双臂关节角度 + 夹爪
- `observation.images.*`: 视频帧
- `action`: [N, 14] — 动作指令

---

## 场景三：纯策略推理 (无遥操)

仅需 slave 臂, 无需 master 臂。可以直接使用物理接口名 (`can1`, `can2`)，
因为 `inference_full_launch.py` 硬编码了 `can1`/`can2`。

```bash
# 1. 激活 slave CAN (无需重命名)
for iface in can1 can2; do
  sudo ip link set "$iface" down
  sudo ip link set "$iface" type can bitrate 1000000
  sudo ip link set "$iface" up
done

# 2. 启动推理服务
cd ~/workspace/deepdive_kai0/kai0
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
  --config pi05_flatten_fold_normal \
  --checkpoint checkpoints/Task_A/mixed_1 \
  --port 8000

# 3. 启动 ROS2 推理节点
cd ~/workspace/deepdive_kai0/ros2_ws
source install/setup.bash
ros2 launch piper inference_full_launch.py
```

推理节点参数:
- `latency_k`: 8 (推理延迟补偿)
- `chunk_size`: 50 (动作序列长度)
- `min_smooth_steps`: 8 (时序平滑重叠步)
- `decay_alpha`: 0.25 (平滑衰减系数)
- `publish_rate`: 30 Hz (控制输出频率)
- `inference_rate`: 3.0 Hz (策略调用频率)

---

## 故障排查

### CAN 接口 DOWN
```bash
# 重新激活 (物理名)
sudo ip link set canX down
sudo ip link set canX type can bitrate 1000000
sudo ip link set canX up

# 或通过 can_activate.sh 重命名+激活
bash can_activate.sh <符号名> 1000000 "<bus-info>"
```

### candump 无数据
- 检查对应机械臂是否上电
- 检查 USB-CAN 适配器 LED 指示
- `lsusb` 确认适配器枚举

### launch 报错找不到 can_left_slave 等接口
- 说明未执行 `can_activate.sh` 重命名步骤
- `ip link show type can` 检查当前接口名, 如果是 `can0`~`can3` 需要先重命名

### piper_sdk 关节读取 NaN
- SDK 0.6.1 API: 使用 `GetArmJointMsgs()` 而非旧版 `GetArmJointMsgNum()`
- ConnectPort() 后至少等 2 秒再读取

### 相机掉线
- `lsusb | grep RealSense` 检查枚举
- 重新插拔 USB 线
- 检查 USB Hub 供电是否充足 (3 个相机共享 Hub 时注意带宽)

### DAgger 采集 QoS 问题
- ROS2 版本需使用 RELIABLE QoS (匹配 ROS1 TCP 行为)
- 控制循环使用 `rate.sleep()` 而非 `time.sleep(1.0/hz)` 以保证时序精度

---

## 关键源码索引

| 文件 | 说明 |
|------|------|
| `ros2_ws/src/piper/launch/start_ms_piper_new_launch.py` | Master-Slave 启动 (4 臂, 使用符号名) |
| `ros2_ws/src/piper/launch/inference_full_launch.py` | 纯推理启动 (2 slave can1/can2 + 3 相机) |
| `ros2_ws/src/piper/scripts/piper_start_ms_node_new.py` | MS 节点实现 (模式切换, 使能, 拖拽示教) |
| `ros2_ws/src/piper/scripts/policy_inference_node.py` | ROS2 策略推理节点 (3 种模式) |
| `kai0/train_deploy_alignment/dagger/agilex/can_activate.sh` | 单个 CAN 接口重命名+激活脚本 |
| `kai0/train_deploy_alignment/dagger/agilex/activate_can_arms.sh` | 批量激活 4 臂 (需更新 bus-info) |
| `kai0/train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect_ros2.py` | DAgger 采集 (ROS2) |
| `kai0/train_deploy_alignment/dagger/agilex/collect_data_ros2.py` | 独立数据录制 (ROS2) |
| `kai0/scripts/serve_policy.py` | WebSocket 策略推理服务 |
| `config/cameras.yml` | 相机配置 (序列号, topic) |
| `config/pipers.yml` | 机械臂配置 (CAN 接口, 符号名, bus-info) |
