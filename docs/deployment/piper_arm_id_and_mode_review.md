# Piper 双臂 — 标识 & 主从模式调研

> 写于 2026-05-23。回答两个问题：
> 1. 当前 `piper_tools/` 如何区分左右 + 主从？
> 2. 是否有更好的方法（不依赖按物理按钮 + 不依赖 USB 口顺序）？

参考代码库：
- `/home/tim/workspace/deepdive_kai0/piper_tools/` — 本仓 CAN 工具
- `/home/tim/workspace/piper_sdk/` — Agilex 官方 SDK
- `/home/tim/workspace/PikaAnyArm/piper/` — Pika 项目的 Piper 封装
- `/home/tim/workspace/pyAgxArm/` — Agilex 多臂统一 SDK
- `/data1/tim/workspace/deepdive_kai0/ros2_ws/src/piper/scripts/arm_*` — 本仓 ROS2 节点

---

## 一、现状

### 1. 左右识别

| 组件 | 机制 | 健壮性 |
|---|---|---|
| `piper_tools/activate_can.sh` (SLAVE_MAPPINGS / MASTER_MAPPINGS) | USB bus-info 静态映射，如 `3-2.2.1:1.0 → can_left_mas` | 脆 — 拔插 USB / 换口立刻错位 |
| `piper_tools/find_n_activate.sh` | 在已知 bus-info 上 `slcand` 起 CAN 接口 | 同上 |
| `piper_tools/calibrate_can_mapping.py` | 一次性校准工具，把当前物理接线写入 `activate_can.sh` | 校准时正确，校准后仍依赖位置 |

### 2. 主从识别

| 组件 | 机制 |
|---|---|
| `piper_tools/calibrate_can_mapping.py:150` | 调用 `MasterSlaveConfig(0xFA / 0xFC, 0,0,0)` 写入 firmware NVM |
| Firmware role 字节 | `0xFA` = 示教输入臂 (master / teach input)；`0xFC` = 运动输出臂 (slave) |
| 切换 master → slave | **必须断电重启**（calibrate_can_mapping.py:506-507 文档化） |
| 物理按钮 | 按下后 master 进入"绿色伺服 / 拖动"模式，由操作员手动驱动 |

### 3. ROS2 节点的实际行为

- `arm_teleop_node.py:203,314,362` —— **已经在用** `MotionCtrl_1(grag_teach_ctrl=0x01)` **软件**触发拖动模式
- `arm_master_servo_node.py:251-279` —— 5 Hz 轮询 `teach_status` 被动感知物理按钮
- 也就是说软件入口已经走通，但没有完全统一替代物理按钮 + firmware role 切换

---

## 二、Piper SDK 提供的关键 API

| API | CAN ID | 用途 | 是否持久 |
|---|---|---|---|
| `MasterSlaveConfig(0xFA, 0,0,0)` | 0x470 | 设为主臂（teach input） | NVM 持久，需重启生效 |
| `MasterSlaveConfig(0xFC, 0,0,0)` | 0x470 | 设为从臂（motion output） | NVM 持久，需重启 |
| **`MotionCtrl_1(grag_teach_ctrl=0x01)`** | **0x150** | **软件触发拖动示教模式** = 等价物理按钮 | 运行时立即生效 |
| `MotionCtrl_1(grag_teach_ctrl=0x02)` | 0x150 | 软件退出拖动 | 同上 |
| `MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01)` | 0x151 | 切到 CAN 控制模式 + MOVE_J | 同上 |
| `EnableArm(7)` | — | 使能 6 关节 + gripper | 同上 |
| `EmergencyStop(0x01)` | 0x150 | 急停 | 同上 |
| `GetArmStatus().arm_status.ctrl_mode` | 0x2A1 byte0 | 实时读模式 | — |
| `.teach_status` | 0x2A1 byte3 | 实时读拖动状态（物理按钮 / 软件触发都反映） | — |

### `ctrl_mode` 完整枚举（参考 `arm_feedback_status.py`）

| 值 | 名 | 含义 |
|---|---|---|
| 0x00 | STANDBY | 待机 |
| 0x01 | CAN_CTRL | CAN 指令驱动 |
| 0x02 | TEACHING_MODE | 示教模式 |
| 0x03/04/05 | ETHERNET / WIFI / REMOTE | 其它通道 |
| 0x06 | LINKAGE_TEACHING_INPUT_MODE | 联动示教（主从在 firmware 里互连时主臂的状态） |
| 0x07 | OFFLINE_TRAJECTORY | 离线轨迹回放 |

---

## 三、其它 SDK 的设计差异

### `pyAgxArm/` — 抽象更清晰
- 改名 master/slave → **leader/follower**
- `set_leader_mode()` / `set_follower_mode()`
- 专门的 `restore_leader_drag_mode()` 用于"home 后恢复拖动"
- 文件：`pyAgxArm/protocols/can_protocol/msgs/piper/default/transmit/arm_leader_follower_config.py`
- 底层仍是 0x470 + 0xFA/0xFC，只是 API 名字更易读

### `PikaAnyArm/piper/` — 无差异
- 仅是 ROS topic 封装（`/left_joint_states` / `/right_joint_states`）
- 主从逻辑完全委托给 `piper_sdk`

---

## 四、推荐改造方向

### A. 左右识别 → USB-CAN dongle iSerial（**已落地**）

**前提澄清**：Piper 机械臂本体没有暴露任何唯一硬件 ID（CAN 协议表里没有 SN 字段，
SDK 只能读固件版本字符串如 `S-V1.5-2`，4 个臂如果固件相同就完全无法区分）。

**实际可用的唯一标识**：candleLight (gs_usb) USB-CAN dongle 每个出厂烧写了 24 位 hex
iSerial（如 `004600434148570C20343133`），4 个 dongle 互不相同。
**把每个 dongle 物理上固定到一只臂**（短 CAN 线绑死），dongle serial 就等价于
那只臂的"唯一 ID"。

#### A.1 落地脚本（已写入 `piper_tools/`）

| 文件 | 作用 |
|---|---|
| **`setup_can_v2.sh`** | **每次重新识别映射时跑**: 走 HITL (依次晃 4 个角色) → 检测 iface→serial → 写 `config/dongle_serials.yml` → 自动激活. 替代 `setup_can.sh` (v1, bus-info) |
| **`activate_can_v2.sh`** | **没改物理链接时直接跑**: 按 YAML 把当前 canX 重命名为符号名 + bitrate up, **不动映射**. 替代 `activate_can.sh` (v1, bus-info) |
| `auto_remap_can.sh` | 智能入口 (可选): serial 集合不变就调 `activate_can_v2.sh`; 变了就拉 `setup_can_v2.sh` |
| `calibrate_serial_hitl.py` | `setup_can_v2.sh` 的 Python 内核 (HITL 检测逻辑) |
| `export_dongle_serials.py` | 兼容 v1: 在已跑 `setup_can.sh` 拿到符号名后, 抓 serial 写 YAML, 跳过 HITL. 仅作历史桥接 |

#### A.2 工作流 (v2)

**两条命令搞定所有场景**:

```bash
# 任何想重新识别 (dongle ↔ 臂) 映射时:
bash piper_tools/setup_can_v2.sh                 # HITL: 晃 4 次 → 写 YAML → 激活

# 没改物理链接时, 直接激活 (开机 / USB 口换位置都用这条):
bash piper_tools/activate_can_v2.sh              # 按 YAML 重命名 + bitrate up
```

**何时跑哪一条**:

| 你做的 | 用哪条 |
|---|---|
| 关机重启 / USB 口换位置 / 整机搬位 | `activate_can_v2.sh` |
| 首次部署 / 换 dongle / 调换 dongle ↔ 臂 / 怀疑映射错位 | `setup_can_v2.sh` |

**可选智能入口 (自动选 v2 中那一条)**:

```bash
bash piper_tools/auto_remap_can.sh               # 检测 serial 集合是否变, 选择 setup/activate
```

| 场景 | 触发条件 | auto_remap 的处理 |
|---|---|---|
| C0 一切不变 | serial 集合 == YAML | 调 `activate_can_v2.sh` |
| C1 USB 口换位置 | serial 集合不变, 顺序变 | 调 `activate_can_v2.sh` (这是 serial 方案的核心收益) |
| C2 dongle 换臂绑定 | serial 集合不变, 但语义错了 | **不可自动检测**, 提示跑 `verify_can_mapping.py` |
| C3 新增/换/拔 dongle | serial 集合变了 | 拉起 `setup_can_v2.sh --no-activate` → 再激活 |

**标志位**:
- `setup_can_v2.sh --no-activate` — 只写 YAML, 不激活
- `auto_remap_can.sh --force-hitl` — 跳过 diff, 强制 HITL
- `auto_remap_can.sh --no-hitl` — serial 变了不进交互, 退出码 3 (CI/守护)

#### A.3 可选: 装 udev rules 走内核级命名

`export_dongle_serials.py` 会打印形如：

```udev
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="gs_usb", \
  ATTRS{serial}=="004600434148570C20343133", NAME="can_left_mas"
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="gs_usb", \
  ATTRS{serial}=="004100484148570C20343133", NAME="can_left_slave"
# ... 共 4 条
```

装到 `/etc/udev/rules.d/99-piper-can.rules` 后 `udevadm control --reload + udevadm trigger`,
之后插上 dongle 直接以符号名出现在 `ip link show`, 连 activate_can_v2.sh 都不用跑。

#### A.4 sysfs 探针 (调试 / 手动查 serial)

```bash
for i in can0 can1 can2 can3; do
  echo "$i → $(udevadm info -q property -p /sys/class/net/$i \
                | grep ID_SERIAL_SHORT)"
done
# 或更原始:
for i in can0 can1 can2 can3; do
  cur="$(readlink -f /sys/class/net/$i/device)"
  while [ -n "$cur" -a "$cur" != "/" ]; do
    [ -f "$cur/serial" ] && { echo "$i → $(cat $cur/serial)"; break; }
    cur="$(dirname $cur)"
  done
done
```

### B. 主从切换 → 软件级（核心）

**统一约定**：所有 4 个臂的 firmware role 全部设为 `0xFC` (slave / motion output)，运行时纯软件按需切换"哪个臂当主"。

| 模式 | firmware role | 软件操作 |
|---|---|---|
| 启动 | 4 × `0xFC` | 一次性写入，永远不改 |
| 遥操（左） | (同上) | `left_master.MotionCtrl_1(grag_teach=0x01)` → 拖动模式 → 读 joint 转发给 `left_slave.JointCtrl(...)` |
| 遥操（右） | (同上) | 同左 |
| 自主 | (同上) | `MotionCtrl_1(grag_teach=0x02)` 退出拖动 → 4 个臂全部 `JointCtrl` 接策略输出 |
| 急停 | (同上) | `EmergencyStop(0x01)` |
| DAgger | (同上) | 部分臂拖动（人接管），其余 `JointCtrl`（策略） |

收益：
- ✅ 不用按物理按钮
- ✅ 不用断电重启切换角色
- ✅ "谁当主"完全可编程，DAgger 场景一行代码切换
- ✅ 物理按钮仍可作 emergency 拖动备用（`teach_status` 持续监听）

### C. 启动健康检查

启动后强制读 `GetArmStatus()` 校验：
- `ctrl_mode == 期望值`
- `teach_status == 期望值`
- `arm_status == NORMAL`（非 `EMERGENCY_STOP` / `JOINT_BRAKE_NOT_RELEASED` 等）

任一失败：报警 + 自动重写 + 等待操作员介入。**禁止静默继续** —— 否则后面策略动作可能在错误模式下执行。

---

## 五、与当前代码的差距 & 落地清单

✅ 已有：
- `arm_teleop_node.py` 已用 `MotionCtrl_1(grag_teach=0x01)` 软件入拖动
- `arm_master_servo_node.py` 已用相同 API

🚧 待做：
1. **firmware role 一次性归一**：写脚本把 4 个臂全部 `MasterSlaveConfig(0xFC)`，断电重启一次后永久生效，弃用"master arm = 0xFA"的概念
2. ✅ **dongle serial 替代 bus-info**：`export_dongle_serials.py` + `activate_can_v2.sh` 已实现（见 §四·A）
3. **ROS service 暴露动态主从**：`/enable_drag <arm_id>` / `/disable_drag <arm_id>` —— 任何臂临时变主
4. **启动健康检查**：在 `arm_teleop_node.py` / `arm_master_servo_node.py` 启动序列里加 `GetArmStatus()` 校验闭环
5. **（可选）pyAgxArm 风格抽象**：内部 wrapper 把 `MasterSlaveConfig` + `MotionCtrl_1` 包装成 `set_leader_mode()` / `set_follower_mode()`，可读性更好
6. **（可选）跑 export 一次, 把现在 4 个 dongle 的 serial 落档**：`bash piper_tools/setup_can.sh && python3 piper_tools/export_dongle_serials.py` —— 当前 sim01 4 个 dongle 的 serial 已经 probe 过（004600.../004100.../004200.../002E00... 后缀），跑这条命令即可绑定到符号名

---

## 附：关键代码定位

- Piper SDK 主接口：`piper_sdk/interface/piper_interface_v2.py:2393` (`MotionCtrl_1`)
- 主从协议消息：`piper_sdk/piper_msgs/msg_v2/transmit/arm_master_slave_config.py`
- 状态反馈枚举：`piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_status.py` (CAN 0x2A1)
- Demo (set master)：`piper_sdk/demo/V2/piper_set_master.py`
- Demo (set slave)：`piper_sdk/demo/V2/piper_set_slave.py`
- Calibrate 工具：`piper_tools/calibrate_can_mapping.py:140-160`
- ROS 节点（teleop）：`ros2_ws/src/piper/scripts/arm_teleop_node.py:200-220, 313-335`
- ROS 节点（master-servo）：`ros2_ws/src/piper/scripts/arm_master_servo_node.py:200-280`
- pyAgxArm leader/follower：`pyAgxArm/protocols/can_protocol/msgs/piper/default/transmit/arm_leader_follower_config.py`
- pyAgxArm driver：`pyAgxArm/protocols/can_protocol/drivers/piper/default/driver.py:1319-1356`
- 本仓 v2 HITL 校准：`piper_tools/setup_can_v2.sh` → `piper_tools/calibrate_serial_hitl.py`
- 本仓 v2 激活：`piper_tools/activate_can_v2.sh`
- 本仓 4 场景自动 remap：`piper_tools/auto_remap_can.sh`
- 本仓 v1 兼容桥：`piper_tools/export_dongle_serials.py`
