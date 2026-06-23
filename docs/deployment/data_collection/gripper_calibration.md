# 夹爪标定 (Gripper Calibration) — 官方 0–70mm 规范

把 4 个 Piper 夹爪(`can_left_mas` / `can_right_mas` / `can_left_slave` / `can_right_slave`)
统一配成**官方规范坐标系**:`max_range = 70mm`,且 **command 0 = 机械全闭**。配完之后:

- 主臂读 `0(闭)..70mm(开)`,从臂命令 `0(闭)..70mm(开)`;
- 遥操 master→slave **纯 1:1 直通**(`arm_teleop_node` / `arm_reader_node` 的
  `joint_6 = round(position[6]*1e6)` 原样下发),**不需要任何软件 remap**;
- 主夹爪捏到底 → 从夹爪闭到底。

工具:[`piper_tools/configure_gripper_official.py`](../../../piper_tools/configure_gripper_official.py)

---

## 何时需要标定

- **首次部署** / 新机器;
- **更换夹爪 / 夹爪固件复位** 之后;
- 出现 **「主夹爪捏到底,从夹爪没闭到底」** 或主从夹爪开合幅度明显不一致;
- `--check` 显示某只夹爪 `max_range != 70`(本机历史出厂值是 100,非官方)。

## 前提

```bash
./start_scripts/start_data_collect.sh stop   # 或 pkill -f ros2 —— 必须先停遥操, 释放 CAN
```
4 路 CAN 接口已激活(`piper_tools/activate_can.sh`),机械臂上电。

## 标定步骤

```bash
cd /data1/tim/workspace/deepdive_kai0
python3 piper_tools/configure_gripper_official.py --role both --arm both
```

依次对 **left-master → left-slave → right-master → right-slave** 各弹一次提示:

> 用手把该夹爪捏合到机械**硬底**并**按住别松**,然后按 Enter 设零。

每只应打印 `设零后读数 ≈ 0`。工具对每只夹爪做两件事:
1. `GripperTeachingPendantParamConfig(100, 70, 1)` + `ArmParamEnquiryAndConfig(4)` → 设 `max_range=70`;
2. 失能夹爪 → 你手压机械底 → `GripperCtrl(0,1000,0x00,0xAE)` 把当前位置设为 0 点。

> 只需重标某几只时:`--role slave --arm both` / `--role master --arm right` 等组合。

## 验证

```bash
python3 piper_tools/configure_gripper_official.py --check   # 4 只都应 max_range=70, 静止 angle≈0
bash start_scripts/kai/start_teleop.sh                      # 主夹爪捏到底 → 从夹爪闭到底, 左右都验
```

`arm_teleop_node` 启动日志应是 `[gripper] no calibration file; using identity 1:1 mapping`
(规范化后就是纯 1:1,无 remap 配置文件)。

---

## 为什么 4 只都要**人手压住**设零(关键坑)

`set_zero(0xAE)` **只在夹爪失能(code `0x00`)时生效**;而失能瞬间从臂夹爪被内部弹力
**顶开几 mm**(右从臂实测回弹 3mm)。所以:

- **自动驱动**顶死机械底后再失能设零 → 失能即回弹 → 零点设在偏开位置 → 命令 0 闭不到底;
- 带力(`0x01`)时发 `0xAE` → 被固件忽略(无效);
- **唯一可靠**:人手把夹爪压在机械硬底、保持不松,失能设零 → 零点 = 真实硬底。

运行时夹爪**始终使能**,电机会主动驱到零点(=硬底,能克服回弹弹力),所以 command 0 必闭到底。
主、从同理,故 4 只全部手动。(早期试过自动驱动从臂 + `set_zero_pressed`,均因回弹失败,
已废弃。)

## 持久化 & 部署注意

- `set_zero` / `max_range` 写入夹爪固件,**掉电不丢**,重启遥操即生效;需要重标时再跑一次即可
  (`set_zero` 可反复重设,无出厂回退)。
- **坐标系变了**:旧的 100mm-range 数据集 / ckpt 的夹爪维度(action/state 第 6、13 维)
  部署到现在的 70mm-range 真机时,需把这两维的 norm_stats 按 **70/100 = 0.7** 重算后再部署
  (用新范围重新解算 q01/q99),否则模型输出的夹爪幅度会偏大。
