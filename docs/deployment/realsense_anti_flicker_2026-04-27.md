# RealSense 抗闪烁修复 — 2026-04-27 分水岭

## TL;DR

2026-04-27 修复了 RealSense RGB 流在 LED 灯下的可见闪烁。**该日之前采集的所有训练数据**（Task A/B/C 的 `base/`、`dagger/` 等）在视觉上与**今天起的部署图像 / 新采集数据**存在系统性差异。

需要在训练 / 评估时对**旧数据**做处理或在 augmentation 上对齐，否则视觉敏感的 policy（如 AWBC）可能在真机上表现出训练—部署 gap。

---

## 1. 修了什么 + 在哪儿

### D435（顶视，rolling-shutter RGB）

**问题：** 50Hz 市电下 LED 灯产生 100Hz 亮度脉动 → rolling shutter 把它积分成横向暗带 ("波纹")。

**修法：** 把 RGB sensor 的 `power_line_frequency` 选项设为 `1`（50Hz），让 AE 在曝光时间选择上避开闪烁频率。

### D405（左右手腕，global-shutter color on depth_module）

**问题：** Global-shutter 不会出 rolling-band，但**LED 驱动器的 kHz 级 PWM**会让整帧亮度脉动。`power_line_frequency` 选项对全局快门 sensor 无效。

**修法：** 关掉 AE，把曝光时间锁定到 **20000 μs (20 ms)**，让单帧曝光跨越多个 PWM 周期，PWM 自然平均掉。20 ms 在 sim01 工位下经过实测：闪烁消失，亮度足够。

### 代码位置

| 用途 | 文件 | 改动 |
|------|------|------|
| 部署 / 推理 | `ros2_ws/src/piper/scripts/multi_camera_node.py:131-148` | `pipeline.start` 后通过 `set_option` 设 PLF=1；`is_d435=False` 分支额外锁 `enable_auto_exposure=0`、`exposure=20000` |
| 数据采集 | `start_scripts/launch_3cam.py:21-58` | `make_camera_node` 加 `is_d405` 参数；D435 设 `rgb_camera.power_line_frequency=1`；D405 额外设 `depth_module.enable_auto_exposure=False` 和 `depth_module.exposure=20000` |

---

## 2. 配置参数对比（只列变化项）

✅ = 显式变更　⚠️ = 间接受影响（因 AE 关闭等副作用）

### 2.1 D435 顶视相机（serial `254622070889`）

| 参数 | 旧（< 2026-04-27） | 新（≥ 2026-04-27） | 状态 |
|------|----|----|---|
| `power_line_frequency` | 驱动默认 | **`1` (50Hz)** | ✅ |
| `exposure`（μs） | AE 选取，1 000–33 000 间漂移 | AE 选取，但被 PLF 约束到 100Hz 整数倍 | ⚠️ |

> **变更要点：** 只动了 PLF 一个值。AE 仍开 → 整体亮度仍随环境光自适应。
> **未变化：** 分辨率 / 帧率 / 色彩格式 / AWB / IR 投射器 / 深度后处理（spatial+temporal filter）/ align_depth — 全部保持原样。

### 2.2 D405 左 / 右手腕相机（serial `409122273074` / `409122271568`）

| 参数 | 旧（< 2026-04-27） | 新（≥ 2026-04-27） | 状态 |
|------|----|----|---|
| `enable_auto_exposure` | true（驱动默认） | **false（锁定）** | ✅ |
| `exposure`（μs） | AE 选取，约 1 000–8 000 间漂移 | **20 000 固定（20 ms）** | ✅ |
| `gain` | AE 控制（16–248 漂移） | 驱动默认值停留在 ≈16 | ⚠️ |
| `power_line_frequency` | 驱动默认 | `1`（**对 global shutter 无效**，仅代码统一） | — |

> **变更要点：** AE 关 + 曝光锁 20 ms。`gain` 因 AE 关闭也间接锁定到驱动默认 ≈16，所以**整体亮度由"曝光 + 增益"双重锁定**，对环境光不再自适应。
> **未变化：** 分辨率 / 帧率 / 色彩格式 / AWB / IR 投射器 / 深度后处理 — 全部保持原样。

### 2.3 关键 4 项一览

| 选项 | D435 旧 → 新 | D405 旧 → 新 |
|------|-------------|-------------|
| `power_line_frequency` | default → **1 (50Hz)** ✅ | default → 1（无效）|
| `enable_auto_exposure` | true → true | **true → false** ✅ |
| `exposure` (μs) | AE 控制（动态） | AE 动态 → **20 000 固定** ✅ |
| `gain` | AE 控制（动态） | AE 动态 → **≈16 固定** ⚠️ |

---

## 3. 旧数据 vs 新数据的图像层面差异

把上面的配置变更翻译成**图像像素层面**会出现什么不同：

### 3.1 总览表（只列发生变化的维度）

| 图像维度 | 旧数据（< 2026-04-27） | 新数据（≥ 2026-04-27） | 来自哪个配置改动 |
|---------|----------------------|----------------------|----------------|
| **D435 横向 banding** | 有（沿 H 方向 100Hz 暗带，幅度 2–8%） | 无 | PLF=1 |
| **D405 帧间整帧亮度脉动** | 有（PWM 高频，相邻帧亮度差 1–3%） | 无 | 曝光 20 ms 跨过 PWM 周期 |
| **D405 episode 内整体亮度** | **随场景动态漂移**（手臂遮挡 → AE 拉高） | **恒定** | AE off |
| **D405 单帧亮度均值（典型）** | 0.30 – 0.65（宽） | 0.40 – 0.50（窄） | AE off + 固定曝光 |
| **D405 曝光时长** | 1–8 ms（AE 实测） | 20 ms（固定） | exposure=20000 |
| **D405 运动模糊** | 几乎没有（短曝光） | 快动作时轻微拖影 | exposure=20000 |
| **D405 暗部噪点** | AE 高 gain 时噪点明显（gain 漂到 100+） | 较稳定（gain≈16） | AE off |

> **未变化（不影响数据适配）：** D435 整体亮度（AE 仍工作）/ 白平衡 / 分辨率 / 帧率 / 深度流 / IR 投射器 — 旧数据和新数据在这些维度上一致，无需处理。

### 3.3 对策略学习的影响排序

按**对模型行为影响从大到小**，需要重点处理的就是前两项：

1. **D405 episode 内亮度漂移消失 ⭐⭐⭐⭐⭐**
   旧数据里 AE 会随场景动态调整亮度（手臂遮挡相机时拉高曝光，物体进入视野时压低）。这种漂移**和动作 / 任务进度强相关**，模型很容易学到"亮度变化 → 状态变化"的**伪信号**。新数据完全没有，部署时如果模型依赖这个伪信号会失灵。
   _训练—部署 gap 的最大来源，必须处理。_

2. **D405 单帧亮度均值分布从宽变窄 ⭐⭐⭐⭐**
   旧 0.30–0.65 vs 新 0.40–0.50。模型见过的明暗极值在新部署中消失，对极端场景鲁棒性下降。

3. **D435 横纹 + D405 PWM 脉动消失 ⭐⭐**
   高频纹理 / 噪声，augmentation 里加 random brightness 已经基本能 cover。

4. **D405 运动模糊增加 + 高 gain 噪点消失 ⭐**
   仅在快动作 / 极暗场景轻微影响。Task A 慢任务可忽略。

---

## 4. 旧数据适配策略

四个思路，按**改动量从小到大**排序。

### 策略 A：训练时 photometric augmentation（推荐起点）

最小改动。已有的 `kai0/src/openpi/models_pytorch/preprocessing_pytorch.py` 已经做 brightness/contrast/saturation 抖动，**只需要在已有 aug 里再加两个能 cover 闪烁的随机扰动**：

```python
# 伪代码示意：在现有 aug pipeline 里追加
def simulate_d435_horizontal_band(img, prob=0.3):
    # 模拟 50Hz 横纹 — 给 H 维度加正弦亮度
    if random.random() < prob:
        h = img.shape[-2]
        amp = random.uniform(0.02, 0.08)         # 振幅 2–8%
        freq = random.uniform(2.0, 6.0)          # 周期数 / 帧
        phase = random.uniform(0, 2*math.pi)
        bands = 1 + amp * np.sin(2*math.pi*freq*np.arange(h)/h + phase)
        img = img * bands[None, :, None]         # 仅 H 维度调制
    return img

def simulate_d405_brightness_drift(img, prob=0.5):
    # 模拟 AE 漂移 — 整帧亮度乘随机系数
    if random.random() < prob:
        img = img * random.uniform(0.7, 1.3)
    return img.clip(0, 1)
```

**作用：** 把"闪烁/AE 漂移"这个分布差异变成模型在训练时见过的常规扰动。新旧数据在模型眼里都是被随机扰动过的版本，gap 被吸收。

**缺点：** 不能消除已有的真实闪烁，只能让模型对它不敏感。

**适用于：** 想保留旧数据 + 不想做离线后处理。AWBC 这种本来就重视觉的策略，这一步几乎是必做。

---

### 策略 B：离线"去闪烁"旧数据（中等改动）

把旧数据物理上处理成"看起来像新数据"。

#### B-1. D405 亮度归一化（必做）

按 episode 计算每帧整体亮度，统一到一个目标值（比如 episode 中位数，或全数据集中位数）：

```python
# 单 episode 处理示意
import numpy as np
import h5py

def normalize_episode_brightness(images, target_mean=0.45):
    """images: (T, H, W, 3) float32 in [0,1]"""
    out = np.empty_like(images)
    for t, img in enumerate(images):
        cur_mean = img.mean()
        gain = target_mean / max(cur_mean, 1e-6)
        gain = np.clip(gain, 0.5, 2.0)  # 限制极端值
        out[t] = (img * gain).clip(0, 1)
    return out
```

把这步加到 `kai0/src/openpi/data_pipeline/`（具体路径以你 dataset loader 实际位置为准）的 transform 链里，**只对旧数据 episode 启用**（用 episode metadata 里的采集时间判断 cutoff）。

#### B-2. D435 横纹去除（可选）

50Hz 横纹是固定空间频率，可以用沿 H 维度的 1D 低通做基线估计后除掉：

```python
def deband_d435(img, sigma=8):
    """img: (H, W, 3) float32. 沿行方向估算亮度基线后除以基线."""
    from scipy.ndimage import gaussian_filter1d
    luma = img.mean(axis=-1)                          # (H, W)
    row_mean = luma.mean(axis=1)                      # (H,)
    smooth = gaussian_filter1d(row_mean, sigma=sigma) # 低通基线
    correction = (smooth / row_mean.mean())[:, None, None]  # (H,1,1)
    return (img / np.clip(correction, 0.5, 2.0)).clip(0, 1)
```

⚠️ 这种简单 deband 会**损伤画面里真实的水平结构**（比如桌面边缘）。建议先在几张样本上目视验证，再决定是否做全量。如果效果不理想，跳过 B-2，靠策略 A 的 aug 兜底。

#### 实施方式

写一个一次性脚本，扫描旧数据集，生成镜像目录 `*_debanded/`：

```bash
python scripts/process_legacy_camera_data.py \
    --input data/task_a/base/ \
    --output data/task_a/base_normalized/ \
    --cutoff-date 2026-04-27 \
    --d405-brightness-norm true \
    --d435-deband false   # 默认关，肉眼验证后再开
```

**优点：** 训练时不需要再加 aug，新旧数据看起来一致。
**缺点：** 数据膨胀一倍；deband 不完美可能引入新 artifact；亮度归一化可能擦掉一些真实的光影变化（虽然这些变化大多本来就是 AE 伪造的）。

---

### 策略 C：让新数据"看起来像旧数据"（不推荐）

往新数据上加合成闪烁，使其与旧数据匹配。技术上等价于策略 A 但只在新数据上做。**不推荐**因为部署时模型看到的是干净图像，让模型适应"脏"分布反而会降低部署性能。

---

### 策略 D：重采全部旧数据（最彻底）

如果数据规模可控（比如总时长 < 10 小时），重采是最干净的选项。每条 episode 用同样的 task prompt + 操作脚本回放一次（如果你有自动回放系统）或人手重采。

**适用于：** 数据量不大，且对未来长期使用价值高的场景（比如要发论文或长期上生产）。

---

## 5. 推荐执行顺序

1. **先做策略 A（必做）** — 在 `preprocessing_pytorch.py` 里追加 D405 brightness drift sim + D435 horizontal band sim。这是最便宜的兜底。
2. **再做策略 B-1（推荐）** — 对所有 < 2026-04-27 的 episode 做 D405 brightness normalization，生成 `*_normalized/` 镜像数据。然后训练里混用归一化后的旧数据 + 原生的新数据。
3. **B-2 D435 deband 默认不做** — 除非你目视确认横纹明显且策略 A 兜不住。
4. **策略 D（重采）** — 仅在小批关键数据集上做，比如 Task A 的最终 eval 数据。

---

## 6. 数据集级别的 cutoff 判断

每条 episode 应该有采集时间戳（来自 hdf5 metadata 或文件 mtime）。判断逻辑：

```python
from datetime import datetime
CUTOFF = datetime(2026, 4, 27)

def needs_legacy_processing(episode_meta) -> bool:
    ts = episode_meta.get('collection_time')  # ISO string or epoch
    if ts is None:
        # 旧数据可能没记 metadata, fallback 到文件 mtime
        ts = os.path.getmtime(episode_meta['path'])
        ts = datetime.fromtimestamp(ts)
    elif isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts < CUTOFF
```

如果 metadata 完全缺失，建议用 `git log` 找出每个 dataset 目录最早被 commit / 拷贝的时间作为 cutoff 的代理。

---

## 7. 验证清单

处理完旧数据后，在训练前抽查：

- [ ] D405 帧亮度直方图：旧（处理后）vs 新（原生），均值应接近、方差应接近
- [ ] 抽 5 条旧 episode 拼成视频，肉眼看亮度漂移是否平稳
- [ ] D435 抽几张帧做行均值曲线，处理后应当平坦
- [ ] 用 `tools/verify_dataset_visual.py`（如果不存在，可以新写一个 5 行的脚本）跑一次 sanity check
- [ ] 新训练的 wandb run 在 val 集上的 image MSE / brightness MSE 与旧训练相比无明显回归

---

## 8. 部署侧反向兼容

如果你想用**旧的 checkpoint**（< 2026-04-27 训练的）跑新的部署画面：

- 旧 checkpoint 没见过"稳定亮度 + 无横纹"的图像分布
- 临时方案：在 `policy_inference_node.py` 的图像 preprocessing 里**反向**注入闪烁 + AE 漂移，让部署画面看起来像旧训练分布
- 这只是临时兜底，应尽快用新数据 + 旧数据归一化重训 checkpoint

---

## 9. 长期建议

- 把 `power_line_frequency` 和 `exposure` 这两个值移到 `config/cameras.yml`，避免分散在多处。
- 在 episode 元数据里记录当时的 RealSense 选项快照（`power_line_frequency`、`exposure`、`gain`、`auto_exposure`、固件版本），未来再有类似分水岭就能精确定位。
- 加一个 `tools/inspect_camera_options.py`，启动后读取所有 RealSense 当前选项打印 + 写到 latest run dir，每次 deploy / 采集自动留痕。
