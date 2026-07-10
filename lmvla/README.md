# LMVLA — Latent-Milestone VLA

> **一句话**:用**零训练的 milestone 结构**（CRAVE）驱动一个**递归里程碑世界模型**（LMWM），
> 把预测出的 latent milestone 子目标注入 π0.5 action expert，得到 **LMWAM = LMWM × kai0 π0.5**，
> 在 RoboTwin 2.0（sim，验方法）+ kai0 叠衣真机（验域）上以 **SR / action-MAE** 为唯一裁决。

这是一个**伞形项目**，把两个可独立运行的子项目串成一条从"感知里程碑"到"策略执行"的流水线：

| 子项目 | 角色 | 一句话 |
|---|---|---|
| [`crave/`](crave/README.md) | **结构/价值引擎（感知侧）** | 零训练从冻结视觉特征挖 milestone 图 + 技能切分 + progress/value（`frames → encoder → cluster → order → readout`） |
| [`lmwm/`](lmwm/README.md) | **递归里程碑世界模型（动态侧）** | 在 CRAVE 的 milestone 状态上训练 next-milestone / latent subgoal 预测器，产出可注入 VLA 的子目标接口 |

顶层（本目录）只做**总纲 + 导航 + 跨子项目约定**，不含运行代码；代码分别住在两个子项目里。

---

## 数据流（端到端）

```text
CRAVE demo videos
   └─► frozen encoder (默认 DINOv3-H)        [crave.encoders]
         └─► KMeans 簇 + 顺序化(precedence/isotonic)   [crave.clustering]
               └─► milestone 图 + 每帧 progress/value    [crave.value]  ← 零训练结构资产
                     └─► milestone id / prototype 序列数据集
                           └─► LMWM 递归世界模型            [lmwm.runtime]
                                 · Greedy / Max-product next milestone
                                 · latent prototype 子目标 + 置信度/熵
                                 └─► 注入 π0.5 (SigLIP 虚拟图像 token 进 prefix + KI)
                                       └─► LMWAM 策略 → RoboTwin 2.0 / 真机  → SR
```

- **CRAVE** 的不可替代资产 = *milestone 图 + 零标签技能切分*，value 是副产品。
- **LMWM** 把静态 milestone 变成*可预测的动态*：给定当前帧特征，输出"下一个 milestone / 子目标"。
- **注入**走 VLM prefix 的 token 化通路（milestone 就在 SigLIP=PaliGemma 视觉空间，近零 distribution-shift），
  保预训练靠 **KI stop-grad**，非冻 backbone。详见 [`docs/architecture_overview.md`](docs/architecture_overview.md)。

---

## 目录结构

```text
lmvla/
├── README.md              ← 你在这里（总纲 + 导航）
├── pyproject.toml         # uv workspace 声明（members = crave, lmwm）；仅统一 tooling，不是运行时环境
├── .gitignore
├── docs/                  # 父层跨子项目文档
│   ├── README.md          #   文档索引
│   ├── architecture_overview.md   #   CRAVE→LMWM→VLA 全景 + 注入机制
│   ├── roadmap.md         #   E0→E3 阶段路线 + kill criteria
│   └── glossary.md        #   术语锁(milestone / Greedy / Max-product / KI / LMWAM …)
├── crave/                 # 子项目 1：零训练 milestone/value 引擎（自带 pyproject/src/docs）
└── lmwm/                  # 子项目 2：递归里程碑世界模型（自带 pyproject/src/docs）
```

---

## 快速开始

两个子项目**运行时环境不同**（顶层 workspace 不提供运行时，只统一 lint/dev tooling）：

```bash
# CRAVE — 在 srpo 环境（torch 2.10 + transformers 4.57, DINOv3-capable）
/home/tim/miniconda3/envs/srpo/bin/python -m pip install -e crave/
/home/tim/miniconda3/envs/srpo/bin/python crave/scripts/generalize.py coffee --encoder dinov3-h

# LMWM — 见 lmwm/README.md（PyTorch，>=2.1）；当前最优 world-model artifact：
#   lmwm/checkpoints/stage3_unified/.../best.pt
python -c "from lmwm.runtime import UnifiedLMWMPredictor"
```

各自的安装/运行/最优 artifact/接口，见子项目 README：[crave](crave/README.md) · [lmwm](lmwm/README.md)。

---

## 文档导航

- **想看整体怎么拼** → [`docs/architecture_overview.md`](docs/architecture_overview.md)
- **想看走到哪了 / 下一步** → [`docs/roadmap.md`](docs/roadmap.md)（E0→E3）+ [`lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md`](lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md)（总执行规划，单一事实源）
- **术语对不上** → [`docs/glossary.md`](docs/glossary.md)
- **CRAVE 方法/定位** → [`crave/docs/CRAVE_positioning_and_roadmap.md`](crave/docs/CRAVE_positioning_and_roadmap.md) · [`crave/docs/STATUS.md`](crave/docs/STATUS.md)
- **LMWM 架构/注入** → [`lmwm/docs/LMWM2_FINAL_ARCHITECTURE.md`](lmwm/docs/LMWM2_FINAL_ARCHITECTURE.md) · [`lmwm/docs/INJECTION_DESIGN_2026-07.md`](lmwm/docs/INJECTION_DESIGN_2026-07.md)
- **仿真评测环境** → [`../docs/deployment/robotwin_sim_env_setup.md`](../docs/deployment/robotwin_sim_env_setup.md)（RoboTwin 2.0 双机 eval）
