# LMWM 版本演进史 + 踩坑/错误路径

> 目的:**记录走过的路和踩过的坑,防止未来重复犯错。** 每条错误路径都写清「曾相信什么 / 为何错 / 正确结论」。
> 覆盖 2026-07-01 → 07-08。当前架构见 `ARCHITECTURE_AND_BASELINE.md` / `FINAL_REPORT.md` / `FINAL_CROSSTASK_PREDICTOR.md`。

---

## A. 版本演进时间线

| 日期 | 阶段 | 关键变化 | 结果/数字 |
|---|---|---|---|
| 07-01 | 立项 Recurrence-State WM | CRAVE 循环 milestone 状态上训世界模型,镜像 LaWM `(r_t,r_{t+h})→inv code→forward` | code_dim 32/64 |
| 07-01 | Stage-1 smoke → DINOv3-H proto→proto → 帧→proto | 逐步换真 DINOv3-H 1280 特征 | val top1 1.0(⚠️查表,非世界模型) |
| 07-01 | 循环概率图 + Stage-2/3 Unified | 帧→簇心→转移图→蒸馏进网 | 334875帧/37 milestone/1232 边;greedy 0.936/proto cos 0.99 |
| **07-02** | **Phase A 真实未来标签(转折)** | 发现 0.94 是"vs 图"循环(图标签与真实未来仅 24.2% 一致);改真实未来 | top1 0.94→**0.383**、NLL 16→**1.98**(首次超经验基线) |
| 07-02 | Phase B 校准+图作软先验 | T=1.30(ECE 0.10→0.005);对数线性池化 λ≈0.3 | top1 **0.417** |
| 07-02 | Phase C 帧历史(负结果) | 拼历史帧过拟合 | 0.383→0.377,单帧已到当前表述天花板 |
| 07-02 | Episode-medoid subgoal | 目标从全局簇心→本集 medoid | subgoal cos 0.832→**0.864** |
| 07-02→03 | 均值+方差 / augin / 优化 L1–L9 | augin 输入(H+prev-ms+state)、异构集成、蒸馏 student | v1 冠军 top1 **0.459**/NLL 1.72;student 部署 0.449 |
| 07-03 | 收紧架构 forward-from-current | 外观由当前观测带入;subgoal 双损失 | top1 **0.465**/subgoal 0.882;可训 ~21M(LaWM 1/11) |
| 07-03→04 | 最优解码器 | flow fixed-noise(首选)+ dec_v2(兜底);patch-grid 解码 L1 0.027 | — |
| 07-04→05 | V1→V2 目标术语 + patch-grid + VAE | V2=progress-next(value 单调);LAM 换 CNN(27M/路);VAE kl≈1e-2 | 容量+0.00、多模态 VAE+0.02 |
| **07-06** | **SigLIP-space 重设计** | 在线编码器 DINOv3-H → **π0.5 SigLIP 同塔**;DINOv3-H+CRAVE 降为离线 label 工厂 | deploy 0.716 ≥ DINOv3-H 0.694,融合零损失 |
| **07-06→07** | **Predictor/Generator 两模型** | inverse-teacher→码 + MDN 部署 + AdaLN 生成器;concat→AdaLN 修 77% persistence 塌缩 | ratio 0.29→0.99;center_w 扫定 **0.1**;gist>grid |
| 07-07 | LaWM baseline 实测 + 报告/网站重构 | 官方 ckpt 补 deploy-predm 测 reach | **reach 1.67s > LaWM 1.48s** |
| **07-07→08** | **跨任务 era** | 多任务 trainer(3锚 3teacher);coffee/xvla/vis bank;P0/P1b/P2/LOO/teacher 消融 | union_ce in-dist 最强;**teacher=proto 定案** |
| **07-08** | **teacher=proto(簇中心)定案** | 码=下一 milestone SigLIP 中心投影;去 InverseEnc+去 CE 锚 | 与 inv 打平、更简/更轻/开放词表;交付 `teach_proto_*.pt` |

---

## B. 踩坑与错误路径(核心:别再犯)

### B1. 评估与结论(方法学坑 —— 最贵)
| 曾相信 | 为何错 | 正确结论 |
|---|---|---|
| **top1 0.94 = 世界模型质量** | "vs 图"循环:图标签只 24.2% 匹配真实未来,图头对现实仅 0.233/NLL 16 | 一律**对真实未来**评估 → 真实 0.383 |
| Hybrid 图回退 0.997 = 改进 | 回退用的图先验正是标签来源 → 循环 | 图降级为软先验,诚实增益仅 +3.4pt |
| **多模态天花板低 / 只回收 +0.02**(反复出现) | **测错了轴** —— grid-cos/best-of-code 对**身份多峰**不敏感;真多峰在"下一个是哪个 milestone" | 换身份 top-N / MDN → best-of-N 随 K 单调回收(+0.29);"天花板低"作废 |
| ~13 分支是固有天花板(Phase C) | 高熵三因,只一种是真瓶颈;簇≠相位混叠 | 加 time-bin +3.5pt / prev-ms +11pt / state +5.2pt;是"当前表述"的天花板非根本 |
| frame-only kNN append prev-ms 无用 | cosine-append 稀释离散信号(方法学假象) | 用**学习型组合器** → prev-ms 突破 kNN 上界 |
| **单变量外推 / 跨模型复用超参**(λ/CVaR/β) | 子集赢≠全量赢;LaWM 的 β/CVaR/λ 搬到我们模型伤 top1 | **一次动一个变量、同 split 同 recipe**;换模型/数据 λ/CVaR 必重扫 |
| latent forecast 精度决定 VLA 收益 | 坐标系错:LaWM 收益由下游 SR 衡量,forecast 只定性 | 抬 world-model 数字 ≠ VLA 收益;**唯一判据=下游 SR 消融(最大缺口)** |
| 官方 LaWM 230M 会碾压 | 实测我们数据 oracle 0.770 未碾压;容量堆 260M +0.00 | oracle→deploy 0.07 gap 大部分是硬信息损失(~0.05),非容量 |

### B2. 解码 / 目标 / 表示
| 曾相信 | 为何错 | 正确结论 |
|---|---|---|
| 解码糊 = 预测不准 | 完美 latent 解码仍糊(锐度 152 vs 992) | 糊在解码器(pooled+L1 求均值);要锐用生成式/检索 |
| pooled 1280 根本有损 | GAN 解码 self-recon 锐到 852≈992 | 单帧级 pooled 够信息;ill-posed 只对**平均/簇心**目标 |
| GAN 解码器(锐)作保真判据 | 锐是幻觉+颜色漂移 | 保真用 **patch-grid 解码**(L1 2.7% vs 6.2%);量化用**再编码 cos**(L1 偏爱模糊) |
| latent cosine 是训练目标 | 与 decode-space loss 近乎正交甚至反相关;grid decode-loss 让特征塌 off-manifold | 按去向选:喂 VLA→特征空间 loss;仅渲染→decode-space。**同头不能两用** |
| 扩散/多假设 subgoal 头捅破 0.874 | oracle 证多模态真实,但 deploy(top-gate)反降;采样只加噪 | frame-only ≈0.874 是结构天花板;**回归≈条件均值最优**,别投生成头(除非有闭环分支信号) |
| 大 backbone / LaWM transformer / patch grid 提 top1 分类 | H7 决定性:真 grid vs 广播 pooled 结果 A≈B(<0.01) | 分类头用 **pooled MLP**;别编码 256-token(130GB 全白费);patch 只对 subgoal/解码有增益 |
| pooled→un-pool→patch-grid 捷径 | un-pool 对真 grid 仅 cos 0.77 | 死路;忠实解码只能 LMWM **直接预测 patch-grid** |

### B3. milestone 目标 / 稳定性
| 曾相信 | 为何错 | 正确结论 |
|---|---|---|
| V1 temporal-next 作目标 | horizon std 0.91、value 可倒退、欠射 ratio 0.42 | 改 **V2 progress-next**(value 单调) → 敢 commit |
| 逐帧重采样 flow 解码 milestone 提示 | milestone 是分段常值,每帧新噪 → 闪烁(违反确定性/连续) | **flow fixed-noise(seed=0)** 退化为确定性连续函数 |
| 检索式 NN 解码(曾按 R1 推荐) | 隐向量分段常值,Voronoi 边界跳帧(违反 R2 连续性) | 检索仅用于"要真实照片";连续预测提示不用 |
| color-aug 强制颜色无关 | 毁颜色相关任务;forward-from-current 无需 | 外观由当前观测带入,不做外观无关硬假设 |

### B4. 跨任务 / 锚 / teacher(本轮 07-07→08)
| 曾相信 | 为何错 | 正确结论 |
|---|---|---|
| **离散 union-CE 词表无限增长 = 立即瓶颈** | ≤~100 类 `Linear(128,99)` 毫无压力;in-dist(所有任务训过)CE 最强 | 词表增长真实但**只在远大规模/开放词表才咬人**;当前 union_ce 最优 |
| 标量进度锚能替代 CE | 标量丢身份/多峰(预测过、实测证实),in-dist+OOD 都输 CE | 进度须配**连续身份**(原型嵌入/簇中心码),别一把梭 |
| **in-dist 联合训练能裁决"离散 vs 连续"** | 闭合完整词表天然偏袒离散 CE | **裁决场是 open-vocab/LOO**:连续锚在 unseen 身份上稳定小胜(coffee .35>.32、xvla .34>.30) |
| 去掉 teacher 直接端到端也行 | teacher=none deploy 掉 **0.07~0.13** | inverse-dynamics 蒸馏有效,teacher 保留 |
| inv 逆向 teacher 是唯一形态 | proto(簇中心码)与 inv 打平,且去 InverseEnc+去 CE 锚、开放词表、轻 5.9M | **teacher=proto 定案**:码=下一 milestone 中心投影,predm 蒸馏,生成器渲染到画布 |

### B5. 工程 / 基础设施坑(踩过就别再踩)
| 坑 | 修法 |
|---|---|
| `kai0 transformers 4.53.2` 载不了 dinov3("Unrecognized") | 纯 torch 门控 DINOv3-H **standalone** 载 safetensors(cos 0.9999);别升级共享 env |
| **现场编码 vs 缓存库 cos 仅 0.86**(OOD 静默退化) | 特征缓存=训练契约,现场编码须**逐比特同路径**(统一 `crave.encoders.encode_pooled`)→ subgoal 0.882→0.909 |
| coffee 视频是 **AV1,cv2 解不了** | 用 **pyav** 解 lerobotv3;xvla 是 HDF5(JPEG bytes)用 cv2.imdecode |
| **exit-144**:`nohup ... & sleep` / `pkill+ssh+nohup` 复合命令被 harness 杀进程组 | 用 `run_in_background=true` 跑**裸命令**;别在前台 `sleep` 等待;别复合 pkill |
| **陈旧日志误导**:重跑被 exit-144 杀掉没覆盖旧 log → 反复读到 pre-fix traceback | 写**新 logfile**;核对时间戳;别信没被覆盖的旧 log |
| **`cut -c1-50` 截断数字**:`frames=21350` 显示成 `frames=215` → 误判"held-out 坏了" | 读数字别 cut 日志行;先 reproduce |
| **误判 OOM**:假设本地 2 大 run OOM,实则本机 **463GB RAM**,是别的 transient | 断言 OOM 前先 `free -g` 看真 RAM |
| **编排器等待而非启动**:orchestrator 等一个预启动的 smoke → 重跑只读到旧失败 | 编排脚本自己**启动**任务,别等已存在的 |
| **gf3 git 代理(ghfast.top)挂** → gf3 pull 不到 → 跑旧代码(`--teacher proto` 无效选项) | 直接 **rsync** 文件绕过 git;或等代理恢复 |
| **val 未 cap** → 每 run 读 kai0 22349–36796 帧 → 加载慢+RAM | 加 `--val_cap`(cap 验证对) |
| **2 个并发 read_imgs 抢盘** 慢 2× | 错峰重解码任务;或先建缓存只传小文件 |
| `train_multitask` 只存 JSON 不存模型 | 加 `--save_ckpt`(存 fwd/predm/inv/anchor + 元数据) |
| `vendor/` gitignore → gf3 缺 LaWAM;`pkill -f 脚本名` 自杀 shell;跨区传 131GB grids 6.4h;H20 **fp32 比 bf16 慢 4×** | vendor rsync 手传;`pgrep\|xargs kill` 按 PID;就地编码只传小文件;训练套 bf16 autocast |

---

## C. 通用铁律(每次实验前默念)
1. **诚实指标**:对**真实未来**评估;报 top-k/NLL/subgoal-cos/deploy,不只 top1;⚠️ 跨 session 引用的数字**必须 fresh 重测**。
2. **单变量**:一次只动一个变量,同 split 同 recipe;绝对值受数据规模影响,**看相对**。
3. **测在正确的场**:要证机制优势,选**能压到该机制**的实验(如离散vs连续 → open-vocab 而非 in-dist)。
4. **探针先行**:kNN 上界判是否触表示天花板;oracle/best-of-N 判多模态是否可分;py-spy 判 hang vs 慢。
5. **保真看像素不看度量**:grid-cos/L1 会被模糊或 off-manifold 游戏;肉眼看解码 + 再编码 cos。
6. **最终判据是下游 SR**:world-model 内部指标再高 ≠ 对 VLA 有用;唯一裁决是接策略测 action-MAE/SR。
