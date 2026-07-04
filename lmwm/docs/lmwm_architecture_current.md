# LMWM 当前架构(2026-07-04 优化后定稿)

面向 LMWAM(接入 kai0 π0.5 的 world-model-augmented VLA)的**冻结** LMWM。全部组件 stop-grad,只作为子目标/条件源。

## 组件

| 组件 | 实现 | 输出 | 指标 |
|---|---|---|---|
| **冻结编码器** | DINOv3-H (ViT-H+/16 lvd1689m) | 256²→ 16×16×1280 patch grid | 纯torch standalone loader,env 无关,cosine 0.9999 vs transformers |
| **milestone 结构** | recurrence 原型 (kai0base_dinov3h);逐帧 argmax(Fn·protoᵀ) | 每帧 stage id + 段 medoid | ⚠️ 细粒度(单 episode ~70-90 段),标签非单调 |
| **子目标预测器 A:milestone+1** | forward-from-current:predm(g_t)→code→fwd(g_t,code) | û_T^m: 16×16×1280 | deploy grid-cos **0.725**(cd128);oracle 0.79 |
| **子目标预测器 B:near-future** | 同上,horizon h3(~1s) | û_T^nf: 16×16×1280 | deploy **0.705**(cd64) |
| **转移码 ẑ(潜在动作)** | inverse(g_t,g_fut)→code(teacher) / predm(g_t)(deploy) | 64/128 维 | 蒸馏进 VLA action expert |
| **patch 解码器(仅可视化)** | make_decoder("big")+GDL 锐度损失 | grid→128² RGB | L1 **0.0206**,sharp **574**(real ~1045);**不喂 VLA** |

## 关键设计判断
- **forward-from-current 是核心机制**:条件在当前 grid + 预测转移码上,比旧无条件 CNN(0.653)全面高 0.05–0.07。oracle→deploy 的 gap(0.79→0.72)是下个杠杆 = 多模态/flow 码头(替确定性回归)。
- **两路子目标喂 VLA**:milestone+1(语义大目标)+ near-future(~1s 动力学),对齐 LaWAM 的 û_T 双角色。
- **解码器仅用于人看**,VLA 直接吃 grid 特征;解码器锐度靠 GDL(非生成式,不幻觉),选 big+GDL(忠实)或 xl+GDL(更锐)。

## 跨数据集证据(kai0 训练 → vis_base 迁移,零微调)
| 数据 | mean pred cos | 判读 |
|---|---|---|
| kai0_base(in-dist) | 0.725 | — |
| vis_base v4/2026-04-23 ep8 | 0.663 | 构型预测吻合 |
| vis_base v4/2026-06-18 ep0 | 0.676 | 构型吻合;**解码器颜色偏 kai0(橙→青)= 外观域差,非预测失败**;预测/真实一致故 cos 有效 |

**结论**:预测器在**特征空间**跨本体/相机/衣物颜色迁移成立(cos ~0.67,掉 ~0.06);像素外观有 kai0 域偏(解码器侧,非预测器)。支撑 [[project_crave_milestone_value]] 跨数据集论点。

## 喂给 LMWAM(下一步 M2)
```
obs_grid (256×1280, frozen)          → π0.5 prefix (替/并 SigLIP)
û_T^m  milestone+1 subgoal (cd128)   → prefix (新视觉流, KV-cache 一次)
û_T^nf near-future subgoal (h3)       → prefix (第二子目标流)
ẑ      转移码 (64)                    → suffix (潜在动作, 蒸馏 ‖ẑ−z‖²)
```
全 stop-grad + KI 梯度隔离(world model 不被污染)。

## 交付件(gf3)
- 预测器:`lmwm/outputs/subgoal_opt/{milestone_cd128,nearfuture_h3_cd64}.pt`
- 解码器:`lmwm/checkpoints/patch_decoder/patch_dec_{big,xl}_gdl0.5.pt`
- 编码器:`crave/encoders/_dino_vit_standalone.py` + hf_dino fallback
- 渲染:`lmwm/scripts/render_milestone_predict_video.py`(支持 `--raw_video` 跨数据集)
