# 教训:静默"喂黑图"的 dataloader 让所有 X-VLA 模型训成 vision-blind

> 日期:2026-06-23 · 类型:数据管线 bug 复盘 / 工程教训
> 关联:[`future_plans/plans/xvla_proprio_shortcut_openloop_fix.md`](../future_plans/plans/xvla_proprio_shortcut_openloop_fix.md) (§0 根因 banner) · [`experiments/xvla_e0_v1_official_results.md`](experiments/xvla_e0_v1_official_results.md) (作废) · memory `reference_xvla_vision_blind_openloop`
> 涉及代码:`train_scripts/xvla/data/multi_domain_dataset.py` (`LeRobotEE6DDataset`)

---

## 0. 一句话

**一个静默的 `except: return torch.zeros` + 一个视频目录命名不匹配 (全名 vs 短名),让 v1 数据集的每一帧图像都变成纯黑;E0/E1 等所有用该 loader 训的 X-VLA 模型全程"没见过图",训成 vision-blind。我们却花了两周把它误诊为 `use_proprio` 早融合捷径,设计了 E0/E1/E1_v1_official 一整条"断 proprio / 换真实 action"的修复实验 —— 全部在黑图上跑,从未真正测过任何假说。**

---

## 1. bug 机制

`LeRobotEE6DDataset` 解码相机帧:

```python
# _video_path 旧版 (bug):
def _video_path(self, ep_idx, cam_key):
    return self.root / self.video_tpl.format(video_key=cam_key, ...)
    # cam_key = "observation.images.top_head" (全名 feature key)
    # 但磁盘上的视频目录是短名: videos/chunk-000/top_head/...
    # → 拼出 videos/chunk-000/observation.images.top_head/...  →  不存在

# __getitem__ 解码 (bug 被静默吞掉):
try:
    frame = decode_frame(self._video_path(ep_idx, cam_key), f_idx)  # av.open FileNotFound
    ...
except Exception as e:
    img_dict[...] = torch.zeros((3, size, size))   # ← 静默返回黑图, 不报错不告警
```

两个独立缺陷叠加:
1. **路径错**:`video_key` 用了全名 feature key,而数据集按短名存目录。视频文件其实存在且能正常解码 (480×640, mean≈104),只是路径拼错。
2. **静默兜底**:解码失败 `except` 直接返回全零张量,不抛错、不打印、不计数。训练照常进行,loss 照常下降 (黑图 + 准静态动作 → 模型轻松学个常量/本体映射),**没有任何信号提示"图是黑的"**。

**作用域**:只命中**短名视频目录**的 LeRobot 数据集 (v1 `A_v1_noRelabel_ee6d`)。x3c `A_new_smooth_800_xvla` 用**全名**目录 → 旧代码恰好拼对 → 见真实图 (其 vision-blind 是另一回事:action≡state 捷径)。官方 soft_fold 走 `XVLAHdf5Dataset` 从 hdf5 直接读图,不经此路径 → 一直正常 (`d_img`=12.87mm)。

---

## 2. 为什么瞒了这么久 (误诊链)

- **离线视觉消融也用同一个 buggy loader** → 测出 `d_img≈0` 被解读为"模型不读视觉",而真相是"喂进去的两张图都是黑的,当然换图无变化"。harness 自验只比过 "真图 vs 黑图" (差异大→以为图被消费),**没比过 "真图 vs 另一张真图"**,所以没暴露"真图其实也是黑的"。
- **官方对照 (`d_img`=12.87) 走的是 hdf5 loader**,不经过 bug → 看似"同架构官方能读、我们不能",强化了"架构/数据语义有问题"的错误方向。
- **loss / MAE 完全测不出**:黑图训练 loss 正常收敛 (~5.5),MAE 也正常。只有视觉消融能测,但消融本身被同一 bug 污染。
- 于是合理但错误地推断出"`action≡state` copycat + `use_proprio` 早融合捷径"根因,并投入 E0/E1/E1_v1_official 三个 50k 训练去验证 —— 全在黑图上。

## 3. 决定性证据 (如何最终查出)

逐级 probe 一帧图像信号 (`stage0 原始图 → stage1 Florence2 → stage2 投影 → stage3 动作`):
- `stage0 原始图 reldiff = 0.0000` —— 两张完全不同的帧喂进模型竟字节一致 → 直接查数据集像素:**所有帧所有相机 min=max=mean=std=0.000 (纯黑)**。
- 视频文件单独拿出来能正常解码 → 锁定是 `_video_path` 路径拼错 + `except` 静默黑图。
- 修好 loader 后真实图重测:
  - E1 (`use_proprio=False`) `d_img`=0.01mm (proprio 关掉、视觉是唯一信号却零响应 → 坐实"训练时没见过图")。
  - **E0 配方修复重训 (仅 2k 步)** `d_img`/`d_state` = **1.763** (黑图版 50k=0.000;官方=0.220;健康线≳0.5) → loader 一修,2k 步内视觉影响就超过本体。

---

## 4. 教训 (可推广,按重要性)

1. **数据管线绝不静默兜底**。`except: return zeros` 是元凶 —— 它把"文件找不到"这种致命配置错降级成"无声的退化输入"。任何 IO/解码失败要么**抛错**,要么**大声告警 + 计数**,绝不能悄悄返回零/默认值让训练继续。
2. **退化输入要有硬门禁**。训练 (和评测) 启动时应断言"首个 batch 的图像 std > 0 / 非全零 / 跨帧有差异"。一行 assert 能省两周。
3. **诊断模型行为前,先证明输入是对的**。所有"模型不读视觉/不学 X"的结论,前提是"X 真的喂进去了"。消融实验要包含**正例对照** (真图 vs 另一张真图,而非只比真图 vs 黑图),否则会把"输入退化"误判成"模型缺陷"。
4. **同结论的多条证据若共用一个上游组件,不算独立验证**。离线消融 + 训练都用同一个 buggy loader → 它们的"一致"只反映共同的 bug,不构成交叉验证。找一条**不经过可疑组件**的独立路径 (这里:官方 hdf5 loader / 直接读像素)。
5. **loss/MAE 收敛 ≠ 训练健康**。黑图照样收敛。窄分布 + 准静态任务下,模型能靠常量/本体把 loss 压低,完全不需要视觉。判 VLA 是否读视觉只能靠视觉消融,且消融本身必须可信。
6. **路径模板的 `{key}` 要确认是"目录用的名字"**。LeRobot 视频目录有的用全名 feature key、有的用短相机名;模板填值前要对齐磁盘实际布局 (修复:先试全名、不存在退短名,兼容两种)。

## 5. 已处置 / Action Items

- ✅ `_video_path` 改为先试全名、不存在退短名 (兼容两种布局);解码失败改 **大声 warn + 计数** (`_decode_fail_count`)。commit `3b46252`。
- ✅ E0/E1 旧结论全部标作废 (plan §0 / E0 results / history README)。
- ✅ E0 配方用修好的 loader 50k 重训 (dev 队列 `t-20260623145309-8clks`, OUT `xvla_e0_v1_official_fixedcam`)。
- ⏳ (建议) 在 `xvla_train.py` 启动处加首-batch 图像非黑断言,作为永久门禁。
- ⏳ (建议) 视觉消融脚本输出里固定打印 `stage0 原始图 reldiff`,reldiff≈0 直接红字报"输入退化,结果不可信"。
