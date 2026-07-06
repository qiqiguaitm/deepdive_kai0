# v2 预测器消融收敛(2026-07)

控制变量消融(每 config 只改一处,同口径)。gf3 8卡 RAM-OOM 后改本地 2卡 fp16 跑通。

## 判据全表
| config | deploy | id_t1 | id_t3 | lag_ratio | fwd% | neg% | smooth | 结论 |
|---|---|---|---|---|---|---|---|---|
| base | 0.721 | 0.208 | 0.474 | 0.75 | 46% | 18% | 0.943 | 参照 |
| concat | 0.714 | 0.230 | 0.470 | 0.52 | 35% | 6% | 0.905 | AdaLN关→lag/smooth掉 |
| nolift | 0.701 | 0.241 | 0.481 | 0.57 | 40% | 8% | 0.873 | lift关→lag/smooth掉 |
| noteacher | 0.645 | 0.178 | 0.401 | — | — | — | — | teacher关→deploy崩 |
| code256 | 0.703 | 0.192 | 0.444 | — | — | — | — | 更差 |
| ccenter | 0.718 | 0.260 | 0.509 | 0.46 | 33% | 5% | 0.914 | 身份↑ 负lag↓ 但保守 |
| tall (allframes) | (val混淆) | — | — | 1.67 | 71% | 13% | 0.953 | 最平滑+最往前 但过射 |
| tc (all+center) | (val混淆) | — | — | 1.52 | 69% | 12% | 0.913 | 过射,未平衡 |

## 收敛结论
**核心锁死(控制变量验证)**:inverse-teacher + AdaLN + lift + code128。四项各自关掉都掉指标。

**两个旋钮 = 相反 trade-off,最优点由 SR 定**:
- 簇中心 center:身份+0.05、负lag 18%→5%,但保守(欠射 0.46)。
- allframes:最平滑(0.953)+最往前(71%),但过射(1.5–1.7,越过 milestone+1)。
- 二者方向相反,intrinsic 指标此消彼长(身份 vs horizon);tc 叠加未平衡。→ **只有下游 SR 能裁决 target/teacher 的最优点。**

**allframes val 混淆说明**:allframes 把段内早期帧(离目标远)纳入 → 自身 val 变难 → deploy/身份"掉"是 val 更难,非模型更差;公共 lag 口径下它其实最平滑+最往前。

脚本:`train_ablation.py`(--target_mode/--teacher/--fwd_arch/--lift_w/--code_dim)、`measure_twomodel_v2_lag.py`(lag+平滑,支持 concat)。
