# Realtime-VLA — V1 Triton 推理优化实施日志 (2026-05-20)

> 本文档是 `realtime_vla_optimization_analysis.md` (1687 行, 已拆) 拆分后的"V1 已实施日志层"。记录 V1 Triton 推理优化的实施过程 (P50 76→32ms)。
>
> **同 series 文档**: `strategy.md` (战略) / `roadmap.md` (5 阶段路线) / `layer_b_plan.md` (下一阶段 Layer B 未来 plan)

---

## 6. V1 Triton 推理优化实施日志 (2026-05-20)

> **路径决策**: 阶段 3.4.2 原计划"自写 `serve_policy_pytorch.py` + 6 个 Triton kernel" (1-2 周 AI 辅助). 实施时改走**直接复用 V1 (arXiv:2510.26742) `pi05_infer.py` (22+ 个 Triton kernel)** + 写 deepdive_kai0 sentencepiece adapter + JAX→pickle 转换. 工程量从"自写 Triton kernel"降到"集成 + 数值对齐 + 5090 重 tune", 实际 ~3 天.
> **代码位置**: `optimize/v1_triton/` (生产 `pi05_infer_tuned.py`, sweep 脚本 `tune_5090_*.py`, benchmark `benchmark_kai0_v1.py`).
> **最终结果**: **P50 = 32.05 ms** (8.00× vs eager baseline, 比 §3.4.2 max-autotune 43.5ms 再快 26%).

### 6.1 总进度表 (Step 0-9)

真 ckpt: `task_a_mix_b6000_p1200_mixed_1_step49999`, 5090 sm_120 bf16 3-view chunk_size=50, 100-iter benchmark.

| Step | 实施 | Mean (ms) | vs 上一步 | vs eager | 文件 |
|:---:|---|---:|---:|---:|---|
| 0 | PyTorch eager (baseline) | 256.5 | — | 1.00× | `optimize/benchmark_pi05_inference.py` |
| 1 | + torch.compile(default) | 110.3 | -57.0% | 2.33× | (backend B) |
| 2 | + 手动 CUDA Graph | 60.7 | -45.0% | 4.23× | (backend C) |
| 3 | + compile(reduce-overhead) | 48.3 | -20.4% | 5.31× | (backend D) |
| 4 | + compile(max-autotune) (deepdive_kai0 当前默认) | 41.0 | -15.1% | 6.26× | (backend E) |
| 5 | V1 Triton 直接复用 (4090-tuned BLOCK_SIZE) | 35.4 | -13.7% | 7.25× | `pi05_infer.py` (upstream V1) |
| **6** | **+ 5090 BLOCK_SIZE autotune (3 hot decoder kernels)** | **32.3** | **-8.8%** ⭐ | **7.94×** | `pi05_infer_tuned.py` + `tune_5090_all.py` |
| 7 | + Triton pipelining sweep (num_warps × num_stages) | 32.3 | -0.2% 噪声 | 7.95× | `tune_5090_pipelining.py` |
| 10 | + Encoder FFN gate+up BLOCK_SIZE sweep | 32.34 | +0.13% 噪声 | 7.93× | `tune_5090_step10_encoder.py` |
| 8 | + Decoder QKV+RoPE BLOCK_SIZE sweep | 32.25 | -0.4% (-0.13 ms) | 7.95× | `tune_5090_step8_qkv.py` |
| **9** | **+ Decoder Attn QK matmul BLOCK_SIZE sweep** | **32.05** | **-0.6%** (-0.20 ms) | **8.00×** | `tune_5090_step9_attn.py` |

**生产版**: `optimize/v1_triton/pi05_infer_tuned.py` (含 Step 6 + 8 + 9 五个 kernel 全 tune), **P50 = 32.05 ms**.

### 6.2 Sweep 方法论

"Sweep" = **参数空间网格搜索**. 给一个 kernel 的可调参数 (`BLOCK_SIZE_N/M/K` / `num_warps` / `num_stages`) 列 N 个候选, 逐个 benchmark, 比较找最优.

#### 实操步骤

1. **列候选**: 写 Python list 含 10-15 组 BLOCK_SIZE
   ```python
   GATE_FFN_CANDIDATES = [
       (128, 64, 32),    # V1 default (4090)
       (32,  64, 128),   # ← 5090 最优
       (256, 64, 32),    # 退化 -41%
       (256, 128, 64),   # OOM
       # ... 共 10 个
   ]
   ```
2. **每候选独立跑**: monkey-patch `pi05_infer.transformer_decoder` → rebuild `Pi05Inference` (重 capture CUDA Graph) → warm-up 10 + 测 50 iter → 还原 + `empty_cache`
3. **排序选最优**, 应用到生产 `pi05_infer_tuned.py`

#### Greedy multi-kernel sweep

3 个 kernel 同时调时用 greedy (固定其他 default, 一个一个调):
```
[Sweep gate]  fixed (ffn=default, attno=default), 10 候选 → best = (32, 64, 128)
[Sweep ffn]   fixed (gate=best, attno=default),   11 候选 → best = (16, 32, 512)
[Sweep attno] fixed (gate=best, ffn=best),        11 候选 → best = (16, 32, 256)
```
Greedy 32 候选 ≈ 30 min vs 全局 1331 候选 ≈ 22 hr.

#### 我们使用的 sweep 脚本

| 脚本 | 调优对象 | 候选数 |
|---|---|---|
| `tune_5090.py` | matmul_small_gate 单 kernel | 10 |
| `tune_5090_all.py` | 3 个 hot kernels greedy | 32 |
| `tune_5090_pipelining.py` | num_warps × num_stages | 24 |
| `tune_5090_step8_qkv.py` | matmul_rope_qkv | 10 |
| `tune_5090_step9_attn.py` | matmul_abT_scale | 9 |
| `tune_5090_step10_encoder.py` | encoder FFN gate+up | 11 |

#### 与 Triton 内置 `@triton.autotune` 区别

| 维度 | 手动 sweep | `@triton.autotune` |
|---|---|---|
| 候选定义 | Python list, 任意修改 | 写在 kernel 装饰器上 |
| 选择时机 | Build phase, 用户主动跑 | 第一次 call 时自动 |
| 可见性 | 每候选 mean/P50 全可见 | 黑盒 |
| 与 CUDA Graph 兼容 | ✅ 完美 | ⚠️ 可能与 graph capture 冲突 |

### 6.3 PyTorch baseline 路径 (Step 0-4)

> 详细 5-backend 实测见 §3.4.2 (核心结论 + 数据表 + 分量分解 + 抖动分析). 此处仅简述每 Step 边际收益.

| Step | 关键改动 | 为什么有效 |
|:---:|---|---|
| 0 (eager) | PyTorch nn.Module 直跑 | (baseline) Python dispatcher + 1000+ kernel 各自 launch |
| 1 (compile-default) | `torch.compile(mode="default")` | TorchDynamo trace → FX graph → Inductor fusion (1000+ kernel → ~200-300), 跳过 Python dispatcher. **-57%** |
| 2 (CUDA Graph manual) | 手动 `torch.cuda.CUDAGraph()` capture-replay | 1000+ cuLaunchKernel → 1 个 cuLaunchGraph. **-45%** |
| 3 (reduce-overhead) | `mode="reduce-overhead"` (Inductor + 自动 graph) | 二合一: fusion + 自动 CUDA Graph. **-20%** |
| 4 (max-autotune) | `mode="max-autotune"` | Inductor 对每个 GEMM 跑 21 个 Triton 模板, 选最快. **-15%, deepdive_kai0 当前默认** |

**Step 0 → 4 累积 6.26× 加速 (256.5 → 41.0 ms)**. 这是 PyTorch 工具链上限 — Step F/G/H/I/J/K 测试均失败 (详 `optimize/results/FINAL_30ms_attempt_summary.md`).

### 6.4 V1 Triton 路径 (Step 5-9)

#### Step 5: V1 Triton 直接复用 (35.4 ms)

复制 V1 `pi05_infer.py` (22+ 手写 Triton kernel + 手动 CUDA Graph + 预分配 buffer) 进 `optimize/v1_triton/`, 写 `convert_kai0_to_v1.py` (sentencepiece adapter, JAX orbax → 6.7 GB pickle), 跑 `benchmark_kai0_v1.py`.

**为什么有效 (-14%)**: 完全 bypass PyTorch + Inductor, "model = dict of weight tensors + 手写 kernel" 极致路径. Triton kernel `tl.constexpr` BLOCK_SIZE 编译器针对 5090 sm_120 自动生成 PTX, 无需修改即可在 5090 跑.

**数值对齐**: rel error 1.42%, per-dim MAE < 0.01 rad (部署级). 修复要点: 移除 V1 test.py 的 `+ ori_state` (deepdive_kai0 训练绝对 action 不是 delta).

**V1 vs PyTorch E 架构对比**:

| 维度 | V1 Triton | PyTorch E (max-autotune) |
|---|---|---|
| 模型构造 | dict of weight tensors | nn.Module + transformers |
| QKV | 手动 concat 到 1 大矩阵 | 3 个独立 nn.Linear |
| Attention | 手写 Triton softmax+matmul | SDPA (cuDNN) |
| GEMM kernel 数 | 22 个 shape-specific 手写 | Inductor 模板 (21 候选 sweep) |
| Memory | 全预分配, 零 cudaMalloc | PyTorch caching allocator |

#### Step 6: 5090 BLOCK_SIZE autotune (32.3 ms) ⭐ 决定性单步

V1 BLOCK_SIZE 是 4090-tuned. 写 `tune_5090_all.py` 给 3 个 decoder hot kernel (180× per forward) sweep, greedy 选最优:

| Kernel | shape | V1 default (4090) | **5090 tuned** | 单独贡献 |
|---|---|---|---|---:|
| `matmul_small_gate` (FFN gate+up) | 1024→4096 | (128, 64, 32) | **(32, 64, 128)** | **-7.8%** ⭐ |
| `matmul_small_res_gate` (FFN down) | 4096→1024 | (16, 32, 256) | (16, 32, 512) | -0.1% |
| `matmul_small_res_gate` (Attn O) | 2048→1024 | (32, 32, 128) | (16, 32, 256) | -0.9% |

**反直觉发现 — "小 N, 大 K" 在 5090 上更优**:

| Rank | BLOCK (N, M, K) | Mean (ms) | 解读 |
|:---:|---|---:|---|
| **1** | **(32, 64, 128)** | **32.6** | 小 N → 2 grid × 64 = 128 program ≈ 5090 SM (170), 利用率高 |
| 4 | (128, 64, 32) | 35.4 | V1 默认 — 仅 64 program, **5090 浪费 100+ SMs** |
| 9 | (256, 64, 32) | 49.9 | grid=1, **几乎所有 SM 闲置** |
| - | (256, 128, 64) | OOM | shared memory 131KB > 5090 SM 上限 101KB |

**关键洞察**: 5090 SM 170 (vs 4090 128), 真正起作用是 **grid 总数 ≈ SM 数**, 不是 BLOCK 大小本身. 大 K (128) 利用 5090 L2 cache (96MB vs 4090 64MB).

#### Step 7: Triton pipelining sweep (32.3 ms, 噪声内)

固定 Step 6 BLOCK_SIZE, 试 24 个 `num_warps × num_stages` 组合. 最优 (gate: warps=8/stages=3, FFN-down: warps=4/stages=3, Attn-O: warps=4/stages=4) → 32.26 ms, 与 Step 6 (32.33 ms) 差 0.07ms 噪声内.

**为什么无效**: pi05 decoder GEMM **memory-bound** (3.6B × 2B / 1.7 TB/s = 4.2 ms floor), pipelining 主要优化 compute-bound 间隙.

#### Step 10: Encoder FFN gate+up sweep (32.34 ms, 噪声内)

Encoder `rms_matmul_n_2048_16384_gate` (FFN gate+up 2048→16384, 18× per inference, 最大单 GEMM). 11 候选全在 32.34-32.46 ms 窗口 (0.4% 内).

**为什么无效**: Encoder seq_len=775, BLOCK_N=128 时 grid=1792 programs ≫ 5090 SM 170, **已 grid-saturated**.

#### Step 8: Decoder QKV+RoPE sweep (32.25 ms, -0.4%)

`matmul_rope_qkv` (1024→2560, 180×). 10 候选 sweep: V1 default (64,32,64) → 32.38 ms; best (64, 32, 128) → 32.25 ms (-0.13 ms).

**为什么微弱**: QKV 比 FFN gate+up 小 (2560 vs 4096), decoder QKV 用 1D persistent grid `(128,)` 已合理饱和; 大 K (128) 仍有小收益.

#### Step 9: Decoder Attention QK matmul sweep (32.05 ms, -0.6%)

`matmul_abT_scale` (Q × K^T, 180×). 9 候选 sweep: V1 default (32,32,64) → 32.19 ms; best (32, 64, 64) → 32.05 ms (-0.14 ms).

**为什么微弱**: total_queries=400 (50 token × 8 head), total_keys=825. BLOCK_N=64 让 grid 沿 keys 维度更细 (cdiv(825,64)=13), 与 5090 SM 数更匹配.

**Step 8 + 9 累积 -0.28 ms** vs Step 6.

### 6.5 累积分析与硬件下限

#### 最终生产 P50 = 32.05 ms

`pi05_infer_tuned.Pi05InferenceTuned` 应用 5 个 kernel BLOCK_SIZE tune:

| Kernel (decoder, 180×) | V1 default (4090) | 5090 tuned |
|---|---|---|
| matmul_small_gate (FFN g+u) | (128, 64, 32) | **(32, 64, 128)** |
| matmul_small_res_gate (FFN down) | (16, 32, 256) | (16, 32, 512) |
| matmul_small_res_gate (Attn O) | (32, 32, 128) | (16, 32, 256) |
| matmul_rope_qkv (QKV+RoPE) | (64, 32, 64) | (64, 32, 128) |
| matmul_abT_scale (Attn QK) | (32, 32, 64) | (32, 64, 64) |

Mean=32.33ms (Std 1.19), **P50=32.05ms**, P95=32.97ms.

#### 距硬件下限分析

**5090 memory-bound 理论下限**: 3.6B × 2 bytes / 1.7 TB/s = **4.2 ms** (理论 8×)

**实际 32.05 ms 拆分**:
- vision encoder (1×) + transformer encoder (18 layer 1×) + transformer decoder (10 step × 18 layer = 180 attention block)
- 每 decoder block ≈ 32 / 180 ≈ 0.18 ms (含 norm + QKV + SDPA + O + norm + FFN), **已接近硬件极限**
- 剩余 28 ms 主要去向: attention softmax + QK + AV 非纯 matmul (难达 memory peak bandwidth); 22 个 Triton kernel 串行 dispatch; layer 间无法完全 hide 的同步开销

#### 距 30 ms 目标分析

当前 32.05 ms, **距 30 ms 差 -2.05 ms (-6.4%)**.

V1 + 5090 BLOCK_SIZE tune 路径已榨干: 单步收益从 Step 6 的 -7.8% 衰减到 Step 7-10 的 < 1%, 累积已逼近 V1+autotune 路径在 5090 上的极限. **触达 30ms 必须结构性优化** (kernel 重写或 fusion), 不是参数 sweep.

### 6.6 待实施 (Step 11+, 结构性优化)

| Step | 实施 | 预期收益 | 工程量 | 优先级 | 风险 |
|:---:|---|---|---|:---:|---|
| **11** | **Kernel fusion**: 合并 `adarms_norm_style_proj` + 后续 matmul, 减 360+ kernel 边界 sync | **-1-3%** | **3-5 天** | 中 | 中 (需写新 Triton kernel + 数值验证) |
| **12** | **Stream overlap**: vision encoder ‖ decoder denoise 并发 (V1 §4.4 提到 3.7%) | **-1-2%** | 2-3 天 | 低 (decoder 串行依赖, overlap 空间小) | 中 |
| **13** | **wgmma 重写主要 GEMM**: Blackwell 原生 `wgmma.mma_async.m64n*` 替代 `mma.sync` | **-5-10%** | **5-10 天** | 高回报 | 高 (Triton 3.x sm_120 实际生成 wgmma 还是 mma 不确定; OOM shared mem 风险) |
| 14 | **FlashAttention 3**: 替换 attention path | 不确定 (-1 ~ +3%) | 2-3 天 | 低 | 高 (短序列 seq=50 收益不明) |
| 15 | **共享 KV cache cross-step**: 10 步 denoise 部分 encoder K/V 可缓存复用 | 不确定 | 1-2 天 | 中 | 低 (V1 已部分实现) |

#### 推荐路径 (按 ROI / 单位工时排序)

1. **Step 11 (Kernel fusion)** ⭐ — 中等工程量, 收益较确定 (-1-3%), 难度可控
2. Step 13 (wgmma) — 收益最大但风险最高, 适合"豁出去"投入
3. Step 14/15 — 收益不确定, 不优先

#### 结论 (V1 路径上限)

pi05 5090 在不改架构 / 不重训 / 不量化约束下:
- **V1 + 5 kernel autotune = P50 32.05 ms (8.00× vs eager)** ← 当前生产
- **再压到 30 ms 需 3-5 天 kernel fusion 或 5-10 天 wgmma 重写**
- 硬件理论下限 4.2 ms (memory-bound), 当前距下限 8× — 长期目标可参考

---

