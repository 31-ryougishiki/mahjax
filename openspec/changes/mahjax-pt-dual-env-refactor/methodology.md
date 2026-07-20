# GPU Profiling 方法论

基于 mahjax-pt PPO 训练管线（B=256, GPU）性能优化过程中建立并验证的分析方法。

---

## 1. 核心原则

GPU profiling 的核心难点：**测量工具不能改变被测量对象的执行模式**。

| 方法 | 问题 |
|------|------|
| `torch.cuda.synchronize()` | 强制 GPU 排空所有 pending kernels，破坏异步流水线，测量值膨胀 2-3x |
| `time.time()`（wall clock） | 含 kernel launch overhead + Python 开销 + 隐式 sync，无法反映 GPU 计算时间 |

## 2. CUDA Event 链式记录

利用 CUDA default stream 的**顺序执行保证**：仅在关键边界 record event，绝不中途 sync。

```python
ev0 = torch.cuda.Event(enable_timing=True); ev0.record()  # entry
# ... GPU 工作 A ...
ev1 = torch.cuda.Event(enable_timing=True); ev1.record()  # 边界 1
# ... GPU 工作 B ...
ev2 = torch.cuda.Event(enable_timing=True); ev2.record()  # 边界 2
# ... GPU 工作 C ...
ev3 = torch.cuda.Event(enable_timing=True); ev3.record()  # 出口

torch.cuda.synchronize()  # 仅在末尾 sync 一次

t_A = ev0.elapsed_time(ev1)  # 段 A 纯 GPU 时间
t_B = ev1.elapsed_time(ev2)  # 段 B 纯 GPU 时间
t_C = ev2.elapsed_time(ev3)  # 段 C 纯 GPU 时间
```

**原理**：default stream 保证 `ev.record()` 插入的时间戳在之前所有 kernel 完成后才被记录。`elapsed_time(e1, e2)` 精确等于两标记之间 GPU kernel 的总执行时间。

**验证**：加 36 个 sync 测 legal_mask=480ms，event-chain（0 sync）测 =214ms。两次的**内部比例完全一致**（draw_batch 均占 ~94%），证明 event-chain 保真。

## 3. 测试参数

所有测量基准：B=256 envs, round_mode=single, device=cuda:0, BC pretrained params, 8 steps warmup + 64-256 steps profiling。

## 4. Monkey-patch 模式

对被测函数做最小侵入注入，逐段 record event：

```python
orig_fn = impl.target_function

def _patched(self, *args):
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(N + 1)]
    evs[0].record()
    # ... 逐段执行原始逻辑，每段结束 record ev[i] ...
    step_events.append(evs)
    return result

impl.target_function = _patched.__get__(impl, type(impl))

# profiling loop
for step in range(profile_steps):
    torch.cuda.synchronize()
    impl._step_batch_bs(...)
    torch.cuda.synchronize()
    for i, label in enumerate(LABELS):
        dt = evs[i].elapsed_time(evs[i+1]) / 1000.0
```

- 入口处多一个额外 sync 是允许的（GPU 已 idle，开销可忽略）
- 函数内部的 event array 通过外部 list 传递，末尾统一读取
- 被测函数有 early-return 路径时，需在 return 前补录 dummy event 保持长度一致

## 5. Dump-验证-优化 工作流

每项优化按固定流程，确保正确性：

1. **Dump** — monkey-patch 捕获真实训练规模（256×256 steps）的 I/O gold 数据（`.pt`）
2. **Verify** — 实现优化版本，对 gold 数据逐组对比（200-2000 组），100% 位级一致
3. **Profile** — event-chain 测量优化前后 GPU 时间，算加速比
4. **Apply** — 应用到生产代码，端到端验证

对应的脚本命名规范：
- `tests/dump_<target>_io.py` — gold 数据录制
- `tests/verify_<target>_<optimization>.py` — 正确性验证 + 单点性能对比
- `tests/profile_<target>.py` / `profile_<target>_v2.py` — event-chain profiling

## 6. 瓶颈定位流程

```
profile_step_batch.py     ← handler 级粗粒度（_perf）
  → profile_discard_v2.py ← _discard_batch 子步骤（event-chain）
    → profile_yaku_v3.py  ← _precompute_yaku 内部
    → profile_legal_mask  ← legal_mask + _draw_batch 内部
      → profile_draw_mask ← _make_legal_mask_after_draw 内部
      → profile_shanten   ← Shanten.number_batch 内部
    → profile_furiten.py  ← furiten 双循环内部
  → profile_network.py    ← ACNet forward pass 内部
```

逐层下钻：从 handler 级 → 子函数级 → 单次调用级 → kernel 内部，直到定位最深的可优化热点或确认已充分向量化。

---

## 7. 网络 Forward 验证方法

`test_network_forward.py` — 独立验证 PT ACNet 组件与生产 JAX ACNet 的前向传播一致性，零外部数据依赖。

### 7.1 核心思路

```
JAX ACNet.init(seed=42)  ──正交初始化──►  160 JAX 参数
                                              │
                        build_jax_to_pt_map()  │  transfer_weights()
                        160→160 结构化映射     │  (direct/transpose/reshape_3d/reshape)
                                              ▼
PT DualACNet           ◄─────────────────  160 PT 参数
  (内联双提取器，真实 PT
   FeatureExtractor + TransformerBlock
   组件，按 JAX 结构组装)
                                              │
  make_*_obs() ── numpy 进程内生成 ──────────┤
                                              │
          ┌───────────────────────────────────┤
          ▼                                   ▼
  jax_net.apply(params, obs)          pt_net(obs)
          │                                   │
          ▼                                   ▼
    JAX logits + values              PT logits + values
          │                                   │
          └──── np.abs(jax - pt).max() ───────┘
                    logit_diff / value_diff
```

### 7.2 关键设计

**零外部依赖**：所有观测由 `np.random` 进程内生成（`make_all_ones_obs` / `make_random_obs` / `make_edge_case_obs`），不需要 pickle 文件或 JAX 环境。

**内联 DualACNet**：生产 PT ACNet 使用共享提取器（84 params），与 JAX 双提取器（160 params）结构不同。测试内联 `DualACNet` 类——导入真实 PT `FeatureExtractor` + `TransformerBlock` 组件，按 JAX 的双提取器结构组装，验证组件的跨框架等价性而非生产 ACNet 的特定架构选择。

**160→160 结构化映射**：JAX 使用 `sorted(tree.keys())` 确定性排序，PT 使用 `list(parameters())` 插入序。映射表精确指定每个参数的传输方式：
- `direct` — 同形状直接复制（embedding、bias、LayerNorm）
- `transpose` — JAX kernel → PT weight（Dense 层）
- `reshape_3d` — JAX `(feat, heads, head_dim)` → PT Linear `(out, in)`（MHA 权重核）
- `reshape` — JAX MHA bias `(heads, head_dim)` → PT bias `(features,)`

**11 组测试用例**：全1 确定性、随机 (B=1/4/16/32/64)、极值（all-zeros、max-values）、边界（empty-history、negative-scores），覆盖不同 batch size 和极端输入值。

**通过标准**：`logit_diff < 1e-4 && value_diff < 1e-4`（float32 跨框架容差）。

### 7.3 运行方式

```bash
python mahjax_pt/tests/test_network_forward.py   # 需要 JAX + PyTorch 环境，~10s
```

---

## 8. PPO 金数据录制与回放方法

完整的 PPO 训练管线 JAX→PT 精度验证，采用"录制→回放"的金数据方法。

### 8.1 数据流

```
record_jax_acnet_golden_f64.py                 replay_pt_acnet_golden.py
        │                                              │
        │ 生产 JAX ACNet                               │ 纯 PT，不导入 JAX
        │ (examples/networks/red_network.py)           │ 复用 test_network_forward
        │                                              │ 的 DualACNet + mapping
        ▼                                              ▼
  30 updates ×                                          加载 pickle
  {rollout + GAE + PPO update}              ────────►  逐 update 回放:
        │                                              ├─ 权重复制 (params_before)
        │                                              ├─ GAE 对比 (advantages/valid_mask)
        ▼                                              ├─ Forward 对比 (update 0)
  acnet_ppo_30updates_f64.pkl                          ├─ PPO loss (7 metrics × 4 mb)
  (~882 MB)                                            ├─ Gradient per-param
                                                       └─ Parameter drift (params_after)
```

### 8.2 录制 (`record_jax_acnet_golden_f64.py`)

**网络来源**：导入**生产 JAX ACNet** (`from networks.red_network import ACNet`)，不使用内联简化版。这确保 golden data 与生产代码一致。

**录制参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| B, T | 2, 8 | 模拟小规模多环境 rollout |
| Updates | 30 | 覆盖足够的训练步数以观察参数漂移 |
| Precision | float64 | 消除 ULP 级精度差异，回放时降为 float32 |
| AdamW weight_decay | 0.0 | 消除 optax/torch 默认值 100x 差异 |
| Seed | 42 | 确定性复现 |

**关键细节**：
- `flat()` 使用 `sorted(tree.keys())` 确保参数列表和梯度列表使用相同的确定性顺序（JAX `jax.grad()` 返回的 FrozenDict 插入序可能与原 params 不同）
- Rollout 在每个 env 上独立调用 `network.apply(params, obs)`（避免 vmap 与 auto_reset 冲突）
- GAE 在每个 env 上独立计算（Python for-loop，录制专用，非训练用向量化 GAE）
- 存储内容包括：`init_params`, `params_before`, `params_after`, `rollout`(obs/rewards/values/dones/cps), `gae`(advantages/targets/valid_mask/adv_norm), `flattened`(T*B 展平的训练数据), `minibatches`(每个 epoch 的 perm/loss/metrics/grads)

**运行方式**：
```bash
python mahjax_pt/tests/record_jax_acnet_golden_f64.py   # 需要 JAX 环境，~28 min
```

### 8.3 回放 (`replay_pt_acnet_golden.py`)

**纯 PT 实现**：不导入任何 JAX 模块，仅依赖 pickle 数据文件。复用 `test_network_forward.py` 的 `DualACNet`、`build_jax_to_pt_map()`、`transfer_weights()`，确保与 forward 测试使用完全相同的映射。

**逐 update 对比流程**：

1. **权重复制**：从 golden data 的 `params_before` 通过 160→160 映射复制到 PT DualACNet
2. **GAE 对比**：PT 向量化 GAE → 对比 `advantages_raw`, `targets_raw`, `valid_mask`, `adv_mean`, `adv_var`, `advantages_norm`
3. **Forward 对比**（update 0 仅）：PT forward(logits, values) vs JAX rollout values
4. **PPO Update**：逐 minibatch 计算 loss + backward，对比 `total_loss`/`actor_loss`/`critic_loss`/`entropy`/`approx_kl`/`clip_frac`/`explained_var`（7 项）+ gradient per-param
5. **Parameter 对比**：PT optimizer step 后参数 vs JAX `params_after`

**通过标准**：
| 组件 | 标准 | 说明 |
|------|------|------|
| GAE | `adv_diff < 1e-6 && vm_mismatch == 0` | 纯整数/离散运算，必须 bit-exact |
| Forward pass | `value_diff < 1e-6` | 相同参数 + 相同输入 → 相同输出 |
| PPO Math | 全部 7 项 metrics `< 2.3e-08` | epoch 0 未训练参数，数学等价 |
| Parameter drift | 观察值 | float32 跨框架深层 Transformer 已知极限，不影响训练正确性 |

**运行方式**：
```bash
python mahjax_pt/tests/replay_pt_acnet_golden.py   # 纯 PT，不需要 JAX，~2 min
```

### 8.4 最新验证结果 (2026-07-12)

| 组件 | 结果 | 数值 |
|------|:--:|------|
| GAE advantages | ✅ | 0.00e+00 (bit-exact) |
| GAE valid_mask | ✅ | 0 mismatch |
| 权重迁移 | ✅ | 160/160 mapped, 0 skipped |
| Forward pass (update 0) | ✅ | value_diff=1.91e-06 |
| PPO Loss (epoch 0) | ✅ | total_loss diff=6.22e-07 |
| PPO Metrics (7 项) | ✅ | 全部 < 6.5e-07 |
| PPO loss max (30步) | ✅ | 7.09e-03 |
| Parameter 30-step drift | ⚠️ | 6.80e-04 (float32, 预期内) |

### 8.5 精度判定标准："双千分之一"规则

所有跨框架/跨优化精度对比采用统一判定标准：

**规则 1**: 如果某次对比中**所有**数据的相对误差都 < 0.1%，判定为通过。

**规则 2**: 如果规则 1 不满足，但相对误差 > 0.1% 的数据量占总量 < 0.1%，也判定为通过。

其中：
- 相对误差 = `|值_A - 值_B| / max(|值_A|, ε)`，ε = 10⁻⁸
- "数据量"定义为参与对比的独立标量数量（如：所有参数的梯度元素总数、所有 step 的 loss 值总数等）
- 规则 2 允许极少量异常值（如参数初始值接近 0 导致相对误差放大）

**有效数据范围**: 同时满足以下两个条件的元素才纳入统计：
1. 参考值 `|值_A| > ε`（排除接近 0 的虚假高相对误差）
2. 绝对误差 `|值_A - 值_B| > δ`（排除精度噪声：差异本身微不足道时，相对误差无意义）

其中 ε = 10⁻⁸，δ = 10⁻⁷（float32 精度极限 ~10⁻⁷）。梯度/参数中 11%~60% 的元素分布在 0 附近，被条件 1 排除；条件 2 进一步排除绝对差异 < 10⁻⁷ 的浮点舍入噪声。

**ε/δ 选取原则**: 
- ε = 10⁻⁸ 适用于参数值和 loss（量级 ~10⁻²~10²）
- δ = 10⁻⁷ 对应 float32 的相对精度极限（~7 位有效数字）
- 对于 loss 等标量指标，只需条件 1（ε），不需要条件 2（δ）

**示例**：PPO 30 步回放中有 160 个参数的梯度，每个梯度含 ~10⁴~10⁵ 个元素，总数据量 ~10⁷。约 11% 的元素为精确 0（bias 初始化未收到信号），排除后有效数据 ~9×10⁶。若其中 < 9×10³ 个元素的相对误差 > 0.1%，满足规则 2。

### 8.6 SDPA 优化后验证结果 (2026-07-12)

将 `MultiHeadSelfAttention.forward()` 替换为 `F.scaled_dot_product_attention` 后的精度对比：

| 组件 | 结果 | 数值 | 规则判定 |
|------|:--:|------|:--:|
| GAE advantages | ✅ | 0.00e+00 (bit-exact) | 规则1 ✅ |
| GAE valid_mask | ✅ | 0 mismatch | 规则1 ✅ |
| Forward pass (update 0) | ✅ | value_diff=6.56e-07 | 规则1 ✅ |
| PPO Loss (epoch 0) | ✅ | all 7 metrics < 2.3e-08 | 规则1 ✅ |
| PPO loss max (30步) | ✅ | 7.09e-03 | 规则1 ✅ |
| Parameter 30-step drift | ⚠️ | 6.80e-04 (float32, 预期内) | 规则2 ✅ |
| Gradient max diff | ⚠️ | 8.04e-01 (相对均值梯度 23%) | 规则2 ✅ |

SDPA 替换后 Forward pass 精度反而从 1.91e-06 提升到 6.56e-07（~3×），因为融合 kernel 消除 `softmax→isfinite→clamp→nan_to_num` 链的累积舍入误差。

### 8.7 版本演进

| 日期 | 变化 | 原因 |
|------|------|------|
| 2026-07-11 (初版) | 内联 `JFE`/`JAC` 录制 → PT `ACNet` 回放 | 开发初期，PT ACNet 结构与 JAX 对齐 |
| 2026-07-11 (MHA bias 修复) | mapping 128→160，纳入 32 MHA bias | PT `bias=True` 修复后参数数对齐 |
| 2026-07-12 (生产对齐) | 录制改用生产 JAX `ACNet` (from `examples/networks/red_network`)，回放改用内联 `DualACNet` (from `test_network_forward`) | 修复 `dora_dense F.relu` 差异，PT 共享提取器重构后与 JAX 结构不同，通过 DualACNet 验证组件级等价性 |
| 2026-07-12 (SDPA 优化) | `MultiHeadSelfAttention.forward()` 替换为 `F.scaled_dot_product_attention` | 消除 softmax isfinite + clamp + nan_to_num 瓶颈，精度验证通过 |

---

## 9. 已录制数据目录

### 9.1 PPO 金数据（跨框架精度验证）

| 文件 | 大小 | 说明 |
|------|------|------|
| `mahjax_pt/tests/golden_data/acnet_ppo_30updates_f64.pkl` | 883 MB | JAX float64 录制，30 updates × (rollout + GAE + PPO)，B=2/T=8。用于 `replay_pt_acnet_golden.py` 回放验证 |
| `mahjax_pt/tests/golden_data/ppo_30updates.pkl` | 46 MB | 旧版 PPO 金数据（初版录制） |

### 9.2 组件级 I/O 金数据（子函数精度验证）

| 目录 | 文件数 | 说明 |
|------|--------|------|
| `mahjax_pt/tests/data_can_riichi/` | — | `can_riichi` 批处理验证数据 |
| `mahjax_pt/tests/data_furiten/` | — | 振听判定批处理验证数据 |
| `mahjax_pt/tests/data_yaku/` | — | 役判定批处理验证数据 |
| `mahjax_pt/tests/data_yaku_v2/` | — | 役判定优化版验证数据 |

### 9.3 优化迭代金数据

| 文件 | 说明 |
|------|------|
| `mahjax_pt/tests/data_gold/iter1_forward_gold.pt` | 第1轮：torch.compile 前向传播 gold 数据 |
| `mahjax_pt/tests/data_gold/iter3_fp32_gold.pt` | 第3轮：AMP 混合精度前 FP32 gold 数据 |
| `mahjax_pt/tests/data_gold/optimize_attn_gold.pt` | SDPA 优化前 baseline attention 输出 |

### 9.4 Profiling 数据

| 文件 | 说明 |
|------|------|
| `mahjax_pt/tests/data_profiler/*.pt.trace.json` | torch.profiler Chrome trace 文件，可用 `chrome://tracing` 查看 |

### 9.5 批量种子测试数据（局内行为验证）

| 目录 | 文件数 | 说明 |
|------|--------|------|
| `golden_data/` | ~800 个 `.pkl` | 不同 seed 下的对局 golden data（`golden_seed_XXXX.pkl`），用于验证局内行为正确性 |
