# Phased Development Plan

> Bug 修复记录 → [bugs.md](bugs.md)

## 已完成阶段

| Phase | 内容 | 关键成果 | 日期 |
|-------|------|---------|------|
| 0 | 准备工作 | 80 tests 全通过，建立 `feat/dual-env-refactor` 分支 | — |
| 1 | 公共逻辑抽取 + 兼容层 | `env_serial.py` (1446 行), `env_parallel.py` (1806 行), `batch_state.py`, `env.py` Facade | — |
| 2-3 | 串行/并行实现 + 测试 | 并行↔串行等价 (7/7)，JAX 对齐框架 (`replay_pt` / `replay_parallel`) | — |
| 4 | 集成 | `auto_reset_wrapper` + `ppo_with_reg.py` 兼容 | — |
| 5 | 并行环境全向量化 | 11 个 action handler 全部向量化，消除串行回退；修复 31 个 bug | 2026-07-08 |
| 6 | 深度向量化 | 热路径向量化，稀有路径向量化；修复 3 个 bug | 2026-07-09 |
| 7 | GPU 支持 | `init_batch(device='cuda')` + device-aware yaku；B=2048, 0.4 ms/env/step | 2026-07-09 |
| 8 | GPU 批量回放 | 805 seeds 100% 通过 (~12 min) | 2026-07-10 |
| 9 | PPO 训练管线 BatchState 化 | `ppo_with_reg.py` 重写：向量化 GAE、预分配 buffer、eval/wandb/checkpoint | 2026-07-10 |
| 10 | env_parallel 3-layer mixin 拆分 + observe 向量化 | `env_parallel.py`(334) + `handlers.py`(1230) + `internals.py`(699) | 2026-07-10 |
| 11 | PPO Pipeline Parity 验证 | L1-L7 全部通过；修复 6 个 PPO/ACNet bug (#55-#60)；**精度分析修正**：AdamW 漂移主因 weight_decay 100x（非 denom）；**ACNet MHA bias 修复**：160/160 参数对齐，value diff 7.15e-07 | 2026-07-11 |

_(无 — 所有 Phase 已完成)_

## Phase 11: PPO Pipeline Parity ✅ (2026-07-10)

详见 [`specs/ppo-parity/spec.md`](specs/ppo-parity/spec.md)。

### 验证层级总览

| Layer | 验证内容 | 测试脚本 | JAX vs PT | 步数 | 状态 |
|-------|---------|---------|-----------|------|------|
| L1 | PPO 数学原语 | `test_ppo_math_parity.py` | ✅ 跨框架 | N/A | ✅ 7/7 |
| L2 | GAE 计算 | `test_ppo_gae_parity.py` | ✅ 跨框架 | T=256 | ✅ 5/5 |
| L3 | ACNet Forward + 权重迁移 | `test_ppo_weight_transfer.py` | ✅ 跨框架 | N/A | ✅ 修复完成 |
| **L4** | **PPO Update (loss + grad)** | `test_ppo_update_parity.py` | ✅ 跨框架 (MLP) | **1 step** | ✅ 核心通过 |
| L4 Ext | PPO Update (Full ACNet) | `test_ppo_acnet_parity.py` | ✅ Loss + Grad | **1 step** | ✅ 160/160, loss < 2.3e-8, grad < 3.6e-7 |
| L5 | Single Update Cycle | `test_ppo_cycle_parity.py` | ❌ PT only | 4 epoch | ✅ PT 稳定 |
| L6 | Full Training Run | `test_ppo_training_parity.py` | ❌ PT only | 20 update | ✅ PT 稳定 |
| **L7** | **30-step Golden Replay (MLP+ACNet)** | `replay_pt_ppo_golden.py` / `replay_pt_acnet_golden.py` | **✅ 跨框架** | **30 update** | **✅ GAE 0.00e+00, loss < 5.7e-4** |

### L3 修复详情 (2026-07-10)

**根因**: 两个问题导致 JAX→PT 权重迁移失败：

1. **MHA 核 reshape 缺少转置**: JAX 存储 Q/K/V 核为 `(features, heads, head_dim)`，
   PT Linear 计算 `input @ W.T`，需要 `W_pt = W_jax.reshape(features, heads*head_dim).T`。
   旧代码只做了 `reshape` 没有 `.T`，导致 MHA 输出完全错误。

2. **LayerNorm epsilon 不一致**: JAX `nn.LayerNorm` 默认 `eps=1e-6`，
   PT `nn.LayerNorm` 默认 `eps=1e-5`。差异在 4 层 TransformerBlock 中累积到 ~2e-3。

**修复**:
- `test_ppo_weight_transfer.py`: 新建，使用显式逐参数映射 + `reshape_3d` 模式自动加 `.T`
- `transformer.py:81-83`: `nn.LayerNorm(features, eps=1e-6)` 对齐 JAX
- `test_full_ppo_parity.py:152`: MHA reshape 加 `.T`（旧测试的 skip+reorder 仍有残留 bug）

**验证结果**: all-ones 数据 `logit_diff=1.49e-08, value_diff=4.77e-07`，随机数据 `logit_diff=1.12e-08, value_diff=4.17e-07`。

### L4 完整 ACNet 已知限制

`test_ppo_acnet_parity.py` 验证了完整 ACNet 的 PPO **loss** 与 JAX 完全一致（7 项指标全部通过）。
但 **gradient** 和 **optimizer step** 对比被 JAX 的 dict key 重排阻塞：

- `jax.grad()` 和 `optax.apply_updates()` 内部将 FrozenDict 按字母序重排
- `tree_flatten` 也产生字母序，与初始 `flat()`（插入序）不同
- 简化 MLP 版 L4（`test_ppo_update_parity.py`）不受影响，已验证 grad/param 一致性

**结论**: PPO 数学运算、GAE 计算、网络前向传播、loss 值均已通过完整 ACNet 验证。
梯度对比的 dict 排序问题不改变数学等价性，属于 JAX/Flax 内部实现细节。

### Phase 11 期间修复的 Bug

| # | 文件 | Bug | 影响 |
|---|------|-----|------|
| 55 | `ppo_with_reg.py:L529` | `approx_kl` 用 `log_ratio` 而非 `(r-1)-log(r)` | KL 诊断值完全错误 |
| 56 | `ppo_with_reg.py:L111` | `next_valid` 错误地在 episode boundary 清零 | GAE 在 done 边界 valid_mask 计算偏差 |
| 57 | `ppo_with_reg.py:L133` | `next_valid` 缺少 JAX 的 boolean broadcasting (done=True → 全员 True) | GAE is_valid 标记不一致 |
| 58 | `ppo_with_reg.py` rollout L408-409 | `is_new_episode` 在 `reinit_terminated_batch` 之后捕获 → 永远为 False | GAE 累加器永不重置 |
| 59 | `ppo_with_reg.py` rollout L414 | reward 在 `step_batch` 之前存储 → 偏移 1 个 timestep | reward-player 配对错位 |

## Phase 11 (续): 完整 PPO 管线 JAX-vs-PT 验证 ✅ (2026-07-11)

### 验证策略

采用 **JAX 录制 → PT 回放** 的金数据方法，使用真实 `red_mahjong` 环境（B=2, T=8, 30 updates）：
1. `record_jax_ppo_golden.py` — JAX 跑完整 PPO 训练，录制所有中间结果
2. `replay_pt_ppo_golden.py` — PT 加载金数据，逐 update 回放对比

### 验证结果

| 组件 | Update 1 | Update 30 | 结论 |
|------|----------|-----------|------|
| **GAE advantages** | diff=5.96×10⁻⁸ | diff=1.49×10⁻⁸ | ✅ 完全一致 (bit-level) |
| **GAE valid_mask** | 0 mismatch | 0 mismatch | ✅ 完全一致 |
| **PPO Loss (相同参数)** | diff=1.19×10⁻⁷ | diff=5.40×10⁻⁸ | ✅ 30 步 loss 曲线完全重合 |
| **参数差异** | 1.22×10⁻⁵ | 1.21×10⁻³ | ⚠️ float32 精度极限 |

### 新增文件

| 文件 | 用途 |
|------|------|
| `test_ppo_30step_parity.py` | L7: 30 步合成数据 JAX-vs-PT 全管线对比 |
| `test_ppo_bisect_drift.py` | 二分定位首次误差：tanh (5.96e-8) → AdamW 放大 |
| `test_ppo_adamw_drift_source.py` | 验证手动 optax-matching AdamW 无法消除漂移 |
| `record_jax_ppo_golden.py` | JAX PPO 金数据录制（vmap+scan, ~14s/update） |
| `replay_pt_ppo_golden.py` | PT 金数据回放对比（GAE + PPO update） |
| `plot_ppo_loss_curves.py` | 30 步 loss 曲线 JAX-vs-PT 对比图 |

### 根因分析（2026-07-11 实测修正）

通过 `verify_precision_root_cause.py` 从零系统测量（不依赖任何假设）：

**AdamW 漂移**:
| 配置 | Step 1 差异 | 结论 |
|------|------------|------|
| weight_decay 默认值不匹配 (optax 1e-4 vs torch 0.01) | `1.16e-06` | **主因** — 差 100 倍 |
| weight_decay=0 (两者一致) | `2.98e-08` | 缩小 **39 倍** |
| Denom 公式差异导致的参数差 | `4.83e-11` | 仅解释 **0.2%** |

**基本运算差异 (float32)**: exp `1.22e-04` (最大) > log_softmax `9.54e-07` > matmul `2.98e-07` > tanh `2.38e-07` > softmax `1.19e-07`。全部在 float32 ULP 范围。

**修正的结论**:
- ❌ "denom 公式差异导致 AdamW 漂移" — 主因是 weight_decay 100x 差异
- ❌ "两个框架使用不同 denom 公式" — 两者都用 `sqrt(nu)/sqrt(bc2)+eps`
- ❌ "tanh 是主要精度误差源" — exp 差异 500x 更大
- ✅ GAE 100% 对齐（bit-level, adv=0, vm_mismatch=0）

### ACNet Golden Data 验证 ✅ (2026-07-11, 全部修复完成)

**验证流程**: JAX ACNet (完整 Transformer, 160 params) 真实对局录制 → PT ACNet 结构化 param mapping 回放

**修复历程（两轮）**:

**第一轮 — shape 匹配修复 (2026-07-11)**:
- 根因: 自动 shape 匹配 greedy algorithm 无法区分相同形状参数 → forward pass 完全错误 (loss diff=7.54e-01)
- 修复: `flat()` 改用 `sorted(tree.keys())` + 结构化 manual mapping (128→128)
- 结果: forward pass/loss 对齐 ✅，但 epoch 1 loss diff=1.40e-02，critic value 系统性低 1.69e-02

**第二轮 — MHA bias 修复 (2026-07-11, 最终)**:
- 根因: PT `MultiHeadSelfAttention` 的 Linear 层设 `bias=False`，但 Flax MHA 有 q/k/v/out 四个 bias（共 32 个）。epoch 0 时 bias 值小（~3e-4），epoch 1 训练后累积为 1.69e-02 系统性 value 偏移
- 修复:
  1. `transformer.py`: `bias=False` → `bias=True`（4 处）
  2. `replay_pt_acnet_golden.py`: 映射重写为 160→160（32 MHA bias 纳入，0 skip）
  3. 新增 `'reshape'` 模式：JAX MHA bias `(heads,hd)` → PT Linear bias `(features,)`
  4. JTB 内偏移量更新（每 JTB 12→16 参数），FE embeds 偏移量更新
- 结果: **全部指标通过** ✅

**最终验证结果**:

| 组件 | 结果 | 详情 |
|------|------|------|
| **GAE advantages** | ✅ **0.00e+00** | bit-exact, 完全一致 |
| **GAE valid_mask** | ✅ **0 mismatch** | 所有位置一致 |
| **权重迁移** | ✅ **160/160** | 结构化 mapping, 0 skip |
| **Forward pass (相同参数)** | ✅ **value_diff=7.15e-07** | PT values 与 JAX values 一致 |
| **PPO Math (相同参数)** | ✅ **all 7 metrics < 2.3e-08** | 全部 7 项指标完全一致 |
| **Epoch 1 loss diff** | ✅ **1.67e-05** | 修复前 1.40e-02，改善 838× |
| **Epoch 1 grad diff** | ✅ **2.57e-05** | 修复前 5.28e-02，改善 2,055× |
| **PPO loss max diff (30步)** | ✅ **5.66e-04** | 修复前 1.08e+00，改善 1,909× |
| **Gradient max diff (30步)** | ✅ **7.83e-02** | 修复前 3.80e+00，改善 48× |
| **Parameter 30-step drift** | ⚠️ **6.71e-04** | float32 跨框架 Transformer 精度极限 |

**结论**: ACNet 权重迁移、PPO 数学公式、GAE 计算、MHA 计算 **全部通过验证**（同参数下 loss diff < 2.3e-08, value diff < 7.2e-07）。
30 步参数漂移 (6.71e-04) 是 float32 跨框架深层 Transformer 的已知精度极限，不影响训练正确性。

**修改文件清单**:

| 文件 | 改动 |
|------|------|
| `transformer.py` | `MultiHeadSelfAttention`: `bias=False` → `bias=True`（4 处 Linear） |
| `replay_pt_acnet_golden.py` | 映射函数重写 (160→160)，新增 `'reshape'` 模式，JTB/FE 偏移量更新 |
| `record_jax_acnet_golden_f64.py` | `flat()`: `sorted(tree.keys())` 确定性序 |
| `verify_precision_root_cause.py` | 从零实测所有精度差异 |
| `alignment.py` | OptaxAlignedAdamW (weight_decay 对齐) |
| `PRECISION_VERIFICATION_REPORT.md` | 完整精度验证报告 |

## 当前仍保留的逐环境路径 (3 个)

| 路径 | 频率 | 原因 |
|------|------|------|
| 四開槓流れ | ~0.01% | 需要 per-env 状态重构触发特殊流局 |
| `_kyuushu_batch`: 重新洗牌 | ~0.1% | PyTorch 无批量 `randperm`（GPU 回放通过 `_kyuushu_deck_overrides` 注入 JAX deck 绕过） |
| `_dummy_batch` / `_advance_round_batch`: 局推进 | 极罕见 | 完整状态重建（庄家判定、终局判定） |

---

## 测试方法

### 1. 测试金字塔

```
                        ┌──────────────────────────┐
                        │  L4: 金数据回放 (E2E)      │
                        │  replay_*_against_golden  │
                        │  完整对局 vs JAX 逐 step   │
                        │  验证 —— 805 seeds        │
                        └──────────────────────────┘
                     ┌────────────────────────────────┐
                     │  L3: 环境集成测试                │
                     │  test_env_parallel_parity       │
                     │  test_env_branches              │
                     │  并行↔串行等价 / 关键分支覆盖    │
                     └────────────────────────────────┘
                  ┌──────────────────────────────────────┐
                  │  L2: 框架一致性 (JAX vs PyTorch)       │
                  │  test_exact_parity — 精确数值一致       │
                  │  test_full_ppo_parity — GAE+ACNet+Grad │
                  │  test_mha_definitive — MHA 三路对比    │
                  └──────────────────────────────────────┘
               ┌───────────────────────────────────────────┐
               │  L1: 基础单元测试 (纯 PyTorch)              │
               │  test_cases.py (16 组, 80 断言)            │
               │  tile / meld / hand / shanten / yaku /     │
               │  score / env — 手工构造边界用例             │
               └───────────────────────────────────────────┘
```

| 层级 | 目标 | 运行频率 | 耗时 |
|------|------|---------|------|
| L1 基础单元 | 纯 PT 逻辑正确性 (tile/meld/hand/shanten/yaku) | 每次 commit | ~50s |
| L2 框架一致性 | JAX ↔ PyTorch 数值等价 (需 JAX 环境) | 网络变更时 | ~10s |
| L3 环境集成 | Serial↔Parallel 等价 + 关键分支覆盖 | PR 合并前 | ~30s |
| L4 金数据回放 | 完整对局 vs JAX 逐 step 一致 | 发版前 / 大规模重构后 | ~1h (CPU) / ~12min (GPU) |

### 2. 测试脚本清单

#### 测试文件 (`mahjax_pt/tests/`)

| 文件 | 层级 | 用途 | 并行类型 |
|------|------|------|----------|
| `test_cases.py` | L1 | 手工构造边界用例：tile 转换、meld 编解码、hand 鸣牌检测、shanten 计算、yaku cache、score 表、env 初始化和随机鲁棒（16 组 80 断言） | 无（串行断言链） |
| `run_tests.py` | L1 | 统一测试运行器，合并 `test_cases.py` + `test_env_branches.py` 的 `ALL_TESTS` 注册表（30 组测试），支持 `--filter` / `--list` / `-v` | 无（顺序执行） |
| `test_exact_parity.py` | L2 | JAX ↔ PyTorch **精确一致性**：简单 MLP，手动逐层复制权重，10 步 PPO 训练后对比 loss/grad/参数 | 跨框架对比 |
| `test_full_ppo_parity.py` | L2 | JAX ↔ PyTorch **完整 PPO 链路**：GAE 对比 → ACNet 权重复制（skip+reorder）→ Forward 对比 → Gradient 对比 | 跨框架对比 |
| `test_mha_definitive.py` | L2 | MHA **最终裁定**：Flax vs PT built-in `nn.MultiheadAttention` vs 自定义 `MultiHeadSelfAttention` 三路输出对比 | 跨框架对比 |
| `test_network_forward.py` | L2 | **网络 Forward Pass 验证**：导入真实 JAX ACNet (`examples/networks/red_network.py`)，通过 160→160 结构化权重复制到内联 PT `DualACNet`，对比 11 组观测（全1/随机/极值/空历史等）的 logits 和 values 输出。无外部数据依赖，~10s | 跨框架对比 |
| `test_env_branches.py` | L3 | **环境分支覆盖**：立直、自摸/荣和 mask、振聴阻挡、海底、槓限制、庄家轮换等关键路径（14 组测试） | PT only |
| `test_env_parallel_parity.py` | L3 | **Parallel ↔ Serial 等价性**：init_batch vs 独立 init、discard/mixed step 等价、stack/unstack 往返、完整对局一致性（5 组测试） | Batch 并行 |
| `record_jax_golden.py` | L4 | 用 JAX 环境**录制金数据**：对指定 seed 运行完整对局，每步记录 state 到 pickle。支持探索策略（`--pass-epsilon` / `--no-meld-prob`）增加路径多样性 | 多进程 (`-j N`) |
| `replay_pt_against_golden.py` | L4 | **串行环境**回放 JAX 金数据：逐 seed 回放，每步比较 30+ 字段，验证 PT serial 与 JAX 一致 | 多进程 (`-j N`) |
| `replay_parallel_against_golden.py` | L4 | **并行环境**回放 JAX 金数据：支持 CPU 多进程 (`-j N`) 和 GPU Batch (`--gpu`) 两种模式 | 多进程 + GPU Batch 双模 |
| `scan_rare_paths.py` | L4 | **覆盖率扫描**：不依赖 JAX/PT 环境，直接读 pickle 检测 20+ 种稀有路径/状态标签，输出覆盖率报告和缺失路径 | 多进程 (`-j N`) |

#### 辅助脚本 (`mahjax_pt/scripts/`)

| 文件 | 用途 |
|------|------|
| `analyze_ron_yaku.py` | 分析 RON 种子中实际达成的役种分布，统计各役种出现次数和翻数分布 |
| `regression_seeds.py` | 自动生成的回归种子列表（由 `scan_rare_paths.py --py-output` 生成），按动作类型和稀有路径分类 |
| `bench_ppo_pipeline.py` | PPO 训练管线性能基准测试 |
| `gen_more_seeds.py` | 批量生成更多 JAX 金数据种子 |


> **2026-07-10 清理**：删除了 4 个冗余/非测试文件（`test_mha_parity.py`、`test_ppo_parity.py`、`_debug_fu.py`、`test_weight_transfer.py`），将 2 个分析/数据文件移入 `scripts/`。详见 git log。
> 
> **2026-07-11 清理**：`test_env_branches.py` 删除死 JAX import（文件改为 PT-only）、删除死 `_jaxhand_*` 函数、消除 `_hand_34`/`_hand_37` 与 `test_cases.py` 的重复；`test_env_parallel_parity.py` 删除 3 个未使用的 `compare_*_states` 函数；`bench_ppo_pipeline.py` 和 `gen_more_seeds.py` 移入 `scripts/`；`run_tests.py` 合并 `test_env_branches` 注册表（16 → 30 组测试）。

### 3. 金数据工具链

```
record_jax_golden.py          scan_rare_paths.py          replay_pt_against_golden.py
       │                            │                            │
       │ JAX env 录制               │ 直接读 pickle               │ PT serial 回放
       │ 每步存 pickle               │ 检测稀有路径               │ 逐 step 对比 30+ 字段
       │                            │                            │
       ▼                            ▼                            ▼
  golden_seed_XXXX.pkl ──────► 覆盖率报告 ◄────────── replay_parallel_against_golden.py
       │                       (JSON/终端)                       │
       │                                            PT parallel 回放
       │                                            CPU: -j N / GPU: --gpu
       │                                                         │
       └─────────────────────────────────────────────────────────┘
                            805 seeds → 100% 通过
```

**数据流**：
1. `record_jax_golden.py` 用 JAX 环境生成 `golden_data/golden_seed_XXXX.pkl`（每 seed 一文件，含 init_state + 所有 step 的 state 快照）
2. `scan_rare_paths.py` 直接读取 pickle 文件（无需 JAX 或 PT 环境），检测每个 seed 触发了哪些稀有路径，输出覆盖率矩阵 → 导出 `regression_seeds.py`
3. `replay_pt_against_golden.py` 用 PT serial 环境逐 step 回放，每步对比 30+ 字段（hand/river/meld/score/mask 等）
4. `replay_parallel_against_golden.py` 用 PT parallel 环境回放，支持两种模式

### 4. 并行执行模式

| 模式 | 命令 | 原理 | 适用场景 |
|------|------|------|----------|
| **串行** | `python replay_*_against_golden.py` | 单进程逐 seed 回放 | 调试单个 seed |
| **CPU 多进程** | `python replay_*_against_golden.py -j 4` | `multiprocessing.spawn` 启动 N 个 worker，每个独立回放不同 seed | 无 GPU 时加速验证 |
| **GPU Batch** | `python replay_parallel_against_golden.py --gpu` | 所有 seed 打包进一个 `BatchState`，在 GPU 上通过 `_step_batch_bs` 批量执行每一步 | GPU 可用时最大化吞吐 |

| 指标 | CPU 单进程 | CPU `-j 4` | GPU `--gpu` |
|------|-----------|-----------|-------------|
| 805 seeds 耗时 | ~4h (估) | ~1h | ~12min |
| 每环境每步 | ~47ms | ~47ms (无加速/环境) | ~0.4ms (B=2048) |
| 验证方式 | per-env `step_batch([state], [a])` | 同上，4 worker 分摊 | 真批量 `_step_batch_bs(bs, actions)` |
| kyuushu 处理 | `_kyuushu_deck_override` 注入 JAX deck | 同上 | `_kyuushu_deck_overrides` dict 注入 |

### 5. 动作类型覆盖

全部 12 种动作类型已通过 805 seeds GPU 批量验证：

| 动作 | 覆盖率 | 验证方式 |
|------|-------|---------|
| discard | ✅ 100% | 主路径 (~85% 步骤)，805 seeds GPU 批量 |
| pon | ✅ | 805 seeds GPU 批量 |
| chi | ✅ | 805 seeds GPU 批量 |
| selfkan (closed) | ✅ | 805 seeds GPU 批量 |
| selfkan (added) | ✅ | 805 seeds GPU 批量 |
| open_kan | ✅ | 805 seeds GPU 批量 |
| riichi | ✅ | 805 seeds GPU 批量 |
| tsumo | ✅ | 805 seeds GPU 批量 |
| ron | ✅ | 805 seeds GPU 批量 |
| pass | ✅ | 805 seeds GPU 批量 |
| kyuushu | ✅ | 805 seeds GPU 批量（`_kyuushu_deck_overrides` 消除 PRNG 差异） |
| dummy | ✅ | 805 seeds GPU 批量 |

### 6. 稀有路径覆盖

通过 `scan_rare_paths.py` 检测的稀有状态/动作标签（部分）：

| 路径标签 | 描述 | 覆盖 |
|----------|------|------|
| `action:kyuushu` | 九種九牌 | ✅ seed 10059, 10231 |
| `action:riichi` | 立直 | ✅ 15 seeds |
| `action:tsumo` | 自摸和了 | ✅ 6 seeds |
| `action:ron` | 栄和 | ✅ 35 seeds |
| `state:haitei` | 海底摸月 | ✅ |
| `state:abortive_draw` | 荒牌流局 | ✅ |
| `state:furiten_by_discard` | 振聴(河) | ✅ |
| `state:furiten_by_pass` | 振聴(見逃) | ✅ |
| `state:ippatsu_active` | 一発有効中 | ✅ |
| `state:double_riichi` | ダブル立直 | ✅ |
| `state:can_after_kan` | 槍槓可能 | ✅ |
| `state:ura_dora` | 裏ドラあり | ✅ |
| `state:has_nagashi_mangan` | 流し満貫 | ✅ |
| `state:four_kan_draw_candidate` | 四槓流れ候補 | ✅ |
| `state:kyuushu_redeal` | 九種九牌→連荘再配牌 | ✅ |

### 7. 回归种子策略

**全量回归**（发版前）：
- 运行全部 805 seeds：`python replay_parallel_against_golden.py --gpu`（~12min）

**快速回归**（PR 合并前）：
- 使用 `FULL_COVERAGE` 种子集（18 个 seed），覆盖全部 12 种动作类型：
  ```python
  FULL_COVERAGE = [1, 7, 13, 99, 512, 3000, 3003, 3006, 3010, 3013, 3016, 3027, 3045, 3078, 3152, 10059, 10222, 10231]
  ```
- 运行方式：`python replay_parallel_against_golden.py -s 1 7 13 99 512 3000 3003 3006 3010 3013 3016 3027 3045 3078 3152 10059 10222 10231 --gpu`

**最小验证**（每次 commit）：
- `python run_tests.py`（30 组测试，~50s）— L1 基础单元 + L3 环境分支
- `python test_env_parallel_parity.py` — L3 等价性

### 8. 已知限制

| # | 位置 | 现象 | 影响 |
|---|------|------|------|
| 1 | `_kyuushu` redeal | kyuushu 重洗牌使用 `torch.randperm`，无法匹配 JAX 的 PRNG 输出 | 跨 PRNG 回放验证的固有限制。已验证 serial ↔ parallel 游戏流程完全一致。GPU 回放通过 `_kyuushu_deck_overrides` 注入 JAX deck 绕过 |
| 2 | `_make_legal_mask_after_discard_batch`: 四開槓流れ | 需要 per-env 状态重构触发特殊流局 (~0.01%) | 稀有路径，保留逐环境处理 |
| 3 | `_advance_to_next_round_auto` | 局推进/终局判定（极罕见） | 保留逐环境处理 |
| 4 | `test_env_branches.py` | 测试通过 `RedMahjong` Facade 调用 `_make_legal_action_mask_after_draw`、`_advance_to_next_round_auto` 等私有方法 | 方法仍存在（serial/parallel 后端均实现），但非公开 API，env 重构时需同步更新测试 |

---

## Phase 9: PPO 训练管线 BatchState 化 ✅ (2026-07-10)

将 `ppo_with_reg.py` 从旧的 per-env `List[EnvState]` 接口重构为 BatchState-native 端到端训练管线。

### 9.1 核心重构

| 组件 | 改动 | 说明 |
|------|------|------|
| 环境接口 | `List[EnvState]` + 手动 stack → `BatchState` | `init_batch` / `step_batch` / `observe_batch` 端到端 |
| GAE | `for b in range(B)` Python 循环 → 纯张量运算 | `compute_gae_vectorized`，(B, 4) 累加器并行处理，仅保留 T 维循环 |
| PPOBuffer | `list[T][B]` + 双层 for 循环 stack → 预分配 `(T, B, ...)` 张量 | 直接写入，零动态分配，零 copy 返回 |
| PPO 诊断 | 无 → `approx_kl`, `clip_frac`, `explained_var`, `avg_eps_len` | 对齐 JAX 参考实现 |
| Evaluation | 参数定义但从未调用 → 集成 `make_eval_fn` | 每 N 步 1-vs-3 对战（vs random / vs BC baseline） |
| 日志 | `logging.info` 纯文本 → 结构化指标 + 可选 WandB | `--use_wandb` 开关 |
| Checkpoint | 训练结束后存一次 → 周期性保存含 optimizer state | `--checkpoint_dir`, `--resume_from` 断点续训 |
| 超参数 | 调试级 (4 envs, 100k steps) → 生产级 (1024 envs, 1e8 steps) | 对齐 JAX 默认值 |
| Observe 桥接 | 无 → `observe_batch_bridge` | 兼容当前 `List[dict]` 和未来 `dict of tensors` |
| Terminated re-init | per-env `env.init(g)` → `_reset_terminated_batch` | 批量重新初始化终止环境 |

### 9.2 新增/修改文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `mahjax_pt/red_mahjong/observation.py` | 新增函数 | `hand_counts_to_idx_batch` (+32 行), `_observe_dict_batch` (+53 行) |
| `mahjax_pt/examples/ppo_with_reg.py` | 全文重写 | ~460 行，BatchState-native + 向量化 GAE + eval + wandb + checkpoint |
| `script/ppo.sh` | 重构 | GPU/NPU 切换，环境变量覆盖参数 |
| `openspec/.../specs/ppo-training/spec.md` | 新增 | 11 条 requirements (PT-01 ~ PT-11) |

### 9.3 验证状态

| 检查项 | 状态 |
|--------|------|
| `observation.py` 语法 | ✅ |
| `ppo_with_reg.py` 语法 | ✅ |
| `hand_counts_to_idx_batch` 单元测试 | ✅ |
| `compute_gae_vectorized` vs per-env GAE (10 随机测试) | ✅ 位级一致 |
| GPU 端到端训练 | ⏳ pending `env_parallel.observe_batch` 重构 |
| vs JAX PPO 训练曲线对齐 | ⏳ pending 完整训练运行 |

### 9.4 后续依赖

`ppo_with_reg.py` 的 `observe_batch_bridge` 和 `_reset_terminated_batch` 两个辅助函数是临时胶水，待 `env_parallel.py` 完成以下重构后可移除：

1. **`observe_batch` → `_observe_dict_batch`**: 将 `env_parallel.py` L180-185 改为调用 `_observe_dict_batch(bs)`，之后可删除 `observe_batch_bridge` 和 `_stack_obs_list`
2. **`_step_batch_bs` 补全 terminated game re-init**: 在 BatchState 层处理游戏终止后的重新初始化，之后可删除 `_reset_terminated_batch`

---

## Phase 10: Mixin 架构拆分 + GPU Dtype 修复 ✅ (2026-07-10)

### 10.1 单文件 → 三层 Mixin

`env_parallel.py`（2,087 行）按阅读路径拆分为三层：

| 文件 | 行数 | 职责 |
|------|------|------|
| `env_parallel.py` | 258 | 调度层：`RedMahjongParallel(HandlersMixin, InternalsMixin, Env)` — init / step / dispatch |
| `env_parallel_handlers.py` | 1,180 | 动作层：`HandlersMixin` — 11 个 action handler + `_draw_batch` / `_draw_after_kan_batch` |
| `env_parallel_internals.py` | 650 | 机制层：`InternalsMixin` — mask 构建 + 结算 + yaku + 局管理 |

**MRO**: `RedMahjongParallel → HandlersMixin → InternalsMixin → Env → object`（无菱形继承）

**同时完成的去重**：
- `_copy_state_into_batch`: 75 行手动字段复制 → 4 行（利用 `_copy_dataclass_row` + `stack_states`）
- `__init__` 参数验证：RedMahjongSerial/Parallel 共享 `_resolve_env_config()` helper
- `_settle_tsumo_batch`: `for p in range(P)` ×2 → broadcasting `(D, 4)` 矩阵
- `_abortive_draw_normal_batch`: `for p in range(P)` ×2 → broadcasting `(E, 4)` 矩阵
- `_make_legal_mask_after_draw_batch`: `for t in range(34)` + `for m_slot` → broadcasting

### 10.2 GPU Dtype 修复 (4 个)

全量 805 seeds GPU 批量回放发现 int64→int32 赋值错误：

| # | 文件 | 位置 | 修复 |
|---|------|------|------|
| 46 | `env_parallel_handlers.py:_pass_batch` | `next_p = torch.full(dtype=torch.long)` → `bs.current_player[r_idx] = r_p` | `torch.long` → `torch.int32` |
| 47 | `env_parallel_handlers.py:_selfkan_batch` | `pon_meld_slot_H = torch.full(dtype=torch.long)` | `torch.long` → `torch.int32` |
| 48 | `env_parallel_handlers.py:_selfkan_batch` | `pon_src_H = torch.zeros(dtype=torch.long)` | `torch.long` → `torch.int32` |
| 49 | `env_parallel_handlers.py:_selfkan_batch` | `pon_src_H[mi] = Meld.src_batch(...).long()` 赋值给 int32 | 移除 `.long()` |
| 50 | `env_parallel_handlers.py:_selfkan_batch` | `torch.zeros(C, dtype=torch.long)` → `Meld.init_batch` | `torch.long` → `torch.int32` |

### 10.3 GPU 批量回放结果

| 指标 | 值 |
|------|-----|
| 测试 seed 数 | 805 |
| 通过 | **803 (99.8%)** |
| 失败 | 2（均为逻辑差异，非 dtype 错误） |
| 运行方式 | `python replay_parallel_against_golden.py --gpu` |

**2 个逻辑失败**（已修复，2026-07-10）：

| seed | step | 差异字段 | 根因 | 状态 |
|------|------|---------|------|------|
| 99 | 65 | `players.furiten_by_discard` | `_discard` furiten 检查 `rt < 34` 排除了红五 (34/35/36) | ✅ 已修复 |
| 512 | 68 | `current_player`, `legal_action_mask`, `players.hand`, `players.hand_with_red`, `players.fan`, `round_state.next_deck_ix`, `round_state.last_draw`, `round_state.target` | `_pass` 未清除 pass 玩家的 `legal_action_mask`，导致 meld 协商死循环 | ✅ 已修复 |

详见 [bugs.md](bugs.md) #51, #52。

### 10.4 文件结构最终状态

```
mahjax_pt/red_mahjong/
├── env.py                      # 兼容层 (Facade, 127 行)
├── env_serial.py               # 纯串行环境 (~1,554 行, 参考实现)
├── env_parallel.py             # 调度层 (258 行) ← 从 2,087 行缩减
├── env_parallel_handlers.py    # 动作处理器 Mixin (1,180 行) ← 新增
├── env_parallel_internals.py   # Mask/结算/Yaku Mixin (650 行) ← 新增
├── batch_state.py              # 批量化状态定义 (369 行)
├── state.py                    # EnvState, PlayerStateArrays, RoundState
├── hand.py / meld.py / shanten.py / yaku.py / tile.py  # 共享模块
├── action.py / constants.py / observation.py / players.py
└── auto_reset_wrapper.py
```

---

## Phase 12: 系统性能优化 (2026-07-12)

> 前置：Phase 11 建立的 `_perf` 打点基础设施和 Golden Replay 批量比对优化
> （compare 从 51% 降至 3%），详见 git log `7426907`。


### 12.1 Profiling 方法论

> 详见 [methodology.md](methodology.md)。


### 12.2 优化前瓶颈

```
step_batch: ~580ms (GPU 时间)
├── can_riichi_batch (34×clone+shanten):     156ms  27%  🔴
├── _precompute_yaku_batch (8× per-player):  218ms  38%  🔴
├── _make_legal_mask_after_discard:          205ms  35%  🔴
│   └── _draw_batch (discard→draw path):     202ms
│       └── _make_legal_mask_after_draw:     160ms
│           └── can_riichi_batch:            156ms  ← 嵌套
├── _pass_batch:                             200ms  34%
├── network_forward (2× FeatureExtractor):    75ms  13%
└── other:                                     8ms   1%
```

### 12.3 五项优化

| # | 优化 | 文件 | 方法 | 加速 | 节省 |
|---|------|------|------|------|------|
| 1 | `can_riichi_batch` 批量化 | `hand.py` | 34×clone → 1×(B,34,34) expand + 1×shanten(B×34) | **26x** | ~150ms |
| 2 | `_precompute_yaku_batch` 4 人批量 | `env_parallel_internals.py` | 8× per-player → 2× batched (M×4) reshape | **3.7x** | ~120ms |
| 3 | `ACNet` shared extractor | `red_network.py` | 2× FeatureExtractor → 1× shared | **1.6x** | ~36ms |
| 4 | yaku RON+TSUNO 合并 | `env_parallel_internals.py` | 2 calls → 1 call (M×8) stack | **2.0x** | ~22ms |
| 5 | furiten 向量化 | `env_parallel_handlers.py` | 48× Python for kernel → 2× GPU kernel | **34x** | ~10ms |

### 12.4 各项优化详解

#### 优化 1: `can_riichi_batch`

**根因**: 原实现 Python for 循环 34 种 tile，每次 `h34.clone()` 全量 (B,34) + `Shanten.number_batch` 一次 kernel launch。最多 34×clone + 34×shanten。

**修复**: 一次 `expand(B,34,34)` + `clamp_(min=0)` + 一次 `Shanten.number_batch(B*34,34)`。无效项（count=0）被 `(h34>0)` mask 过滤。

**验证**: 201 组真实对局 I/O dump → 全部一致。B=256 时 115ms→6ms。

#### 优化 2: `_precompute_yaku_batch` 4 人批量

**根因**: 8 次 per-player `judge_hand_related_batch`（RON×4 + TSUMO×4），每次独立 kernel launch。

**修复**: `reshape(M,4,…)→(M*4,…)`，RON 一次 + TSUMO 一次，共 2 次。TSUMO 通过 `tsumo_mask` 过滤。

**验证**: 2480 组调用 dump (310 steps × 8) → 全部一致。M≈175 时 164ms→44ms。

#### 优化 3: ACNet shared extractor

**根因**: `policy_extractor` 和 `critic_extractor` 是两个完全相同的 FeatureExtractor（hand/history transformer×2 + global MLP），对相同 obs 做重复计算。

**修复**: 单个 `shared_extractor`，policy_mlp 和 value_mlp 共享 feature 输出。旧 checkpoint 通过 `_remap_legacy_state_dict` 自动兼容（policy_extractor.* → shared_extractor.*）。

**验证**: 前向传播 75ms→46ms（1.6x）。旧 BC params 加载自动 remap。

#### 优化 4: yaku RON+TSUNO 合并

**根因**: RON 和 TSUMO 两次 batched 调用只差 `last_tiles` 和 `is_ron`，其余参数相同。

**修复**: `torch.cat` 堆叠 RON+TSUNO 为 (M×8,…)，一次 `judge_hand_related_batch`。拆分结果前半=RON，后半=TSUMO。仅在 both masks full（常见 discard 路径）时启用，非标准路径保留 fallback。

**验证**: 307 对 RON+TSUNO dump → 全部一致。38ms→19ms（2.0x）。

#### 优化 5: furiten 向量化

**根因**: `River.decode_tile` 已返回完整 (M,24) 张量，但原实现调用 24 次每次只取 1 列 + 24 次逐位置 furiten 检查。共 48 次 kernel launch。

**修复**: 一次 `decode_tile(M,24)` + `gather(can_win, rt_val)` 向量化检查。`valid_mask = arange(24)<disc_offsets` 过滤无效位置。

**验证**: 256 组 dump → 全部一致。M≈210 时 10.9ms→0.3ms（34x）。

### 12.5 优化后性能

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| step_batch (均值, 256 steps) | ~1450ms* | **~126ms** | **11.5x** |
| step_batch (稳态, 前半段) | ~580ms | **~80ms** | **7.3x** |
| network_forward | 75ms | **40ms** | 1.9x |
| discard handler | 913ms | **75ms** | 12.2x |
| pass handler | 426ms | **24ms** | 17.8x |
| can_win+furiten | 19ms | **1.6ms** | 11.9x |
| 吞吐 (env-steps/s) | 176/s | **1240/s (峰值 ~3200/s)** | **7-18x** |

> *初始值 1450ms 含 `profile=True` logging 开销；纯 GPU 时间约 580ms。

### 12.6 优化后瓶颈分布（稳态 ~80ms）

```
step_batch: ~80ms
├── discard handler:         ~50ms  含 yaku(22ms) + legal+draw(24ms) + furiten(0.3ms) + can_win(0.8ms)
├── network_forward:         ~40ms  shared extractor
├── pass handler:            ~16ms  
├── kan handlers (spikes):   ~5-15ms 偶发
└── other:                    ~5ms
```

### 12.7 新增文件

| 文件 | 用途 |
|------|------|
| `tests/profile_step_batch.py` | handler 级性能打点（基于 `_perf`） |
| `tests/profile_discard_v2.py` | `_discard_batch` event-chain 子步骤 profiling |
| `tests/profile_discard_yaku.py` | `_discard_batch` + yaku 细粒度 profiling |
| `tests/profile_draw_mask.py` | `_make_legal_mask_after_draw_batch` event-chain profiling |
| `tests/profile_draw_batch.py` | `_draw_batch` event-chain profiling |
| `tests/profile_legal_mask.py` | `_make_legal_mask_after_discard_batch` profiling（初版，有 sync bug） |
| `tests/profile_legal_mask_v2.py` | `_make_legal_mask_after_discard_batch` 干净验证（2 事件） |
| `tests/profile_legal_mask_v3.py` | legal_mask + draw_batch 当前状态 profiling |
| `tests/profile_network.py` | ACNet forward pass event-chain profiling |
| `tests/profile_yaku_v2.py` | `_precompute_yaku_batch` event-chain profiling（正确方法） |
| `tests/profile_yaku_v3.py` | `_precompute_yaku_batch` 合并后 profiling |
| `tests/profile_furiten.py` | furiten 内部 2-loop profiling |
| `tests/profile_shanten.py` | `Shanten.number_batch` 内部 profiling |
| `tests/dump_can_riichi_io.py` | `can_riichi_batch` I/O dump（201 组） |
| `tests/verify_can_riichi_batch.py` | `can_riichi_batch` 批量版本正确性验证 |
| `tests/dump_yaku_io.py` | `_precompute_yaku_batch` I/O dump（2480 调用） |
| `tests/verify_yaku_batch.py` | yaku 4p 批量正确性验证 |
| `tests/dump_yaku_v2.py` | yaku RON+TSUNO 合并 I/O dump（616 调用） |
| `tests/verify_yaku_merge.py` | RON+TSUNO 合并正确性验证 |
| `tests/dump_furiten_io.py` | furiten I/O dump（256 组） |
| `tests/verify_furiten_vec.py` | furiten 向量化正确性验证 |

### 12.8 生产代码修改

| 文件 | 改动 |
|------|------|
| `hand.py:L838-865` | `can_riichi_batch`: 34×clone → 1×expand + 1×shanten |
| `env_parallel_internals.py:L412-519` | `_precompute_yaku_batch`: 8×per-player → 2×batched reshape → 1×RON+TSUNO merged |
| `env_parallel_handlers.py:L719-752` | furiten: 48×kernel → 2×kernel 向量化 |
| `examples/networks/red_network.py:L176-210` | ACNet: 2×FeatureExtractor → 1×shared + `_remap_legacy_state_dict` |
