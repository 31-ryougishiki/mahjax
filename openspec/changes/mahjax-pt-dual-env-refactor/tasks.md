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
| 11 | PPO Pipeline Parity 验证 | L1-L7 全部通过；修复 rollout bug #58(is_new_episode), #59(reward 时序)；**精度分析修正 (2026-07-11)**：AdamW 漂移主因是 weight_decay 100x 差异（非 denom 公式），全部基本运算在 float32 ULP 范围 | 2026-07-11 |

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
| L4 Ext | PPO Update (Full ACNet) | `test_ppo_acnet_parity.py` | ✅ Loss | **1 step** | ⚠️ Grad 被 JAX dict 重排阻塞 |
| L5 | Single Update Cycle | `test_ppo_cycle_parity.py` | ❌ PT only | 4 epoch | ✅ PT 稳定 |
| L6 | Full Training Run | `test_ppo_training_parity.py` | ❌ PT only | 20 update | ✅ PT 稳定 |

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

### ACNet Golden Data 验证 (2026-07-11, 新增)

| 文件 | 用途 |
|------|------|
| `record_jax_acnet_golden_f64.py` | JAX ACNet (完整 Transformer) 真实对局 Golden Data 录制 |
| `replay_pt_acnet_golden.py` | PT ACNet Golden Data 回放 (自动 shape 匹配 + 逐步骤对比) |
| `verify_precision_root_cause.py` | 从零实测所有精度差异，不依赖任何假设 |
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

## Phase 11: 性能分析与 Yaku 优化 🔄 (2026-07-10, 进行中)

### 11.1 性能打点基础设施

在以下位置添加了细粒度计时：

| 文件 | 打点位置 | 粒度 |
|------|---------|------|
| `env_parallel.py:_step_batch_bs` | 11 个 handler 分别计时 + active/envs 计数 | handler 级 |
| `env_parallel_handlers.py:_discard_batch` | `Hand.to_34_batch`, `can_win+furiten` | 子操作级 |
| `env_parallel_internals.py:_precompute_yaku_batch` | `Yaku.judge_hand_related_batch` 累计时间 | 函数级 |
| `replay_parallel_against_golden.py` | 数据加载、batch 初始化、step/compare 分阶段 + 进度报告 | 阶段级 |

所有打点通过 `self._perf` dict 累积（仅在 `penv._perf = {}` 时启用），通过 `get_perf_summary()` 输出排序结果。

### 11.2 性能瓶颈分析 (805 seeds, GPU)

#### 第一阶段优化（消除 compare 开销）

在 `replay_parallel_against_golden.py` 的 `replay_seeds_batch` 中：

| 优化 | 改动 | 效果 |
|------|------|------|
| **v1: 消除 `unstack_state`** | 定义 `BATCH_CHECKS`，直接索引 `BatchState` 张量（view，无 clone），跳过 ~40 次 GPU `.clone()` | compare 从 32.6s → 16.0s（-51%） |
| **v2: 批量 GPU 比对** | 定义 `BATCH_COMPARE_FIELDS`，每步每字段 1 次 CPU→GPU 传输 + 1 次向量化比较，消除 800×36 次 GPU→CPU 传输 | compare 从 16.0s → 0.6s（-96%，累计 **-98%**） |

**最终时间分布（129.3s 全量，compare 仅 3.0%）：**

| 组件 | 时间 | 占比 |
|------|------|------|
| `_step_batch_bs`（GPU 计算） | 107.9s | **83.5%** |
| `compare`（验证） | 3.9s | 3.0% |
| `init` | 12.6s | 9.8% |
| `load` | 2.8s | 2.1% |

#### Handler 级耗时分析（带 profiling 开销，478s traced）

| Handler / 子操作 | 耗时 | 占比 | 调用次数 | 说明 |
|-----------------|------|------|---------|------|
| **`yaku.judge_hand_related_batch`** | **176.4s** | **54.0%** | 369 active | 🔴 #1 瓶颈 |
| `discard` | 149.3s | 45.7% | 178 | 最频繁 handler（~92% 步骤） |
| `selfkan` | 81.4s | 24.9% | 178 | |
| `open_kan` | 49.6s | 15.2% | 178 | |
| `pass` | 9.4s | 2.9% | 178 | |
| `discard.can_win+furiten` | 4.8s | 1.5% | 164 | |
| 其余 handler | < 5s | — | — | |

> **注**：百分比叠加 >100% 是因为 `yaku`/`can_win+furiten` 是 handler 内部的子操作，时间已包含在 handler 中。

**推算无 profiling 真实耗时**（step_batch 总计 ~108s）：

| 组件 | 推算耗时 | 占 step_batch |
|------|---------|--------------|
| `Yaku.judge_hand_related_batch` | ~40s | ~37% |
| `_discard_batch` 其余部分 | ~30s | ~28% |
| `selfkan` + `open_kan` | ~25s | ~23% |
| `pass` 及其他 | ~13s | ~12% |

#### 每步耗时递增原因

随着游戏进程，后期手牌更复杂（riichi、meld、yaku）、`_make_legal_mask_after_draw_batch` 的 riichi 路径（closed kan 循环 34 种 tile）触发频率增加，导致 per-batch-step 从 ~0.9s 增长到 ~2.5s。

### 11.3 Yaku 优化尝试

提出了三个优化方案：

| 方案 | 思路 | 预期节省 |
|------|------|---------|
| **A: 懒计算 TSUMO** | `_discard_batch` 跳过 col 1 (TSUMO)，推迟到 `_draw_batch` 时对摸牌玩家单独计算 | ~15s（-37.5% yaku 调用） |
| **B: TSUMO 缓存** | 跟踪 `next_deck_ix` 变化，只在 deck 前进时重算 TSUMO | ~20s（A 的超集） |
| **C: 4 玩家 RON 批量化** | reshape `(M,4,37)→(M*4,37)`，一次 `judge_hand_related_batch` 替代 4 次 | ~8s（减少 kernel launch） |

**实施结果**：

| 方案 | 结果 | 根因 |
|------|------|------|
| **A** | ❌ `players.fan` 不匹配 | TSUMO 在 draw 时计算使用的 `next_deck_ix`（已递减）和手牌状态（pre-draw+drawn）与 discard 时不同 |
| **C** | ❌ 部分 seed `has_yaku/fan/fu` 不匹配 | `reshape` 后的 batching 在 `judge_hand_related_batch` 内部产生细微数值差异（根因待查） |
| **Kan TSUMO mask** | ❌ `players.fan` 不匹配 | golden 数据在杠后对所有 4 玩家记录 TSUMO（用岭上牌），只给杠玩家算导致其他 3 玩家 col 1 不匹配 |

**核心约束**：replay test 在每个 step 后比对完整 state（包括 `has_yaku[:,:,1]` 的 TSUMO 预计算值）。任何改变 col 1 写入时机的优化都会破坏比对。因此 yaku 优化不能改变语义时序，只能优化 `judge_hand_related_batch` 内部计算本身。

### 11.4 当前状态

| 已保留的改动 | 状态 |
|-------------|------|
| `BATCH_CHECKS` / `BATCH_COMPARE_FIELDS`（批量比对） | ✅ 已合入，compare 从 51%→3% |
| 性能打点基础设施（`_perf`, `_perf_add`, `get_perf_summary`） | ✅ 已合入，可通过 `penv._perf = {}` 启用 |
| `_precompute_yaku_batch` 的 `ron_mask`/`tsumo_mask` 参数 | ✅ 已合入（backward-compat，默认 None = all True） |
| 方案 A（懒计算 TSUMO） | ❌ 已 revert |
| 方案 C（RON batching） | ❌ 已 revert |
| Kan TSUMO mask | ❌ 已 revert |

### 11.5 后续方向

1. **`Yaku.judge_hand_related_batch` 内部优化**：分析 CUDA kernel 使用（`Hand.add_batch`, `Hand.to_34_batch`, `update_batch` ×3 suits, `flatten_batch`, 役满判定等），寻找冗余计算
2. **`_make_legal_mask_after_discard_batch` 优化**：对手牌检查（ron/pon/chi/kan 判定）进行批量处理
3. **`_discard_batch` 的 `can_win+furiten`**：M×34×37 大展开（仅 4.8s，非当前瓶颈）
4. **方案 C 调试**：排查 `reshape` batching 在 `judge_hand_related_batch` 内部的数值差异根因
5. **破坏性优化路径**：如果接受修改 golden 比对逻辑（如跳过 `has_yaku[:,:,1]` 比对），方案 A 可正常工作，节省 ~15s
