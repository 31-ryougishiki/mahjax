## Why

当前 `mahjax_pt/red_mahjong/env.py` 采用混合架构：`step()` 处理单环境，`step_batch()` 对约 90% 的 discard 动作做批处理，其余稀有动作回退串行。这种设计带来以下问题：

1. **正确性验证困难**：混合代码与 JAX 参考实现 (`mahjax/red_mahjong/env.py`) 的逐行对比需要同时理解两条执行路径（串行 + 批处理）
2. **性能瓶颈不可见**：串行回退路径成为隐式热点，在 GPU/NPU 上造成 kernel launch 风暴
3. **代码维护复杂**：同一逻辑在 `_draw`/`_draw_batch`、`_discard`/`_discard_batch` 中重复，修改需要同步两处
4. **架构方向不清晰**：混合设计无法明确表达"这是参考实现"还是"这是生产实现"的意图

## What Changes

将 `mahjax_pt/red_mahjong/env.py` 拆分为两个独立的、职责清晰的实现：

### 1. 纯串行版本 (`env_serial.py`)

- **目标**：正确性验证，与 JAX 参考实现逐函数对比
- **特点**：单环境、逐步骤、Python 原生控制流
- **用途**：单元测试、集成测试、JAX 对比验证、调试
- **设计原则**：代码结构与 JAX `mahjax/red_mahjong/env.py` 保持 1:1 对应，每行标注 JAX 对应行号

### 2. 纯并行版本 (`env_parallel.py`)

- **目标**：GPU/NPU 硬件加速训练
- **特点**：完全向量化、batch-first 张量操作、eager 模式
- **用途**：RL 训练 (PPO/BC)、大规模 rollout
- **设计原则**：所有操作在 batch 维度上并行，控制流分歧通过 mask 解决

### 3. 公共模块抽取

- 将两个版本共享的**纯函数**逻辑抽取到公共模块
- 状态定义、常量、工具函数等不重复维护

## Capabilities

### New Capabilities

- `serial-env`: 纯串行麻将环境，一步一环境，可直接与 JAX 版本逐函数对比验证
- `parallel-env`: 纯并行麻将环境，所有 N 个环境通过 batch-first 张量操作同步推进，eager 模式适配 GPU/NPU

### Modified Capabilities

- `env.py`: 保留为兼容层（导入重定向到 serial 或 parallel，由配置决定）
- `step_batch`: 从混合实现迁移到纯并行版本，消除串行回退路径

### Refactored Capabilities

- 公共逻辑（常量、状态定义、手牌操作、面子操作、役种判定、向听计算）抽取为共享模块

## Impact

依赖关系：
```
env_serial.py ──→ hand.py, meld.py, shanten.py, yaku.py, tile.py, state.py, action.py, constants.py (已有，不变)
env_parallel.py ──→ 同上 + batch_state.py (新增，批量化状态)
env.py (兼容层) ──→ env_serial.py, env_parallel.py
```

- **新增文件**:
  - `mahjax_pt/red_mahjong/env_serial.py` — 纯串行环境 (~1446 行)
  - `mahjax_pt/red_mahjong/env_parallel.py` — 纯并行环境 (~2150 行, 全向量化 + GPU 支持, 805 seeds 100% 通过)
  - `mahjax_pt/red_mahjong/batch_state.py` — 批量化状态定义 (~370 行)
  - `mahjax_pt/tests/replay_pt_against_golden.py` — 串行环境 JAX 金数据回放验证
  - `mahjax_pt/tests/replay_parallel_against_golden.py` — 并行环境 JAX 金数据回放验证（支持 -j 多进程）
  - `mahjax_pt/tests/test_env_parallel_parity.py` — 并行环境 vs 串行环境一致性测试
  - `mahjax_pt/tests/test_ppo_math_parity.py` — L1: PPO 数学原语 JAX/PT 对比 (Phase 11)
  - `mahjax_pt/tests/test_ppo_gae_parity.py` — L2: GAE vs JAX calculate_gae (Phase 11)
  - `mahjax_pt/tests/test_ppo_weight_transfer.py` — L3: ACNet 权重迁移显式映射 (Phase 11)
  - `mahjax_pt/tests/test_ppo_update_parity.py` — L4: PPO loss/grad/param 对比 (Phase 11)
  - `mahjax_pt/tests/test_ppo_acnet_parity.py` — L4 Ext: 完整 ACNet PPO loss 对比 (Phase 11)
  - `mahjax_pt/tests/test_ppo_cycle_parity.py` — L5: 单次 update cycle (Phase 11)
  - `mahjax_pt/tests/test_ppo_training_parity.py` — L6: 多步训练稳定性 (Phase 11)

- **修改文件**:
  - `mahjax_pt/red_mahjong/env.py` — 改为兼容层，内部委托到 serial/parallel
  - `mahjax_pt/red_mahjong/yaku.py` — GPU device-aware 常量 (`_get_cache`, `_FAN.to(device)` 等)
  - `mahjax_pt/examples/ppo_with_reg.py` — 适配新接口 + Phase 11 GAE/approx_kl bug 修复
  - `mahjax_pt/examples/networks/transformer.py` — LayerNorm eps=1e-6 对齐 JAX (Phase 11)
  - `mahjax_pt/red_mahjong/auto_reset_wrapper.py` — 适配两种环境

- **不变文件**:
  - `hand.py`, `meld.py`, `shanten.py`, `tile.py`, `state.py`, `action.py`, `constants.py`, `observation.py`, `players.py`

## Constraints

- **不引入 JIT 编译**：并行版本坚持 eager 模式，不依赖 `torch.compile` 或 `jax.jit`
- **PyTorch 兼容**：依赖 `torch>=2.0`，不引入新的大依赖
- **API 兼容**：现有 `make()` 工厂函数和 `Env` 基类接口保持兼容
- **JAX 对照**：串行版本代码结构与 JAX `mahjax/red_mahjong/env.py` 一一对应
- **正确性优先**：并行版本的每个 batch 操作必须与串行版本等价（mask 语义正确）

## Non-Goals

- 不修改 JAX `mahjax/` 的任何代码
- 不修改核心游戏逻辑（hand, yaku, meld, shanten, tile）
- 不引入分布式训练支持
- 不为串行版本添加性能优化
- 不修改网络模型定义
