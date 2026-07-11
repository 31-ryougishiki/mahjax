## Overview

将 `mahjax_pt/red_mahjong/env.py` 重构为两个独立的、职责清晰的环境实现：
- **`env_serial.py`**：纯串行、单环境、用于正确性验证
- **`env_parallel.py`**：纯并行、batch-first、用于 GPU/NPU 训练

两者通过 `env.py` 兼容层统一对外接口。

## Architecture

```
                        ┌──────────────────────┐
                        │      env.py          │
                        │   (兼容层 / Facade)    │
                        │  make(backend=...)    │
                        └──────┬───────┬───────┘
                               │       │
                  ┌────────────┘       └────────────┐
                  ▼                                 ▼
        ┌──────────────────┐             ┌──────────────────┐
        │  env_serial.py   │             │  env_parallel.py │
        │  RedMahjongSerial│             │ RedMahjongParallel│
        │                  │             │                  │
        │  step(state)     │             │  step_batch(     │
        │  init(key)       │             │    batch_states, │
        │  observe(state)  │             │    actions)      │
        │                  │             │  init_batch(keys)│
        └───────┬──────────┘             │  observe_batch(  │
                │                        │    batch_states) │
                │                        └────────┬─────────┘
                │                                 │
        ┌───────┴─────────────────────────────────┴─────────┐
        │              共享模块 (不变)                        │
        │  hand.py   meld.py   shanten.py   yaku.py         │
        │  tile.py   state.py  action.py    constants.py    │
        │  observation.py   players.py                      │
        └──────────────────────────────────────────────────┘
```

## Component Design

### 1. `env_serial.py` — Pure Serial Env

**职责**：单环境、逐步骤的正确性参考实现。

**核心类**：`RedMahjongSerial(Env)`

```python
class RedMahjongSerial(Env):
    """纯串行版麻将环境。

    每次调用 step(state, action) 处理恰好一个环境的一个步骤。
    代码结构与 JAX mahjax/red_mahjong/env.py 保持 1:1 对应，
    每行关键逻辑标注 JAX 对应行号。

    State 使用现有的 EnvState (单环境可变数据类)。
    """

    def init(self, key=None) -> EnvState:
        """初始化一个游戏状态。"""

    def step(self, state: EnvState, action, key=None) -> EnvState:
        """执行一个步骤。"""
        # 动作分发: _discard / _riichi / _ron / _tsumo /
        #           _pon / _chi / _open_kan / _selfkan /
        #           _pass / _kyuushu / _dummy

    def observe(self, state: EnvState) -> dict:
        """构建观察。"""
```

**关键设计决策**：
- 使用现有的 `EnvState` 可变数据类（与当前 `env.py` 共享）
- 复用现有的 `_draw`, `_discard`, `_ron`, `_tsumo` 等辅助方法（从当前 `env.py` 提取）
- 每行代码标注 JAX 对应实现的行号范围，格式：`# JAX: env.py L800-L820`
- 所有标量为 Python 原语 (int/bool)，张量为 `torch.Tensor`（单环境维度）
- 无 `step_batch` 方法
- 无 `_draw_batch`、`_discard_batch` 等方法

**从当前 env.py 提取的方法**（纯串行部分）：
- `_draw(state)` — 摸牌
- `_discard(state, tile)` — 舍牌
- `_riichi(state)` — 立直
- `_ron(state)` — 荣和
- `_tsumo(state)` — 自摸
- `_pon(state, action)` — 碰
- `_chi(state, action)` — 吃
- `_open_kan(state)` — 大明杠
- `_selfkan(state, action, is_added)` — 暗杠/加杠
- `_pass(state)` — 过
- `_kyuushu(state)` — 九种九牌
- `_dummy(state)` — dummy 步
- `_make_legal_action_mask_after_draw(state)` — 摸牌后合法动作
- `_make_legal_action_mask_after_discard(state)` — 舍牌后合法动作
- `_make_legal_action_mask_after_draw_riichi(state, cp)` — 立直后合法动作
- `_settle_ron/tsumo` — 结算
- `_flip_dora` — 翻宝牌
- `_draw_after_kan` — 杠后摸牌
- `_abortive_draw_normal` — 流局
- `_advance_to_next_round_auto` — 局推进
- `_finalize_game` — 游戏结束

### 2. `env_parallel.py` — Pure Parallel Env

**职责**：batch-first、全向量化的生产环境实现。

**核心类**：`RedMahjongParallel(Env)`

```python
class BatchState:
    """批量化状态 — 所有字段形状为 (B, ...) 或 (B, 4, ...)。

    将 B 个独立环境的 EnvState 打包为单个批量化结构。
    所有操作在此结构的张量上进行，避免 Python 循环。
    """
    B: int
    current_player: torch.Tensor        # (B,) int
    legal_action_mask: torch.Tensor     # (B, 87) bool
    players: BatchPlayerState           # (B, 4, ...) 结构
    round_state: BatchRoundState        # (B,) 结构
    step_count: torch.Tensor            # (B,) int
    rewards: torch.Tensor               # (B, 4) float
    terminated: torch.Tensor            # (B,) bool
    truncated: torch.Tensor             # (B,) bool

class RedMahjongParallel(Env):
    """纯并行版麻将环境。

    所有方法在 batch 维度上并行操作。
    控制流分歧通过 boolean mask 解决。
    坚持 eager 模式，不使用 torch.compile。
    """

    def init_batch(self, keys=None, num_envs=None) -> BatchState:
        """批量初始化 B 个游戏状态。"""

    def step_batch(self, states: BatchState, actions: torch.Tensor) -> BatchState:
        """批量执行 B 个环境的步骤。

        actions: (B,) int32 — 每个环境的动作

        实现策略：
        1. 按动作类型分组 (discard, tsumogiri, riichi, ron, tsumo,
           pon, chi, open_kan, selfkan, pass, kyuushu, dummy)
        2. 每组通过 mask 选择环境子集
        3. 对每个子集调用专门的 batch handler
        4. 合并结果回主 BatchState
        """

    def observe_batch(self, states: BatchState) -> dict:
        """批量构建观察。"""
```

**关键设计决策**：

#### 状态批量化
- `EnvState` → `BatchState`：所有字段从标量/单环境张量提升为 batch-first 张量
- `PlayerStateArrays` → `BatchPlayerState`：`(4, N)` → `(B, 4, N)`
- `RoundState` → `BatchRoundState`：标量 → `(B,)` 张量

#### 控制流处理
- 所有 `if/else` 分支通过 boolean mask + tensor indexing 实现
- 动作分发：`mask = (action_type == X)`，`states = handler(states, mask)`
- 优先级逻辑（ron > kan > pon > chi）：通过 argsort 替代串行查找

#### 批量操作模式
每个动作处理器遵循统一模式：
```python
def _handler_batch(self, batch_state, action_mask):
    """action_mask: (B,) bool — 哪些环境执行此动作"""
    if not action_mask.any():
        return batch_state

    # 1. 提取子集（如果需要写回，使用 scatter）
    idx = action_mask.nonzero().squeeze(-1)  # (K,)

    # 2. 批量计算（全部在 GPU 上）
    result = vectorized_op(batch_state, idx, ...)

    # 3. Scatter 回主状态
    batch_state = scatter_update(batch_state, idx, result)
    return batch_state
```

#### Eager Mode 保证
- 不调用 `torch.compile`
- 不调用 `torch.jit.script/trace`
- 所有操作即时执行
- 使用 `torch.where`, `torch.gather`, `torch.scatter` 等原生 eager op

### 3. `env.py` — 兼容层 (Facade)

```python
class RedMahjong(Env):
    """向后兼容的环境类。

    根据 backend 参数委托到序列化或并行实现。
    默认使用 serial（用于测试），可通过 backend='parallel' 切换。
    """

    def __init__(self, backend='serial', **kwargs):
        if backend == 'serial':
            self._impl = RedMahjongSerial(**kwargs)
        elif backend == 'parallel':
            self._impl = RedMahjongParallel(**kwargs)

    def step(self, state, action, key=None):
        return self._impl.step(state, action, key)

    def step_batch(self, states, actions, profile=False):
        return self._impl.step_batch(states, actions, profile)
```

### 4. `batch_state.py` — 批量化状态定义

新增模块，定义 `BatchState`, `BatchPlayerState`, `BatchRoundState` 及转换函数：

```python
def stack_states(states: List[EnvState]) -> BatchState:
    """将 B 个独立 EnvState 打包为 BatchState。"""

def unstack_state(batch_state: BatchState, index: int) -> EnvState:
    """从 BatchState 中提取第 index 个 EnvState。"""
```

## Data Flow

### Serial Path (正确性验证)
```
test_env_serial_compare.py
  │
  ├─→ env_serial.init(key) → EnvState
  ├─→ env_serial.step(state, action) → EnvState
  │     └─→ 逐函数与 JAX 对比 (相同输入 → 相同输出)
  └─→ assert torch.allclose(pt_result, jax_result)
```

### Parallel Path (训练)
```
ppo_with_reg.py
  │
  ├─→ make(backend='parallel', ...)
  ├─→ env.init_batch(keys, num_envs=128) → BatchState
  ├─→ network.forward(batch_state) → actions, log_probs, values
  └─→ env.step_batch(batch_state, actions) → BatchState
        │
        ├─→ 按 action type 分组 (O(1) via bucketize)
        ├─→ _discard_batch(states, discard_mask)   # ~90%
        ├─→ _tsumo_batch(states, tsumo_mask)       # ~5%
        ├─→ _pass_batch(states, pass_mask)         # ~3%
        ├─→ _ron_batch(states, ron_mask)           # ~1%
        └─→ ... (其他稀有动作)
```

## Verification Strategy

详细的测试方法、脚本清单、金数据工具链、并行执行模式、覆盖率矩阵和回归种子策略见 [tasks.md § 测试方法](tasks.md#测试方法)。以下为架构层面的验证策略摘要：

### 四层验证金字塔

| 层级 | 目标 | 关键测试 |
|------|------|---------|
| L1 基础单元 | 纯 PT 逻辑正确性 | `test_cases.py` — 16 组 80 断言 |
| L2 框架一致性 | JAX ↔ PyTorch 数值等价 | `test_exact_parity.py`, `test_full_ppo_parity.py`, `test_mha_definitive.py` |
| L3 环境集成 | Serial↔Parallel 等价 + 关键分支 | `test_env_parallel_parity.py`, `test_env_branches.py` |
| L4 金数据回放 | 完整对局 vs JAX 逐 step 一致 | `replay_pt_against_golden.py`, `replay_parallel_against_golden.py` — 805 seeds 100% 通过 |

### 核心验证原则

1. **串行 vs JAX 逐函数对比**：相同随机种子，单步执行，比较每次 step 后的状态字段（hand, river, meld, score, legal_action_mask, rewards 等）
2. **并行 vs 串行等价性**：相同随机种子初始化 B 个环境，并行 step_batch 结果 vs 逐环境串行 step 结果，全字段 exact match (float 允许 1e-5 误差)

性能基准、GPU 兼容性、批量回放验证详情见 [tasks.md § 测试方法](tasks.md#测试方法)。

## File Structure After Refactor

```
mahjax_pt/red_mahjong/
├── env.py                      # 兼容层 (Facade, 127 行)
├── env_serial.py               # 纯串行环境 (~1,554 行, 参考实现)
├── env_parallel.py             # 调度层 (258 行) — RedMahjongParallel orchestrator
├── env_parallel_handlers.py    # 动作处理器 Mixin (1,180 行) — 11 action handlers
├── env_parallel_internals.py   # Mask/结算/Yaku Mixin (650 行) — 内部机制
├── batch_state.py              # 批量化状态定义 (369 行)
├── state.py                    # EnvState, PlayerStateArrays, RoundState (不变)
├── hand.py                     # 手牌操作 (chi_batch/open_kan_batch/closed_kan_batch 等)
├── meld.py                     # 面子编码 (已有 batch 版本)
├── shanten.py                  # 向听计算 (已有 batch 版本)
├── yaku.py                     # 役种判定 (judge_hand_related_batch, GPU device-aware 常量)
├── tile.py                     # 牌面操作 (已有 batch 版本)
├── action.py                   # 动作常量 (不变)
├── constants.py                # 常量 (不变)
├── observation.py              # 观察构建 (不变)
├── players.py                  # 玩家策略 (不变)
└── auto_reset_wrapper.py       # 自动重置 (不变)
```

### Architecture (Mixin Split)

```
RedMahjongParallel(HandlersMixin, InternalsMixin, Env)
  │
  ├── env_parallel.py (258 lines)
  │   └── Orchestrator: __init__, init_batch, step_batch, _step_batch_bs
  │
  ├── env_parallel_handlers.py (1,180 lines)
  │   └── HandlersMixin: _discard_batch, _pon_batch, _chi_batch,
  │       _selfkan_batch, _open_kan_batch, _ron_batch, _tsumo_batch,
  │       _riichi_batch, _pass_batch, _kyuushu_batch, _dummy_batch,
  │       _accept_riichi_batch, _draw_batch, _draw_after_kan_batch
  │
  └── env_parallel_internals.py (650 lines)
      └── InternalsMixin: _make_legal_mask_after_discard_batch,
          _make_legal_mask_after_draw_batch, _precompute_yaku_batch,
          _settle_ron_batch, _settle_tsumo_batch,
          _abortive_draw_normal_batch, _advance_round_batch,
          _copy_state_into_batch, _score_batch
```
