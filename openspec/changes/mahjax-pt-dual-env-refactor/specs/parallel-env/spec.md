# Parallel Env Specification

## Overview

纯并行版麻将环境 `RedMahjongParallel`，所有操作在 batch 维度上并行处理 B 个环境。坚持 eager 模式，不使用 JIT 编译。控制流分歧通过 boolean mask 张量解决。

## Requirements

### PE-01: 批量状态 (BatchState)
- `BatchState` 包含 B 个环境的全部状态
- 所有字段形状为 `(B, ...)`, `(B, 4, ...)` 或 scalars
- 提供 `stack_states(List[EnvState]) → BatchState` 转换
- 提供 `unstack_state(BatchState, int) → EnvState` 转换
- 状态字段与 `EnvState` 语义等价

### PE-02: 批量初始化 (init_batch)
- `init_batch(keys=None, num_envs=B)` → `BatchState`
- 批量生成洗牌后的牌墙 `(B, 136)`
- 批量发牌 `(B, 4, 13)`
- 批量庄家第一摸
- 批量设置 `legal_action_mask` `(B, 87)`
- keys 支持 `List[torch.Generator]` 或 `None`（自动生成）

### PE-03: 批量步进 (step_batch)
- `step_batch(batch_state, actions)` → `BatchState`
- `actions`: `(B,)` int32
- 自动按动作类型分组处理
- 已终止环境跳过（mask 过滤）
- 自动局间过渡（auto 模式）
- 所有分支通过 mask + indexing 实现，零 Python 循环

### PE-04: 批量动作分发
- 将 actions 按类型分为最多 12 组：
  - `discard` (0-36): 普通舍牌
  - `tsumogiri` (71): 手摸切
  - `selfkan` (37-70): 暗杠/加杠
  - `riichi` (72): 立直
  - `ron` (73): 荣和
  - `tsumo` (74): 自摸
  - `pon/pon_red` (75-76): 碰
  - `open_kan` (77): 大明杠
  - `chi` (78-83): 吃
  - `pass` (84): 过
  - `kyuushu` (85): 九种九牌
  - `dummy` (86): dummy
- 每组通过专门的 batch handler 处理
- 无动作类型回退到串行（全部批处理）

### PE-05: 批量摸牌 (_draw_batch)
- 批量立直接受
- 批量特殊流局检查
- 批量摸牌：`(B,)` 牌墙索引 → `(B,)` 新牌
- 批量更新 `last_draw`, `last_player`, `next_deck_ix`
- 批量手牌添加（scatter add）
- 批量 mask 构建（区分 riichi 和非 riichi）
- 批量向听计算

### PE-06: 批量舍牌 (_discard_batch)
- 批量手牌减法（scatter subtract）
- 批量河更新（`add_discard_batch`）
- 批量行动历史更新
- 批量振听检测
- 批量构建 4 玩家鸣牌/荣和 mask（全张量运算）
- 批量确定下一个行动玩家和 draw_next 标志

### PE-07: 批量鸣牌/荣和 Mask 构建
- RON: `Hand.can_ron_batch(hands_37_b, target_tts_b)` → `(B, 4)` bool
- CHI: `Hand.can_chi_matrix_batch_4p(...)` → `(B, 4, 6)` bool
- PON: `Hand.can_no_red_pon_batch_4p(...)`, `Hand.can_red_pon_batch_4p(...)` → `(B, 4)` bool
- OPEN_KAN: `Hand.can_open_kan_batch_4p(...)` → `(B, 4)` bool
- 优先级确定：argsort 替代串行搜索
- PASS: 任何有行动的玩家自动可用

### PE-08: 批量稀有动作
每种稀有动作实现 batch handler：
- `_riichi_batch`: 批量计算立直后合法 mask
- `_ron_batch`: 批量役种判定 + 批量结算
- `_tsumo_batch`: 批量役种判定 + 批量自摸分摊结算
- `_pon_batch`: 批量碰牌
- `_chi_batch`: 批量吃牌
- `_open_kan_batch`: 批量大明杠 + 翻宝牌
- `_selfkan_batch`: 批量暗杠/加杠 + 翻宝牌
- `_pass_batch`: 批量过牌 + 振听设置
- `_kyuushu_batch`: 批量九种九牌
- `_dummy_batch`: 批量 dummy 步

### PE-09: 批量局管理
- `_flip_dora_batch`: 批量翻宝牌
- `_draw_after_kan_batch`: 批量岭上摸牌
- `_abortive_draw_normal_batch`: 批量荒牌流局
- `_advance_to_next_round_batch`: 批量局推进/终局判定
- `_finalize_game_batch`: 批量终局顺位点

### PE-10: 批量观察构建
- `observe_batch(batch_state)` → `dict` of `(B, ...)` tensors
- 批量构建 hand, action_history, shanten 等观察字段

### PE-11: Eager Mode 保证
- 不使用 `torch.compile`
- 不使用 `torch.jit.script/trace`
- 不使用 `jax.jit`、`jax.vmap`
- 所有操作即时执行
- 使用原生 PyTorch eager ops

### PE-12: 可恢复性
- 支持从 `List[EnvState]` 构建 `BatchState`
- 支持将 `BatchState` 拆解为 `List[EnvState]`
- 用于 debug 时：并行运行 → 发现异常环境 → 拆解为串行 → 单步调试

### PE-13: Profiling 支持
- `step_batch(..., profile=True)` 输出各阶段耗时
- 分类统计：动作分组耗时、各 handler 耗时
- 与当前 `ppo_with_reg.py` 的 profile 兼容
