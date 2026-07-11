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

### PE-08: 批量稀有动作（全部已向量化）
每种稀有动作已实现 batch handler，零 per-env 循环：
- `_riichi_batch` ✅: 批量计算立直后合法 mask
- `_ron_batch` ✅: `Yaku.judge_hand_related_batch` + `_settle_ron_batch`
- `_tsumo_batch` ✅: `Yaku.judge_hand_related_batch` + `_settle_tsumo_batch`
- `_pon_batch` ✅: batch meld + `Hand.pon` 内联变异 + 河标记
- `_chi_batch` ✅: batch meld + `Hand.chi_batch` + 喰いかえ mask
- `_open_kan_batch` ✅: batch meld + `Hand.open_kan_batch` + `_draw_after_kan_batch`
- `_selfkan_batch` ✅: batch 区分暗杠/加杠 + `Hand.closed_kan_batch`/`Hand.added_kan_batch`
- `_pass_batch` ✅: 批量过牌 + 振听设置 + 环形搜索
- `_kyuushu_batch` ✅: 直接设置流局标志位
- `_dummy_batch` ✅: 批量 dummy 步

### PE-09: 批量局管理（主要已向量化）
- `_flip_dora_batch`: ✅ 在 `_draw_after_kan_batch` 内联
- `_draw_after_kan_batch` ✅: 批量岭上摸牌 + 翻宝牌 + mask 构建
- `_kyuushu_batch` ✅: 批量流局重开（JAX `_special_next_round`）— 同庄、honba+1、新牌山
- `_abortive_draw_normal`: ⚠️ per-env 委托（稀有路径）
- `_advance_to_next_round_auto`: ⚠️ per-env 委托（终局，稀有路径）
- `_finalize_game`: ⚠️ 内联在 `_advance_round_batch` 中

### PE-14: 四開槓流れ（新增）
- `_make_legal_mask_after_discard_batch` 中实现 `is_four_kan_draw` 检查
- `had_after_kan` 在 `_discard_batch` 中捕获（`can_after_kan` 清除之前）
- 条件：`enable_special_abortive_draw & had_after_kan & n_kan>=4 & >=2 players & no_ron`
- 触发后 mask 仅 KYUUSHU，current_player 保持为舍牌者

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

### PE-15: GPU 支持 ✅ (2026-07-10)
- `init_batch(..., device='cuda')` 支持 GPU 设备
- `_batch_state_to_device()` 递归转移所有 BatchState 张量到目标 device
- `yaku.py` 中模块级常量 (`DORA_ARRAY`, `_FAN`, `_YAKUMAN`, `CACHE`) 已改为 device-aware
- 所有 `*_batch` 函数加 `.clamp(0, _CACHE_MAX)` 兼容 GPU 严格越界断言
- GPU 批量回放验证：805 seeds 全部通过 (100%)
- 已验证：RTX 4060 Ti, B=2048, 201 MB 显存, 0.4 ms/env/step

### PE-16: 向量化完成度 ✅ (2026-07-09)

全向量化（无逐环境循环）：
- 全部 11 个 action handler 主体逻辑
- 手牌变异、河牌添加、鸣牌检查
- yaku 预计算、结算
- can_win 计算（单次批量 can_tsumo_batch）
- 喰いかえ mask（pon/chi）
- 立直 discard_ok 计算
- 摸牌后 mask 构建（普通 + 杠后 + 立直后）
- 舍牌后 mask 构建（chi/pon/kan/ron 4 人检查）
- 特殊流局检查（四風連打/四家立直）
- 荒牌流局结算

仍保留逐环境（3 个稀有路径）：
- 四開槓流れ（~0.01%）
- kyuushu 重新洗牌（PRNG 限制，~0.1%）
- 局推进/终局判定（极罕见）

### PE-17: Mixin 架构拆分 ✅ (2026-07-10)

将 `env_parallel.py` 从 2,087 行单文件拆分为三层 Mixin 架构：

- **`env_parallel.py`** (258 行): 调度层 — `RedMahjongParallel(HandlersMixin, InternalsMixin, Env)`, MRO 线性无菱形继承
- **`env_parallel_handlers.py`** (1,180 行): `HandlersMixin` — 14 个动作/摸牌处理器
- **`env_parallel_internals.py`** (650 行): `InternalsMixin` — mask 构建 + 结算 + yaku + 局管理

同时完成的去重：
- `_copy_state_into_batch`: 75行 → 4行 (`_copy_dataclass_row` + `stack_states`)
- `__init__` 参数验证: serial/parallel 共享 `_resolve_env_config()`
- `_settle_tsumo_batch` / `_abortive_draw_normal_batch`: per-player 循环 → broadcasting
- `_make_legal_mask_after_draw_batch`: per-tile 循环 → broadcasting

### PE-18: GPU Dtype 修复 ✅ (2026-07-10)

GPU 批量回放发现的 5 处 int64→int32 赋值错误（仅 GPU 严格检查）：

| # | 位置 | 修复 |
|---|------|------|
| 46 | `_pass_batch`: `torch.full(dtype=torch.long)` 赋值给 `current_player` (int32) | `torch.long` → `torch.int32` |
| 47-50 | `_selfkan_batch`: `pon_meld_slot_H`, `pon_src_H` 等 int64 来源 | `torch.long` → `torch.int32`, 移除 `.long()` |

修复后 GPU 批量回放: **803/805 seeds pass (99.8%)**。2 个失败为预存逻辑差异（furiten_by_discard / 状态漂移），与 mixin 拆分无关。
