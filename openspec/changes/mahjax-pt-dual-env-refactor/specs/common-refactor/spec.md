# Common Refactor Specification

## Overview

将 `env.py` 中共用的辅助函数和状态管理逻辑规范化，使 serial 和 parallel 版本可以独立引用而不重复。

## Requirements

### CR-01: 辅助函数模块化
以下函数从 `env.py` 提取到独立位置（或标记为共享工具）：
- `_resolve_game_config(game_config)` → 已有 `state.py`
- `_live_wall_end_ix(state)` → 工具函数
- `_set_tile_type_action(mask, tile_type, value)` → 工具函数
- `_has_red_discard_action(mask)` → 工具函数
- `_special_abortive_draw_mask()` → 工具函数
- `_trigger_special_abortive_draw(state)` → 工具函数
- `_append_meld_to_player(state, meld, player, discard_idx, src)` → 工具函数
- `_accept_riichi(state)` → 工具函数（串行版），batch 版在 parallel 中
- `_is_waiting_tile(can_ron, tile)` → 工具函数
- `_calc_wind(east_player)` → 工具函数
- `_is_first_turn(next_deck_ix)` → 工具函数
- `_append_action_history(state, action)` → 工具函数
- `CHI_ACTIONS` → 常量，已有 `action.py`

### CR-02: 动作分发常量
- `ACTION_FUN_MAP`：(87,) int32 映射 action→handler index
- 在串行版用于 switch/if-elif，在并行版用于 bucketize

### CR-03: 兼容层 (env.py)
- `RedMahjong` 类保持为 Facade
- `make(env_name, backend='serial', **kwargs)` 工厂函数
- 支持 `backend='serial'` 或 `backend='parallel'`
- 自动选择对应的实现类
- 接口兼容现有 `auto_reset_wrapper.py` 和 `ppo_with_reg.py`

### CR-04: 测试基础设施
- JAX 对比测试框架：加载 JAX 环境和 PT 串行环境，相同 seed 运行对比
- 并行 vs 串行等价性测试框架：B 个串行环境 vs 1 个并行环境 (B envs)
- 性能基准测试框架：测量串行/并行/混合的 throughput
