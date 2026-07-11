# Serial Env Specification

## Overview

纯串行版麻将环境 `RedMahjongSerial`，每次调用 `step(state, action)` 处理恰好一个环境的一个步骤。代码结构与 JAX `mahjax/red_mahjong/env.py` 保持 1:1 对应。

## Requirements

### SE-01: 单环境初始化
- `init(key=None)` 返回 `EnvState`
- key 支持 `torch.Generator` 或 `int` seed
- 生成洗牌后的牌墙、发牌、庄家第一摸
- 设置 `legal_action_mask` 和 `shanten_current_player`

### SE-02: 单步执行 (step)
- `step(state, action, key=None)` → `EnvState`
- 动作分发到对应 handler：`_discard` / `_riichi` / `_ron` / `_tsumo` / `_pon` / `_chi` / `_open_kan` / `_selfkan` / `_pass` / `_kyuushu` / `_dummy`
- 非法动作：调用 `_step_with_illegal_action` 终止游戏
- 已终止状态：返回零 reward
- 自动处理局间过渡 (`_advance_to_next_round_auto`)

### SE-03: 摸牌 (_draw)
- 处理立直接受 (`_accept_riichi`)
- 检查特殊流局（四风连打、四家立直）
- 从牌墙摸牌、更新 `last_draw` / `last_player`
- 构建合法动作 mask（区分立直后和非立直后）
- 更新向听数、清除振听被动标记

### SE-04: 舍牌 (_discard)
- 从手牌移除 tile
- 更新河 (River) 和舍牌列表
- 标记手摸切 (tsumogiri) 在行动历史中
- 振听检测：听牌时舍出待牌 → 设置振听
- 构建其他玩家的鸣牌/荣和 mask
- 确定下一个行动玩家（优先级：ron > open_kan > pon > chi）

### SE-05: 立直 (_riichi)
- 只允许舍出后仍听牌的 tile
- 设置 `riichi_declared` 标记
- 实际支付在下一轮 `_accept_riichi` 中处理

### SE-06: 荣和 (_ron)
- 役种判定 + 附加役种（一发、双立直、枪杠、河底）
- 结算支付、供托奖励
- 标记 `has_won` 和 `terminated_round`

### SE-07: 自摸 (_tsumo)
- 同荣和逻辑，但支付模式为自摸分摊
- 附加役种（一发、双立直、岭上开花、海底）

### SE-08: 鸣牌 (_pon, _chi, _open_kan)
- PON: 碰牌、打包面子、更新手牌
- CHI: 吃牌（仅上家非字牌）、打包面子
- OPEN_KAN: 大明杠、翻宝牌、杠后摸牌

### SE-09: 暗杠/加杠 (_selfkan)
- 暗杠 (closed_kan) 和加杠 (added_kan) 区分
- 打包面子、翻宝牌、杠后摸牌

### SE-10: 其他动作
- `_pass`: 过、设置振听被动、找下一个有行动的玩家
- `_kyuushu`: 流局重开（JAX `_special_next_round`）— 同庄、honba+1、保留分数、重新洗牌发牌。不同于通常的 `terminated_round`
- `_dummy`: 过渡局 dummy 步

### SE-11: 局管理
- `_flip_dora`: 杠后翻宝牌指示牌
- `_draw_after_kan`: 岭上摸牌
- `_abortive_draw_normal`: 荒牌流局结算（听牌/不听）
- `_advance_to_next_round_auto`: 局推进、庄家续行判定、终局判定
- `_finalize_game`: 最终顺位点结算

### SE-12: 合法动作 Mask 构建
- 摸牌后 mask: 舍牌 (37 种) + 手摸切 + 暗杠/加杠 (34 种) + 自摸 + 立直 + 九种九牌
- 九种九牌条件（JAX 对齐）：`enable_special_abortive_draw` ∧ `is_first_turn` ∧ `can_kyuushu` ∧ `meld_counts.sum() == 0`
- 舍牌后 mask: 荣和 + 碰/吃/大明杠 + 过
- 四開槓流れ（JAX 对齐）：`enable_special_abortive_draw & had_after_kan & n_kan>=4 & >=2 players have kan & no_ron` → 仅 KYUUSHU 可选
- `is_abortive_draw_normal` 条件设置：仅当 `no_meld_player | (is_abort & no_ron_player)` 为真
- 立直后 mask: 仅限听牌舍牌 + 不变向听的暗杠

### SE-13: 代码标注
- 每个方法标注 JAX 对应行号范围
- 格式：`# JAX: env.py L800-L820`
- 关键循环/条件与 JAX 一一对应

### SE-14: 观察构建
- `observe(state)` → dict
- 与当前 `_observe_dict` 兼容
- 包含 hand, action_history, shanten, furiten, scores 等字段
