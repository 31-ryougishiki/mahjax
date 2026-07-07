# Phased Development Plan

## Phase 0: 准备工作 ✅

- [x] **P0-1**: 确认模块状态 — 80 tests 全通过
- [x] **P0-2**: 分支 `feat/dual-env-refactor`
- [x] **P0-3**: JAX 参考代码 + yaku cache 验证

## Phase 1: 公共逻辑抽取 + 兼容层 ✅

- [x] `env_serial.py` — 纯串行环境 (~1200 行)
- [x] `env_parallel.py` — 纯并行环境 (~680 行)
- [x] `batch_state.py` — 批量化状态
- [x] `env.py` → Facade 兼容层 (120 行)
- [x] `cpu_env.py` — JAX 纯 CPU 模式 (jax_disable_jit)

## Phase 2-3: 串行/并行实现 + 测试 ✅

- [x] 基本功能完整
- [x] 并行 vs 串行等价性测试 (7/7 通过)
- [x] 多 seed JAX 对齐测试框架 (`test_multi_seed.py`)

## Phase 4: 集成 ✅

- [x] `auto_reset_wrapper` 兼容
- [x] `ppo_with_reg.py` import 正常

---

## JAX vs PT 对齐 — Bug 修复记录

基于 10 个 seed 的完整对比测试，共发现并修复以下 bug：

### 已修复 (13 个)

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 1 | `_append_meld_to_player` | meld marker 写到鸣牌者而非被鸣者的河，src 用绝对位置而非 `(discarder-caller)%4` | PON/CHI 后 river 分叉 |
| 2 | `_pon` / `_chi` | `Meld.init(src=1)` 硬编码，应为相对位置 | meld 值编码不一致 |
| 3 | `_draw` | 多余 `last_player=cp`，JAX 不设 | 摸牌后 player 轮转偏移 |
| 4 | `_draw_after_kan` | 同上，多余 `last_player=cp` | 杠后 player 偏移 |
| 5 | `yaku.py` L581 | `~` (位取反) 误用作逻辑取反 → 断幺九始终 True | PT 稳定多 1 翻 |
| 6 | `_discard` | 缺少 yaku 预计算 (`_precompute_yaku`) 调用 | has_yaku 始终为 False |
| 7 | `_precompute_yaku` | 传 14 张手牌而非 13 张+state target | 双重计算赢张 |
| 8 | `_precompute_yaku` | 列顺序反了 (col0=TSUMO col1=RON → col0=RON col1=TSUMO) | mask 判断反 |
| 9 | `_precompute_yaku` | RON 没加舍牌到手牌 (只剩 13 张) | yaku 计算用错手牌 |
| 10 | `_precompute_yaku` | 临时 state target/last_draw 未对齐 JAX 的 ron_state/tsumo_state | Yaku.judge 读错赢张 |
| 11 | `_pon`/`_chi`/`_open_kan`/`_selfkan` | meld 后 target 未重置为 -1 | 后续 pon/chi 用错 target |
| 12 | `_discard` | `target = to_tile_type(tile)` 转成了 34-type，JAX 存原始 tile | 红五 target 值不同 |
| 13 | `_precompute_yaku` | 用 `fan>0` 而非 `yaku_vec.any()` 判断 has_yaku (fan 含 dora，dora 不算役) | has_yaku 误判 |
| 14 | `yaku.py` L349 | Pinfu 运算符优先级：`&` > `|` 导致 `pung==0` 被短路 | 平和误判 |
| 15 | `yaku.py` L529 | Fu 开牌修正用 `.any()` 而非 per-pattern | Fu 可能偏高 |
| 16 | `yaku.py` L702 | Nine Gates 缓存只读 `[0]` 列 | 漏掉其他分解模式 |
| 17 | `_abortive_draw_normal` | 流局结算公式 3000*n_noten//100 而非 JAX 的 30//n_tenpai | 分数差 3 倍 |
| 18 | `_precompute_yaku` | `nxt<lst` 时 early return，JAX 在牌山耗尽时仍计算 | 末巡 has_yaku 缺失 |
| 19 | `step()` | `is_added` 用 `Hand.can_added_kan` (检查手牌)，JAX 检查 meld 列表 PON | 加杠/暗杠误判 |
| 20 | `_selfkan` | 加杠追加新面子而非替换 PON | meld_counts 不一致 |
| 21 | `_selfkan` | 多余 `last_player=cp` | 杠后 player 偏移 |
| 22 | `n_kan` 递增时机 | PT 在 `_selfkan` 内递增，JAX 在 `_draw_after_kan` 读 rinshan **后** 递增 | rinshan tile 偏移 |
| 23 | `step()` | `last_draw=-1` / `ippatsu=False` / `can_after_kan=False` 重置缺失 | 状态漂移 |

### 待确认 (1 个)

| # | 位置 | 现象 |
|---|------|------|
| 24 | step 84 `is_haitei` | selfkan 修复后，流局前 is_haitei 差 1 步 — 应在 `_draw` 而非 `_discard` 后设置 |

---

## 测试覆盖

### 动作类型覆盖

| 动作 | 覆盖？ | 备注 |
|------|-------|------|
| discard | ✅ | 主路径 |
| pon | ✅ | |
| chi | ✅ | |
| selfkan (closed) | ✅ | 新策略触发 |
| selfkan (added) | ⚠️ | 未触发，需先有 PON |
| open_kan | ⚠️ | 条件苛刻 |
| riichi | ⚠️ | 需手牌听牌 |
| tsumo | ⚠️ | 需手牌能赢 |
| ron | ⚠️ | 需鸣牌阶段出现 |
| pass | ⚠️ | 鸣牌阶段 |
| kyuushu | ⚠️ | 仅第一巡 |

### 测试脚本

| 脚本 | 用途 | 用法 |
|------|------|------|
| `test_multi_seed.py` | 多 seed 对齐主测试 | `python test_multi_seed.py -j 4` |
| `verify_jax_pt_alignment.py` | 单 seed 128 步详细测试 | `python verify_jax_pt_alignment.py` |
| `test_env_parallel_parity.py` | 并行 vs 串行等价性 | `python test_env_parallel_parity.py` |
