# Bug 修复记录

基于 250+ 个 seed 的多进程 JAX 金数据回放验证，共修复 45 个 bug。

## Phase 1-4: 串行环境 (23 个)

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

## Phase 5: 向量化 (8 个)

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 24 | `can_tsumo_batch` | honor code 错误编码到 suit cache 查询，超出 cache 边界被 clamp | batch 版 can_tsumo 偶发漏判 |
| 25 | `can_chi_matrix_batch_4p` | 红五 chi 检测：`base_ok` 要求普通五存在 + `has_red` 检查红五 → 应允许红五**替代**普通五 | CHI_RED 选项漏判 |
| 26 | `can_chi_matrix_batch_4p` | `red_ok & ~base_ok` 阻止普通五+红五同时可用 → 应双双可用 | CHI_RED 被错误排除 |
| 27 | `can_no_red_pon_batch_4p` | 使用 `to_34_batch` 计数（含红五合并），JAX 用 37-type 计数（仅普通五） | PON 选项误判 |
| 28 | `_make_legal_mask_after_discard_batch` | ron 检测用缓存 `can_win`，非舍牌者可能过期 → 应使用 `Hand.can_ron_batch` 实时计算 | RON 选项漏判 |
| 29 | `torch.argmax` 返回 | `torch.argmax` 返回 int64，assign 到 int32 张量后 dtype 不匹配 | 运行时崩溃 |
| 30 | `_draw_after_kan_batch` 参数名 | `is_closed_kan` → `pre_flip_dora` | 运行时崩溃 |
| 31 | `can_kyuushu` (serial + parallel) | 缺少 `meld_counts.sum() == 0` 条件（JAX 要求全场无人鸣牌过） | KYUUSHU 在非第一巡误出现 |

## Phase 6: 新种子验证 (3 个)

基于 100 个新种子 (10000-10099) 的多进程验证发现并修复：

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 32 | `_discard` / `_make_legal_action_mask_after_discard` (serial + parallel) | `is_abortive_draw_normal` 无条件设置；JAX 仅在 `no_meld_player \| (is_abort & no_ron_player)` 时条件设置 | 流局标记过早出现 |
| 33 | `_make_legal_action_mask_after_discard` (serial + parallel) | 缺少 `is_four_kan_draw`（四開槓流れ）检查；`had_after_kan` 需在清除 `can_after_kan` **之前**捕获 | 四杠流局无法触发 |
| 34 | `_kyuushu` (serial + parallel) | KYUUSHU 语义错误：JAX 映射到 `_special_next_round`（同庄、honba+1、重新洗牌发牌）；PT 错误设为 `terminated_round=True` | 九种九牌/四杠流局后续对局无法继续 |
| — | `_trigger_special_abortive_draw` (serial) | `legal_action_mask` 被赋值为 2D `(4, 87)` 而非 1D `(87,)` | 特殊流局 mask 形状错误 |

## Phase 8: GPU 批量回放 (6 个)

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 40 | `tile.py:add_discard_batch` | `batch_idx = arange(rivers.shape[0])` 假定全批量 B，调用方传入子集 K 时索引广播失败 | GPU 批量回放崩溃 |
| 41 | `tile.py:add_meld_batch` | 同上 | GPU 批量回放崩溃 |
| 42 | `hand.py:chi_batch` L448 | `full_rm_idx[red_idx_mask][has_red]` 将环境索引当作子集内位置索引使用 | GPU 越界断言 |
| 43 | `env_parallel.py:_tsumo_batch` | 重新调用 `judge_hand_related_batch`，导致 `last_draw` 重复加牌（14→15 张） | TSUMO 结算 fan/fu 计算错误 |
| 44 | `env_parallel.py:_kyuushu_batch` | 缺少 `has_yaku`/`fan`/`fu`/`can_win`/`legal_action_mask` 重置 | kyuushu 后 yaku 预计算残留 |
| 45 | `env_parallel.py:_pon_batch` | kuikae mask 对红五额外禁止（JAX 只禁止原始 target 值，不转换 `to_tile_type`） | PON 后 mask 缺少红五 discard 选项 |

## GPU 兼容性 (5 个)

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 35 | `yaku.py` L540 | `DORA_ARRAY[dt.long()]`：CPU 常量被 GPU 张量索引 | `DORA_DEV = DORA_ARRAY.to(device)` |
| 36 | `yaku.py` L746 | `torch.where(condition_gpu, _FAN_cpu, _FAN_cpu)` 混合 device | `FAN_DEV = _FAN.to(device)` |
| 37 | `yaku.py` L808 | `int32 @ float32` matmul 不支持 CUDA | 转 `.float()` 再 matmul |
| 38 | `yaku.py` L389-413 | `Yaku.CACHE[codes]`：CPU 常量被 GPU 张量索引 | `_get_cache(device)` 按 device 懒缓存 |
| 39 | `yaku.py` L389-413 | GPU 严格断言越界索引（CPU 静默忽略） | `codes.clamp(0, _CACHE_MAX)` |

## Phase 10+: 串行回放验证 (2 个，2026-07-10)

基于串行环境 `replay_pt_against_golden.py` 回放 seed 99 和 512 发现的遗留逻辑 bug：

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 51 | `env_serial.py:_discard` L711 | `furiten_by_discard` 河流检查 `rt < 34` 排除红五 (34/35/36)，导致红五在河中但 `can_ron` 以 tile type 索引时漏检 | 玩家听红五所在 tile type 时 furiten 漏判 |
| 52 | `env_serial.py:_pass` L1175-1183 | PASS 时未清除 pass 玩家的 `legal_action_mask`，后续搜索时其 mask 仍有效，导致 meld 协商陷入循环（cp 在玩家间无意义轮转）| 鸣牌协商阶段 pass 后 current_player 错误、target 不清、draw 不触发 |

**同时修复了并行环境中的相同问题**：
- `env_parallel_handlers.py:_discard_batch` L739: `rt < 34` → `rt <= 36` + `Tile.to_tile_type_tensor` 转换
- `env_parallel_handlers.py:_pass_batch`: 添加 mask 清零逻辑；改用 JAX 优先级搜索 (RON > OPEN_KAN > PON > CHI)；无响应者时清理 target

**串行验证**：18/18 FULL_COVERAGE seeds 通过（含 seed 99 和 512）

## 性能分析与 Yaku 优化 (Phase 11-12)

> 性能打点基础设施（`_perf`）和瓶颈分析 → [tasks.md](tasks.md) Phase 12。
> 方案 A/C 最初因正确性问题被 revert（#53, #54），后通过 dump-验证-优化工作流
> 在 Phase 12 中成功实现等效优化（yaku 4p 批量 + RON+TSUNO 合并）。

## Phase 11: PPO Training (3 个，2026-07-10)

基于 L1-L6 PPO 对比流程发现并修复：

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 55 | `ppo_with_reg.py:compute_gae_vectorized` L529 | `approx_kl = masked_mean(log_ratio, mask)` 用 `log_ratio` 而非 JAX 的 `(ratio-1.0)-log_ratio`（Taylor KL 近似） | KL divergence 诊断值完全错误（差 ~exp(log_ratio) 倍） |
| 56 | `ppo_with_reg.py:compute_gae_vectorized` L111 | `next_valid[done] = False` — 错误地在 episode boundary 重置 `next_valid` state，但 JAX 从不重置 `next_valid_mask` | GAE `valid_mask` 在 episode boundary 后偏差 |
| 57 | `ppo_with_reg.py:compute_gae_vectorized` L133 | `next_valid[b_idx, cp] = is_valid \| done` — 当 `done=True` 时仅设置当前玩家，但 JAX 使用 `next_valid_mask.at[player].set(is_valid) \| done`（scalar bool `\|` 数组广播到所有玩家） | GAE `is_valid` 标记不一致，导致优势计算偏差 |

## Phase 11 (续): Rollout Loop (2 个，2026-07-11)

基于 JAX vs PT 训练循环逐行对比发现并修复：

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 58 | `ppo_with_reg.py` rollout loop L408-409 | `is_new_episode = bs.terminated \| bs.truncated` 在 `reinit_terminated_batch` 之后捕获——此时 terminated 状态已被替换为 fresh init 状态（`terminated=False`），导致 `is_new_episode` **永远为 False**。JAX 的 `auto_reset` 在 scan 内部工作，`is_new_episode` 捕获的是 reset 之前的状态 | GAE 累加器（`gae_acc`/`reward_accum` 等）永不重置，value estimate 和 reward 跨 episode 泄漏 |
| 59 | `ppo_with_reg.py` rollout loop L414 | `bs.rewards.clone()` 在 `step_batch` **之前**存储 reward。JAX 存储的是 `next_state.rewards`（step 之后的 reward）。PT 的 reward 偏移了 1 个 timestep（step t 存储的 reward 来自 step t-1） | GAE 的 `(reward, value, player)` 逐 timestep 配对不一致；GAE 累加器部分补偿但并非完全等价 |

## Phase 11 (续): ACNet MHA Bias (1 个, 2026-07-11)

| # | 位置 | Bug | 影响 |
|---|------|-----|------|
| 60 | `transformer.py:MultiHeadSelfAttention` L44-47 | PT `MultiHeadSelfAttention` 的 4 个 Linear 层（q/k/v/out_proj）设为 `bias=False`；Flax `MultiHeadDotProductAttention` 有对应 4 个 bias（shape: q/k/v=(heads,hd), out=(feat,)）。共 32 个 MHA bias 参数缺失（4 heads × 8 JTBs） | epoch 0 时 bias 值小（~3e-4），epoch 1 训练后累积为 1.69e-02 系统性 value 偏移；逐层对比发现 history JTB 差异达 0.29-0.40 |

**修复**:
- `transformer.py`: `bias=False` → `bias=True`（4 处 Linear）
- `replay_pt_acnet_golden.py`: 映射重写 (128→160)，纳入 32 MHA bias；新增 `'reshape'` 模式处理 JAX bias `(heads,hd)` → PT Linear bias `(features,)` 的 reshape
- 结果: value diff 从 1.69e-02 → 7.15e-07，epoch 1 loss diff 从 1.40e-02 → 1.67e-05

## 已知限制

| # | 位置 | 现象 |
|---|------|------|
| — | `_kyuushu` redeal | kyuushu 重洗牌使用 `torch.randperm`，无法匹配 JAX 的 PRNG 输出。此为非逻辑 bug，属于跨 PRNG 回放验证的固有限制。GPU 回放通过 `_kyuushu_deck_overrides` 注入 JAX deck 绕过 |
| — | `meld.py:_calc_addition_batch` L441 | 某些场景下 meld target 值为 63（超出 0-33），导致 `addition[b_idx, tgt]` 越界。预存于 uncommitted 代码中 |
| — | AdamW float32 精度 | `optax.adamw` 与 `torch.optim.AdamW` 的 denom 计算顺序不同（`sqrt(nu/bc2)+eps` vs `sqrt(nu)/sqrt(bc2)+eps`），导致纯 AdamW 一步产生 ~1.5×10⁻⁶ 参数差异。但在真实训练中被前向/反向传播的 float32 差异（tanh、matmul）完全淹没，实际参数漂移 ~1.2×10⁻³/30步 |
| — | 前向/反向传播 float32 | JAX 和 PyTorch 的 `tanh` 实现不同（~6×10⁻⁸ 差异），经多层网络传播后在梯度上积累到 ~3×10⁻⁴ 的差异。这是跨框架 float32 训练的物理极限，不影响训练正确性（loss 一致） |
