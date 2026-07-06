# Phased Development Plan

## Phase 0: 准备工作 (Pre-requisites)

- [ ] **P0-1**: 确认 `mahjax_pt/` 下所有依赖模块状态良好
  - 确认 `hand.py`, `meld.py`, `shanten.py`, `yaku.py`, `tile.py` 无未提交修改
  - 确认现有测试全部通过：`python mahjax_pt/tests/run_tests.py`
  - 确认 `ppo.sh` 可以正常启动

- [ ] **P0-2**: 创建开发分支
  ```bash
  git checkout -b feat/dual-env-refactor
  ```

- [ ] **P0-3**: 阅读 JAX 参考实现完整代码
  - 通读 `mahjax/red_mahjong/env.py`（约 2000+ 行）
  - 标注关键行号范围（init, draw, discard, meld, ron, tsumo, riichi, round-transition）
  - 建立 JAX → PT 串行版的函数映射表

## Phase 1: 抽取公共逻辑 + 创建兼容层

**目标**：在不修改核心游戏逻辑的前提下，将 `env.py` 中的纯辅助函数和动作处理器解耦，为分拆做准备。

### Task 1.1: 创建目录结构 (0.5h)

- [ ] 确认 `mahjax_pt/red_mahjong/` 目录结构
- [ ] 创建空文件 `env_serial.py`, `env_parallel.py`, `batch_state.py`

### Task 1.2: 将辅助函数抽取到工具模块 (1h)

- [ ] 将以下函数从 `env.py` 提取到 `env_serial.py`（作为模块级函数）：
  - `_resolve_game_config`, `_live_wall_end_ix`, `_set_tile_type_action`
  - `_has_red_discard_action`, `_special_abortive_draw_mask`, `_trigger_special_abortive_draw`
  - `_append_meld_to_player`, `_accept_riichi`, `_is_waiting_tile`
  - `_calc_wind`, `_is_first_turn`, `_append_action_history`
- [ ] 保持 JAX 对应行号注释
- [ ] 验证：现有测试通过

### Task 1.3: 创建兼容层 env.py (1h)

- [ ] 在 `env.py` 中保留 `Env` 基类
- [ ] `RedMahjong.__init__` 支持 `backend='serial'|'parallel'`
- [ ] 内部委托：`self._impl.step(...)`, `self._impl.init(...)`
- [ ] `make()` 工厂函数支持 `backend` 参数
- [ ] 验证：`make(backend='serial')` 和 `make(backend='parallel')` 均可正常导入

## Phase 2: 实现纯串行版本 (env_serial.py)

**目标**：从当前 `env.py` 提炼出纯串行实现，逐函数标注 JAX 对应关系。

### Task 2.1: 创建 RedMahjongSerial 类骨架 (1h)

- [ ] 定义 `RedMahjongSerial(Env)` 类
- [ ] 实现 `__init__`（参数与当前 RedMahjong 一致）
- [ ] 实现 `init()`, `step()`, `observe()` 方法签名
- [ ] 将 Phase 1.2 的辅助函数绑定为模块级函数

### Task 2.2: 迁移 _draw + 合法动作 Mask (2h)

- [ ] 从 `env.py` 迁移 `_draw` 方法（当前行 556-621）
- [ ] 迁移 `_make_legal_action_mask_after_draw`（行 648-702）
- [ ] 迁移 `_make_legal_action_mask_after_draw_riichi`（行 623-646）
- [ ] 标注所有 JAX 对应行号
- [ ] **验证**：单步 draw 后状态与 JAX 一致

### Task 2.3: 迁移 _discard + 鸣牌 Mask (2h)

- [ ] 迁移 `_discard`（行 704-752）
- [ ] 迁移 `_make_legal_action_mask_after_discard`（行 754-858）
  - 包含优先级逻辑：ron > open_kan > pon > chi
  - 包含多 ron 候选选择（离舍牌者最近）
- [ ] 标注 JAX 对应行号
- [ ] **验证**：discard 后 mask 与 JAX 一致

### Task 2.4: 迁移鸣牌动作 (2h)

- [ ] 迁移 `_pon`（行 1005-1028）
- [ ] 迁移 `_chi`（行 1079-1100）
- [ ] 迁移 `_open_kan`（行 1030-1053）
- [ ] 迁移 `_selfkan` + `_draw_after_kan` + `_flip_dora`（行 1055-1183）
- [ ] 标注 JAX 对应行号
- [ ] **验证**：鸣牌后手牌 + 面子状态与 JAX 一致

### Task 2.5: 迁移和了动作 + 结算 (2h)

- [ ] 迁移 `_ron` + `_settle_ron`（行 890-924, 960-974）
- [ ] 迁移 `_tsumo` + `_settle_tsumo`（行 926-1003）
- [ ] 迁移 `_riichi`（行 860-888）
- [ ] 迁移 `_pass` + `_kyuushu` + `_dummy`（行 1102-1148）
- [ ] 标注 JAX 对应行号
- [ ] **验证**：和了后奖励 + 分数与 JAX 一致

### Task 2.6: 迁移局管理 (2h)

- [ ] 迁移 `_abortive_draw_normal`（行 1185-1210）
- [ ] 迁移 `_advance_to_next_round_auto`（行 1212-1316）
- [ ] 迁移 `_finalize_game`（行 1669-1682）
- [ ] 标注 JAX 对应行号
- [ ] **验证**：多局推进与 JAX 一致

### Task 2.7: env_serial.py 完整性测试 (2h)

- [ ] 编写 `tests/test_env_serial_compare.py`
  - 测试：相同 seed，JAX vs PT 串行，100 步全字段对比
  - 测试：每个动作类型单独对比（discard/ron/tsumo/pon/chi/kan/riichi/pass）
  - 测试：完整 1 局从开始到结束对比
- [ ] **验证**：所有 JAX 对比测试通过

## Phase 3: 实现纯并行版本 (env_parallel.py)

**目标**：实现 batch-first 的完全向量化环境，消除所有串行回退路径。

### Task 3.1: 定义 BatchState (2h)

- [ ] 在 `batch_state.py` 中定义 `BatchState`, `BatchPlayerState`, `BatchRoundState`
- [ ] 所有字段形状：`(B, ...)`, `(B, 4, ...)` 或 batch 标量
- [ ] 实现 `stack_states(List[EnvState]) → BatchState`
- [ ] 实现 `unstack_state(BatchState, int) → EnvState`
- [ ] 实现 `_default_batch_state(B, device)` 初始化
- [ ] **验证**：stack → unstack round-trip 无损

### Task 3.2: RedMahjongParallel 类骨架 + init_batch (2h)

- [ ] 定义 `RedMahjongParallel(Env)` 类
- [ ] 实现 `init_batch(keys, num_envs)` → `BatchState`
- [ ] 批量洗牌：`torch.randperm` → `(B, 136)`
- [ ] 批量发牌：`make_init_hand_batch` → `(B, 4, 37)`
- [ ] 批量庄家第一摸
- [ ] **验证**：init_batch(128) 产生 128 个合法初始状态

### Task 3.3: step_batch 动作分发框架 (2h)

- [ ] 实现 `step_batch(batch_state, actions)` 主函数
- [ ] 动作分组：通过比较/索引将 `(B,)` actions 分为最多 12 组
- [ ] 每组的 mask + handler 调用范式
- [ ] 已终止环境跳过
- [ ] **验证**：动作分组正确（统计各组数量与预期一致）

### Task 3.4: _draw_batch 完全向量化 (3h)

- [ ] `_accept_riichi_batch`：批量立直接受
- [ ] 特殊流局检查（批量四风、批量四立直）
- [ ] 批量摸牌：`(B,)` → advanced indexing
- [ ] 批量手牌添加：`scatter_add_` or index_put
- [ ] 批量 kan check：`can_closed_kan_batch`, `can_added_kan_batch`, `can_tsumo_batch`, `can_kyuushu_batch`, `can_riichi_batch`
- [ ] 批量 mask 构建（riichi 和非 riichi 分支）
- [ ] 批量向听计算：`Shanten.number_batch`
- [ ] **验证**：draw_batch 结果与 B 次 serial draw 完全一致

### Task 3.5: _discard_batch 完全向量化 (3h)

**这是最关键的性能热点（~90% 动作），需要仔细优化。**

- [ ] 批量手牌减法：`(B, 37)` scatter subtract
- [ ] 批量河更新：`River.add_discard_batch`
- [ ] 批量行动历史更新
- [ ] 批量海底线检查
- [ ] 批量振听检测
- [ ] 批量 4 玩家鸣牌/荣和 mask：
  - `Hand.can_ron_batch` → `(B, 4)` bool
  - `Hand.can_chi_matrix_batch_4p` → `(B, 4, 6)` bool
  - `Hand.can_no_red_pon_batch_4p` / `can_red_pon_batch_4p` → `(B, 4)` bool
  - `Hand.can_open_kan_batch_4p` → `(B, 4)` bool
- [ ] 批量优先级确定（ron > kan > pon > chi）
- [ ] 批量 "无鸣牌 → draw" 和 "有鸣牌 → 下一玩家" 分支
- [ ] **验证**：discard_batch 结果与 B 次 serial discard 完全一致

### Task 3.6: _ron_batch, _tsumo_batch (2h)

- [ ] `_ron_batch`：批量役种判定 → 批量结算
- [ ] `_tsumo_batch`：批量役种判定 → 批量自摸分摊结算
- [ ] 附加役种（一发/双立直/枪杠/海底/岭上）批量处理
- [ ] **验证**：ron/tsumo batch 结果与 serial 一致

### Task 3.7: 其他批量动作 (2h)

- [ ] `_riichi_batch`：批量立直 mask 构建
- [ ] `_pon_batch` + `_open_kan_batch` + `_chi_batch`：批量鸣牌
- [ ] `_selfkan_batch`：批量暗杠/加杠 + 翻宝牌
- [ ] `_pass_batch`：批量过 + 振听设置
- [ ] `_kyuushu_batch` + `_dummy_batch`
- [ ] **验证**：每种动作 batch 与 serial 等价

### Task 3.8: 批量局管理 (2h)

- [ ] `_flip_dora_batch` + `_draw_after_kan_batch`
- [ ] `_abortive_draw_normal_batch`
- [ ] `_advance_to_next_round_batch`
- [ ] `_finalize_game_batch`
- [ ] **验证**：多局过渡与 serial 等价

### Task 3.9: observe_batch (1h)

- [ ] `observe_batch(batch_state)` → `{key: (B, ...) tensor}`
- [ ] 批量构建各观察字段
- [ ] **验证**：observe_batch 每个环境与 serial observe 等价

### Task 3.10: 并行 vs 串行等价性测试 (2h)

- [ ] 编写 `tests/test_env_parallel_parity.py`
  - 测试：相同 seeds, init_batch → 拆解，与 serial init 逐个对比
  - 测试：随机 action 序列，parallel step_batch vs serial step × B，全字段对比
  - 测试：1000 步随机 rollout，parallel vs serial 等价
  - 测试：所有动作类型全覆盖
- [ ] **验证**：所有 parity 测试通过

### Task 3.11: 性能基准测试 (2h)

- [ ] 编写 `tests/test_env_parallel_perf.py`
- [ ] 基准测试：
  - Serial: B=1 throughput (steps/sec)
  - Parallel: B=128 throughput (steps/sec)
  - 计算加速比：parallel / (serial × 128)
  - GPU 利用率监测
  - 各阶段耗时分析（draw, discard, meld, rare_actions）
- [ ] 目标：parallel 在 GPU 上达到 serial 串行吞吐的 30× 以上

## Phase 4: 集成 + 训练验证

**目标**：确保新实现与现有训练管道兼容。

### Task 4.1: 适配 auto_reset_wrapper (1h)

- [ ] 检查 `auto_reset_wrapper.py` 对两种 env 的兼容性
- [ ] 如需要，添加 `auto_reset_batch` 包装器
- [ ] **验证**：auto_reset 在 parallel 模式下正常工作

### Task 4.2: 适配 ppo_with_reg.py (1.5h)

- [ ] 修改 `ppo_with_reg.py` 使用 `make(backend='parallel')`
- [ ] 适配 buffer：观察存储从 `List[T][B]` 变为 batch tensor
- [ ] 适配 GAE 计算：确保 current_player 正确传递
- [ ] **验证**：PPO 训练可完整运行 1000 步

### Task 4.3: 训练正确性验证 (2h)

- [ ] 相同 seed 下：serial PPO (B=1×128 steps) vs parallel PPO (B=128)
- [ ] 对比：loss 曲线、策略熵、value loss
- [ ] 容许浮点误差内的数值一致性
- [ ] **验证**：训练指标在合理误差内一致

### Task 4.4: NPU 适配验证 (1h)

- [ ] 在 NPU 设备上运行 `ppo.sh`
- [ ] 确认无设备不兼容的 op
- [ ] 确认 eager mode 正常工作
- [ ] 如有 NPU 不支持的 op，进行替换或 polyfill
- [ ] **验证**：`bash script/ppo.sh` 运行成功

## Phase 5: 清理 + 文档

### Task 5.1: 清理 env.py (0.5h)

- [ ] `env.py` 保留为纯兼容层（删除不再需要的实现代码）

### Task 5.2: 更新测试入口 (0.5h)

- [ ] `tests/run_tests.py` 添加新测试文件
- [ ] `tests/test_plan.md` 更新状态

### Task 5.3: 代码审查 + 性能分析 (1h)

- [ ] 并行版关键路径 kernel launch 分析
- [ ] 确认热点函数（discard_batch, draw_batch）无隐藏串行循环
- [ ] 检查 GPU 显存使用

## Phase 6: 后续优化（可选）

以下为并行版本可能的后续优化方向，不作为本次交付目标：

- [ ] **O6-1**: 使用 CUDA Graph 减少 kernel launch overhead
- [ ] **O6-2**: 役种判定的完全批量计算（全 `(B,)` 张量无 Python loop）
- [ ] **O6-3**: 优化 BatchState 内存布局（padding/alignment 改善 coalesced access）
- [ ] **O6-4**: 混合精度（FP16）状态存储
- [ ] **O6-5**: Stream 并行（多 CUDA stream 处理不同动作组）

---

## 工作量估算

| Phase | 内容 | 预估工时 |
|-------|------|---------|
| Phase 0 | 准备工作 | 2h |
| Phase 1 | 公共逻辑抽取 + 兼容层 | 2.5h |
| Phase 2 | 纯串行版本实现 + 测试 | 12h |
| Phase 3 | 纯并行版本实现 + 测试 | 21h |
| Phase 4 | 集成 + 训练验证 | 5.5h |
| Phase 5 | 清理 + 文档 | 2h |
| **总计** | | **~45h** |

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| JAX→PT 数值差异 | 串行版与 JAX 不完全一致 | 通过逐函数对比测试定位差异，确认是预期差异（如 float 精度）还是 bug |
| 并行版控制流复杂度 | 某些动作极难完全向量化 | 对极稀有动作（如 kyuushu）允许小幅串行回退，但主路径 100% 向量化 |
| NPU op 兼容性 | 某些 op 在 Ascend 上不支持 | 使用 PT adapter 层抽象设备差异，fallback 到兼容 op |
| 训练性能回退 | 重构后比混合版更慢 | 通过 profiling 对比，确保关键路径（discard_batch）性能不退化 |
