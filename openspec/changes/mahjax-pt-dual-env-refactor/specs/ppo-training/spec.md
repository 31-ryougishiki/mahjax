# PPO Training Pipeline Specification

## Overview

将 `ppo_with_reg.py` 从旧的 per-env 串行接口重构为 **BatchState-native** 训练管线，打通 GPU 批量并行环境与 PPO 强化学习之间的端到端链路。

此前 `env_parallel.py` 已完成全向量化（Phase 1-8，805 seeds 100% JAX 对齐），但训练脚本仍使用 `List[EnvState]` + 手动 stack/unstack 的胶水模式，导致 GPU 批量并行的核心优势完全未发挥。本次重构消除这一鸿沟。

## Requirements

### PT-01: BatchState 端到端训练
- `train_ppo()` 使用 `make(backend='parallel')` + `BatchState`
- `init_batch(num_envs=B, device=device)` 一次性批量初始化
- `step_batch(bs, actions)` 直接消费/返回 `BatchState`
- 消除所有 `List[EnvState]` ↔ `BatchState` 的 stack/unstack 转换
- Terminated 环境通过 `env.reinit_terminated_batch(bs)` 批量重新初始化（使用 `_copy_dataclass_row` 递归 splice）

### PT-02: 向量化 GAE（零 per-env Python 循环）
- `compute_gae_vectorized` 使用 `(B, 4)` 累加器张量，一次性处理全部环境
- 仅保留 `for t in reversed(range(T))` 不可并行的时间维度循环
- 已验证与原始 per-env 实现位级一致（10 个随机测试通过）
- `is_new_episode`（step 前捕获）正确重置 episode 边界的 GAE 累加器

### PT-03: 预分配 PPOBuffer
- `PPOBuffer(num_steps, num_envs)` 预分配所有 `(T, B, ...)` 张量
- `store(t, ...)` 直接写入预分配张量（零动态分配）
- `get_batch()` 返回已按 `(T, B, ...)` 排列的引用（零 copy）
- Observation 张量按第一次 `store` 的 shape/dtype 自动初始化

### PT-04: PPO 诊断指标
- `approx_kl`: `masked_mean(log_ratio, vmask)` — 策略变化幅度
- `clip_frac`: `masked_mean(|ratio-1| > clip_eps, vmask)` — clip 触发比例
- `explained_var`: `1 - Var(target - value) / Var(target)` — 价值函数解释力
- `avg_eps_len`: `1 / mean(dones)` — 平均 episode 长度
- 所有指标对齐 JAX 参考实现

### PT-05: Evaluation 对战评估
- 每 `eval_interval` 步执行 1-vs-3 对战（串行 env，小规模 1000 envs）
- `eval/vs_rand/*`: agent vs random — `hora_rate`, `deal_in_rate`, `riichi_rate`, `meld_rate`, `tenpai_at_ryuu_rate`, `avg_rank`, `avg_gain`
- `eval/vs_baseline/*`: agent vs BC pretrained baseline
- `hand/hora_finish_rate`, `hand/ryuukyoku_rate`, `hand/total`
- 复用现有 `make_eval_fn`（utils.py）

### PT-06: WandB 结构化日志
- 可选 `--use_wandb` 开关
- 记录训练指标（loss, entropy, kl, clip_frac, explained_var, reward）和 eval 指标
- 记录时间分解（rollout, gae, update）
- 未安装 wandb 时降级为 stdout 日志

### PT-07: Checkpoint / Resume
- `--checkpoint_dir` 指定保存目录，每 `eval_interval` 步保存
- Checkpoint 包含 `network.state_dict()`, `optimizer.state_dict()`, `update` 序号
- `--resume_from` 从 checkpoint 恢复训练（网络 + 优化器 + 步数）
- 训练结束后保存最终模型至 `params/red_mahjong-seed={seed}.pt`

### PT-08: 默认超参数对齐 JAX 参考
| 参数 | 旧 PT 默认 | 新 PT 默认 | JAX 默认 |
|------|-----------|-----------|---------|
| `num_envs` | 4 | 1024 | 1024 |
| `total_timesteps` | 100,000 | 100,000,000 | 1e8 |
| `minibatch_size` | 256 | 4096 | 4096 |
| `num_steps` | 256 | 256 | 256 |
| `update_epochs` | 4 | 4 | 4 |
| `lr` | 3e-4 | 3e-4 | 3e-4 |
| `ent_coef` | 0.01 | 0.01 | 0.01 |
| `clip_eps` | 0.2 | 0.2 | 0.2 |
| `vf_coef` | 0.5 | 0.5 | 0.5 |
| `mag_coef` | 0.2 | 0.2 | 0.2 |

### PT-09: 向量化 observe_batch ✅
- `env_parallel.observe_batch` 直接调用 `_observe_dict_batch`，返回 `dict of (B, ...) tensors`
- 消除旧的 `unstack → serial observe → List[dict]` 路径（~650ms → ~1.4ms，~400x 加速）
- `ppo_with_reg.py` 中不再需要桥接代码
- `env.py` Facade 透传 `observe_batch` 和 `reinit_terminated_batch`

### PT-10: 批量观察构建（observation.py）
- `hand_counts_to_idx_batch(counts)` — 向量化 `(B, 37)` 直方图 → `(B, 14)` 牌索引
- `_observe_dict_batch(bs)` — 直接构建批量观察 dict，全部操作在 `(B, ...)` 张量上完成：
  - `hand`: `(B, 14)` 从 `hand_with_red` 批量构建
  - `action_history`: `(B, 3, 200)` 批量相对化 player index
  - `scores`: `(B, 4)` 批量 rotate 到 current_player 视角
  - `furiten`, `shanten_count`, `round`, `honba`, `kyotaku`, `prevalent_wind`, `seat_wind`, `dora_indicators` 等直接取自 BatchState 张量

### PT-11: 启动脚本（ppo.sh）
- 支持 `gpu` / `npu` 平台切换：`bash script/ppo.sh gpu 1024`
- 关键参数通过环境变量覆盖：`DEVICE`, `NUM_ENVS`, `NUM_STEPS`, `TOTAL_STEPS`, `ROUND_MODE`
- 可选 `--use_wandb`, `--checkpoint_dir`

## Non-Goals

- ~~不修改 `env_parallel.py` 的 step 逻辑~~ — 已完成（Phase 10: 3-layer mixin split）
- 不实现多 GPU DDP/FSDP 分布式训练
- 不修改网络模型架构
- 不实现 `update_magnet` 动态更新（JAX 高级特性，后续版本）
- 不实现 `mag_divergence_type: l2`（仅支持 kl）

## Dependencies

```
ppo_with_reg.py ──→ env.py (make, backend='parallel')
                ├── batch_state.py (BatchState)
                ├── observation.py (_observe_dict_batch)
                ├── utils.py (make_eval_fn)
                ├── common.py (get_network_cls, default_bc_params_path)
                └── networks/red_network.py (ACNet)

env_parallel.py ──→ env_parallel_handlers.py (HandlersMixin)
                 ├── env_parallel_internals.py (InternalsMixin, _copy_dataclass_row)
                 └── observation.py (_observe_dict_batch)
```

## Verification

- [x] `observation.py` 语法检查通过
- [x] `ppo_with_reg.py` 语法检查通过
- [x] `hand_counts_to_idx_batch` 单元测试通过
- [x] `compute_gae_vectorized` vs 原始 per-env GAE 10 随机测试位级一致
- [x] `observe_batch` 向量化 → 返回 `dict of (B, ...) tensors`
- [x] GPU 端到端集成测试（init → observe → network → step → reinit, B=8）
- [ ] vs JAX PPO 训练完整性验证（pending: Phase 11 step-by-step parity）
