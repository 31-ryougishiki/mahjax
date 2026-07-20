# MahJax 新手阅读指南

> 面向刚接触 MahJax 项目的开发者，提供从零到全面理解整个代码库的阅读路径。

## 项目概览

**MahJax** 是一个基于 JAX 的 GPU 加速日本立直麻将模拟器，用于强化学习（RL）研究。核心学术论文发表在 arXiv：[Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX](https://arxiv.org/abs/2605.20577)（2026）。

### 两大运行时

| 运行时 | 框架 | 定位 | 速度 |
|--------|------|------|------|
| [`mahjax/`](../mahjax/) | JAX | 官方参考实现，GPU 加速 | ~1-2M steps/sec |
| [`mahjax_pt/`](../mahjax_pt/) | PyTorch eager | 非 JAX 硬件（NPU）移植 | 精度验证对齐 JAX |

### 两种麻将规则

| 规则 | 标识 | 特点 |
|------|------|------|
| 赤麻将 | `red_mahjong` | 天凤标准，含赤五牌，完整规则（包牌、双响、特殊流局等） |
| 无赤麻将 | `no_red_mahjong` | 无赤五牌，简化规则（无特殊流局、包牌、双响），约 2x 更快 |

### 当前分支背景

分支 `feat/dual-env-refactor` 已完成 PyTorch 端从「混合串行/并行架构」到「清洁双轨架构」的重构：

- **串行轨**（`env_serial.py`）：纯串行、单环境、与 JAX 1:1 对照、用于正确性验证
- **并行轨**（`env_parallel.py`）：全向量化、batch-first 张量操作、用于 GPU/NPU 训练
- **Facade**（`env.py`）：兼容层，根据 `backend` 参数路由到串行或并行实现

相关设计文档见 [openspec/changes/mahjax-pt-dual-env-refactor/](../openspec/changes/mahjax-pt-dual-env-refactor/)。

---

## 总览：项目文件结构

```
mahjax/
├── mahjax/                          # JAX 核心包
│   ├── __init__.py                  # 公开 API 导出
│   ├── core.py                      # Env/State 基类、make() 工厂
│   ├── _src/                        # 共享基础设施
│   │   ├── struct.py                # @dataclass 装饰器
│   │   ├── types.py                 # Array、PRNGKey 类型别名
│   │   └── visualizer.py            # SVG 可视化器
│   ├── red_mahjong/                 # 赤麻将（JAX 参考实现）
│   │   ├── env.py                   # ★ 环境主引擎（2,257 行）
│   │   ├── state.py                 # 状态定义（GameConfig, PlayerStateArrays, State）
│   │   ├── action.py                # 动作常量（87 种动作）
│   │   ├── constants.py             # 全局常量
│   │   ├── tile.py                  # 牌编码、牌河
│   │   ├── hand.py                  # 手牌操作
│   │   ├── meld.py                  # 面子编码
│   │   ├── shanten.py               # 向听数计算
│   │   ├── yaku.py                  # 役种判定
│   │   ├── observation.py           # 观察构建
│   │   ├── players.py               # 内置玩家策略
│   │   ├── visualization.py         # 回合 SVG 渲染
│   │   ├── env_optim.py             # 环境优化（CPU 版）
│   │   └── cpu_env.py               # CPU 串行版环境
│   ├── no_red_mahjong/              # 无赤麻将（简化版）
│   │   └── ...                      # 与 red_mahjong 镜像结构
│   ├── ui/                          # FastAPI 网页 UI
│   │   ├── app.py                   # 应用入口
│   │   ├── game_manager.py          # 游戏管理器
│   │   ├── agents.py                # 代理注册
│   │   └── utils.py                 # 工具函数
│   └── wrappers/                    # 环境包装器
│       └── auto_reset_wrapper.py    # 自动重置
│
├── mahjax_pt/                       # PyTorch 移植（当前分支重点）
│   ├── red_mahjong/
│   │   ├── env.py                   # Facade 兼容层（委托到 serial/parallel）
│   │   ├── env_serial.py            # ★ 纯串行环境（1,601 行，JAX 1:1 对照）
│   │   ├── env_parallel.py          # ★ 并行编排层（375 行）
│   │   ├── env_parallel_handlers.py # ★ 并行动作处理器（1,219 行）
│   │   ├── env_parallel_internals.py# ★ 并行内部机制（727 行）
│   │   ├── batch_state.py           # ★ 批量化状态定义（369 行）
│   │   ├── alignment.py             # JAX/PyTorch 算子对齐
│   │   ├── auto_reset_wrapper.py    # 自动重置包装器
│   │   ├── state.py                 # EnvState 等数据类
│   │   ├── hand.py                  # 手牌操作（含 batch 版本）
│   │   ├── meld.py                  # 面子编码（含 batch 版本）
│   │   ├── shanten.py               # 向听计算
│   │   ├── yaku.py                  # 役种判定（GPU device-aware）
│   │   ├── tile.py                  # 牌面操作（含 batch 版本）
│   │   ├── observation.py           # 观察构建
│   │   ├── players.py               # 玩家策略
│   │   ├── action.py                # 动作常量
│   │   ├── constants.py             # 常量
│   │   └── types.py                 # 类型别名
│   ├── examples/                    # PyTorch RL 示例
│   │   ├── bc.py                    # 行为克隆训练
│   │   ├── ppo_with_reg.py          # PPO 训练
│   │   ├── collect_offline_data.py  # 离线数据收集
│   │   ├── common.py                # 公共训练工具
│   │   ├── utils.py                 # 工具函数
│   │   └── networks/                # 网络定义
│   │       ├── red_network.py       # 赤麻将 Actor-Critic 网络
│   │       └── transformer.py       # Transformer 编码器
│   ├── scripts/                     # 辅助脚本
│   │   ├── bench_ppo_pipeline.py    # PPO 流水线基准测试
│   │   ├── analyze_ron_yaku.py      # 荣和役种分析
│   │   ├── gen_more_seeds.py        # 种子生成
│   │   └── regression_seeds.py      # 回归种子
│   ├── tests/                       # ★ 精度验证测试（重点）
│   │   ├── PRECISION_VERIFICATION_REPORT.md  # 精度验证报告
│   │   ├── test_plan.md             # 测试计划
│   │   ├── test_cases.py            # L1: 16 组 80 断言
│   │   ├── test_aligned_math.py     # L1: JAX/PT 数学对齐
│   │   ├── test_exact_parity.py     # L2: JAX ↔ PT 精确等价
│   │   ├── test_full_ppo_parity.py  # L2: 完整 PPO 等价
│   │   ├── test_mha_definitive.py   # L2: MHA 精度
│   │   ├── test_env_parallel_parity.py  # L3: Serial ↔ Parallel 等价
│   │   ├── test_env_branches.py     # L3: 关键分支覆盖
│   │   ├── replay_pt_against_golden.py  # L4: 串行金数据回放
│   │   ├── replay_parallel_against_golden.py  # L4: 并行金数据回放
│   │   ├── test_ppo_math_parity.py  # PPO L1: 数学原语对比
│   │   ├── test_ppo_gae_parity.py   # PPO L2: GAE 对比
│   │   ├── test_ppo_weight_transfer.py  # PPO L3: 权重迁移
│   │   ├── test_ppo_update_parity.py    # PPO L4: loss/grad/param 对比
│   │   ├── test_ppo_acnet_parity.py     # PPO L4 Ext: 完整 ACNet PPO
│   │   ├── test_ppo_cycle_parity.py     # PPO L5: 单次 update cycle
│   │   ├── test_ppo_training_parity.py  # PPO L6: 多步训练稳定性
│   │   ├── test_ppo_adamw_drift.py      # PPO: AdamW 漂移分析
│   │   ├── test_ppo_bisect_drift.py     # PPO: 二分定位漂移
│   │   ├── test_ppo_30step_parity.py    # PPO: 30 步对比
│   │   ├── verify_precision_root_cause.py  # 精度根因分析
│   │   ├── run_tests.py             # 测试运行器
│   │   ├── scan_rare_paths.py       # 稀有路径扫描
│   │   ├── plot_ppo_loss_curves.py  # 损失曲线绘图
│   │   └── record_jax_golden.py     # JAX 金数据录制
│   ├── verify_game.py               # 游戏逻辑验证
│   ├── visualize.py                 # 可视化工具
│   ├── inference.py                 # 推理工具
│   └── compare_e2e.py               # 端到端对比
│
├── examples/                        # JAX RL 示例
│   ├── bc.py                        # 行为克隆
│   ├── ppo_with_reg.py              # PPO 训练
│   ├── collect_offline_data.py      # 离线数据收集
│   ├── common.py                    # 公共工具
│   ├── utils.py                     # 工具函数
│   └── networks/                    # JAX 网络定义
│       ├── red_network.py           # 赤麻将网络
│       ├── no_red_network.py        # 无赤麻将网络
│       └── transformer.py           # Transformer
│
├── tests/                           # JAX 环境单元测试
│   ├── red_mahjong/                 # 赤麻将测试
│   │   ├── test_env.py              # 环境 test
│   │   ├── test_hand.py             # 手牌 test
│   │   ├── test_meld.py             # 面子 test
│   │   ├── test_yaku.py             # 役种 test
│   │   ├── test_shanten.py          # 向听 test
│   │   ├── test_tile.py             # 牌 test
│   │   ├── test_observe.py          # 观察 test
│   │   ├── test_play.py             # 对局 test
│   │   ├── test_parity.py           # 等价性 test
│   │   ├── test_special_case.py     # 特殊情形 test
│   │   └── test_visualize.py        # 可视化 test
│   └── no_red_mahjong/              # 无赤麻将测试（镜像结构）
│
├── docs/                            # 文档
│   ├── index.md                     # 文档索引
│   ├── api.md                       # API 文档
│   ├── rule.md                      # 支持的麻将规则
│   ├── mahjong-basics.md            # 麻将基础入门
│   ├── red_mahjong.md               # 赤麻将详解
│   ├── no_red_mahjong.md            # 无赤麻将详解
│   ├── ui.md                        # UI 使用说明
│   ├── visualization.md             # 可视化配置
│   └── onboarding-guide.md          # 本文档
│
├── openspec/                        # 设计规格文档
│   └── changes/mahjax-pt-dual-env-refactor/
│       ├── proposal.md              # 重构提案
│       ├── design.md                # 架构设计
│       ├── tasks.md                 # 任务清单
│       ├── methodology.md           # 方法论
│       └── specs/                   # 各模块规格
│
├── script/                          # 项目辅助脚本
├── Mortal/                          # Mortal 机器人参考代码
├── pyproject.toml                   # Python 项目配置
├── Makefile                         # 开发命令
├── mkdocs.yml                       # 文档站点配置
├── README.md                        # 项目首页
└── LICENSE                          # Apache-2.0
```

**总代码量**：核心代码约 20,000+ 行（不含测试和第三方代码）。

---

## 阅读路线图

建议按以下七个阶段循序渐进，预估总耗时约 **15-20 小时**（约 3-4 周，每周 4-5 小时）。

---

## 第一阶段：概念入门（约 30 分钟）

> **目标**：理解"做什么"和"为什么"，形成整体认知。无需阅读代码。

### 1.1 麻将规则入门

如果你不熟悉日本立直麻将，先了解基本概念：

- [mahjong-basics.md](mahjong-basics.md) — 34 种牌、4 面子 + 1 雀头的和牌条件、役种概念、点数计算
- [rule.md](rule.md) — 赤麻将 vs 无赤麻将的区别、天凤（Tenhou）规则标准
- [red_mahjong.md](red_mahjong.md) — 赤麻将详细规则
- [no_red_mahjong.md](no_red_mahjong.md) — 无赤麻将详细规则

### 1.2 项目定位

- [README.md](../README.md) — 项目整体介绍、快速开始、支持规则、文献引用
- [docs/index.md](index.md) — 文档索引页
- [pyproject.toml](../pyproject.toml) — 了解依赖项（JAX、FastAPI、svgwrite）

### 1.3 当前分支背景

- [openspec/changes/mahjax-pt-dual-env-refactor/proposal.md](../openspec/changes/mahjax-pt-dual-env-refactor/proposal.md) — 为什么需要重构：混合架构的四大问题（验证困难、性能瓶颈、维护复杂、方向不清）
- [openspec/changes/mahjax-pt-dual-env-refactor/design.md](../openspec/changes/mahjax-pt-dual-env-refactor/design.md) — 架构全景图：三组件（Facade、Serial、Parallel）与 Mixin 拆分

---

## 第二阶段：核心数据模型（约 1-2 小时）

> **目标**：理解麻将的"数据结构"——牌、面子、手牌、状态定义。这是所有游戏逻辑的基石。

### 阅读顺序（按依赖关系排列）

#### 2.1 常量定义
**[mahjax/red_mahjong/constants.py](../mahjax/red_mahjong/constants.py)**（84 行）

关注以下关键数字的来源与含义：
- `NUM_PLAYERS = 4`，`NUM_TILE_TYPES = 34`，`NUM_TILE_TYPES_WITH_RED = 37`
- `NUM_PHYSICAL_TILES = 136`（34 × 4）
- `LEGAL_ACTION_SIZE = 87`
- `MAX_DISCARDS_PER_PLAYER = 60`，`MAX_MELDS_PER_PLAYER = 4`
- `COPIES_PER_TILE = 4`，`DEAD_WALL_TILES = 14`

#### 2.2 牌的编码
**[mahjax/red_mahjong/tile.py](../mahjax/red_mahjong/tile.py)**（189 行）

- `Tile`：34 种基本牌（万/筒/索/字），编码 0-33；3 种赤五牌（赤 5m/5p/5s），编码 34-36
- `River`：牌河的 16-bit 压缩编码，包含牌值、来源（自摸/吃/碰/杠）、立直信息
- `EMPTY_RIVER`：空牌河标记

#### 2.3 动作常量
**[mahjax/red_mahjong/action.py](../mahjax/red_mahjong/action.py)**（41 行）

87 种动作的分类：
- 舍牌：0-36（包括 `TSUMOGIRI`）
- 杠（暗杠/加杠）：37-73
- `RIICHI`：74
- `RON`：75
- `TSUMO`：76
- `PON`：77-78（含 `PON_RED`）
- `CHI`：79-82
- `OPEN_KAN`：83
- `PASS`：84
- `KYUUSHU`：85
- `DUMMY`：86

#### 2.4 面子编码
**[mahjax/red_mahjong/meld.py](../mahjax/red_mahjong/meld.py)**（202 行）

- `Meld`：16-bit 压缩编码，包含类型（顺子/刻子/杠子）、牌值、来源、红五信息
- `EMPTY_MELD`：空面子标记
- 理解 `meld_to_str()` 等辅助函数

#### 2.5 手牌操作
**[mahjax/red_mahjong/hand.py](../mahjax/red_mahjong/hand.py)**（374 行）

- `Hand` 类的核心方法：
  - `draw(tile)` — 摸牌
  - `discard(tile_idx)` — 舍牌
  - `can_ron(tile)` / `can_tsumo()` — 和牌判定
  - `can_riichi()` — 立直判定
- JAX 中如何使用 `jax.vmap` 对 player 维度并行化

#### 2.6 状态定义 ★
**[mahjax/red_mahjong/state.py](../mahjax/red_mahjong/state.py)**（141 行）

这是最重要的数据结构文件：

- `GameConfig`：游戏配置（食断允许、食替禁止、赤牌使用、双响、包牌、特殊流局）
- `PlayerStateArrays`：**struct-of-arrays 模式**，所有玩家数据打包为 `(4, N)` 的 `jnp.ndarray`
  - 手牌（`hand`, `hand_with_red`, `hand_ids`）
  - 副露（`melds`, `meld_tiles`, `meld_info`）
  - 牌河（`river`, `discards`, `discard_info`）
  - 立直（`riichi`, `riichi_declared`, `riichi_step`, `double_riichi`, `ippatsu`）
  - 振听（`furiten_by_discard`, `furiten_by_pass`）
  - 役种预备（`can_win`, `has_yaku`, `fan`, `fu`）
- `State`：完整环境状态，继承自 `mahjax.core.State`，包含 `current_player`, `rewards`, `terminated`, `legal_action_mask`

> **核心设计模式**：JAX 使用不可变的 `@dataclass`（struct-of-arrays）表示状态。PlayerStateArrays 将所有 4 人的数据按 player 维打包，利用 `jax.vmap` 沿该维度并行计算——这是高性能的关键。

---

## 第三阶段：JAX 参考实现——核心引擎（约 3-4 小时）

> **目标**：深入理解 JAX 版麻将环境。这是整个项目的"金标准"，PyTorch 端的一切都对照它验证。

### 3.1 工厂与基类
**[mahjax/core.py](../mahjax/core.py)**（280 行）

- `State` 抽象基类：6 个公共属性（`current_player`, `rewards`, `terminated`, `truncated`, `legal_action_mask`, `_step_count`）
- `Env` 抽象基类：`init`, `step`, `observe` 三核心方法
- `make()` 工厂函数：根据 `env_id` 字符串实例化环境
- `available_envs()` 函数

### 3.2 JAX 赤麻将环境 ★★★
**[mahjax/red_mahjong/env.py](../mahjax/red_mahjong/env.py)**（2,257 行，全项目最大文件）

**建议分模块阅读**，每个模块聚焦一个游戏阶段的逻辑：

| 模块 | 大致区域 | 行数 | 功能 |
|------|---------|------|------|
| 导入 & 动作映射 | 1-50 | ~50 | `ACTION_FUN_MAP` 将 87 种动作映射为 6 个处理函数 |
| 配置解析 | `_resolve_*` | ~60 | `round_mode`, `observe_type`, `order_points`, `GameConfig` |
| `RedMahjong.__init__` | class 头部 | ~50 | 回合模式、观察类型、顺位点初始化 |
| 牌山初始化 | `_init_wall` | ~80 | Fisher-Yates 洗牌、王牌区 14 张、宝牌指示牌 |
| 配牌 | `_init_hand` | ~120 | 各 13 张配牌、庄家 14 张、`_sort_hand` |
| 宝牌翻示 | `_init_dora_indicators` / `_flip_dora` | ~60 | 宝牌指示牌翻示逻辑 |
| 一局初始化 | `_init_round` | ~100 | 组合以上步骤，生成初始 `State` |
| 摸牌 | `_draw` / `_draw_after_kan` | ~120 | 从牌山摸牌、王牌区管理、杠后摸牌 |
| 舍牌 | `_discard` / `_riichi` | ~150 | 舍牌到牌河、立直宣言（耗 1000 点供托）、w立直判定 |
| 副露 | `_pon` / `_chi` / `_open_kan` | ~180 | 碰/吃/大明杠，包含赤牌调整、食替禁止检查 |
| 杠 | `_selfkan` | ~100 | 暗杠/加杠，宝牌翻示时机 |
| 和牌 | `_ron` / `_tsumo` | ~200 | 荣和（放铳）与自摸和，多家和牌处理 |
| 结算 | `_settle_ron` / `_settle_tsumo` | ~280 | 点数计算（符/翻→基本点→授受）、本场棒、供托、飞人判定 |
| 流局 | `_abortive_draw_normal` / `_trigger_special_abortive_draw` | ~100 | 正常流局（荒牌）、特殊流局（四风连打/四杠散了/九种九牌/四家立直） |
| 合法动作 Mask | `_make_legal_action_mask_after_draw` / `_make_legal_action_mask_after_discard` | ~180 | 摸牌后/舍牌后的合法动作生成，含 chi_shift 计算 |
| 推进 & 终局 | `_advance_to_next_round_auto` / `_finalize_game` | ~150 | 局推进、游戏结束、最终排名、顺位点 |
| 主步进 | `step` | ~80 | 动作分发（通过 `ACTION_FUN_MAP` + `jax.lax.switch`） |

> **阅读技巧**：`_init_round` → `_draw` → `_discard` → `_pon`/`_chi` → `_ron`/`_tsumo` → `_settle_ron`/`_settle_tsumo` → `_finalize_game` 构成了一局游戏的完整生命周期。建议沿着 `step()` 的调用链追踪，而非按行号顺序阅读。

### 3.3 辅助模块

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| [shanten.py](../mahjax/red_mahjong/shanten.py) | 159 | 向听数计算——距听牌还差几张。基于手牌分割 + 面子候选集搜索 |
| [yaku.py](../mahjax/red_mahjong/yaku.py) | 702 | 役种判定——断幺九/立直/平和/三色同顺/一气通贯等。使用预计算缓存（`.npz`）加速 |
| [observation.py](../mahjax/red_mahjong/observation.py) | 95 | 观察构建：`_observe_dict`（dict 格式给 Transformer）、`_observe_2D`（2D 网格给 CNN） |
| [players.py](../mahjax/red_mahjong/players.py) | 378 | 内置玩家策略：随机、`rule_based_player`（基于启发式规则） |
| [visualization.py](../mahjax/red_mahjong/visualization.py) | 668 | 回合 SVG 渲染，含双语牌面支持 |
| [env_optim.py](../mahjax/red_mahjong/env_optim.py) | 217 | 针对特定硬件优化的环境配置 |
| [cpu_env.py](../mahjax/red_mahjong/cpu_env.py) | 34 | CPU 串行版环境（调试用） |

---

## 第四阶段：PyTorch 移植——双轨架构（约 3-4 小时）

> **目标**：理解 PyTorch 双轨架构的设计思路和实现细节。这是当前分支的核心工作。

### 4.1 架构全景

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

### 4.2 兼容层入口
**[mahjax_pt/red_mahjong/env.py](../mahjax_pt/red_mahjong/env.py)**（140 行）

- `RedMahjong` Facade 类：根据 `backend` 参数（`"serial"` 或 `"parallel"`）委托到对应实现
- `make()` 工厂函数
- 属性转发：`id`, `version`, `num_players`, `num_actions`, `observation_shape`

### 4.3 串行参考实现 ★
**[mahjax_pt/red_mahjong/env_serial.py](../mahjax_pt/red_mahjong/env_serial.py)**（1,601 行）

- 与 JAX `env.py` 保持 1:1 结构对应，**每行标注 JAX 对应行号**
- 关键翻译模式：
  - `jax.numpy` → `torch`
  - `jax.lax.switch` → Python `if/elif`
  - `jnp.bool_` → `torch.bool`
  - 不可变状态 → 可变 `EnvState`（`dataclass` + 直接属性赋值）
  - JAX PRNGKey → Python `random.Random` + `np.random.Generator`
- 所有标量为 Python 原语（`int`/`bool`），张量为单环境维度的 `torch.Tensor`
- 阅读策略：与 JAX `env.py` 对照阅读，理解翻译模式后可以主要参考 JAX 版

### 4.4 并行训练实现（分三层阅读）

#### 4.4.1 批量化状态
**[mahjax_pt/red_mahjong/batch_state.py](../mahjax_pt/red_mahjong/batch_state.py)**（369 行）

- `BatchState`：所有字段从单环境提升为 `(B, ...)` 或 `(B, 4, ...)` 的 batch 维度
- `BatchPlayerState`：玩家状态的 batch 版
- `BatchRoundState`：回合状态的 batch 版
- `stack_states()`：将 B 个 `EnvState` 打包为一个 `BatchState`
- `unstack_state()`：从 `BatchState` 中提取第 i 个 `EnvState`

#### 4.4.2 编排层
**[mahjax_pt/red_mahjong/env_parallel.py](../mahjax_pt/red_mahjong/env_parallel.py)**（375 行）

- `RedMahjongParallel(HandlersMixin, InternalsMixin, Env)` — 通过 Mixin 继承组合功能
- `init_batch()`: 批量初始化 B 个游戏状态
- `step_batch()`: 核心分发逻辑

**并行 step 的执行模式**：
```
step_batch(states, actions)
  ├── 计算 action_type = bucketize(actions, boundaries)
  ├── discard_mask = (action_type == DISCARD)       → _discard_batch      (~90%)
  ├── tsumo_mask   = (action_type == TSUMO)          → _tsumo_batch       (~5%)
  ├── pass_mask    = (action_type == PASS)           → _pass_batch        (~3%)
  ├── ron_mask     = (action_type == RON)            → _ron_batch         (~1%)
  ├── riichi_mask  = (action_type == RIICHI)         → _riichi_batch
  ├── pon_mask     = (action_type in [PON, PON_RED]) → _pon_batch
  ├── chi_mask     = (action_type in CHI range)      → _chi_batch
  ├── kan_mask     = (action_type in KAN range)      → _selfkan_batch
  ├── open_kan_mask                                   → _open_kan_batch
  ├── kyuushu_mask                                    → _kyuushu_batch
  └── dummy_mask                                      → _dummy_batch
```

#### 4.4.3 动作处理器 Mixin
**[mahjax_pt/red_mahjong/env_parallel_handlers.py](../mahjax_pt/red_mahjong/env_parallel_handlers.py)**（1,219 行）

每个 handler 遵循统一模式：
```python
def _xxx_batch(self, batch_state, action_mask):
    """action_mask: (B,) bool — 哪些环境执行此动作"""
    if not action_mask.any():
        return batch_state
    # 1. 提取子集
    idx = action_mask.nonzero().squeeze(-1)  # (K,)
    # 2. 批量计算
    result = vectorized_op(batch_state, idx, ...)
    # 3. Scatter 回主状态
    batch_state = scatter_update(batch_state, idx, result)
    return batch_state
```

包含 13 个处理方法：
`_discard_batch`, `_riichi_batch`, `_pon_batch`, `_chi_batch`, `_open_kan_batch`, `_selfkan_batch`, `_ron_batch`, `_tsumo_batch`, `_pass_batch`, `_kyuushu_batch`, `_dummy_batch`, `_draw_batch`, `_draw_after_kan_batch`

#### 4.4.4 内部机制 Mixin
**[mahjax_pt/red_mahjong/env_parallel_internals.py](../mahjax_pt/red_mahjong/env_parallel_internals.py)**（727 行）

- `_make_legal_mask_after_discard_batch` / `_make_legal_mask_after_draw_batch`：合法动作掩码的批量化构建
- `_settle_ron_batch` / `_settle_tsumo_batch`：批量结算
- `_precompute_yaku_batch`：批量役种预计算
- `_abortive_draw_normal_batch`：批量流局
- `_advance_round_batch`：批量局推进

### 4.5 PyTorch 共享模块

这些模块与 JAX 对应模块功能等价，但实现为 PyTorch eager 模式，部分已包含 batch 版本的方法：

| 文件 | 行数 | 与 JAX 的差异 |
|------|------|-------------|
| [hand.py](../mahjax_pt/red_mahjong/hand.py) | 886 | 添加 `chi_batch`, `open_kan_batch`, `closed_kan_batch` 等方法 |
| [meld.py](../mahjax_pt/red_mahjong/meld.py) | 455 | 已有 batch 版本的面子编码 |
| [shanten.py](../mahjax_pt/red_mahjong/shanten.py) | 233 | 已有 batch 版本 |
| [yaku.py](../mahjax_pt/red_mahjong/yaku.py) | 1,182 | **GPU device-aware 常量**：`_get_cache()` 和 `_FAN.to(device)` 等适配逻辑 |
| [tile.py](../mahjax_pt/red_mahjong/tile.py) | 298 | 已有 batch 版本 |
| [alignment.py](../mahjax_pt/red_mahjong/alignment.py) | 218 | **JAX ↔ PyTorch 算子对齐函数**，用于精度对比 |
| [observation.py](../mahjax_pt/red_mahjong/observation.py) | 191 | 添加 `_observe_dict_batch` |
| [players.py](../mahjax_pt/red_mahjong/players.py) | 339 | 等价实现 |
| [state.py](../mahjax_pt/red_mahjong/state.py) | 145 | 使用 Python `dataclass` 替代 JAX `@dataclass` |

---

## 第五阶段：RL 训练管线（约 1-2 小时）

> **目标**：理解如何在麻将上做强化学习——数据收集、网络结构、PPO 训练循环。

### 5.1 JAX 示例
| 文件 | 内容 |
|------|------|
| [examples/common.py](../examples/common.py) | 训练公共工具（`load_mortal_demo_data` 等） |
| [examples/bc.py](../examples/bc.py) | 行为克隆：从 Mortal 离线数据学习 |
| [examples/ppo_with_reg.py](../examples/ppo_with_reg.py) | PPO + 多样性正则化训练 |
| [examples/collect_offline_data.py](../examples/collect_offline_data.py) | 离线数据收集脚本 |
| [examples/networks/red_network.py](../examples/networks/red_network.py) | 赤麻将 Actor-Critic 网络 |
| [examples/networks/transformer.py](../examples/networks/transformer.py) | Transformer 编码器 |

### 5.2 PyTorch 示例
| 文件 | 内容 |
|------|------|
| [mahjax_pt/examples/bc.py](../mahjax_pt/examples/bc.py) | 行为克隆训练 |
| [mahjax_pt/examples/ppo_with_reg.py](../mahjax_pt/examples/ppo_with_reg.py) | PPO 训练（已适配新接口） |
| [mahjax_pt/examples/networks/red_network.py](../mahjax_pt/examples/networks/red_network.py) | 赤麻将 Actor-Critic 网络 |
| [mahjax_pt/examples/networks/transformer.py](../mahjax_pt/examples/networks/transformer.py) | Transformer 编码器（LayerNorm eps=1e-6 对齐 JAX） |

---

## 第六阶段：测试与验证体系（约 1-2 小时）

> **目标**：理解项目的正确性保证——多层测试金字塔与金数据回放机制。

### 6.1 JAX 环境单元测试
**[tests/red_mahjong/](../tests/red_mahjong/)**

- `test_tile.py` — 牌编码正确性
- `test_hand.py` — 手牌操作（摸牌、舍牌、听牌判定）
- `test_meld.py` — 面子编码/解码
- `test_shanten.py` — 向听数计算
- `test_yaku.py` — 役种判定
- `test_env.py` — 环境初始化与 step
- `test_observe.py` — 观察构建
- `test_play.py` — 完整对局
- `test_special_case.py` — 特殊情形（九种九牌、四风连打等）
- `test_parity.py` — Red/No-Red 等价性
- `test_visualize.py` — SVG 渲染

### 6.2 PyTorch 精度验证 ★★★

这是项目的质量控制核心，采用四层验证金字塔：

```
         ┌─────────────────────────────────┐
         │  L4: 金数据回放                  │
         │  805 seeds × 完整对局            │
         │  逐 step 与 JAX 对比             │
         │  replay_pt_against_golden.py     │
         │  replay_parallel_against_golden.py│
         ├─────────────────────────────────┤
         │  L3: 环境集成                    │
         │  Serial ↔ Parallel 等价          │
         │  关键分支覆盖                    │
         │  test_env_parallel_parity.py     │
         │  test_env_branches.py            │
         ├─────────────────────────────────┤
         │  L2: 框架一致性                  │
         │  JAX ↔ PyTorch 数值等价          │
         │  test_exact_parity.py            │
         │  test_full_ppo_parity.py         │
         │  test_mha_definitive.py          │
         ├─────────────────────────────────┤
         │  L1: 基础单元                    │
         │  纯 PyTorch 逻辑正确性           │
         │  test_cases.py (16组80断言)      │
         │  test_aligned_math.py            │
         └─────────────────────────────────┘
```

#### PPO 精度专项（六层递进）

| 层级 | 测试文件 | 验证目标 |
|------|---------|---------|
| L1 | `test_ppo_math_parity.py` | 数学原语（log_prob, entropy, advantage 等）JAX vs PT 对比 |
| L2 | `test_ppo_gae_parity.py` | GAE 计算 vs JAX `calculate_gae` |
| L3 | `test_ppo_weight_transfer.py` | ACNet 权重迁移显式映射 |
| L4 | `test_ppo_update_parity.py` | PPO loss/grad/param 对比 |
| L4 Ext | `test_ppo_acnet_parity.py` | 完整 ACNet PPO loss 对比 |
| L5 | `test_ppo_cycle_parity.py` | 单次 update cycle |
| L6 | `test_ppo_training_parity.py` | 多步训练稳定性 |

#### 金数据工具链

| 文件 | 功能 |
|------|------|
| `record_jax_golden.py` | 录制 JAX 环境的金数据（输入、输出、中间状态） |
| `record_jax_acnet_golden_f64.py` | 录制 ACNet 金数据（float64 精度） |
| `record_jax_ppo_golden.py` | 录制 PPO 金数据 |
| `replay_pt_against_golden.py` | 用 PyTorch 串行环境回放金数据，逐 step 对比 |
| `replay_parallel_against_golden.py` | 用 PyTorch 并行环境回放金数据，支持 `-j` 多进程 |

> **关键事实**：805 seeds 的金数据回放 100% 通过，验证了 PyTorch 移植的完全正确性。

---

## 第七阶段：周边设施（约 1 小时）

### 7.1 网页 UI
**[mahjax/ui/](../mahjax/ui/)**

- [app.py](../mahjax/ui/app.py) — FastAPI 应用，`create_app()` 工厂
- [game_manager.py](../mahjax/ui/game_manager.py) — 游戏状态机管理
- [agents.py](../mahjax/ui/agents.py) — 代理注册表，支持动态加载自定义 agent

### 7.2 环境包装器
- [mahjax/wrappers/auto_reset_wrapper.py](../mahjax/wrappers/auto_reset_wrapper.py) — 对局结束自动重置
- [mahjax_pt/red_mahjong/auto_reset_wrapper.py](../mahjax_pt/red_mahjong/auto_reset_wrapper.py) — PyTorch 版自动重置，兼容 serial/parallel 两种后端

### 7.3 SVG 可视化
- [mahjax/_src/visualizer.py](../mahjax/_src/visualizer.py) — 通用 SVG 可视化器基类
- [mahjax/red_mahjong/visualization.py](../mahjax/red_mahjong/visualization.py) — 麻将专用的 `render_round_svg`，支持 `standard` 和 `bilingual` 两种牌面风格

### 7.4 无赤麻将
**[mahjax/no_red_mahjong/](../mahjax/no_red_mahjong/)**

赤麻将的简化版，规则更少、速度更快（~2x）。理解红麻将后对比阅读可以加深对二者差异的理解。

### 7.5 其余文件
| 路径 | 说明 |
|------|------|
| [Mortal/](../Mortal/) | Mortal 机器人代码（参考用，非本项目核心） |
| [script/](../script/) | 项目级辅助脚本 |
| [openspec/](../openspec/) | 设计规格文档 |
| [Makefile](../Makefile) | 开发命令（`install-dev`, `format`, `check`, `test` 等） |
| [mkdocs.yml](../mkdocs.yml) | 文档站点配置 |

---

## 最小阅读集

如果时间极其有限，以下 8 个文件构成理解本项目的最短路径：

| 优先级 | 文件 | 行数 | 为什么 |
|--------|------|------|--------|
| 🔴 | [README.md](../README.md) | 199 | 项目全貌、API、规则、引用 |
| 🔴 | [design.md](../openspec/changes/mahjax-pt-dual-env-refactor/design.md) | 318 | 架构全景图，理解双轨设计 |
| 🔴 | [mahjax/red_mahjong/state.py](../mahjax/red_mahjong/state.py) | 141 | 核心状态定义 |
| 🔴 | [mahjax/red_mahjong/constants.py](../mahjax/red_mahjong/constants.py) | 84 | 所有关键数字的来源 |
| 🔴 | [mahjax/red_mahjong/env.py](../mahjax/red_mahjong/env.py) | 2,257 | JAX 参考实现（主引擎） |
| 🟡 | [mahjax/core.py](../mahjax/core.py) | 280 | 工厂和基类 API |
| 🟡 | [mahjax_pt/red_mahjong/env_serial.py](../mahjax_pt/red_mahjong/env_serial.py) | 1,601 | PT 串行参考（JAX 1:1 对照） |
| 🟡 | [mahjax_pt/red_mahjong/env_parallel.py](../mahjax_pt/red_mahjong/env_parallel.py) | 375 | PT 并行编排层 |

---

## 核心设计模式总结

理解以下模式会极大加速代码阅读：

### 1. Struct-of-Arrays（JAX）
```python
# 不可变的 struct-of-arrays：所有玩家数据打包为 (4, N) 张量
@dataclass
class PlayerStateArrays:
    hand: jnp.ndarray          # (4, 34)   — 每人的手牌计数
    river: jnp.ndarray         # (4, 60)   — 每人的牌河
    melds: jnp.ndarray         # (4, 4)    — 每人的副露
    # ...

# 沿 player 维并行计算
v_hand_op = jax.vmap(hand_op, in_axes=0)  # 自动沿 axis=0 并行
```

### 2. 可变 Dataclass（PyTorch Serial）
```python
# 可变 EnvState：复用同一结构但直接修改属性
@dataclass
class EnvState:
    hand: torch.Tensor         # (4, 34)
    river: torch.Tensor        # (4, 60)
    # ...

# 直接赋值修改（对标 JAX 的不可变创建）
state.hand[cp] = new_hand
```

### 3. Mask 驱动的批量化控制流（PyTorch Parallel）
```python
# 所有 if/else 转为 boolean mask + tensor indexing
mask = (action_type == DISCARD)  # (B,) bool
idx = mask.nonzero().squeeze(-1) # (K,)  — 需要处理的环境子集
batch_state = handler(batch_state, idx)  # 只在 K 个环境上操作
```

### 4. Facade 模式
```python
# env.py 不包含业务逻辑，只做路由
class RedMahjong(Env):
    def __init__(self, backend="serial", **kwargs):
        self._impl = (RedMahjongSerial if backend == "serial"
                      else RedMahjongParallel)(**kwargs)
    def step(self, state, action, key=None):
        return self._impl.step(state, action, key)
```

### 5. Mixin 模式
```python
# 并行环境的三个 Mixin 各自独立，职责清晰
class RedMahjongParallel(HandlersMixin, InternalsMixin, Env):
    # env_parallel.py:         编排层 (258 行)
    # env_parallel_handlers.py: 动作处理器 (1,180 行)
    # env_parallel_internals.py: 内部机制 (650 行)
```

---

## 建议阅读时间表

```
Week 1: 阶段 1（概念入门）+ 阶段 2（数据模型）
        └── 产出：能读/写麻将状态的代码

Week 2: 阶段 3（JAX 核心引擎）
        └── 产出：理解 env.py 的完整生命周期

Week 3: 阶段 4（PyTorch 双轨架构）
        └── 产出：理解 serial/parallel 的转换方式

Week 4: 阶段 5（RL 管线）+ 阶段 6（测试体系）+ 阶段 7（周边）
        └── 产出：端到端理解训练→推理→部署全链路
```

---

## 相关资源

- [项目仓库](https://github.com/nissymori/mahjax)
- [在线文档](https://nissymori.github.io/mahjax/)
- [学术论文](https://arxiv.org/abs/2605.20577)
- [天凤规则（英文）](https://tenhou.net/0/mj/mjlog/en/mjlog-en-rules.html)
- [欧洲立直麻将规则（2025）](http://mahjong-europe.org/portal/images/docs/Riichi-rules-2025-EN.pdf)
- [Pgx（API 设计参考）](https://github.com/sotetsuk/pgx)
