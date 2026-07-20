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
│   ├── __init__.py                  # 公开 API 导出（23 项）
│   ├── core.py                      # Env/State 基类、make() 工厂
│   ├── _src/                        # 共享基础设施
│   │   ├── struct.py                # @dataclass 装饰器（flax.struct 包装）
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

## 使用指南：从安装到训练

> **目标**：能跑起来——安装、测试、训练、UI、精度验证的完整实操命令。

### 环境安装

#### 作为用户安装（使用 PyPI 发布版）

```bash
pip install mahjax
```

PyPI 包依赖 `jax>=0.4.28`，请根据自己的硬件（CPU/GPU/TPU）安装对应的 `jaxlib`。

#### 作为开发者安装（从源码）

```bash
git clone https://github.com/nissymori/mahjax.git
cd mahjax
pip install -e ".[dev]"
# 或者用 Makefile：
make install-dev
```

`make install-dev` 会安装 dev、lint、typing、test、coverage 五组依赖：
- **lint**: `ruff>=0.14.11`, `blackdoc>=0.3.9`
- **typing**: `mypy>=1.19.1`
- **test**: `pytest>=8.4.2`, `pytest-xdist>=3.8.0`
- **coverage**: `pytest-cov>=7.0.0`
- **核心**: `jax>=0.4.28`, `fastapi>=0.128.0`, `svgwrite>=1.4.3`, `uvicorn>=0.39.0`

PyTorch 端需要额外安装 `torch>=2.0`（不作为 pip 依赖自动安装，因为不同硬件的 torch 包名不同）。

---

### JAX 环境基本使用

#### 最简示例（API 对标 Pgx）

```python
import jax
import jax.numpy as jnp
import mahjax

# 创建环境
env = mahjax.make(
    "red_mahjong",                         # 或 "no_red_mahjong"
    round_mode="single",                   # "single" | "east" | "half"
    observe_type="dict",                   # "dict" (Transformer) | "2D" (CNN, 未完成)
    order_points=[30, 10, -10, -30],       # 顺位点 (uma)
    next_round_style="auto",               # "auto" (训练) | "dummy_share" (交互/UI)
)

# JIT + vmap：批量运行
batch_size = 10
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))
obs_fn  = jax.jit(jax.vmap(env.observe))

# 初始化
rng = jax.random.PRNGKey(0)
rng, subrng = jax.random.split(rng)
state = init_fn(jax.random.split(subrng, batch_size))

# 执行一步（tsumogiri）
rng, subrng = jax.random.split(rng)
action = jnp.full((batch_size,), mahjax.Action.TSUMOGIRI, dtype=jnp.int32)
state = step_fn(state, action, jax.random.split(subrng, batch_size))

# 获取观察和奖励
obs = obs_fn(state)          # Dict 格式，可直接送入 Transformer
reward = state.rewards       # (batch_size, 4) 每步的即时奖励

# 可视化（仅单个 state，不 batch）
single_state = env.init(jax.random.PRNGKey(1))
single_state.save_svg("state.svg", tile_style="bilingual")
```

#### round_mode 说明

| 模式 | 局数 | 说明 |
|------|------|------|
| `"single"` | 1 局 | 单局结束即终止，适合快速实验 |
| `"east"` | 4 局 | 东风战（tonpuusen），`round_limit=4` |
| `"half"` | 8 局 | 半庄战（hanchan），`round_limit=8` |

#### next_round_style 说明

| 模式 | 默认 | 适用场景 |
|------|------|---------|
| `"auto"` | ✅ | RL 训练：一局结束的 step 返回下一局的 init state，rewards 携带结算奖励 |
| `"dummy_share"` | | 交互式 UI / mjai 兼容回放：每局结束后需要 4 家各打一个 DUMMY 才能推进 |

---

### PyTorch 环境基本使用

#### 串行模式（正确性验证）

```python
from mahjax_pt.red_mahjong.env import make

env = make(backend="serial", round_mode="half")
state = env.init()                         # key=None 自动生成随机种子

# 取合法动作
mask = state.legal_action_mask
legal_actions = [i for i, m in enumerate(mask.tolist()) if m]

state = env.step(state, legal_actions[0])  # 单步执行
obs = env.observe(state)                   # 获取观察
```

#### 并行模式（GPU/NPU 批量训练）

```python
from mahjax_pt.red_mahjong.env import make

env = make(backend="parallel", round_mode="half")

# 批量初始化 128 个环境
batch_state = env.init_batch(num_envs=128, device="cuda")  # 或 "npu" / "cpu"

# 批量执行一步：actions 是 (B,) tensor
import torch
actions = torch.randint(0, 37, (128,))
batch_state = env.step_batch(batch_state, actions)

# 批量观察：直接返回 (B, ...) 张量，无需 stack/unstack
batch_obs = env.observe_batch(batch_state)

# 自动重置已终止的环境
batch_state = env.reinit_terminated_batch(batch_state)
```

---

### 运行测试

#### JAX 环境单元测试（mahjax/）

```bash
# 全量测试（4 进程并行，含 doctest）
make test

# 等价于：
python3 -m pytest -n 4 -vv tests --doctest-modules mahjax --ignore mahjax/experimental

# 带覆盖率报告
make test-with-codecov

# 只跑赤麻将测试
python3 -m pytest tests/red_mahjong/ -v

# 只跑无赤麻将测试
python3 -m pytest tests/no_red_mahjong/ -v

# 只跑某个特定测试
python3 -m pytest tests/red_mahjong/test_env.py -v
```

#### PyTorch 精度验证测试（mahjax_pt/）

```bash
# L1 基础单元测试
python mahjax_pt/tests/test_cases.py

# L2 JAX ↔ PyTorch 等价测试
python mahjax_pt/tests/test_exact_parity.py

# L3 Serial ↔ Parallel 等价测试
python mahjax_pt/tests/test_env_parallel_parity.py

# L4 金数据回放（805 seeds，串行模式）
python mahjax_pt/tests/replay_pt_against_golden.py

# L4 金数据回放（并行模式，多进程加速）
python mahjax_pt/tests/replay_parallel_against_golden.py -j 8

# PPO 精度全套
python mahjax_pt/tests/test_ppo_math_parity.py
python mahjax_pt/tests/test_ppo_gae_parity.py
python mahjax_pt/tests/test_ppo_update_parity.py
python mahjax_pt/tests/test_ppo_acnet_parity.py
python mahjax_pt/tests/test_ppo_cycle_parity.py
python mahjax_pt/tests/test_ppo_training_parity.py

# 一键运行所有 PT 测试
python mahjax_pt/tests/run_tests.py
```

---

### RL 训练

#### 第一步：收集离线数据（JAX 端）

```bash
cd examples
python collect_offline_data.py \
    env_name=no_red_mahjong \
    dataset_path=./offline_data/no_red_offline_data.pkl
```

用 rule_based agent 自对弈，收集 (observation, action, mask) 三元组。

#### 第二步：行为克隆（BC）预训练

```bash
# JAX 版 BC
cd examples
python bc.py \
    env_name=no_red_mahjong \
    dataset_path=./offline_data/no_red_offline_data.pkl \
    batch_size=1024 lr=3e-4 num_epochs=5 seed=42

# PyTorch 版 BC
cd mahjax_pt
python examples/bc.py \
    --env_name red_mahjong \
    --dataset_path ./data/red_offline_data.pkl \
    --batch_size 1024 --lr 3e-4 --num_epochs 5
```

输出：训练好的模型参数文件（`.pkl` 或 `.pt`）。

#### 第三步：PPO 强化学习训练

```bash
# JAX 版 PPO（使用 wandb 记录日志）
cd examples
python ppo_with_reg.py \
    env_name=no_red_mahjong \
    round_mode=single \
    seed=0 \
    num_envs=1024 num_steps=256 \
    pretrained_model_path=./params/no_red_mahjong_bc_params.pkl \
    wandb_project=mahjax-ppo-with-reg

# PyTorch 版 PPO（支持 GPU/NPU 加速）
cd mahjax_pt
python examples/ppo_with_reg.py \
    --env_name red_mahjong \
    --round_mode single \
    --seed 0 \
    --num_envs 1024 --num_steps 256 \
    --backend parallel \
    --device cuda \
    --use_wandb
```

**关键超参数默认值**（JAX 和 PT 两端一致）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 1024 | 并行环境数 |
| `num_steps` | 256 | 每次 rollout 步数 |
| `total_timesteps` | 100,000,000 | 总训练步数 |
| `update_epochs` | 4 | 每轮更新的 epoch 数 |
| `minibatch_size` | 4096 | 小批次大小 |
| `gamma` | 1.0 | 折扣因子（全 episode 范围） |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `lr` | 3e-4 | 学习率（AdamW） |
| `clip_eps` | 0.2 | PPO clip 范围 |
| `ent_coef` | 0.01 | 熵正则化系数 |
| `vf_coef` | 0.5 | 价值损失系数 |
| `mag_coef` | 0.2 | MAGNET 正则化系数 |

---

### 启动 Web UI

```bash
# 安装后直接启动（默认 agent：rule_based、random）
uvicorn mahjax.ui.app:create_app --host 0.0.0.0 --port 8000

# 或开发模式（自动重载）
uvicorn mahjax.ui.app:create_app --host 0.0.0.0 --port 8000 --reload
```

打开浏览器访问 `http://localhost:8000`，即可：
- 选择 agent、局数、座位
- 手动打牌对抗 AI
- 支持双语牌面切换
- 查看对局日志和结算摘要

#### 注册自定义 Agent

```python
# my_ui_app.py
from pathlib import Path
from mahjax.ui.app import create_app

app = create_app()
app.state.manager.registry.load_callable_from_path(
    file_path=Path("path/to/my_agent.py"),
    attribute="act",               # 函数名：act(state, rng) -> action_id
    description="My Custom Agent",
)
```

```bash
uvicorn my_ui_app:app --host 0.0.0.0 --port 8000
```

---

### 精度验证流程

当你修改了 PyTorch 端代码后，建议按以下顺序验证正确性：

```bash
# 1. 基础单元（秒级）
python mahjax_pt/tests/test_cases.py

# 2. JAX ↔ PT 逐函数对比（分钟级）
python mahjax_pt/tests/test_exact_parity.py

# 3. Serial ↔ Parallel 等价（分钟级）
python mahjax_pt/tests/test_env_parallel_parity.py

# 4. 环境分支覆盖率（分钟级）
python mahjax_pt/tests/test_env_branches.py

# 5. 805 seeds 金数据回放（分钟级，多进程）
python mahjax_pt/tests/replay_pt_against_golden.py -j 8

# 6. 并行版金数据回放
python mahjax_pt/tests/replay_parallel_against_golden.py -j 8

# 7. PPO 精度全套（分钟级）
python mahjax_pt/tests/test_ppo_math_parity.py
python mahjax_pt/tests/test_ppo_gae_parity.py
python mahjax_pt/tests/test_ppo_update_parity.py
python mahjax_pt/tests/test_ppo_cycle_parity.py
python mahjax_pt/tests/test_ppo_training_parity.py
```

**如果全部通过**：PT 端所有逻辑与 JAX 参考实现一致。

### 录制新的金数据

如果修改了 JAX 端的环境逻辑，需要重新录制金数据：

```bash
# 录制环境金数据
python mahjax_pt/tests/record_jax_golden.py --num_seeds 1000

# 录制 ACNet 金数据（float64 精度，用于定位 fp32 漂移）
python mahjax_pt/tests/record_jax_acnet_golden_f64.py

# 录制 PPO 训练金数据
python mahjax_pt/tests/record_jax_ppo_golden.py
```

---

### 代码质量

```bash
# 格式化（ruff + blackdoc）
make format
# 等价于：
ruff format mahjax
blackdoc mahjax
ruff check mahjax --fix

# 静态检查（ruff + mypy）
make check
# 等价于：
ruff format mahjax --check
blackdoc mahjax --check
ruff check mahjax
mypy mahjax

# 清理构建产物
make clean
```

配置说明：
- 行宽：120 字符（`pyproject.toml` 中 `line-length = 120`）
- Python 最低版本：3.9
- Ruff 规则：`C90`（mccabe 复杂度，上限 18）、`E`（pycodestyle）、`F`（pyflakes）、`I`（isort）、`W`（pycodestyle warning）

---

## 阅读路线图

建议按以下七个阶段循序渐进，预估总耗时约 **15-20 小时**（约 3-4 周，每周 4-5 小时）。

---

## 第一阶段：概念入门（约 30 分钟）

> **目标**：理解"做什么"和"为什么"，形成整体认知。无需阅读代码。

### 1.1 麻将规则入门

如果你不熟悉日本立直麻将，先了解基本概念：

- [mahjong-basics.md](mahjong-basics.md) — 34 种牌（万/筒/索/字）、4 面子 + 1 雀头的和牌条件、役种概念、点数计算
- [rule.md](rule.md) — 赤麻将 vs 无赤麻将的区别、天凤（Tenhou）规则标准
- [red_mahjong.md](red_mahjong.md) — 赤麻将详细规则（含赤五牌、包牌、双响）
- [no_red_mahjong.md](no_red_mahjong.md) — 无赤麻将详细规则（简化版）

**核心概念速查**：

- **34 种基本牌**：万子 (0-8)、筒子 (9-17)、索子 (18-26)、字牌 (27-33)
- **3 种赤牌**：赤 5m (34)、赤 5p (35)、赤 5s (36)，共计 37 种牌
- **137 张物理牌**：34×4 + 3 赤牌（替换对应普通 5）
- **王牌区 14 张**：岭上牌 (4) + 宝牌指示牌 (5) + 里宝牌指示牌 (5)
- **和牌条件**：4 面子 + 1 雀头 + 至少 1 役种

### 1.2 项目定位

- [README.md](../README.md) — 项目整体介绍、快速开始示例代码、支持的 API
- [docs/index.md](index.md) — 文档索引页
- [pyproject.toml](../pyproject.toml) — 了解依赖项：`jax>=0.4.28`、`fastapi>=0.128.0`、`svgwrite>=1.4.3`
- [Makefile](../Makefile) — 开发命令：`make install-dev`（安装）、`make test`（测试，含 `-n 4` 并行）、`make format`（ruff + blackdoc）、`make check`（lint + mypy）

### 1.3 当前分支背景（必读）

- [openspec/changes/mahjax-pt-dual-env-refactor/proposal.md](../openspec/changes/mahjax-pt-dual-env-refactor/proposal.md) — 为什么需要重构：混合架构的四大问题
    1. **正确性验证困难**：混合代码需同时理解串行+批处理两条路径
    2. **性能瓶颈不可见**：串行回退路径成为 GPU/NPU 隐式热点
    3. **代码维护复杂**：同一逻辑在 `_draw`/`_draw_batch` 中重复
    4. **架构方向不清晰**：无法区分"参考实现"和"生产实现"
- [openspec/changes/mahjax-pt-dual-env-refactor/design.md](../openspec/changes/mahjax-pt-dual-env-refactor/design.md) — 架构全景图：三组件（Facade、Serial、Parallel）与 Mixin 拆分方案

---

## 第二阶段：核心数据模型（约 1-2 小时）

> **目标**：理解麻将的"数据结构"——牌、面子、手牌、状态定义。这是所有游戏逻辑的基石。

### 2.1 常量定义
**[mahjax/red_mahjong/constants.py](../mahjax/red_mahjong/constants.py)**（84 行）

关注以下关键常量的精确值及推导逻辑：

| 常量 | 值 | 含义 |
|------|-----|------|
| `NUM_PLAYERS` | 4 | 玩家数 |
| `NUM_TILE_TYPES` | 34 | 基本牌种数（万 0-8 + 筒 9-17 + 索 18-26 + 字 27-33） |
| `NUM_TILE_TYPES_WITH_RED` | 37 | 含赤牌的牌种数（34 + 赤 5m/5p/5s） |
| `NUM_PHYSICAL_TILES` | 136 | 物理牌总数（不含赤牌替代逻辑） |
| `LEGAL_ACTION_SIZE` | 87 | 动作空间大小 |
| `MAX_DISCARDS_PER_PLAYER` | 60 | 每玩家最大舍牌数 |
| `MAX_MELDS_PER_PLAYER` | 4 | 每玩家最大副露数 |
| `MAX_HAND_TILES` | 14 | 手牌最大张数 |
| `MAX_DORA_INDICATORS` | 5 | 最多宝牌指示牌 |
| `COPIES_PER_TILE` | 4 | 每种牌的数量 |
| `DEAD_WALL_TILES` | 14 | 王牌区张数（岭上 4 + 表宝牌指示 5 + 里宝牌指示 5） |
| `STARTING_POINTS` | 250 | 起始点数（×100 = 25000 点） |
| `TARGET_POINTS` | 300 | 目标点数（×100 = 30000 点，飞人判定） |
| `RIICHI_BET` | 1000 | 立直供托（点） |
| `HONBA_BONUS` | 100 | 本场棒（每本场 +100 点） |

其他重要数组常量：
- `DORA_ARRAY`: 34 元素数组，将每种牌映射到其宝牌指示牌类型
- `ZERO_MASK_1D` / `ZERO_MASK_2D`: 预分配的零掩码张量
- `TILE_RANGE`: `jnp.arange(NUM_PHYSICAL_TILES)`，牌索引范围
- `FIRST_DRAW_IDX`: 首轮摸牌时的牌山索引（83）
- `FALSE` / `TRUE`: `jnp.bool_(False/True)`，类型安全的布尔常量

### 2.2 牌的编码
**[mahjax/red_mahjong/tile.py](../mahjax/red_mahjong/tile.py)**（189 行）

#### Tile 类（静态方法集合）

核心编码方案：
- **基本牌编码 0-33**：`(suit * 9 + rank)`，万=0 筒=1 索=2 字=3
- **赤牌编码 34-36**：赤 5m=34, 赤 5p=35, 赤 5s=36
- **红/黑转换**：`to_red(tile_type)` 将普通 5 转为赤 5；`to_tile_type(tile)` 将赤牌转回普通 5

关键方法：
- `to_tile_type(tile)` — 赤牌→普通牌（L30+）
- `to_red(tile_type)` — 普通 5→赤 5（L40+）
- `is_tile_type_five(t)` — 判断是否为 5（用于处理赤牌特殊情况）
- `tile_to_str(tile)` — 牌的文本表示

#### River 类

牌河的 **16-bit 压缩编码**：
```
bit 0-5:   牌值 (0-36)
bit 6-7:   来源 (0=自摸, 1=吃, 2=碰, 3=大明杠)
bit 8-11:  来源玩家相对位置 (0-3)
bit 12:    是否为立直宣言后的第一张舍牌
bit 13-15: 保留
```
- `EMPTY_RIVER`: 空牌河标记
- `add_meld(river, action, discarder, discard_idx, rel_src)` — 标记被鸣的牌
- `is_riichi_discard(river_entry)` — 判断是否为立直宣言牌

### 2.3 动作常量
**[mahjax/red_mahjong/action.py](../mahjax/red_mahjong/action.py)**（41 行）

87 种动作的完整分类：

| 动作 | 编码范围 | 数量 | 说明 |
|------|---------|------|------|
| 舍牌 | 0-36 | 37 | 0-36 对应 37 种牌（含赤牌） |
| `TSUMOGIRI` | 36 | 1 | 将刚摸的牌直接舍出 |
| 杠（暗杠/加杠） | 37-70 | 34 | 每种普通牌对应一个杠动作 |
| `RIICHI` | 74 | 1 | 立直宣言（在不换牌听牌时） |
| `RON` | 75 | 1 | 荣和（和别人的舍牌） |
| `TSUMO` | 76 | 1 | 自摸和 |
| `PON` | 77 | 1 | 碰（普通 5） |
| `PON_RED` | 78 | 1 | 碰赤 5 |
| `CHI_L/L_RED` | 79-80 | 2 | 吃左侧（普通/赤） |
| `CHI_M/M_RED` | 81-82 | 2 | 吃中间（普通/赤） |
| `CHI_R/R_RED` | 83-84 | 2 | 吃右侧（普通/赤） |
| `OPEN_KAN` | 85 | 1 | 大明杠 |
| `PASS` | 86 | 1 | 过（不鸣牌/不和） |
| `KYUUSHU` | 87 | 1 | 九种九牌流局 |
| `DUMMY` | 88 | 1 | 占位动作（用于 dummy_share 模式） |

关键常量：`NUM_ACTION = 89`（实际 0-88 共 89 个）

### 2.4 面子编码
**[mahjax/red_mahjong/meld.py](../mahjax/red_mahjong/meld.py)**（202 行）

#### Meld 类的 16-bit 压缩编码

```
Bits 0-5:   牌类型 (0-36, 赤牌感知)
Bits 6-8:   面子来源 (0=暗/1=吃/2=碰/4=大明杠/8=暗杠/16=加杠)
Bits 9-11:  来源玩家相对位置 (0-3)
Bits 12-13: 赤牌标记
Bits 14-15: 保留
```

关键方法：
- `action(meld)` — 从面子编码中提取来源类型
- `src(meld)` — 从面子编码中提取来源玩家
- `tile_type(meld)` — 提取面子中的牌类型
- `meld_to_str(meld)` — 面子文本表示
- `EMPTY_MELD` — 空面子标记值

### 2.5 手牌操作
**[mahjax/red_mahjong/hand.py](../mahjax/red_mahjong/hand.py)**（374 行）

#### Hand 类（静态方法集合，使用预计算缓存 `CACHE`）

核心方法：
- `draw(hand_37, tile)` — 摸牌：`hand_37[tile] += 1`
- `discard(hand_37, tile_idx)` — 舍牌：`hand_37[discard_tile] -= 1`
- `sub(hand_37, tile)` — 减去一张指定牌
- `can_ron(hand_37, tile)` — 判断是否能用 `tile` 和牌（`number(hand_37 + tile) <= 0`）
- `can_tsumo(hand_37)` — 判断是否能自摸和（对任意牌 `can_ron` 为真）
- `can_riichi(hand_37, tile)` — 判断舍弃 `tile` 后是否听牌（用于立直判定）
- `to_34(hand_37)` — 将 37 维手牌转为 34 维（合并赤牌）
- `can_kyuushu(hand_37)` — 九种九牌判定：幺九牌种类 ≥ 9
- `can_pon(hand_37, tile)` — 判断是否能碰（手牌中该牌 ≥ 2 张）
- `can_chi(hand_37, tile, chi_idx)` — 判断是否能吃
- `can_open_kan(hand_37, tile)` — 判断是否能大明杠（≥ 3 张）
- `can_closed_kan(hand_37)` — 判断是否有暗杠
- `can_added_kan(hand_37, melds)` — 判断是否有加杠
- `_chi_index(action)` / `_base_chi_action(chi_idx)` — 吃的 action 编码/解码
- `is_hand_concealed(melds)` — 判断手牌是否为门清

预计算缓存：
- `CACHE = load_hand_cache()` — 从 `mahjax/_src/cache/hand_cache.npz` 加载
- `THIRTEEN_ORPHAN_IDX` — 十三幺的 13 种幺九牌的索引
- `POWERS_OF_5_FULL` — 5 的各次幂数组，用于手牌哈希
- `KYUUSHU_MASK` — 九种九牌的幺九牌掩码

> **JAX 并行化关键**：所有 Hand 方法都是纯函数（输入手牌数组，返回结果数组），通过 `jax.vmap(Hand.xxx, in_axes=(0, ...))` 自动沿 player 维并行计算。

### 2.6 状态定义 ★★★
**[mahjax/red_mahjong/state.py](../mahjax/red_mahjong/state.py)**（141 行）

这是最重要的数据结构文件，定义了三层嵌套状态：

#### GameConfig（游戏配置）

```python
@dataclass
class GameConfig:
    allow_open_tanyao: jnp.bool_ = TRUE     # 允许食断（副露后断幺九依然有效）
    allow_kuikae: jnp.bool_ = FALSE         # 允许食替（吃后立即舍同种牌）
    use_red_fives: jnp.bool_ = TRUE         # 使用赤五牌
    allow_double_ron: jnp.bool_ = TRUE      # 允许双响（两人同时荣和）
    enable_special_abortive_draw: jnp.bool_ = TRUE  # 启用特殊流局
    enable_pao: jnp.bool_ = TRUE            # 启用包牌规则
    seed_wall_from_key: jnp.bool_ = TRUE    # 从 JAX key 控制牌山随机
    starting_points: jnp.int32 = 250        # 起始点数（×100）
    target_points: jnp.int32 = 300          # 目标点数（×100）
    honba_bonus: jnp.int32 = 100            # 本场棒
    riichi_bet: jnp.int32 = 1000            # 立直供托
```

#### PlayerStateArrays（struct-of-arrays 核心）

所有玩家数据打包为 `(4, ...)` 形状的不可变 `jnp.ndarray`：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `hand` | (4, 34) | int8 | 每人的手牌计数（34 维，赤牌合并） |
| `hand_with_red` | (4, 37) | int8 | 每人的手牌计数（37 维，赤牌分离） |
| `hand_ids` | (4, 14) | int16 | 每人的手牌 ID 列表（有序） |
| `hand_counts` | (4,) | int8 | 每人的手牌张数 |
| `drawn_tile` | (4,) | int16 | 每人刚摸到的牌 |
| `legal_action_mask` | (4, 87) | bool | 每人的合法动作掩码 |
| `can_win` | (4, 34) | bool | 每人每种牌是否能荣和 |
| `has_yaku` | (4, 2) | bool | 是否有役（0=荣和, 1=自摸） |
| `fan` | (4, 2) | int32 | 番数 |
| `fu` | (4, 2) | int32 | 符数 |
| `melds` | (4, 4) | uint16 | 每人的副露（压缩编码） |
| `meld_tiles` | (4, 4, 4) | int16 | 每人的副露中的具体牌 |
| `meld_info` | (4, 4, 3) | int8 | 每人的副露元信息 |
| `meld_counts` | (4,) | int8 | 每人的副露数 |
| `river` | (4, 60) | uint16 | 每人的牌河（压缩编码） |
| `discards` | (4, 60) | int16 | 每人的舍牌列表 |
| `discard_info` | (4, 60, 4) | int8 | 每人的舍牌元信息 |
| `discard_counts` | (4,) | int8 | 每人的舍牌数 |
| `riichi` | (4,) | bool | 是否已立直 |
| `riichi_declared` | (4,) | bool | 本轮是否宣言立直 |
| `riichi_step` | (4,) | int8 | 立直宣言后的步数（用于一发判定） |
| `double_riichi` | (4,) | bool | 是否为 w 立直 |
| `ippatsu` | (4,) | bool | 是否在一发巡内 |
| `furiten_by_discard` | (4,) | bool | 是否因舍牌而振听 |
| `furiten_by_pass` | (4,) | bool | 是否因放过而振听 |
| `is_hand_concealed` | (4,) | bool | 手牌是否为门清 |
| `pon` | (4, 34) | int32 | 碰的计数 |
| `has_won` | (4,) | bool | 是否已和牌 |
| `n_kan` | (4,) | int8 | 杠的数量 |
| `has_nagashi_mangan` | (4,) | bool | 是否可能流局满贯 |

#### RoundState（回合级状态）

| 字段 | 形状 | 说明 |
|------|------|------|
| `action_history` | (3, 200) | 动作历史：[玩家, 动作, tsumogiri] |
| `round` | 标量 | 当前局数 (0-7 或 0-3) |
| `round_limit` | 标量 | 总局数限制 |
| `dealer` | 标量 | 庄家索引 (0-3) |
| `honba` | 标量 | 本场数 |
| `kyotaku` | 标量 | 供托数（立直棒累积） |
| `score` | (4,) | 各玩家分数 |
| `deck` | (136,) | 牌山 |
| `next_deck_ix` | 标量 | 下一个要摸的牌的索引 |
| `last_deck_ix` | 标量 | 王牌区截止索引（杠后递减） |
| `dora_indicators` | (5,) | 宝牌指示牌列表 |
| `ura_dora_indicators` | (5,) | 里宝牌指示牌列表 |
| `draw_next` | 标量 | 是否为摸牌回合（而非鸣牌响应回合） |
| `init_wind` | (4,) | 各玩家初始风位 |
| `seat_wind` | (4,) | 各玩家座风 (0=东, 1=南, 2=西, 3=北) |

#### State（顶层状态，继承 `mahjax.core.State`）

```python
@dataclass
class State(mahjax.core.State):
    current_player: Array       # 当前行动玩家
    rewards: Array              # (4,) 奖励
    terminated: Array           # 游戏是否结束
    truncated: Array            # 是否被截断
    legal_action_mask: Array    # (89,) 当前玩家合法动作
    _step_count: Array          # 步数计数
    players: PlayerStateArrays  # 玩家状态
    round_state: RoundState     # 回合状态
```

> **核心设计模式**：JAX 使用不可变 `@dataclass`（源自 flax.struct）实现 struct-of-arrays。PlayerStateArrays 将所有 4 人的数据沿 player 轴打包，通过 `jax.vmap` 沿该轴并行——这是整个系统高性能的关键。

---

## 第三阶段：JAX 参考实现——核心引擎（约 3-4 小时）

> **目标**：深入理解 JAX 版麻将环境。这是整个项目的"金标准"，PyTorch 端的一切都对照它验证。

### 3.1 工厂与基类
**[mahjax/core.py](../mahjax/core.py)**（280 行）

#### State 抽象基类

```python
@dataclass
class State(abc.ABC):
    current_player: Array       # 当前行动玩家
    rewards: Array              # 奖励数组
    terminated: Array           # 终止标志
    truncated: Array            # 截断标志
    legal_action_mask: Array    # 合法动作掩码
    _step_count: Array          # 步数
```

- `env_id` 抽象属性 — 环境标识符
- `to_svg()` — 在 Notebook 中渲染 SVG
- `save_svg()` — 保存 SVG 到文件

#### Env 抽象基类

```python
class Env(abc.ABC):
    def init(self, key: PRNGKey) -> State: ...
    def step(self, state: State, action: Array, key=None) -> State: ...
    def observe(self, state: State) -> Array: ...
    @property
    def id(self) -> EnvId: ...
    @property
    def num_players(self) -> int: ...
    @property
    def num_actions(self) -> int: ...
    @property
    def observation_shape(self) -> Tuple[int, ...]: ...
```

#### make() 工厂函数
- 根据 `env_id` 字符串路由：`"no_red_mahjong"` → `NoRedMahjong`, `"red_mahjong"` → `RedMahjong`
- `EnvId = Literal["no_red_mahjong", "red_mahjong"]`
- `available_envs()` 枚举所有注册环境

### 3.2 JAX 赤麻将环境 ★★★
**[mahjax/red_mahjong/env.py](../mahjax/red_mahjong/env.py)**（2,257 行，全项目最大文件）

#### 核心架构

**动作分发机制**（L39-54）：
```python
ACTION_FUN_MAP = jnp.zeros(Action.NUM_ACTION, dtype=jnp.int32)
# 0=discard, 1=kan, 2=riichi, 3=ron, 4=tsumo, 5=pon,
# 6=chi, 7=pass, 8=kyuushu, 9=dummy
```
通过 `jax.lax.switch(ACTION_FUN_MAP[action], branches, state, action)` 实现 JIT 兼容的动作分发。

**状态更新辅助函数**：

- `_replace_state(state, **updates)`（L122）：核心工具函数。根据字段名自动将更新路由到 `state.players`、`state.round_state` 或 `state` 本体。依赖 `_PLAYER_FIELDS`（L56-87）和 `_ROUND_FIELDS`（L89-119）两个集合进行分类。
- `_make_state(**updates)`（L156）：从默认状态创建新状态并应用更新

#### 完整函数索引（按行号）

##### 初始化管线

| 函数 | 行号 | 功能 |
|------|------|------|
| `_init(rng, game_config)` | L402 | **顶层初始化**：创建初始 State，包含配牌、宝牌翻示、向听计算 |
| `_init_wall(rng)` 内嵌 | ~L410 | Fisher-Yates 洗牌算法（用 `jax.random.permutation`），生成 136 张牌山 |
| `_init_hand(deck)` 内嵌 | ~L420 | 配牌：庄家 14 张，其余 13 张，调用 `_sort_hand` |
| `_init_dora_indicators` 内嵌 | ~L440 | 翻第一张宝牌指示牌 |
| `_init_for_next_round(key, state, game_config)` | L461 | 为下一局初始化：重置牌山、重新配牌 |
| `_prepare_next_round_assets(key, state, game_config)` | L473 | 预计算下一局所需资源（牌山、初始手牌） |
| `_init_for_next_round_from_prepared(state, ...)` | L488 | 从预准备的资源构建下一局状态 |
| `_calc_wind(east_player)` | L538 | 计算各玩家座风 |
| `_is_first_turn(next_deck_ix)` | L544 | 判断是否为首巡（用于 w 立直和九种九牌） |

##### 牌山与摸牌

| 函数 | 行号 | 功能 |
|------|------|------|
| `_live_wall_end_ix(state)` | L173 | 活墙截止索引（`last_deck_ix`，杠后递减） |
| `_draw(state, game_config)` | L788 | **摸牌主函数**：从牌山摸一张牌、处理立直接受、一发失效 |
| `_draw_after_kan(state, game_config)` | L1233 | 杠后摸牌（从王牌区摸） |

##### 舍牌与立直

| 函数 | 行号 | 功能 |
|------|------|------|
| `_discard(state, tile, game_config)` | L939 | **舍牌主函数**：将牌放入牌河、处理 tsumogiri、更新振听状态 |
| `_riichi(state)` | L1675 | 立直宣言：设置 `riichi_declared` 标志 |
| `_accept_riichi(state)` | L1176 | 接受立直：扣 1000 点、设置 riichi/ippatsu/double_riichi 标志 |

##### 副露（鸣牌）

| 函数 | 行号 | 功能 |
|------|------|------|
| `_pon(state, action)` | L1497 | 碰：复制手牌、添加面子、赤牌调整 |
| `_chi(state, action)` | L1539 | 吃：根据 CHI_L/CHI_M/CHI_R 构造顺子 |
| `_open_kan(state)` | L1466 | 大明杠：从其他玩家的舍牌声明杠 |
| `_kan(state, action, game_config)` | L1322 | 杠的总入口：分派到暗杠/加杠/大明杠 |
| `_selfkan(state, action, is_added_kan)` | L1412 | 暗杠/加杠分派 |
| `_closed_kan(state, target)` | L1429 | 暗杠实现 |
| `_added_kan(state, target)` | L1442 | 加杠实现 |

##### 和牌与结算

| 函数 | 行号 | 功能 |
|------|------|------|
| `_ron(state, game_config)` | L1701 | **荣和主函数**：多家和牌处理、双响逻辑 |
| `_tsumo(state, game_config)` | L1800 | **自摸主函数**：庄家/闲家不同点数分配 |
| `_settle_ron` 内嵌 | ~L1730 | 荣和点数结算（符×翻×4/6→基本点→授受） |
| `_settle_tsumo` 内嵌 | ~L1820 | 自摸点数结算（庄家支付双倍） |
| `_pao(state, winner)` | L1967 | 包牌逻辑（大三元/大四喜包牌） |
| `_mangan_tsumo(winner, dealer, honba)` | L1993 | 满贯自摸的点数分配 |
| `_next_ron_player(legal_action_mask_4p, discarded_player)` | L1110 | 寻找下一个可以和牌的玩家（优先级：ron > kan > pon > chi） |
| `_next_meld_player(legal_action_mask_4p, discarded_player)` | L1130 | 寻找下一个可以鸣牌的玩家 |

##### 合法动作 Mask

| 函数 | 行号 | 功能 |
|------|------|------|
| `_make_legal_action_mask_after_draw(state, game_config)` | L855 | 摸牌后的合法动作：舍牌判断（听牌→可立直）、自摸/暗杠/加杠/九种九牌 |
| `_make_legal_action_mask_after_draw_w_riichi(state)` | L917 | 立直后摸牌：只能 tsumogiri 或自摸 |
| `_make_legal_action_mask_after_discard(state, ...)` | L1053 | 舍牌后的合法动作（给其他玩家）：ron/pon/chi/open_kan/pass |
| `_make_legal_action_mask_after_chi(state, action)` | L1577 | 吃后的合法舍牌（不能食替） |
| `_mask_for_chi(hand, tile)` | L1090 | 计算可以吃的三种位置 |
| `_mask_for_pon_open_kan(hand, tile, cannot_kan)` | L1099 | 计算可以碰/大明杠的动作 |
| `_has_red_discard_action(mask)` | L188 | 判断是否有赤牌舍牌动作 |
| `_set_tile_type_action(mask, tile_type, value)` | L178 | 在 mask 中设置/清除指定牌的动作位 |

##### 流局

| 函数 | 行号 | 功能 |
|------|------|------|
| `_abortive_draw_normal(state)` | L1894 | 正常流局（荒牌）：听牌/不听牌结算 |
| `_trigger_special_abortive_draw(state)` | L215 | 触发特殊流局流程 |
| `_special_abortive_draw_mask()` | L211 | 特殊流局的合法动作 mask（只有 KYUUSHU） |

##### 局推进与终局

| 函数 | 行号 | 功能 |
|------|------|------|
| `_advance_to_next_round_auto(state, game_config)` | L2156 | **回合推进**：判断是否需要进入下一局（有人和牌/流局），重新初始化 |
| `_next_round(state, key, game_config)` | L2009 | 下一局初始化 |
| `_finalize_game` 内嵌 | ~L2072 | **游戏结束**：计算最终排名、应用顺位点、返回 rewards |
| `_special_next_round(state, game_config)` | L1939 | 特殊流局后的局推进 |
| `_dora_array(state)` | L2236 | 构建宝牌计数数组 |

##### 步进函数

| 函数 | 行号 | 功能 |
|------|------|------|
| `_step_auto(state, action, game_config)` | L583 | `auto` 模式主步进（标准流局，本分支默认） |
| `_step_dummy_share(state, action, game_config)` | L567 | `dummy_share` 模式主步进（用 dummy 动作替代直接局推进） |
| `_dispatch_action_auto(state, action, game_config)` | L647 | auto 模式的动作分发 |
| `_dispatch_action_dummy_share(state, action, game_config)` | L604 | dummy_share 模式的动作分发 |
| `_finalize_step_state(state, game_config)` | L751 | 步进后的状态收尾（检查特殊流局、更新合法动作 mask） |

##### 辅助函数

| 函数 | 行号 | 功能 |
|------|------|------|
| `_resolve_game_config(game_config)` | L160 | 解析/默认化游戏配置 |
| `_apply_red_five_config(deck, game_config)` | L164 | 根据配置决定是否使用赤牌 |
| `_set_player_hand(state, player, hand_37)` | L192 | 设置指定玩家的手牌（同时更新 34 维和 37 维） |
| `_is_waiting_tile(can_ron, tile)` | L1225 | 判断 tile 是否为听牌之一 |
| `_append_action_history(state, action)` | L548 | 追加动作到历史记录 |
| `_append_meld(state, meld, player)` | L1167 | 追加副露到玩家状态 |

##### RedMahjong 类方法

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__(...)` | L256 | 初始化环境配置 |
| `init(key)` | L286 | 创建初始 State |
| `step(state, action, key)` | L303 | 主步进：非法动作处理 → 自动局推进 → 终止状态处理 |
| `observe(state)` | L361 | 构建观察（dict 或 2D） |
| `_step_with_illegal_action(state, loser)` | L395 | 非法动作惩罚 |
| `id` | L366 | `"red_mahjong"` |
| `version` | L370 | `"beta"` |

##### 役种预计算

| 函数 | 行号 | 功能 |
|------|------|------|
| `yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(state, tile, next_tile)` | L226 | **JIT 编译的役种预计算**：对 4 人 × 2 种方式（荣和/自摸）批量计算役种 |

> **阅读技巧**：核心生命周期是 `_init` → `_draw` → `_discard` → `_ron`/`_tsumo` → `_settle` → `_advance_to_next_round_auto`。建议先通读 `step()` 方法理解分发流程，然后按生命周期追踪每个函数的调用链。`_replace_state` 是理解 JAX 不可变状态更新的关键——始终创建新状态对象而非修改原对象。

### 3.3 辅助模块

#### 向听计算
**[mahjax/red_mahjong/shanten.py](../mahjax/red_mahjong/shanten.py)**（159 行）

- `Shanten.CACHE`：预计算的向听数查找表，从 `shanten_cache.npz` 加载
- `Shanten.number(hand_34)` → 0-6 向听数（0=听牌，6=离听牌最远）
- `Shanten.discard(hand_34)` → (34,) 数组，每张牌舍弃后的向听数
- `Shanten.detailed_number(hand_34)` → (3,) 数组，普通/七对/国士的向听数
- 算法参考：[pgx PR #123](https://github.com/sotetsuk/pgx/pull/123)

#### 役种判定
**[mahjax/red_mahjong/yaku.py](../mahjax/red_mahjong/yaku.py)**（702 行）

- 使用预计算缓存 `yaku_cache.npz` 加速役种查找
- `Yaku.judge(hand_37, is_ron, player, state)` — 核心函数：返回 (yaku_array, fan, fu)
- `_Internal` 类：52 种天凤役种的枚举常量
  - 1 番役：立直(1)、一发(2)、断幺九(8)、平和(7)、自风(10-13)、场风(14-17)、一发(2)、岭上开花(4)、海底捞月(5)、河底捞鱼(6)、枪杠(3)
  - 2 番役：混全带幺九(18-20)、一气通贯(21-23)、三色同顺(24-26)、三色同刻(27)、对对和(28)、三暗刻(29)、三杠子(30)、七对子(31)、混老头(32)、小三元(33)、双立直(34)
  - 3+ 番役：混一色(35-36)、纯全带幺九(37-38)、二杯口(39)、清一色(40-41)
  - 6 番役：人和(42)
  - 满贯：流局满贯(43)
  - 役满：天和(44)、地和(45)、大三元(46)、四暗刻(47)、字一色(48)、绿一色(49)、清老头(50)、九莲宝灯(51)、国士无双(52)
- `_dora_array_from_state(state)` — 从状态计算宝牌计数
- 符计算函数：`_calc_fu` — 基于和牌方式、面子构成、雀头类型计算

#### 观察构建
**[mahjax/red_mahjong/observation.py](../mahjax/red_mahjong/observation.py)**（95 行）

- `_observe_dict(state)` → `Dict[str, jnp.ndarray]`：
  - `"hand"`: (14,) 手牌 ID 数组（-1 表示空位）
  - `"last_draw"`: (1,) 最后摸到的牌
  - `"action_history"`: (3, 200) 动作历史（玩家/动作/tsumogiri），玩家索引已转为相对视角
  - `"shanten_count"`: (1,) 向听数
  - `"furiten"`: (1,) 振听标志
  - `"scores"`: (4,) 分数（从当前玩家视角排列：cp, right, across, left）
  - `"round"`: (1,) 当前局数
  - `"honba"`: (1,) 本场数
  - `"kyotaku"`: (1,) 供托数
  - `"prevalent_wind"`: (1,) 场风
  - `"seat_wind"`: (1,) 座风
  - `"dora_indicators"`: (5,) 宝牌指示牌（最多 5 个）
- `_observe_2D(state)` — 2D CNN 格式观察（未实现）

#### 内置玩家
**[mahjax/red_mahjong/players.py](../mahjax/red_mahjong/players.py)**（378 行）

- `random_player(state, key)` — 从合法动作中随机选择
- `rule_based_player(state, key)` — 基于启发式规则：
  - 能荣和就荣和
  - 能自摸就自摸
  - 听牌后立直
  - 否则选择使向听数最小的舍牌

#### 可视化
**[mahjax/red_mahjong/visualization.py](../mahjax/red_mahjong/visualization.py)**（668 行）

- `render_round_svg(state, show_all_hands, tile_style)` — 渲染一局游戏的 SVG
- 支持 `"standard"` 和 `"bilingual"` 两种牌面风格
- 使用 `svgwrite` 库生成 SVG

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
        │  (1,601 行)       │             │ (375 行编排)      │
        │                  │             │                  │
        │  step(state)     │             │  step_batch(     │
        │  init(key)       │             │    batch_states, │
        │  observe(state)  │             │    actions)      │
        │                  │             │  init_batch(keys)│
        └───────┬──────────┘             │  observe_batch(  │
                │                        │    batch_states) │
                │                        └────────┬─────────┘
                │                                 │
                │              ┌──────────────────┴──────────┐
                │              │  env_parallel_handlers.py   │
                │              │  HandlersMixin (1,219 行)    │
                │              │  11 个 _xxx_batch 方法       │
                │              ├─────────────────────────────┤
                │              │  env_parallel_internals.py  │
                │              │  InternalsMixin (727 行)     │
                │              │  mask/结算/yaku/局推进       │
                │              └─────────────────────────────┘
                │                                 │
        ┌───────┴─────────────────────────────────┴─────────┐
        │              共享模块 (不变)                        │
        │  hand.py   meld.py   shanten.py   yaku.py         │
        │  tile.py   state.py  action.py    constants.py    │
        │  observation.py   players.py   batch_state.py     │
        └──────────────────────────────────────────────────┘
```

### 4.2 兼容层入口
**[mahjax_pt/red_mahjong/env.py](../mahjax_pt/red_mahjong/env.py)**（140 行）

```python
class RedMahjong(Env):
    def __init__(self, backend="serial", **kwargs):
        if backend == "parallel":
            self._impl = RedMahjongParallel(**kwargs)
        else:
            self._impl = RedMahjongSerial(**kwargs)
```

- 属性转发：`id`, `version`, `num_players`, `num_actions`, `observation_shape`
- `step_batch()` / `init_batch()` — 仅在 `backend="parallel"` 时可用
- `make(env_name, backend, **kwargs)` — 工厂函数

### 4.3 串行参考实现 ★
**[mahjax_pt/red_mahjong/env_serial.py](../mahjax_pt/red_mahjong/env_serial.py)**（1,601 行）

#### 关键翻译模式

| JAX | PyTorch | 说明 |
|-----|---------|------|
| `jnp.bool_` | `torch.bool` / Python `bool` | 类型转换 |
| `jnp.int8/int16/int32` | `torch.int8/int16/int32` | 整数类型 |
| `jax.lax.cond(cond, true_fn, false_fn)` | Python `if cond: ... else: ...` | JIT 条件→Python 原生分支 |
| `jax.lax.switch(idx, branches, ...)` | Python `if/elif` 链 | JIT 多路分发→Python 条件链 |
| `jax.random.PRNGKey(seed)` | `random.Random(seed)` + `np.random.Generator(seed)` | 随机数 |
| `state.replace(**updates)` → 新 State | `state.field = new_value` 直接修改 | 不可变→可变 |
| `jnp.zeros((4, 34), dtype=jnp.int8)` | `torch.zeros(4, 34, dtype=torch.int8)` | 张量创建 |
| `hand.at[cp].set(new_hand)` | `hand[cp] = new_hand` | 索引更新 |

#### 公共辅助函数（L56-216）

这些函数与 JAX 版本 1:1 对应，标记了 JAX 行号范围：

| 函数 | 行号 | JAX 对应行号 | 功能 |
|------|------|-------------|------|
| `_resolve_game_config` | L58 | JAX L160 | 解析游戏配置 |
| `_resolve_env_config` | L62 | — | 验证并解析环境配置（Serial/Parallel 共享） |
| `_live_wall_end_ix` | L88 | JAX L173 | 活墙截止索引 |
| `_set_tile_type_action` | L92 | JAX L178 | 设置 mask 中指定牌的动作位 |
| `_has_red_discard_action` | L102 | JAX L188 | 判断是否有赤牌舍牌动作 |
| `_special_abortive_draw_mask` | L111 | JAX L211 | 特殊流局 mask |
| `_trigger_special_abortive_draw` | L117 | JAX L215 | 触发特殊流局 |
| `_append_meld_to_player` | L129 | — | 记录副露到玩家状态 |
| `_accept_riichi` | L151 | JAX L1176 | 接受立直（扣点、设标志） |
| `_is_waiting_tile` | L186 | JAX L1225 | 判断牌是否为听牌 |
| `_calc_wind` | L192 | JAX L538 | 计算座风 |
| `_is_first_turn` | L197 | JAX L544 | 判断首巡 |
| `_append_action_history` | L202 | JAX L548 | 追加动作历史 |

#### RedMahjongSerial 类完整方法索引

| 方法 | 行号 | JAX 对应行号 | 功能 |
|------|------|-------------|------|
| `__init__` | L232 | JAX L256 | 初始化环境 |
| `init` | ~L270 | JAX L286 | 初始化一局游戏 |
| `step` | ~L380 | JAX L303 | 主步进函数 |
| `observe` | ~L430 | JAX L361 | 构建观察 |
| `_step_with_illegal_action` | L447 | JAX L395 | 非法动作惩罚 |
| `_draw` | L457 | JAX L788 | 摸牌 |
| `_make_legal_action_mask_after_draw_riichi` | L528 | JAX L917 | 立直后的合法动作 mask |
| `_make_legal_action_mask_after_draw` | L555 | JAX L855 | 摸牌后合法动作 mask |
| `_make_legal_action_mask_after_pon` | L619 | — | 碰后合法舍牌 mask |
| `_make_legal_action_mask_after_chi` | L635 | JAX L1577 | 吃后合法舍牌 mask |
| `_discard` | L661 | JAX L939 | 舍牌 |
| `_make_legal_action_mask_after_discard` | L751 | JAX L1053 | 舍牌后合法动作 mask |
| `_riichi` | L873 | JAX L1675 | 立直 |
| `_ron` | L903 | JAX L1701 | 荣和 |
| `_tsumo` | L935 | JAX L1800 | 自摸 |
| `_settle_ron` | L968 | JAX ~L1730 | 荣和点数结算 |
| `_settle_tsumo` | L983 | JAX ~L1820 | 自摸点数结算 |
| `_pon` | L1014 | JAX L1497 | 碰 |
| `_open_kan` | L1046 | JAX L1466 | 大明杠 |
| `_selfkan` | L1077 | JAX L1412 | 暗杠/加杠 |
| `_chi` | L1133 | JAX L1539 | 吃 |
| `_pass` | L1168 | JAX L1599 | 过 |
| `_kyuushu` | L1227 | — | 九种九牌 |
| `_dummy` | L1312 | — | 占位动作 |
| `_flip_dora` | L1323 | JAX ~L1280 | 翻宝牌 |
| `_draw_after_kan` | L1338 | JAX L1233 | 杠后摸牌 |
| `_abortive_draw_normal` | L1401 | JAX L1894 | 正常流局 |
| `_advance_to_next_round_auto` | L1431 | JAX L2156 | 局推进 |
| `_precompute_yaku` | L1535 | JAX L226 | 预计算役种 |
| `_finalize_game` | L1592 | JAX ~L2072 | 游戏结束 |

### 4.4 并行训练实现（分四层阅读）

#### 4.4.1 批量化状态 ★
**[mahjax_pt/red_mahjong/batch_state.py](../mahjax_pt/red_mahjong/batch_state.py)**（369 行）

##### BatchState（顶层）

```python
@dataclass
class BatchState:
    B: int                                                # batch size
    current_player: torch.Tensor                          # (B,) int
    legal_action_mask: torch.Tensor                       # (B, 87) bool
    players: BatchPlayerState                             # 嵌套的玩家状态
    round_state: BatchRoundState                          # 嵌套的回合状态
    step_count: torch.Tensor                              # (B,) int
    rewards: torch.Tensor                                 # (B, 4) float32
    terminated: torch.Tensor                              # (B,) bool
    truncated: torch.Tensor                               # (B,) bool
```

##### BatchPlayerState

与 JAX `PlayerStateArrays` 完全对应，但所有 `(4, N)` 变为 `(B, 4, N)`：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `hand` | (B, 4, 34) | int8 | 手牌计数 |
| `hand_with_red` | (B, 4, 37) | int8 | 赤牌感知手牌 |
| `hand_ids` | (B, 4, 14) | int16 | 手牌 ID 列表 |
| `hand_counts` | (B, 4) | int8 | 手牌张数 |
| `drawn_tile` | (B, 4) | int16 | 刚摸的牌 |
| `legal_action_mask` | (B, 4, 87) | bool | 合法动作掩码 |
| `can_win` | (B, 4, 34) | bool | 能荣和的牌 |
| `has_yaku` / `fan` / `fu` | (B, 4, 2) | bool/int32 | 役种/番/符 |
| `melds` | (B, 4, 4) | int32 | 副露 |
| `meld_tiles` / `meld_info` | (B, 4, 4, 4/3) | int16/int8 | 副露详细 |
| `river` / `discards` / `discard_info` | (B, 4, 60/...) | int32/int16/int8 | 牌河 |
| `riichi` / `riichi_declared` / `riichi_step` | (B, 4) | bool/bool/int8 | 立直状态 |
| `double_riichi` / `ippatsu` | (B, 4) | bool | 双立直/一发 |
| `furiten_by_discard` / `furiten_by_pass` | (B, 4) | bool | 振听 |
| `is_hand_concealed` | (B, 4) | bool | 门清 |
| `pon` | (B, 4, 34) | int32 | 碰计数 |
| `has_won` | (B, 4) | bool | 已和牌 |
| `n_kan` | (B, 4) | int8 | 杠数 |
| `has_nagashi_mangan` | (B, 4) | bool | 流局满贯可能 |

##### BatchRoundState

| 字段 | 形状 | 说明 |
|------|------|------|
| `action_history` | (B, 3, 200) | 动作历史 |
| `round` / `round_limit` | (B,) | 局数/局限制 |
| `dealer` / `honba` / `kyotaku` | (B,) | 庄家/本场/供托 |
| `score` | (B, 4) | 各玩家分数 |
| `deck` | (B, 136) | 牌山 |
| `next_deck_ix` / `last_deck_ix` | (B,) | 牌山索引 |
| `dora_indicators` / `ura_dora_indicators` | (B, 5) | 宝牌指示牌 |
| `draw_next` / `kan_declared` / `can_after_kan` / `can_robbing_kan` | (B,) | 流程控制标志 |

##### 转换函数

- `_default_batch_state(B, device)`（L127）：创建所有字段初始化的空 `BatchState`
- `stack_states(states: List[EnvState])`（L206）：将 B 个独立 `EnvState` 打包为 `BatchState`，逐字段 `[i] = state.field`
- `unstack_state(batch_state, index)`（L289）：从 `BatchState` 中提取单环境 `EnvState`，调用 `.clone()` 确保独立

> **关键理解**：`stack_states` 和 `unstack_state` 是 Serial ↔ Parallel 桥接的核心。`init_batch` 内部循环调用 `_serial.init()` 然后用 `stack_states` 打包；`step_batch` 对 `List[EnvState]` 的输入也先 `stack_states` → 并行处理 → `unstack_state`。

#### 4.4.2 编排层 ★
**[mahjax_pt/red_mahjong/env_parallel.py](../mahjax_pt/red_mahjong/env_parallel.py)**（375 行）

```python
class RedMahjongParallel(HandlersMixin, InternalsMixin, Env):
```

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__` | L82 | 解析配置，创建内部 `_serial` 引用（用于 `init` 等委托） |
| `init_batch(keys, num_envs, device)` | L150 | 循环调用 `_serial.init()` 然后 `stack_states` |
| `init(key)` | L147 | 委托到 `_serial.init` |
| `step(state, action)` | L176 | 委托到 `_serial.step` |
| `step_batch(states, actions)` | L242 | 接受 `List[EnvState]` 或 `BatchState` |
| `_step_batch_bs(bs, actions)` | L270 | **核心并行步进**（详见下文） |
| `observe(state)` | L132 | 委托到 `_serial.observe` |
| `observe_batch(batch_state)` | L135 | 直接构建 (B, ...) 维度观察（跳过 stack/unstack） |
| `reinit_terminated_batch(bs, keys)` | L201 | 重置已终止的环境 |

##### `_step_batch_bs` 核心流程（L270-375）

```
_step_batch_bs(bs, actions)
  ├── 1. 处理已终止 env：清空 rewards
  ├── 2. 处理已终止 round：auto 模式下推进到下一局
  ├── 3. 动作分类（全向量化，无循环）：
  │     is_discard    = active & (a < 37)
  │     is_tsumogiri  = active & (a == Action.TSUMOGIRI)
  │     is_selfkan    = active & (a >= 37) & (a < 71)
  │     is_riichi     = active & (a == Action.RIICHI)
  │     is_ron        = active & (a == Action.RON)
  │     is_tsumo      = active & (a == Action.TSUMO)
  │     is_pon        = active & ((a == PON) | (a == PON_RED))
  │     is_open_kan   = active & (a == Action.OPEN_KAN)
  │     is_chi        = active & (a >= CHI_L) & (a <= CHI_R_RED)
  │     is_pass       = active & (a == Action.PASS)
  │     is_kyuushu    = active & (a == Action.KYUUSHU)
  │     is_dummy      = active & (a == Action.DUMMY)
  ├── 4. 按游戏逻辑顺序处理动作：
  │     riichi → ron → tsumo → pon → open_kan → chi →
  │     selfkan → discard → pass → kyuushu → dummy
  ├── 5. 更新 step_count[active] += 1
  ├── 6. 单局模式：terminated |= terminated_round
  ├── 7. auto 模式：推进已终止的 round
  └── 8. 终止 env 的 legal_action_mask 设为全 True
```

**动作处理顺序很重要**：ron 和 tsumo 优先于任何鸣牌，因为和牌会终止当前局。在 JAX 中这通过 `_next_ron_player` 的优先级查找实现；在 Parallel 中通过 handler 的处理顺序实现。

#### 4.4.3 动作处理器 Mixin ★
**[mahjax_pt/red_mahjong/env_parallel_handlers.py](../mahjax_pt/red_mahjong/env_parallel_handlers.py)**（1,219 行）

`HandlersMixin` 包含 13 个方法，每个方法处理一种动作类型在 B 个环境上的并行执行：

| 方法 | 功能 | 核心张量操作 |
|------|------|-------------|
| `_discard_batch(bs, mask, actions)` | 舍牌/tsumogiri（~90% 动作） | 337 维手牌更新 + 牌河写入 |
| `_riichi_batch(bs, mask)` | 立直宣言 | 137 张牌 × M 环境的 tenpai 批量检查 |
| `_pon_batch(bs, mask, actions)` | 碰 | (M, 2) vs (M, 34) vs (M, 37) 手牌复制 |
| `_chi_batch(bs, mask, actions)` | 吃 | 三种 chi shift + 顺子构造 |
| `_open_kan_batch(bs, mask)` | 大明杠 | 3 张手牌→面子 |
| `_selfkan_batch(bs, mask, actions)` | 暗杠/加杠 | 4 张/1 张手牌→面子 |
| `_ron_batch(bs, mask)` | 荣和 | 手动结算（不调用 settle_ron_batch 共享版本） |
| `_tsumo_batch(bs, mask)` | 自摸 | 手动结算 |
| `_pass_batch(bs, mask)` | 过（不鸣牌） | 清除 draw_next、推进玩家 |
| `_kyuushu_batch(bs, mask)` | 九种九牌 | 流局处理 |
| `_dummy_batch(bs, mask)` | 占位动作 | 无操作 |
| `_draw_batch(bs, m_idx)` | 摸牌（内部） | 从牌山 `next_deck_ix` 摸牌 |
| `_draw_after_kan_batch(bs, m_idx)` | 杠后摸牌（内部） | 从王牌区摸牌 |

**统一执行模式**：
```python
def _xxx_batch(self, bs, mask, actions=None):
    if not mask.any():
        return bs           # 无环境需要此动作，直接返回
    m_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (M,) 需要处理的环境索引
    # ... 批量张量操作，只在 m_idx 指定的行上执行 ...
    return bs               # 修改后的 BatchState
```

`_discard_batch` 的详细流程（最复杂的 handler）：
1. 合并非 tsumogiri 和 tsumogiri 路径
2. 区分 `draw_next=True`（摸牌→舍牌）和 `draw_next=False`（吃/碰后的舍牌）
3. 更新手牌：`hand[m_idx, cp] -= 1`（对于 tsumogiri，先 sub 再 restore）
4. 写入牌河：`river[m_idx, cp, dc] = River.make(tile, ...)`
5. 更新 riichi_step/ippatsu/furiten
6. 调用 `_make_legal_action_mask_after_discard_batch` 构建其他玩家的合法动作
7. 设置 `current_player = next_player`

#### 4.4.4 内部机制 Mixin ★
**[mahjax_pt/red_mahjong/env_parallel_internals.py](../mahjax_pt/red_mahjong/env_parallel_internals.py)**（727 行）

`InternalsMixin` 包含：

| 方法 | 行号 | 功能 |
|------|------|------|
| `_make_legal_mask_after_discard_batch` | L54 | 批量构建舍牌后的合法动作 mask（ron/pon/chi/open_kan） |
| `_make_legal_mask_after_draw_batch` | L231 | 批量构建摸牌后的合法动作 mask（discard/tsumo/riichi/selfkan/kyuushu） |
| `_precompute_yaku_batch` | L412 | 批量计算荣和/自摸的役种（4 人 × 2 方式 × M 环境） |
| `_score_batch` | L554 | 静态方法：根据 fan/fu 计算基本点 |
| `_settle_ron_batch` | L573 | 批量荣和点数结算 |
| `_settle_tsumo_batch` | L594 | 批量自摸点数结算 |
| `_abortive_draw_normal_batch` | L665 | 批量正常流局（荒牌流局） |
| `_advance_round_batch` | L710 | 批量局推进 |
| `_copy_state_into_batch` | L721 | 复制 EnvState 到 BatchState 的指定行 |
| `_copy_dataclass_row` | L27 | 工具函数：在两个 dataclass 之间按行拷贝张量数据 |

### 4.5 PyTorch 共享模块

这些模块与 JAX 对应模块功能等价，但实现为 PyTorch eager 模式，部分已包含 batch 版本的方法：

| 文件 | 行数 | 与 JAX 的关键差异 |
|------|------|------------------|
| [hand.py](../mahjax_pt/red_mahjong/hand.py) | 886 | 添加 `chi_batch`, `open_kan_batch`, `closed_kan_batch`, `sub_batch`, `to_34_batch` 等 batch 方法 |
| [meld.py](../mahjax_pt/red_mahjong/meld.py) | 455 | 已有 batch 版本的面子编码/解码 |
| [shanten.py](../mahjax_pt/red_mahjong/shanten.py) | 233 | `Shanten.number_batch(hand_34_batch)` — 批量向听数计算 |
| [yaku.py](../mahjax_pt/red_mahjong/yaku.py) | 1,182 | **GPU device-aware 常量**：`_get_cache()` 检测张量所在设备并缓存到对应设备；`_FAN.to(device)` |
| [tile.py](../mahjax_pt/red_mahjong/tile.py) | 298 | 已有 batch 版本的牌操作 |
| [alignment.py](../mahjax_pt/red_mahjong/alignment.py) | 218 | **JAX ↔ PyTorch 算子对齐函数**：`aligned_gather`, `aligned_scatter`, `aligned_where` 等 |
| [observation.py](../mahjax_pt/red_mahjong/observation.py) | 191 | 添加 `_observe_dict_batch(batch_state)` — 直接构建 (B, ...) 观察 |
| [players.py](../mahjax_pt/red_mahjong/players.py) | 339 | 等价实现 |
| [state.py](../mahjax_pt/red_mahjong/state.py) | 145 | 使用 Python `dataclass` + 可变属性（`@dataclass` 不带 frozen） |

---

## 第五阶段：RL 训练管线（约 1-2 小时）

> **目标**：理解如何在麻将上做强化学习——数据收集、网络结构、PPO 训练循环。

### 5.1 JAX 训练管线

#### 行为克隆（BC）
**[examples/bc.py](../examples/bc.py)**（约 200 行）

- `TrainConfig` 数据类：`env_name`, `batch_size=1024`, `lr=3e-4`, `num_epochs=5`
- 使用 `optax.adam` 优化器 + `flax.training.train_state.TrainState`
- 从 Mortal 离线数据（`.pkl` 格式）加载状态-动作对
- 损失函数：交叉熵损失 + IL 正则化
- 支持 wandb 日志记录

#### PPO 训练
**[examples/ppo_with_reg.py](../examples/ppo_with_reg.py)**（约 300 行）

- `PPOWithRegArgs` Pydantic 配置模型（L39-78）：
  - 环境：`env_name="no_red_mahjong"`, `round_mode="single"`, `num_envs=1024`
  - PPO：`num_steps=256`, `gamma=1.0`, `gae_lambda=0.95`, `clip_eps=0.2`
  - 训练：`lr=3e-4`, `update_epochs=4`, `minibatch_size=4096`
  - 正则化：`mag_coef=0.2`（MAGNET 多样性正则化）
- 核心组件：
  - `RolloutRunner` — 并行 rollout：`vmap(env.step)` + `vmap(network.apply)`
  - `calculate_gae` — GAE 优势估计（`gamma=1.0` 意味着全 episode 范围）
  - `update_network` — PPO clipped objective + value loss + entropy bonus + magnet loss
- 评估：定期与随机玩家和 rule_based 玩家对战

#### ACNet 网络
**[examples/networks/red_network.py](../examples/networks/red_network.py)**（约 150 行）

`ACNet(nn.Module)` 架构：
```
Observation → FeatureExtractor → ActorHead + CriticHead
```

`FeatureExtractor` 的输入处理：
1. **手牌编码**（`hand_emb_size=128`）：
   - Embedding(38, 128) → TransformerBlock(128, 4 heads) × 2 层 → mean pooling
2. **动作历史编码**（`history_emb_size=192`）：
   - Embedding(90, 192) → TransformerBlock(192, 4 heads) × 2 层 → mean pooling
3. **全局特征**（`global_emb_size=64`）：
   - 分数（标准化 `(score - 250) / 1250`）
   - 向听数（标准化 `/6.0`）
   - 局数（标准化 `/12.0`）
   - 本场/供托（标准化 `/10.0`）
   - 场风/座风（标准化 `/3.0`）
   - 振听标志
   - 宝牌指示牌 Embedding
4. **特征融合**：
   - Concat(hand_feat, history_feat, global_feat)
   - MLP(256) → 最终特征向量

`ActorHead`：MLP(256) → Linear(256 → 89) → 策略 logits
`CriticHead`：MLP(256) → Linear(256 → 1) → 价值

#### Transformer
**[examples/networks/transformer.py](../examples/networks/transformer.py)**（约 80 行）

- 标准 Pre-LayerNorm Transformer block
- `orthogonal_init()`：正交初始化（`nn.initializers.orthogonal()`）
- 关键超参数：`HAND_EMB_SIZE=128`, `HISTORY_EMB_SIZE=192`, `GLOBAL_EMB_SIZE=64`, `FINAL_MLP_DIM=256`

### 5.2 PyTorch 训练管线

**[mahjax_pt/examples/ppo_with_reg.py](../mahjax_pt/examples/ppo_with_reg.py)**
- 已适配新的双轨 API：使用 `make(backend="parallel")` 进行训练，`make(backend="serial")` 进行验证
- LayerNorm `eps=1e-6` 对齐 JAX（PyTorch 默认为 `1e-5`）
- GAE/approx_kl 的 bug 修复（Phase 11 的工作成果）

---

## 第六阶段：测试与验证体系（约 1-2 小时）

> **目标**：理解项目的正确性保证——多层测试金字塔与金数据回放机制。

### 6.1 JAX 环境单元测试
**[tests/red_mahjong/](../tests/red_mahjong/)**

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_tile.py` | 牌编码/解码、赤牌转换、River 编码 |
| `test_hand.py` | 摸牌、舍牌、听牌判定、can_ron/can_tsumo |
| `test_meld.py` | 面子编码/解码、subcode 一致性 |
| `test_shanten.py` | 向听数计算、已知手牌的向听数回归测试 |
| `test_yaku.py` | 役种判定、fan/fu 计算、预计算缓存正确性 |
| `test_env.py` | 环境 init/step、状态一致性 |
| `test_observe.py` | 观察构建、特征维度 |
| `test_play.py` | 完整对局的正确性（随机 rollout） |
| `test_special_case.py` | 九种九牌、四风连打、四杠散了、四家立直 |
| `test_parity.py` | Red ↔ No-Red 等价性（在 No-Red 配置下） |
| `test_visualize.py` | SVG 渲染输出正确 |

### 6.2 PyTorch 精度验证 ★★★

#### 四层验证金字塔

```
         ┌──────────────────────────────────────────────┐
         │  L4: 金数据回放                                │
         │  805 seeds × 完整对局，逐 step 与 JAX 对比      │
         │  100% 通过！                                   │
         │  replay_pt_against_golden.py                   │
         │  replay_parallel_against_golden.py              │
         ├──────────────────────────────────────────────┤
         │  L3: 环境集成                                  │
         │  Serial ↔ Parallel 逐字段 exact match          │
         │  关键分支（立直/荣和/自摸/流局）全覆盖           │
         │  test_env_parallel_parity.py                   │
         │  test_env_branches.py                          │
         ├──────────────────────────────────────────────┤
         │  L2: 框架一致性                                │
         │  相同输入 → JAX/PyTorch 输出逐元素对比           │
         │  test_exact_parity.py                          │
         │  test_full_ppo_parity.py                       │
         │  test_mha_definitive.py                        │
         ├──────────────────────────────────────────────┤
         │  L1: 基础单元                                  │
         │  纯 PyTorch 逻辑正确性                          │
         │  16 组 80 断言                                 │
         │  test_cases.py                                 │
         │  test_aligned_math.py                          │
         └──────────────────────────────────────────────┘
```

#### PPO 精度专项（六层递进）

| 层级 | 测试文件 | 验证目标 |
|------|---------|---------|
| PPO L1 | `test_ppo_math_parity.py` | 数学原语（log_prob, entropy, advantage, masked_softmax 等） |
| PPO L2 | `test_ppo_gae_parity.py` | GAE 计算 vs JAX `calculate_gae` 逐元素对比 |
| PPO L3 | `test_ppo_weight_transfer.py` | ACNet 权重从 JAX 到 PyTorch 的显式参数映射 |
| PPO L4 | `test_ppo_update_parity.py` | PPO loss/grad/param 更新后逐层对比 |
| PPO L4 Ext | `test_ppo_acnet_parity.py` | 完整 ACNet 上的 PPO loss 对比 |
| PPO L5 | `test_ppo_cycle_parity.py` | 单次完整 update cycle（rollout→advantage→update） |
| PPO L6 | `test_ppo_training_parity.py` | 多步训练的数值稳定性 |

#### 金数据工具链

| 文件 | 功能 |
|------|------|
| `record_jax_golden.py` | 录制 JAX 环境金数据（输入 action + 输出 state 的完整字段） |
| `record_jax_acnet_golden_f64.py` | 录制 ACNet 金数据（float64 精度，用于定位 fp32 漂移） |
| `record_jax_ppo_golden.py` | 录制 PPO 训练循环金数据 |
| `replay_pt_against_golden.py` | 用 PT 串行环境逐 step 回放，全字段 `torch.allclose` |
| `replay_parallel_against_golden.py` | 用 PT 并行环境回放，支持 `-j` 多进程加速 |

> **关键事实**：805 seeds 的金数据回放 100% 通过，覆盖率覆盖了 discard (~90%), tsumo (~5%), pass (~3%), pon (~1%), chi (~0.5%), ron (~0.3%), riichi (~0.2%) 等所有动作类型。

---

## 第七阶段：周边设施（约 1 小时）

### 7.1 网页 UI
**[mahjax/ui/](../mahjax/ui/)**

- [app.py](../mahjax/ui/app.py) —
    - FastAPI 应用，`create_app()` 工厂函数
    - REST API：`POST /games`（创建游戏）、`POST /games/{id}/action`（执行动作）、`GET /games/{id}/state`（获取状态）
    - `CreateGameRequest` 模型：`env_id`, `agent_id`, `mode`（single/east/half）, `human_seat`, `seed`
    - 静态文件挂载（`STATIC_DIR`）
- [game_manager.py](../mahjax/ui/game_manager.py) —
    - `GameManager` 类：管理多个游戏实例
    - 状态机：等待玩家动作 → 执行动作 → AI 自动响应 → 等待玩家
    - 分数历史追踪
- [agents.py](../mahjax/ui/agents.py) —
    - `AgentRegistry`：支持动态注册 agent
    - `load_callable_from_path(file_path, attribute)` — 从外部 Python 文件动态加载 act 函数

### 7.2 环境包装器
- [mahjax/wrappers/auto_reset_wrapper.py](../mahjax/wrappers/auto_reset_wrapper.py) — JAX 版：对局终止时自动调用 `env.init()` 创建新状态
- [mahjax_pt/red_mahjong/auto_reset_wrapper.py](../mahjax_pt/red_mahjong/auto_reset_wrapper.py) — PyTorch 版：兼容 serial/parallel 两种后端

### 7.3 SVG 可视化
- [mahjax/_src/visualizer.py](../mahjax/_src/visualizer.py) — 通用 SVG 可视化器
    - `Visualizer` 类：基于 `svgwrite` 的灵活渲染
    - `set_visualization_config(color_theme, scale)` — 全局配置
    - `save_svg(state, filename)` / `save_svg_animation(states, filename)` — 静态/动画导出
- [mahjax/red_mahjong/visualization.py](../mahjax/red_mahjong/visualization.py) — 麻将专用
    - `render_round_svg(state, show_all_hands, tile_style)` — 渲染回合
    - 支持 `"standard"` 和 `"bilingual"` 牌面风格（双语牌面适合不熟悉汉字的玩家）

### 7.4 无赤麻将
**[mahjax/no_red_mahjong/](../mahjax/no_red_mahjong/)**

- 与 `red_mahjong` 镜像结构
- 主要差异：无赤五牌（`NUM_TILE_TYPES_WITH_RED == NUM_TILE_TYPES == 34`）
- 更少的规则：无特殊流局、无包牌、无双响、无赤牌
- 性能约 2 倍（简化规则 → 更少的条件分支）

### 7.5 其余文件
| 路径 | 说明 |
|------|------|
| [Mortal/](../Mortal/) | Mortal 机器人代码（参考用，Rust 实现，非本项目核心） |
| [script/](../script/) | 项目级辅助脚本 |
| [mahjax_pt/scripts/](../mahjax_pt/scripts/) | PT 专用脚本（bench、regression seeds 等） |
| [openspec/](../openspec/) | 设计规格文档和任务清单 |

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

理解以下 5 个模式会极大加速代码阅读：

### 1. Struct-of-Arrays（JAX 高性能核心）
```python
# 不可变的 struct-of-arrays：所有玩家数据打包为 (4, N) 张量
@dataclass
class PlayerStateArrays:
    hand: jnp.ndarray          # (4, 34)   — 每人的手牌计数
    river: jnp.ndarray         # (4, 60)   — 每人的牌河
    melds: jnp.ndarray         # (4, 4)    — 每人的副露
    # ...共 33 个字段

# 沿 player 维并行计算
v_hand_op = jax.vmap(hand_op, in_axes=0)  # 自动沿 axis=0 (player) 并行 4 路
v_can_win = jax.vmap(jax.vmap(Hand.can_ron, in_axes=(None, 0)), in_axes=(0, None))
# 外层 vmap: 对每个玩家; 内层 vmap: 对每种牌
```

### 2. 不可变状态更新（JAX `_replace_state`）
```python
# JAX: 永远创建新 State，不修改原对象
def _replace_state(state: State, **updates) -> State:
    # 根据字段名路由到 players / round_state / env 三层
    # 关键：每次 step 返回全新的 State，原 State 不变
    return state.replace(players=players.replace(...), round_state=round_state.replace(...))
```

### 3. 可变 Dataclass（PyTorch Serial）
```python
# PyTorch Serial: 直接修改属性，对标 JAX 的状态语义
@dataclass
class EnvState:
    hand: torch.Tensor         # (4, 34)
    river: torch.Tensor        # (4, 60)

# 直接赋值修改（"对标 JAX 的不可变创建"）
state.players.hand[cp] = new_hand  # 而非 state = state.replace(hand=new_hand)
```

### 4. Mask 驱动的批量化控制流（PyTorch Parallel）
```python
# 所有 if/else 分支转为 boolean mask + tensor indexing
# B=128 个环境的动作类型不同，但一步同时处理

mask = (action_type == DISCARD)  # (B,) bool — 哪些环境要执行 discard
if not mask.any(): return bs     # 无环境需要此动作，提前返回
m_idx = mask.nonzero().squeeze(-1)  # (M,) — 需要处理的环境子集
# 批量张量操作仅在 M 个环境上执行
bs.players.hand[m_idx, cp] = new_hand[m_idx]
```

### 5. Facade + Mixin 组合
```python
# Facade (env.py): 不含业务逻辑，只做路由
class RedMahjong(Env):
    def __init__(self, backend="serial", **kwargs):
        self._impl = (RedMahjongSerial if backend == "serial"
                      else RedMahjongParallel)(**kwargs)

# Mixin (env_parallel): 职责拆分为三个独立文件
class RedMahjongParallel(HandlersMixin, InternalsMixin, Env):
    # env_parallel.py:           编排层 (375 行)
    # env_parallel_handlers.py:  动作处理器 (1,219 行)
    # env_parallel_internals.py: 内部机制 (727 行)
```

---

## 建议学习时间表

```
Week 0: 使用指南（实操先行）
        ├── Day 1: pip install + JAX/PyTorch 基本使用 + 跑通测试
        ├── Day 2: 启动 UI 玩几局 + 跑通 BC 训练
        └── Day 3: 跑通 PPO 训练 + 精度验证流程
        产出：能跑起来整个项目，对工作流有感性认识

Week 1: 阶段 1（概念入门）+ 阶段 2（数据模型）
        ├── Day 1-2: 麻将规则 + README + proposal/design
        ├── Day 3-4: constants → tile → action → meld → hand
        └── Day 5: state.py 精读
        产出：能读/写麻将状态的代码

Week 2: 阶段 3（JAX 核心引擎）
        ├── Day 1-2: core.py → env.py 的 init/_draw/_discard
        ├── Day 3-4: env.py 的 _pon/_chi/_kan/_ron/_tsumo
        └── Day 5: shanten → yaku → observation
        产出：理解 env.py 的完整生命周期

Week 3: 阶段 4（PyTorch 双轨架构）
        ├── Day 1-2: env.py facade → env_serial.py（对照 JAX）
        ├── Day 3-4: batch_state → env_parallel → handlers
        └── Day 5: internals → 共享模块（hand/meld/yaku/tile 的 batch 版本）
        产出：理解 serial/parallel 的转换方式

Week 4: 阶段 5 + 6 + 7
        ├── Day 1-2: RL 管线（bc → ppo → networks）
        ├── Day 3-4: 测试体系（JAX tests → PT 精度验证金字塔）
        └── Day 5: 周边（UI + wrappers + visualization）
        产出：端到端理解训练→推理→部署全链路
```

---

## 相关资源

- [项目仓库](https://github.com/nissymori/mahjax)
- [在线文档](https://nissymori.github.io/mahjax/)
- [学术论文](https://arxiv.org/abs/2605.20577)
- [天凤规则（英文）](https://tenhou.net/0/mj/mjlog/en/mjlog-en-rules.html)
- [欧洲立直麻将规则（2025）](http://mahjong-europe.org/portal/images/docs/Riichi-rules-2025-EN.pdf)
- [Pgx（API 设计参考）](https://github.com/sotetsuk/pgx)
- [向听数算法实现参考](https://github.com/sotetsuk/pgx/pull/123)
