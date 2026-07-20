# MahJax Training Scripts

按训练硬件分类的即用型训练脚本。**所有参数统一由 `config.json` 管理。**

## 为什么 GPU 用 JAX，NPU 用 PyTorch？

| | GPU (NVIDIA) | NPU (Ascend) |
|---|---|---|
| **训练脚本** | `gpu/` 目录 | `npu/` 目录 |
| **底层框架** | **JAX** | **PyTorch eager** |
| **环境包** | `mahjax/` | `mahjax_pt/` |
| **核心优化** | `jax.jit` 编译 CUDA kernel | eager 模式，适配 torch_npu |
| **速度** | ~1M+ steps/sec | 取决于 NPU 算力 |
| **CLI 格式** | `key=value` (OmegaConf) | `--key value` (argparse) |
| **默认 env** | `no_red_mahjong` | `red_mahjong` |

JAX 在 NVIDIA GPU 上能利用 XLA 编译器直接将计算图编译为 CUDA kernel，配合 `jax.vmap` 自动向量化，性能远超 Python 循环的 eager 模式。PyTorch eager 路径的存在是为了支持 JAX 不支持的硬件（如 Ascend NPU）。

## 目录结构

```
scripts/train/
├── _common.sh                 # 共享 config.json 加载器
├── README.md
├── npu/                       # Ascend NPU 训练
│   ├── config.json            # ★ 唯一配置入口
│   ├── run_bc.sh
│   ├── run_ppo.sh
│   ├── run_all.sh
│   ├── offline_data/          # [产物]
│   ├── params/                # [产物]
│   ├── checkpoints/           # [产物]
│   ├── logs/                  # [产物]
│   └── fig/                   # [产物]
│
└── gpu/                       # NVIDIA GPU 训练
    ├── config.json            # ★ 唯一配置入口
    ├── run_bc.sh
    ├── run_ppo.sh
    ├── run_all.sh
    ├── offline_data/          # [产物]
    ├── params/                # [产物]
    ├── checkpoints/           # [产物]
    ├── logs/                  # [产物]
    └── fig/                   # [产物]
```

## config.json 参数说明

训练前只需修改 `config.json`，无需改动脚本。完整字段：

```json
{
    "env": {
        "name": "red_mahjong",         // "red_mahjong" | "no_red_mahjong"
        "round_mode": "single",        // "single" | "east" | "half"
        "observe_type": "dict"         // "dict" (Transformer) | "2D" (CNN)
    },
    "device": {
        "type": "npu",                 // "npu" | "cuda" | "cpu"
        "id": 0                        // 卡号
    },
    "bc": {
        "batch_size": 1024,
        "lr": 0.0003,
        "num_epochs": 5,
        "seed": 42,
        "val_split": 0.1
    },
    "ppo": {
        "seed": 0,
        "num_envs": 1024,
        "num_steps": 256,
        "total_timesteps": 100000000,
        "lr": 0.0003,
        "ent_coef": 0.01,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "update_epochs": 4,
        "minibatch_size": 4096,
        "mag_coef": 0.2,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "eval_interval": 10,
        "eval_num_envs": 1000
    },
    "paths": {
        "offline_data": "offline_data",
        "params": "params",
        "checkpoints": "checkpoints",
        "logs": "logs",
        "fig": "fig"
    },
    "logging": {
        "use_wandb": false,
        "wandb_project": "mahjax-ppo"
    },
    "pipeline": {
        "skip_data": false,            // 跳过离线数据收集
        "skip_bc": false,              // 跳过 BC 预训练
        "skip_ppo": false              // 跳过 PPO 训练
    }
}
```

**NPU 和 GPU 的 config.json 差异：**

| 字段 | NPU | GPU |
|------|-----|-----|
| `env.name` | `red_mahjong` | `no_red_mahjong` |
| `device.type` | `npu` | `cuda` |

## 快速开始

### 1. 修改参数

```bash
vim scripts/train/npu/config.json     # 调参：lr, num_envs, gae_lambda ...
```

### 2. 一键运行

```bash
# NPU
bash scripts/train/npu/run_all.sh

# GPU
bash scripts/train/gpu/run_all.sh
```

### 3. 分步执行

```bash
bash scripts/train/npu/run_bc.sh      # 只跑 BC
bash scripts/train/npu/run_ppo.sh     # 只跑 PPO（需要 BC 模型已存在）
```

### 4. 跳过某些阶段

在 `config.json` 中设置：

```json
"pipeline": {
    "skip_data": false,    // 已有数据，跳过收集
    "skip_bc": false,      // 已有 BC 模型，跳过预训练
    "skip_ppo": false
}
```

或用环境变量临时覆盖（不影响 config.json）：

```bash
SKIP_BC=1 bash scripts/train/npu/run_all.sh   # 只跑 PPO
SKIP_PPO=1 bash scripts/train/npu/run_all.sh  # 只跑 BC
```

### 5. 常用调参场景

```bash
# 场景 1：改 dev 参数 → 直接改 config.json
vim scripts/train/npu/config.json  # 把 num_envs 从 1024 改为 128

# 场景 2：临时换 seed → 环境变量
SEED=99 bash scripts/train/npu/run_all.sh

# 场景 3：多点实验 → 拷贝配置
cp scripts/train/npu/config.json scripts/train/npu/config_high_lambda.json
# 手动修改 gae_lambda，然后用自定义脚本加载
```

## 产物说明

| 目录 | 内容 | 文件格式 |
|------|------|---------|
| `offline_data/` | 离线对局数据 | `{env}_offline_data.pkl` |
| `params/` | 模型权重 | NPU: `.pt`, GPU: `.pkl` / `.ckpt` |
| `checkpoints/` | PPO 周期快照 | NPU: `ppo_ckpt_{N}.pt`, GPU: `ppo_ckpt_{N}.pkl` |
| `logs/` | 训练日志 | `bc_train.log`, `ppo_train.log` |
| `fig/` | 可视化 | `*.svg` |

## 依赖

- `torch >= 2.0`（NPU 训练额外需要 `torch_npu`）
- `jax >= 0.4.28`（GPU 训练）
- 可选：`wandb`（日志记录）
