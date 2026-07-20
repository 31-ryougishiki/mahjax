# MahJax Training Scripts

按训练硬件分类的即用型训练脚本。

## 为什么 GPU 用 JAX，NPU 用 PyTorch？

| | GPU (NVIDIA) | NPU (Ascend) |
|---|---|---|
| **训练脚本** | `gpu/` 目录 | `npu/` 目录 |
| **底层框架** | **JAX** | **PyTorch eager** |
| **环境包** | `mahjax/` | `mahjax_pt/` |
| **核心优化** | `jax.jit` 编译 CUDA kernel | eager 模式，适配 torch_npu |
| **速度** | ~1M+ steps/sec | 取决于 NPU 算力 |
| **CLI 格式** | `key=value` (OmegaConf) | `--key value` (argparse) |

JAX 在 NVIDIA GPU 上能利用 XLA 编译器直接将计算图编译为 CUDA kernel，配合 `jax.vmap` 自动向量化，性能远超 Python 循环的 eager 模式。PyTorch eager 路径的存在是为了支持 JAX 不支持的硬件（如 Ascend NPU）——**如果你有 NVIDIA GPU，请用 `gpu/` 目录下的 JAX 脚本。**

## 目录结构

```
scripts/train/
├── README.md
├── npu/                        # Ascend NPU 训练
│   ├── run_bc.sh               # BC 预训练（含离线数据收集）
│   ├── run_ppo.sh              # PPO 强化学习训练
│   ├── run_all.sh              # 一键全流程
│   ├── offline_data/           # [产物] 离线数据集
│   ├── params/                 # [产物] 模型参数
│   ├── checkpoints/            # [产物] PPO 周期 snapshot
│   ├── logs/                   # [产物] 训练日志
│   └── fig/                    # [产物] 可视化 SVG
│
└── gpu/                        # NVIDIA GPU 训练
    ├── run_bc.sh
    ├── run_ppo.sh
    ├── run_all.sh
    ├── offline_data/
    ├── params/
    ├── checkpoints/
    ├── logs/
    └── fig/
```

## 快速开始

### NPU 训练

```bash
# 一键全流程（离线数据 → BC → PPO）
bash scripts/train/npu/run_all.sh

# 或分步执行
bash scripts/train/npu/run_bc.sh      # BC 预训练
bash scripts/train/npu/run_ppo.sh     # PPO 训练
```

### GPU 训练

```bash
# 一键全流程
bash scripts/train/gpu/run_all.sh

# 或分步执行
bash scripts/train/gpu/run_bc.sh
bash scripts/train/gpu/run_ppo.sh
```

## 环境变量控制

每个脚本支持通过环境变量覆盖默认行为：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SKIP_DATA` | — | 设为 `1` 跳过离线数据收集 |
| `SKIP_BC` | — | 设为 `1` 跳过 BC 预训练 |
| `SKIP_PPO` | — | 设为 `1` 跳过 PPO 训练 |
| `USE_WANDB` | — | 设为 `1` 启用 wandb 日志 |
| `SEED` | `0` | 随机种子 |
| `DEVICE` | `npu:0` / `cuda:0` | 训练设备 |
| `NUM_ENVS` | `1024` | 并行环境数 |
| `NUM_STEPS` | `256` | 每次 rollout 步数 |
| `LR` | `3e-4` | 学习率 |
| `MAG_COEF` | `0.2` | Magnet 正则化系数 |

### 使用示例

```bash
# 只跑 PPO（假设已有 BC 模型），多卡 NPU，调高 λ
SKIP_BC=1 SKIP_DATA=1 DEVICE=npu:1 NUM_ENVS=2048 \
    bash scripts/train/npu/run_ppo.sh

# 带 wandb 的全流程，指定 seed
SEED=42 USE_WANDB=1 bash scripts/train/npu/run_all.sh

# GPU 训练，跳过数据收集（已有数据），调试规模
SKIP_DATA=1 NUM_ENVS=128 NUM_STEPS=64 SEED=0 \
    bash scripts/train/gpu/run_all.sh
```

## 产物说明

每次训练的中间产物全部存放在对应子目录：

| 目录 | 内容 | 文件格式 |
|------|------|---------|
| `offline_data/` | 离线对局数据 | `{env_name}_offline_data.pkl` |
| `params/` | 模型权重 | `{env_name}_bc_params.pt`（BC）, `{env_name}-seed={N}.pt`（PPO） |
| `checkpoints/` | PPO 周期快照 | `ppo_ckpt_{update}.pt` |
| `logs/` | 训练日志 | `bc_train.log`, `ppo_train.log` |
| `fig/` | 可视化 | `*.svg` |

## 恢复训练

```bash
# 从 checkpoint 恢复 PPO 训练
cd <PROJECT_ROOT>
python mahjax_pt/examples/ppo_with_reg.py \
    --resume_from scripts/train/npu/checkpoints/ppo_ckpt_100.pt \
    --device npu:0
```

## 依赖

- `torch >= 2.0`
- NPU 训练额外需要 `torch_npu`
- GPU 训练额外需要 CUDA 版 `torch`
- 可选：`wandb`（日志记录）
