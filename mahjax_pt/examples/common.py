"""Shared utilities for MahJax PyTorch examples."""

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
OFFLINE_DATA_DIR = EXAMPLES_DIR / "offline_data"
PARAMS_DIR = EXAMPLES_DIR / "params"
FIG_DIR = EXAMPLES_DIR / "fig"


def default_dataset_path(env_name="red_mahjong"):
    return str(OFFLINE_DATA_DIR / f"{env_name}_offline_data.pkl")


def default_bc_params_path(env_name="red_mahjong"):
    return str(PARAMS_DIR / f"{env_name}_bc_params.pt")


def default_rl_params_path(env_name="red_mahjong", seed=0):
    return str(PARAMS_DIR / f"{env_name}-seed={seed}.pt")


def get_network_cls(env_name="red_mahjong"):
    if env_name == "red_mahjong":
        from .networks.red_network import ACNet
        return ACNet
    raise ValueError(f"Unsupported env_name: {env_name}")


def attach_dataset_metadata(dataset, env_name="red_mahjong"):
    dataset["env_name"] = env_name
    dataset["observe_type"] = "dict"
    return dataset
