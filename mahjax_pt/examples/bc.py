#!/usr/bin/env python3
"""Behavior Cloning trainer (PyTorch port).

Ported from examples/bc.py.
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from mahjax_pt.examples.common import (
    default_dataset_path,
    default_bc_params_path,
    get_network_cls,
    FIG_DIR,
)


def train_bc(
    env_name="red_mahjong",
    dataset_path=None,
    batch_size=1024,
    lr=3e-4,
    num_epochs=5,
    seed=42,
    val_split=0.1,
    save_model_path=None,
    viz_out_dir=None,
    device=None,
):
    if dataset_path is None:
        dataset_path = default_dataset_path(env_name)
    if save_model_path is None:
        save_model_path = default_bc_params_path(env_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "npu":
        import torch_npu
    print(f"Using device: {device}")

    # 1. Load data
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    # Convert to tensors on device
    obs_data = {k: v.to(device) for k, v in data["observation"].items()}
    act_data = torch.tensor(data["action"], dtype=torch.long, device=device)
    mask_data = data["legal_action_mask"].to(device)

    # Filter: remove samples where action is not in the legal mask
    valid = mask_data[torch.arange(mask_data.shape[0]), act_data]
    n_bad = (~valid).sum().item()
    if n_bad > 0:
        print(f"  Filtering out {n_bad} bad samples (action not in legal mask)")
        obs_data = {k: v[valid] for k, v in obs_data.items()}
        act_data = act_data[valid]
        mask_data = mask_data[valid]
    num_samples = act_data.shape[0]

    # 2. Train/Val split
    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split_idx = int(num_samples * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    print(f"Loaded {num_samples} samples. Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 3. Init model
    net_cls = get_network_cls(env_name)
    model = net_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    torch.manual_seed(seed)

    steps_per_epoch = len(train_idx) // batch_size
    val_steps = max(len(val_idx) // batch_size, 1)

    for epoch in range(num_epochs):
        # Train
        model.train()
        rng.shuffle(train_idx)
        train_loss = 0.0
        train_acc = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1} [Train]", mininterval=2.0)
        for i in pbar:
            batch = train_idx[i*batch_size:(i+1)*batch_size]
            obs = {k: v[batch] for k, v in obs_data.items()}
            act = act_data[batch]
            mask = mask_data[batch]

            logits = model.get_action_logits(obs)
            logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
            loss = F.cross_entropy(logits, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=-1)
            acc = (pred == act).float().mean().item()
            train_loss += loss.item()
            train_acc += acc
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        train_loss /= steps_per_epoch
        train_acc /= steps_per_epoch

        # Val
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for i in range(val_steps):
                batch = val_idx[i*batch_size:(i+1)*batch_size]
                if len(batch) == 0:
                    break
                obs = {k: v[batch] for k, v in obs_data.items()}
                act = act_data[batch]
                mask = mask_data[batch]

                logits = model.get_action_logits(obs)
                logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
                loss = F.cross_entropy(logits, act)
                pred = logits.argmax(dim=-1)
                acc = (pred == act).float().mean().item()
                val_loss += loss.item()
                val_acc += acc

        val_loss /= val_steps
        val_acc /= val_steps

        print(f"Ep {epoch+1:02d} | Tr Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 5. Save
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")
    return model, save_model_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="red_mahjong")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model_path", default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda:0, npu:0")
    args = parser.parse_args()
    train_bc(**vars(args))
