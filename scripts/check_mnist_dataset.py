# scripts/check_mnist_dataset.py
import argparse
import logging
import os
import sys

import torch
import torchvision.utils as vutils

# Add project root to sys.path for convenient imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data import create_mnist_dataloaders
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Check MNIST dataset & dataloaders.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnist_dataset.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _ensure_image_shape(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has shape [B, C, H, W] with H = W = 28.

    Supports:
      - [B, 1, 28, 28] (no-op)
      - [B, 28, 28]    -> [B, 1, 28, 28]
      - [B, 784]       -> [B, 1, 28, 28]

    Raises ValueError for unexpected shapes.
    """
    if x.dim() == 4:
        # [B, C, H, W]
        return x
    if x.dim() == 3 and x.shape[1] == 28 and x.shape[2] == 28:
        # [B, H, W]
        return x.unsqueeze(1)
    if x.dim() == 2 and x.shape[1] == 28 * 28:
        # [B, 784]
        return x.view(-1, 1, 28, 28)

    raise ValueError(f"Unexpected tensor shape for image: {x.shape}")


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()

    # Seed
    seed = cfg_dict.get("seed", 42)
    set_seed(seed)

    # Logger
    log_cfg = cfg_dict.get("logging", {})
    log_level_str = log_cfg.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = log_cfg.get("log_dir", None)

    logger = setup_logger(name="mnist_dataset", log_level=log_level, log_dir=log_dir)

    logger.info("===== MNIST dataset check started =====")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Effective seed: {seed}")

    data_cfg = cfg_dict.get("data", {})
    train_loader, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=data_cfg.get("batch_size", 64),
        p_missing=data_cfg.get("p_missing", 0.5),
        binarize_threshold=data_cfg.get("binarize_threshold", 0.5),
        flatten=data_cfg.get("flatten", False),
        train_val_split=data_cfg.get("train_val_split", 0.9),
        num_workers=data_cfg.get("num_workers", 2),
        download=data_cfg.get("download", True),
    )

    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Val loader batches:   {len(val_loader)}")

    # Take one batch from train loader and inspect
    batch = next(iter(train_loader))

    x_clean = batch["x_clean"]  # [B, 1, 28, 28] or [B, 784], etc.
    x_obs = batch["x_obs"]
    obs_mask = batch["obs_mask"]
    labels = batch["label"]

    logger.info(f"x_clean shape: {tuple(x_clean.shape)}")
    logger.info(f"x_obs shape:   {tuple(x_obs.shape)}")
    logger.info(f"obs_mask shape:{tuple(obs_mask.shape)}")
    logger.info(f"labels shape:  {tuple(labels.shape)}")

    # Pixel statistics
    with torch.no_grad():
        # fraction of ones in clean images
        frac_ones_clean = (x_clean == 1.0).float().mean().item()

        # fraction of missing pixels (mask == 0)
        frac_missing = (obs_mask == 0.0).float().mean().item()

        # sanity check for sentinel value in x_obs
        frac_sentinel = (x_obs == -1.0).float().mean().item()

    logger.info(f"Fraction of 1s in x_clean: {frac_ones_clean:.4f}")
    logger.info(f"Fraction of missing pixels: {frac_missing:.4f}")
    logger.info(f"Fraction of sentinel (-1.0) in x_obs: {frac_sentinel:.4f}")

    # ------------------------------------------------------------------
    # Create and save visual examples
    # ------------------------------------------------------------------
    samples_dir = os.path.join(PROJECT_ROOT, "outputs", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    num_visual = min(8, x_clean.shape[0])  # сколько картинок показать

    with torch.no_grad():
        # bring tensors to CPU and ensure correct shape [B, 1, 28, 28]
        x_clean_vis = _ensure_image_shape(x_clean[:num_visual].cpu())
        x_obs_vis = _ensure_image_shape(x_obs[:num_visual].cpu())
        obs_mask_vis = _ensure_image_shape(obs_mask[:num_visual].cpu())

        # Для x_obs_vis: заменим sentinel -1.0 на 0.5 (серый),
        # чтобы пропуски были наглядно видны.
        x_obs_vis = x_obs_vis.clone()
        missing_pixels = x_obs_vis < 0.0
        x_obs_vis[missing_pixels] = 0.5

        clean_path = os.path.join(samples_dir, "mnist_clean.png")
        obs_path = os.path.join(samples_dir, "mnist_observed.png")
        mask_path = os.path.join(samples_dir, "mnist_obs_mask.png")

        vutils.save_image(
            x_clean_vis,
            clean_path,
            nrow=num_visual,
            normalize=False,
        )
        vutils.save_image(
            x_obs_vis,
            obs_path,
            nrow=num_visual,
            normalize=False,
        )
        vutils.save_image(
            obs_mask_vis,
            mask_path,
            nrow=num_visual,
            normalize=False,
        )

    logger.info(f"Saved clean samples to: {clean_path}")
    logger.info(f"Saved observed samples to: {obs_path}")
    logger.info(f"Saved observation masks to: {mask_path}")

    logger.info("===== MNIST dataset check finished =====")


if __name__ == "__main__":
    main()
