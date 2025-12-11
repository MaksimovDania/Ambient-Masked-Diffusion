# scripts/train_baseline_mdm.py
import argparse
import logging
import os
import sys

import torch

# Добавляем корень проекта в sys.path для удобного импорта src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data import create_mnist_dataloaders
from src.models.mdm_unet import MDMUNet, MDMUNetConfig
from src.models.mdm_scheduler import MaskSchedule, MaskScheduleConfig
from src.models.mdm_model import MaskedDiffusionModel, MaskedDiffusionConfig
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline masked diffusion model (MDM) on binary MNIST."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_baseline.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def build_checkpoint_prefix(cfg_dict) -> str:
    """
    Строим человекочитаемый префикс имени чекпоинтов,
    включающий некоторые важные гиперпараметры.

    Пример:
      mdm_mnist_baseline_lr0p001_bs128_T100_ch64
    """
    exp_name = cfg_dict.get("experiment_name", "mdm_experiment")
    data_cfg = cfg_dict.get("data", {})
    model_cfg = cfg_dict.get("model", {})
    optim_cfg = cfg_dict.get("optim", {})

    lr = optim_cfg.get("lr", 1e-3)
    batch_size = data_cfg.get("batch_size", 128)
    T = model_cfg.get("num_timesteps", 100)
    base_ch = model_cfg.get("base_channels", 64)

    # Строка lr без точки
    lr_str = str(lr).replace(".", "p")

    prefix = f"{exp_name}_lr{lr_str}_bs{batch_size}_T{T}_ch{base_ch}"
    return prefix


def main():
    args = parse_args()

    # --------- Конфиг, сиды, логгер ---------
    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()

    seed = cfg_dict.get("seed", 42)
    set_seed(seed)

    log_cfg = cfg_dict.get("logging", {})
    log_level_str = log_cfg.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = log_cfg.get("log_dir", "outputs/logs")

    logger = setup_logger(
        name="train_mdm_baseline",
        log_level=log_level,
        log_dir=log_dir,
    )

    logger.info("===== MDM baseline training started =====")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Seed: {seed}")

    # --------- Устройство ---------
    device = cfg_dict.get("device", "mps")
    logger.info(f"Using device: {device}")

    # --------- Данные (чистый бинарный MNIST, p_missing=0.0) ---------
    data_cfg = cfg_dict.get("data", {})
    train_loader, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=data_cfg.get("batch_size", 128),
        p_missing=data_cfg.get("p_missing", 0.0),  # baseline: без порчи
        binarize_threshold=data_cfg.get("binarize_threshold", 0.5),
        flatten=data_cfg.get("flatten", False),
        train_val_split=data_cfg.get("train_val_split", 0.9),
        num_workers=data_cfg.get("num_workers", 4),
        download=data_cfg.get("download", True),
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --------- Модель: UNet + MaskSchedule + MDM wrapper ---------
    model_cfg = cfg_dict.get("model", {})

    unet_config = MDMUNetConfig(
        image_channels=model_cfg.get("image_channels", 3),
        base_channels=model_cfg.get("base_channels", 64),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        time_emb_dim_mult=model_cfg.get("time_emb_dim_mult", 4),
    )
    unet = MDMUNet(unet_config)

    schedule_config = MaskScheduleConfig(
        num_timesteps=model_cfg.get("num_timesteps", 100),
        alpha_min=model_cfg.get("alpha_min", 0.01),
        schedule_type=model_cfg.get("schedule_type", "linear"),
    )
    schedule = MaskSchedule(schedule_config)

    mdm_cfg = MaskedDiffusionConfig(
        neg_infinity=cfg_dict.get("model", {}).get("neg_infinity", -1e9)
    )
    model = MaskedDiffusionModel(
        unet=unet,
        schedule=schedule,
        cfg=mdm_cfg,
    )

    # --------- Оптимайзер ---------
    optim_cfg = cfg_dict.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )

    # --------- Trainer ---------
    checkpoint_prefix = build_checkpoint_prefix(cfg_dict)

    trainer_cfg = TrainerConfig(
        num_epochs=optim_cfg.get("num_epochs", 20),
        grad_clip=optim_cfg.get("grad_clip", None),
        log_interval=optim_cfg.get("log_interval", 100),
        checkpoint_dir="outputs/checkpoints",
        sample_dir="outputs/samples",
        experiment_name=cfg_dict.get("experiment_name", "mdm_mnist_baseline"),
        checkpoint_prefix=checkpoint_prefix,
        use_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_cfg,
        logger=logger,
    )

    # --------- Старт обучения ---------
    trainer.fit()

    logger.info("===== MDM baseline training finished =====")


if __name__ == "__main__":
    main()
