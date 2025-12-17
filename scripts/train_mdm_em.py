# scripts/train_mdm_em.py
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
from src.training.em_trainer import EMTrainer, EMTrainerConfig
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="EM-training of masked diffusion model (MDM) on partially observed binary MNIST."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_em.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint (.pt) to initialize the model (e.g. ambient-pretrained).",
    )
    return parser.parse_args()


def build_checkpoint_prefix(cfg_dict) -> str:
    """
    Строим человекочитаемый префикс имени чекпоинтов,
    включающий некоторые важные гиперпараметры.

    Пример:
      mdm_mnist_em_lr0p001_bs128_T100_ch64_pm0p5_EM3_cons0p1
    """
    exp_name = cfg_dict.get("experiment_name", "mdm_mnist_em")
    data_cfg = cfg_dict.get("data", {})
    model_cfg = cfg_dict.get("model", {})
    optim_cfg = cfg_dict.get("optim", {})
    em_cfg = cfg_dict.get("em", {})

    lr = optim_cfg.get("lr", 1e-3)
    batch_size = data_cfg.get("batch_size", 128)
    T = model_cfg.get("num_timesteps", 100)
    base_ch = model_cfg.get("base_channels", 64)
    p_missing = data_cfg.get("p_missing", 0.5)
    num_em = em_cfg.get("num_em_iters", 3)
    consistency_weight = model_cfg.get("consistency_weight", 0.0)

    # Строка lr / p_missing без точки
    lr_str = str(lr).replace(".", "p")
    pm_str = str(p_missing).replace(".", "p")

    prefix = (
        f"{exp_name}_lr{lr_str}_bs{batch_size}_T{T}_ch{base_ch}"
        f"_pm{pm_str}_EM{num_em}"
    )
    if consistency_weight > 0.0:
        cons_str = str(consistency_weight).replace(".", "p")
        prefix = f"{prefix}_cons{cons_str}"

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
        name="train_mdm_em",
        log_level=log_level,
        log_dir=log_dir,
    )

    logger.info("===== MDM EM training started =====")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Seed: {seed}")
    
    # Логируем consistency_weight заранее (будет использован при создании модели)
    model_cfg = cfg_dict.get("model", {})
    consistency_weight = model_cfg.get("consistency_weight", 0.0)
    if consistency_weight > 0.0:
        logger.info(f"Consistency loss enabled: weight={consistency_weight}, pair_offset={model_cfg.get('consistency_pair_offset', 1)}")
    else:
        logger.info("Consistency loss disabled (consistency_weight=0.0)")

    # --------- Prefixes & output dirs (for checkpoints/samples) ---------
    # checkpoint_prefix используется как fallback, если input_checkpoint_name не задан
    checkpoint_prefix = build_checkpoint_prefix(cfg_dict)

    # --------- Устройство ---------
    device_str = cfg_dict.get("device", "mps")
    # torch принимает строки 'cpu', 'cuda', 'mps'
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # --------- Данные: частично наблюдаемый бинарный MNIST ---------
    data_cfg = cfg_dict.get("data", {})
    train_loader, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=data_cfg.get("batch_size", 128),
        p_missing=data_cfg.get("p_missing", 0.5),  # теперь >0: есть пропуски
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

    consistency_weight = model_cfg.get("consistency_weight", 0.0)
    consistency_pair_offset = model_cfg.get("consistency_pair_offset", 1)
    mdm_cfg = MaskedDiffusionConfig(
        neg_infinity=model_cfg.get("neg_infinity", -1e9),
        training_mode="baseline",  # explicit: M-step trains baseline on x_hat
        consistency_weight=consistency_weight,
        consistency_pair_offset=consistency_pair_offset,
    )
    model = MaskedDiffusionModel(
        unet=unet,
        schedule=schedule,
        cfg=mdm_cfg,
    )
    model.to(device)

    # --------- При необходимости загружаем стартовый чекпоинт ---------
    input_checkpoint_name = None
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint {args.checkpoint} not found.")
            sys.exit(1)
        logger.info(f"Loading initial weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            logger.info("Loaded model_state from checkpoint.")
        else:
            logger.warning("No 'model_state' in checkpoint, skipping model load.")
        # при желании можно восстановить optimizer_state, но EM часто начинают с новым оптимизатором
        
        # Извлекаем имя чекпоинта (без пути и расширения)
        checkpoint_basename = os.path.basename(args.checkpoint)
        if checkpoint_basename.endswith(".pt"):
            input_checkpoint_name = checkpoint_basename[:-3]  # убираем .pt
        else:
            input_checkpoint_name = checkpoint_basename
        logger.info(f"Input checkpoint name: {input_checkpoint_name}")

    # --------- Оптимайзер ---------
    optim_cfg = cfg_dict.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )

    # --------- EMTrainer ---------
    em_raw_cfg = cfg_dict.get("em", {})
    em_cfg = EMTrainerConfig(
        num_em_iters=em_raw_cfg.get("num_em_iters", 3),
        m_epochs_per_iter=em_raw_cfg.get("m_epochs_per_iter", 2),
        cond_num_steps=em_raw_cfg.get("cond_num_steps", 50),
        uncond_num_steps=em_raw_cfg.get("uncond_num_steps", 50),
        log_interval=em_raw_cfg.get("log_interval", 100),
        sample_dir="outputs/samples",
        checkpoint_prefix=checkpoint_prefix,
        input_checkpoint_name=input_checkpoint_name,
        consistency_weight=consistency_weight,
    )

    em_trainer = EMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=em_cfg,
        logger=logger,
    )

    # --------- Запускаем EM-цикл ---------
    em_trainer.fit()

    # --------- Сохраняем финальный чекпоинт ---------
    ckpt_dir = "outputs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    if input_checkpoint_name is not None:
        final_filename = f"em_final_{input_checkpoint_name}.pt"
    else:
        final_filename = f"{checkpoint_prefix}_final.pt"
    final_path = os.path.join(ckpt_dir, final_filename)

    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg_dict,
    }
    torch.save(state, final_path)
    logger.info(f"Saved final EM checkpoint to: {final_path}")
    logger.info("===== MDM EM training finished =====")


if __name__ == "__main__":
    main()
