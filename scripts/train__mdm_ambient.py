# scripts/train_mdm_ambient.py
import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.utils as vutils

# Добавляем корень проекта в sys.path для удобного импорта src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data import create_mnist_dataloaders
from src.models.mdm_unet import MDMUNet, MDMUNetConfig
from src.models.mdm_scheduler import MaskSchedule, MaskScheduleConfig
from src.models.mdm_model import MaskedDiffusionModel, MaskedDiffusionConfig
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed
from src.training.visualization import make_xt_from_xclean, reconstruct_from_xt


# ---------------------------------------------------------------------
# Аргументы и вспомогательные функции
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ambient masked diffusion model (MDM) on partially observed binary MNIST."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_ambient.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def build_checkpoint_prefix(cfg_dict) -> str:
    """
    Строим человекочитаемый префикс имени чекпоинтов,
    включающий некоторые важные гиперпараметры.

    Пример:
      mdm_mnist_ambient_lr0p001_bs128_T100_ch64_pmiss0p5_cons0p1
      mdm_mnist_ambient_strict_lr0p001_bs128_T100_ch64_pmiss0p5_cons0p1
    """
    exp_name = cfg_dict.get("experiment_name", "mdm_ambient")
    data_cfg = cfg_dict.get("data", {})
    model_cfg = cfg_dict.get("model", {})
    optim_cfg = cfg_dict.get("optim", {})

    lr = optim_cfg.get("lr", 1e-3)
    batch_size = data_cfg.get("batch_size", 128)
    T = model_cfg.get("num_timesteps", 100)
    base_ch = model_cfg.get("base_channels", 64)
    p_missing = data_cfg.get("p_missing", 0.5)
    consistency_weight = model_cfg.get("consistency_weight", 0.0)
    training_mode = model_cfg.get("training_mode", "ambient_oracle")

    lr_str = str(lr).replace(".", "p")
    pmiss_str = str(p_missing).replace(".", "p")

    # Добавляем суффикс режима для ambient_strict
    mode_suffix = ""
    if training_mode == "ambient_strict":
        mode_suffix = "_strict"

    prefix = (
        f"{exp_name}{mode_suffix}_lr{lr_str}_bs{batch_size}_T{T}_ch{base_ch}_pmiss{pmiss_str}"
    )
    if consistency_weight > 0.0:
        cons_str = str(consistency_weight).replace(".", "p")
        prefix = f"{prefix}_cons{cons_str}"

    return prefix


def move_batch_to_device(batch, device):
    """
    Достаём из батча x_clean, x_obs и obs_mask и переносим на device.
    """
    if "x_clean" not in batch or "obs_mask" not in batch or "x_obs" not in batch:
        raise KeyError("Batch must contain 'x_clean', 'x_obs' and 'obs_mask' keys.")
    x_clean = batch["x_clean"].to(device)
    x_obs = batch["x_obs"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    return x_clean, x_obs, obs_mask


def reconstruct_from_masked(model, x_clean, obs_mask, device, t_value=None):
    """
    Реконструкция изображений из дополнительной маскированной версии.

    x_clean : [B,1,H,W]
    obs_mask: [B,1,H,W] (1 - наблюдаемый пиксель, 0 - пропуск)

    Алгоритм:
      1) выбираем t (по умолчанию t = T//2),
      2) q(z_t | x0) через MaskSchedule.forward_mask,
      3) форсим MASK там, где obs_mask == 0 (ambient-оператор A),
      4) UNet + SUBS-инференс.
    """
    model.eval()
    B, C, H, W = x_clean.shape
    if C != 1:
        raise ValueError(
            f"reconstruct_from_masked: expected x_clean with C=1, got C={C}"
        )

    if t_value is None:
        t_value = model.schedule.num_timesteps // 2

    xt_states, xt_vis = make_xt_from_xclean(
        model=model,
        x_clean=x_clean,
        t_value=t_value,
        obs_mask=obs_mask,
    )
    t = torch.full((B,), fill_value=t_value, dtype=torch.long, device=device)
    x_recon = reconstruct_from_xt(model, xt_states, t)
    return xt_vis, x_recon


def save_visualizations(
    model,
    val_loader: DataLoader,
    device,
    epoch: int,
    sample_dir: str,
    checkpoint_prefix: str,
    logger,
    num_vis: int = 8,
    uncond_num_steps: int = 50,
):
    """
    Сохраняет:
      - unconditional samples,
      - clean / masked / recon в одной картинке.
    """
    os.makedirs(sample_dir, exist_ok=True)
    model.eval()

    # ---------- 1) Unconditional сэмплы ----------
    with torch.no_grad():
        samples = model.sample(num_samples=num_vis, device=device, num_steps=uncond_num_steps)
    samples = samples.clamp(0.0, 1.0)

    samples_path = os.path.join(
        sample_dir,
        f"{checkpoint_prefix}_epoch{epoch:03d}_samples.png",
    )
    vutils.save_image(
        samples,
        samples_path,
        nrow=num_vis,
        normalize=False,
    )

    # ---------- 2) clean / masked / recon ----------
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        logger.warning("save_visualizations: val_loader is empty, skipping.")
        return

    x_clean = batch["x_clean"].to(device)[:num_vis]       # [B,1,H,W]
    obs_mask = batch["obs_mask"].to(device)[:num_vis]     # [B,1,H,W]

    # ambient-маска: xt учитывает и forward-маску, и obs_mask
    xt_vis, x_recon = reconstruct_from_masked(
        model=model,
        x_clean=x_clean,
        obs_mask=obs_mask,
        device=device,
        t_value=model.schedule.num_timesteps // 2,
    )

    # Собираем: первая строка — x_clean, вторая — xt, третья — recon
    all_imgs = torch.cat([x_clean, xt_vis, x_recon], dim=0)

    recon_path = os.path.join(
        sample_dir,
        f"{checkpoint_prefix}_epoch{epoch:03d}_recon.png",
    )
    vutils.save_image(
        all_imgs,
        recon_path,
        nrow=num_vis,
        normalize=False,
    )

    logger.info(f"Saved samples to:        {samples_path}")
    logger.info(f"Saved reconstructions to:{recon_path}")


# ---------------------------------------------------------------------
# Основной цикл обучения (без использования Trainer)
# ---------------------------------------------------------------------
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
        name="train_mdm_ambient",
        log_level=log_level,
        log_dir=log_dir,
    )

    logger.info("===== MDM ambient training started =====")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Seed: {seed}")
    
    # Логируем training_mode и consistency_weight заранее (будут использованы при создании модели)
    model_cfg = cfg_dict.get("model", {})
    training_mode = model_cfg.get("training_mode", "ambient_oracle")
    consistency_weight = model_cfg.get("consistency_weight", 0.0)
    logger.info(f"Training mode: {training_mode}")
    if consistency_weight > 0.0:
        logger.info(f"Consistency loss enabled: weight={consistency_weight}, pair_offset={model_cfg.get('consistency_pair_offset', 1)}")
    else:
        logger.info("Consistency loss disabled (consistency_weight=0.0)")

    # --------- Устройство ---------
    device_str = cfg_dict.get("device", "mps")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # --------- Данные: частично наблюдаемый бинарный MNIST ---------
    data_cfg = cfg_dict.get("data", {})
    p_missing = data_cfg.get("p_missing", 0.5)

    train_loader, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=data_cfg.get("batch_size", 128),
        p_missing=p_missing,
        binarize_threshold=data_cfg.get("binarize_threshold", 0.5),
        flatten=data_cfg.get("flatten", False),
        train_val_split=data_cfg.get("train_val_split", 0.9),
        num_workers=data_cfg.get("num_workers", 4),
        download=data_cfg.get("download", True),
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | p_missing={p_missing}"
    )

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
    training_mode = model_cfg.get("training_mode", "ambient_oracle")
    if training_mode not in {"ambient_oracle", "ambient_strict"}:
        raise ValueError(
            f"train__mdm_ambient.py: training_mode must be 'ambient_oracle' or 'ambient_strict', "
            f"got '{training_mode}'"
        )
    mdm_cfg = MaskedDiffusionConfig(
        neg_infinity=model_cfg.get("neg_infinity", -1e9),
        training_mode=training_mode,  # ambient_oracle or ambient_strict
        consistency_weight=consistency_weight,
        consistency_pair_offset=consistency_pair_offset,
    )
    model = MaskedDiffusionModel(
        unet=unet,
        schedule=schedule,
        cfg=mdm_cfg,
    )
    model.to(device)

    # --------- Оптимайзер ---------
    optim_cfg = cfg_dict.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )

    num_epochs = optim_cfg.get("num_epochs", 20)
    log_interval = optim_cfg.get("log_interval", 100)
    grad_clip = optim_cfg.get("grad_clip", None)

    # --------- Папки и префиксы ---------
    checkpoint_prefix = build_checkpoint_prefix(cfg_dict)
    ckpt_dir = "outputs/checkpoints"
    sample_dir = os.path.join("outputs/samples", checkpoint_prefix)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # --------- Цикл обучения ---------
    logger.info(
        f"Starting ambient training for {num_epochs} epochs on device: {device}"
    )

    for epoch in range(1, num_epochs + 1):
        # ----- TRAIN -----
        model.train()
        running_loss = 0.0
        num_batches = 0

        train_iter = tqdm(
            train_loader,
            desc=f"Train ambient epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(train_iter, start=1):
            x_clean, x_obs, obs_mask = move_batch_to_device(batch, device)

            # Для ambient_strict используем x_obs, для ambient_oracle - x_clean
            if training_mode == "ambient_strict":
                loss = model.compute_loss(x_obs, obs_mask=obs_mask)
            else:  # ambient_oracle
                loss = model.compute_loss(x_clean, obs_mask=obs_mask)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if step % log_interval == 0:
                avg_loss = running_loss / max(num_batches, 1)
                logger.info(
                    f"[Train] Epoch {epoch} | Step {step}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f} | AvgLoss: {avg_loss:.4f}"
                )

        train_loss = running_loss / max(num_batches, 1)
        logger.info(
            f"[Train] Epoch {epoch} finished | AvgLoss: {train_loss:.4f}"
        )

        # ----- VAL -----
        model.eval()
        val_running_loss = 0.0
        val_num_batches = 0

        with torch.no_grad():
            val_iter = tqdm(
                val_loader,
                desc=f"Val   ambient epoch {epoch}",
                leave=False,
            )
            for batch in val_iter:
                x_clean, x_obs, obs_mask = move_batch_to_device(batch, device)
                # Для ambient_strict используем x_obs, для ambient_oracle - x_clean
                if training_mode == "ambient_strict":
                    loss = model.compute_loss(x_obs, obs_mask=obs_mask)
                else:  # ambient_oracle
                    loss = model.compute_loss(x_clean, obs_mask=obs_mask)
                val_running_loss += loss.item()
                val_num_batches += 1

        val_loss = val_running_loss / max(val_num_batches, 1)
        logger.info(
            f"[Val]   Epoch {epoch} finished | AvgLoss: {val_loss:.4f}"
        )

        logger.info(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        # ----- Сохранение чекпоинта -----
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg_dict,
        }
        ckpt_path = os.path.join(
            ckpt_dir,
            f"{checkpoint_prefix}_epoch{epoch:03d}.pt",
        )
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # ----- Визуализации -----
        uncond_num_steps = model_cfg.get("uncond_num_steps", 50)
        save_visualizations(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            sample_dir=sample_dir,
            checkpoint_prefix=checkpoint_prefix,
            logger=logger,
            num_vis=8,
            uncond_num_steps=uncond_num_steps,
        )

    logger.info("===== MDM ambient training finished =====")


if __name__ == "__main__":
    main()
