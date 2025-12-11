# scripts/train_mdm_ambient.py
import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
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
      mdm_mnist_ambient_lr0p001_bs128_T100_ch64_pmiss0p5
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

    lr_str = str(lr).replace(".", "p")
    pmiss_str = str(p_missing).replace(".", "p")

    prefix = (
        f"{exp_name}_lr{lr_str}_bs{batch_size}_T{T}_ch{base_ch}_pmiss{pmiss_str}"
    )
    return prefix


def move_batch_to_device(batch, device):
    """
    Достаём из батча x_clean и obs_mask и переносим на device.
    """
    if "x_clean" not in batch or "obs_mask" not in batch:
        raise KeyError("Batch must contain 'x_clean' and 'obs_mask' keys.")
    x_clean = batch["x_clean"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    return x_clean, obs_mask


def subs_inference_from_xt(model, xt_states, logits):
    """
    SUBS-подобный шаг инференса (без torch.where).

    xt_states: [B,1,H,W] с {0,1,MASK}
    logits:    [B,C,H,W] с сырыми логитами по классам {0,1,MASK}

    Возвращает:
        x_recon: [B,1,H,W] с {0,1}
    """
    if xt_states.dim() != 4 or logits.dim() != 4:
        raise ValueError(
            f"subs_inference_from_xt: expected xt_states [B,1,H,W] and logits [B,C,H,W], "
            f"got {tuple(xt_states.shape)}, {tuple(logits.shape)}"
        )

    B, C, H, W = logits.shape
    mask_id = model.mask_token_id
    if C != model.num_classes:
        raise ValueError(
            f"subs_inference_from_xt: expected logits C={model.num_classes}, got C={C}"
        )

    neg_inf = model.cfg.neg_infinity

    # 1) Запрещаем MASK-класс
    logits = logits.clone()
    logits[:, mask_id, :, :] += neg_inf

    # 2) log-softmax
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # 3) Для незамаскированных позиций форсим вырожденное распределение,
    #    копирующее xt_states.
    unmasked = (xt_states != mask_id)  # [B,1,H,W]
    if unmasked.any():
        target = xt_states.squeeze(1).long()  # [B,H,W]

        log_probs_unmasked = torch.full_like(
            log_probs, fill_value=neg_inf
        )  # [B,C,H,W]

        log_probs_unmasked_flat = (
            log_probs_unmasked.permute(0, 2, 3, 1).contiguous().view(-1, C)
        )
        target_flat = target.view(-1)  # [B*H*W]
        row_idx = torch.arange(
            log_probs_unmasked_flat.shape[0],
            device=logits.device,
        )
        log_probs_unmasked_flat[row_idx, target_flat] = 0.0

        log_probs_unmasked = (
            log_probs_unmasked_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        )

        unmasked_broadcast = unmasked.expand_as(log_probs)
        log_probs[unmasked_broadcast] = log_probs_unmasked[unmasked_broadcast]

    # 4) Берём argmax по классам {0,1} (MASK ≈ -inf)
    probs = log_probs.exp()
    x_recon = probs[:, :2, :, :].argmax(dim=1, keepdim=True).float()
    return x_recon


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

    t = torch.full(
        (B,),
        fill_value=t_value,
        dtype=torch.long,
        device=device,
    )

    # 1) forward-процесс
    xt_states, _ = model.schedule.forward_mask(x_clean, t)  # [B,1,H,W]

    # 2) учтём исходную маску наблюдений: obs_mask == 0 → MASK
    missing = (obs_mask == 0)
    if missing.any():
        xt_states = xt_states.clone()
        xt_states[missing] = model.mask_token_id

    # 3) one-hot
    states_flat = xt_states.view(B, H, W)
    xt_one_hot = F.one_hot(
        states_flat, num_classes=model.num_classes
    ).permute(0, 3, 1, 2).float()

    # 4) UNet + SUBS-инференс
    logits = model.unet(xt_one_hot, t)
    x_recon = subs_inference_from_xt(model, xt_states, logits)
    return xt_states, x_recon


def save_visualizations(
    model,
    val_loader: DataLoader,
    device,
    epoch: int,
    sample_dir: str,
    checkpoint_prefix: str,
    logger,
    num_vis: int = 8,
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
        samples = model.sample(num_samples=num_vis, device=device)
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
    xt_states, x_recon = reconstruct_from_masked(
        model=model,
        x_clean=x_clean,
        obs_mask=obs_mask,
        device=device,
        t_value=model.schedule.num_timesteps // 2,
    )

    # Визуализируем xt: MASK → 0.5 (серый)
    xt_vis = xt_states.float()
    mask_token = model.mask_token_id
    mask_positions = (xt_states == mask_token)
    if mask_positions.any():
        xt_vis = xt_vis.clone()
        xt_vis[mask_positions] = 0.5

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

    mdm_cfg = MaskedDiffusionConfig(
        neg_infinity=cfg_dict.get("model", {}).get("neg_infinity", -1e9)
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
            x_clean, obs_mask = move_batch_to_device(batch, device)

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
                x_clean, obs_mask = move_batch_to_device(batch, device)
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
        save_visualizations(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            sample_dir=sample_dir,
            checkpoint_prefix=checkpoint_prefix,
            logger=logger,
            num_vis=8,
        )

    logger.info("===== MDM ambient training finished =====")


if __name__ == "__main__":
    main()
