# scripts/train_baseline_mdm.py
import argparse
import os
import sys

import torch
import torchvision.utils as vutils

# Добавляем project_root в sys.path для удобных импортов
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed
from src.data import create_mnist_dataloaders
from src.models.mdm_unet import MDMUNet, MDMUNetConfig
from src.models.mdm_scheduler import MaskSchedule, MaskScheduleConfig
from src.models.mdm_model import MaskedDiffusionModel
from src.training.trainer import Trainer, TrainerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline masked diffusion model on MNIST."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mdm_baseline.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def build_experiment_name(model_cfg: dict, data_cfg: dict, optim_cfg: dict) -> str:
    """
    Строим имя эксперимента, в которое зашиты гиперпараметры.

    Пример:
      mdm_mnist_T100_ch64_bs128_lr0.001
    """
    T = model_cfg.get("num_timesteps", 100)
    ch = model_cfg.get("base_channels", 64)
    bs = data_cfg.get("batch_size", 128)
    lr = optim_cfg.get("lr", 1e-3)

    exp_name = f"mdm_mnist_T{T}_ch{ch}_bs{bs}_lr{lr}"
    return exp_name


@torch.no_grad()
def sanity_check_initial_loss(
    model: MaskedDiffusionModel,
    train_loader,
    device: torch.device,
    logger,
) -> None:
    """
    Простейший sanity-check: один проход compute_loss по маленькому батчу.

    Проверяем:
      - что лосс считается,
      - что он конечный (не NaN/Inf).
    """
    model.eval()
    batch = next(iter(train_loader))
    x_clean = batch["x_clean"][:8].to(device)

    loss = model.compute_loss(x_clean)
    logger.info(f"Initial sanity-check loss: {loss.item():.6f}")

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Sanity check failed: initial loss is not finite "
            f"(got {loss.item()}). Check model/schedule implementation."
        )


@torch.no_grad()
def save_generated_samples(
    model: MaskedDiffusionModel,
    epoch: int,
    samples_dir: str,
    device: torch.device,
    num_samples: int = 16,
) -> None:
    """
    Вызывает model.sample и сохраняет сэмплы.

    Сэмплы сохраняются как:
      <samples_dir>/generated_epoch_{epoch:03d}.png
    """
    model.eval()
    os.makedirs(samples_dir, exist_ok=True)

    samples = model.sample(num_samples=num_samples, device=device)  # [B,1,H,W]
    if samples.dim() != 4 or samples.shape[1] != 1:
        raise RuntimeError(
            "save_generated_samples: expected samples with shape [B,1,H,W], "
            f"got {tuple(samples.shape)}"
        )

    out_path = os.path.join(samples_dir, f"generated_epoch_{epoch:03d}.png")
    vutils.save_image(
        samples.cpu(),
        out_path,
        nrow=int(num_samples ** 0.5),
        normalize=False,
    )


@torch.no_grad()
def save_reconstructions(
    model: MaskedDiffusionModel,
    schedule: MaskSchedule,
    val_loader,
    epoch: int,
    samples_dir: str,
    device: torch.device,
    num_examples: int = 8,
    t_step: int | None = None,
) -> None:
    """
    Сохраняет картинку с оригиналами, маскированными и восстановленными версиями.

    Идея:
      - Берём первый батч из val_loader.
      - Выбираем num_examples примеров.
      - Применяем forward_mask на фиксированном t_step (по умолчанию середина).
      - Прогоняем через модель, получаем реконструкцию.
      - Собираем одну картинку из 3*num_examples изображений:
          [x_clean, x_masked, x_recon].

    Картинка сохраняется как:
      <samples_dir>/recon_epoch_{epoch:03d}.png
    """
    model.eval()
    os.makedirs(samples_dir, exist_ok=True)

    batch = next(iter(val_loader))
    x_clean = batch["x_clean"][:num_examples].to(device)  # [B,1,28,28]
    B, C, H, W = x_clean.shape
    if C != 1:
        raise RuntimeError(
            f"save_reconstructions: expected x_clean with C=1, got {C}"
        )

    # Выбираем шаг t для маскирования
    if t_step is None:
        t_val = max(schedule.num_timesteps // 2, 1)
    else:
        t_val = max(min(t_step, schedule.num_timesteps), 1)

    t = torch.full((B,), t_val, dtype=torch.long, device=device)

    # Применяем forward-процесс q(z_t | x0)
    xt_states, mask_positions = schedule.forward_mask(x_clean, t)
    # xt_states: [B,1,H,W] в {0,1,MASK}, mask_positions: [B,1,H,W] bool

    # Кодируем xt в one-hot
    xt_one_hot = model._encode_states(xt_states)  # [B,3,H,W]

    # Прогоняем UNet
    logits = model.unet(xt_one_hot, t)  # [B,3,H,W]

    # SUBS-параметризация с использованием истинного x_clean как x0
    x0_states = x_clean.long()
    log_probs = model._subs_parameterization(
        logits=logits,
        xt_states=xt_states,
        x0_states=x0_states,
    )  # [B,3,H,W]

    # Получаем реконструкцию как argmax по {0,1}, игнорируя MASK-класс
    probs = log_probs.exp()  # [B,3,H,W]
    x_recon = probs[:, :2, :, :].argmax(dim=1, keepdim=True).float()  # [B,1,H,W]

    # Готовим "видимую" маскированную картинку:
    #   0 -> 0.0 (чёрный), 1 -> 1.0 (белый), MASK -> 0.5 (серый).
    xt_vis = torch.zeros_like(x_clean)
    xt_vis[xt_states == 0] = 0.0
    xt_vis[xt_states == 1] = 1.0
    xt_vis[xt_states == model.mask_token_id] = 0.5

    # Sanity-check: формы совпадают
    if not (x_clean.shape == xt_vis.shape == x_recon.shape):
        raise RuntimeError(
            "save_reconstructions: shape mismatch between x_clean, xt_vis, x_recon: "
            f"{tuple(x_clean.shape)}, {tuple(xt_vis.shape)}, {tuple(x_recon.shape)}"
        )

    # Собираем grid из 3 * B картинок: сначала оригиналы, потом маскированные,
    # затем реконструкции. При nrow=num_examples получим 3 строки:
    #   1-я строка — оригиналы,
    #   2-я — маскированные,
    #   3-я — реконструкции.
    grid = torch.cat(
        [
            x_clean.cpu(),
            xt_vis.cpu(),
            x_recon.cpu(),
        ],
        dim=0,
    )  # [3B,1,H,W]

    out_path = os.path.join(samples_dir, f"recon_epoch_{epoch:03d}.png")
    vutils.save_image(
        grid,
        out_path,
        nrow=num_examples,
        normalize=False,
    )


def main():
    args = parse_args()

    # 1) Загружаем конфиг
    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()

    # 2) Ставим сид
    seed = cfg_dict.get("seed", 42)
    set_seed(seed)

    # 3) Логгер
    log_cfg = cfg_dict.get("logging", {})
    log_level_str = log_cfg.get("level", "INFO").upper()
    log_dir = log_cfg.get("log_dir", "outputs/logs")
    logger = setup_logger(
        name="train_mdm",
        log_level=getattr(torch.logging, log_level_str, None)
        if hasattr(torch, "logging")
        else __import__("logging").getLogger().level,
        log_dir=log_dir,
    )
    # Лайфхак: если вдруг выше получилось странно — просто используем logging.INFO
    import logging

    logger.setLevel(getattr(logging, log_level_str, logging.INFO))

    logger.info("===== Training baseline MDM on MNIST =====")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Seed: {seed}")

    # 4) Разбираем секции конфига
    data_cfg = cfg_dict.get("data", {})
    model_cfg = cfg_dict.get("model", {})
    optim_cfg = cfg_dict.get("optim", {})

    # 5) Определяем устройство
    device_str = optim_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # 6) Создаём даталоадеры
    train_loader, val_loader = create_mnist_dataloaders(
        root=data_cfg.get("root", "data"),
        batch_size=data_cfg.get("batch_size", 128),
        p_missing=data_cfg.get("p_missing", 0.0),  # baseline: без порчи A
        binarize_threshold=data_cfg.get("binarize_threshold", 0.5),
        flatten=data_cfg.get("flatten", False),
        train_val_split=data_cfg.get("train_val_split", 0.9),
        num_workers=data_cfg.get("num_workers", 4),
        download=data_cfg.get("download", True),
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 7) Строим UNet
    unet_cfg = MDMUNetConfig(
        image_channels=model_cfg.get("image_channels", 3),
        base_channels=model_cfg.get("base_channels", 64),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        time_emb_dim_mult=model_cfg.get("time_emb_dim_mult", 4),
    )
    unet = MDMUNet(unet_cfg)

    # 8) Строим расписание маскирования
    schedule_cfg = MaskScheduleConfig(
        num_timesteps=model_cfg.get("num_timesteps", 100),
        alpha_min=model_cfg.get("alpha_min", 0.01),
        schedule_type=model_cfg.get("schedule_type", "linear"),
    )
    schedule = MaskSchedule(schedule_cfg)

    # 9) Собираем masked diffusion модель
    model = MaskedDiffusionModel(unet=unet, schedule=schedule)

    # 10) Оптимизатор
    lr = optim_cfg.get("lr", 1e-3)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 11) Конфиг тренера
    exp_name = build_experiment_name(model_cfg, data_cfg, optim_cfg)
    trainer_cfg = TrainerConfig(
        num_epochs=optim_cfg.get("num_epochs", 20),
        grad_clip=optim_cfg.get("grad_clip", 1.0),
        log_interval=optim_cfg.get("log_interval", 100),
        checkpoint_dir="outputs/checkpoints",
        experiment_name=exp_name,
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

    # 12) Sanity-check: один шаг compute_loss
    sanity_check_initial_loss(model, train_loader, device, logger)

    # 13) Подготовка директорий для чекпоинтов и картинок
    ckpt_dir = trainer_cfg.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    samples_dir = os.path.join("outputs", "samples", exp_name)
    os.makedirs(samples_dir, exist_ok=True)

    logger.info(f"Checkpoints will be saved to: {ckpt_dir}")
    logger.info(f"Samples will be saved to: {samples_dir}")
    logger.info(f"Experiment name (used in filenames): {exp_name}")

    # 14) Цикл обучения с ручным управлением (чтобы после каждой эпохи делать сэмплинг/визуализацию)
    best_val_loss = None

    for epoch in range(1, trainer_cfg.num_epochs + 1):
        # --- Обучение ---
        train_loss = trainer.train_epoch(epoch)
        if not torch.isfinite(torch.tensor(train_loss)):
            logger.error(
                f"Train loss at epoch {epoch} is not finite ({train_loss}). Stopping."
            )
            break

        # --- Валидация ---
        val_loss = trainer.validate(epoch)
        if not torch.isfinite(torch.tensor(val_loss)):
            logger.error(
                f"Val loss at epoch {epoch} is not finite ({val_loss}). Stopping."
            )
            break

        # --- Обновляем лучший валид. лосс ---
        is_best = False
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
            logger.info(
                f"New best val loss: {val_loss:.6f} at epoch {epoch}"
            )

        # --- Сохранение чекпоинтов ---
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": cfg_dict,
        }

        # Имя файла с гиперпараметрами + номер эпохи
        ckpt_name = f"{exp_name}_epoch{epoch:03d}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # Обновляем "last" чекпоинт
        last_path = os.path.join(ckpt_dir, f"{exp_name}_last.pt")
        torch.save(state, last_path)

        # Обновляем "best" чекпоинт (если нужно)
        if is_best:
            best_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")
            torch.save(state, best_path)

        # --- Сэмплинг и реконструкции ---
        try:
            save_generated_samples(
                model=model,
                epoch=epoch,
                samples_dir=samples_dir,
                device=device,
                num_samples=16,
            )
            save_reconstructions(
                model=model,
                schedule=schedule,
                val_loader=val_loader,
                epoch=epoch,
                samples_dir=samples_dir,
                device=device,
                num_examples=8,
            )
            logger.info(
                f"Saved generated samples and reconstructions for epoch {epoch}."
            )
        except Exception as e:
            logger.error(
                f"Error during sampling/reconstruction at epoch {epoch}: {e}"
            )

    logger.info("Training loop finished.")


if __name__ == "__main__":
    main()
