# src/training/trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.utils as vutils

from src.models.mdm_model import MaskedDiffusionModel
from src.training.visualization import make_xt_from_xclean, reconstruct_from_xt, save_triplet_grid


@dataclass
class TrainerConfig:
    """
    Конфигурация тренера.

    num_epochs:
        Количество эпох обучения.
    grad_clip:
        Максимальная норма градиента (если None, клиппинг не используется).
    log_interval:
        Как часто логировать train loss (в шагах).
    checkpoint_dir:
        Папка для сохранения чекпоинтов.
    sample_dir:
        Папка для сохранения картинок (сэмплы, реконструкции).
    experiment_name:
        Имя эксперимента (будет частью префикса).
    checkpoint_prefix:
        Префикс для имени файлов чекпоинтов/картинок.
        Если None, будет использовано experiment_name.
    use_tqdm:
        Использовать ли tqdm для прогресс-баров.
    uncond_num_steps:
        Количество итераций для unconditional sampler (только для `*_samples.png`).
    """
    num_epochs: int = 10
    grad_clip: Optional[float] = None
    log_interval: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    sample_dir: str = "outputs/samples"
    experiment_name: str = "mdm_experiment"
    checkpoint_prefix: Optional[str] = None
    use_tqdm: bool = True
    uncond_num_steps: int = 50


class Trainer:
    """
    Простой тренер для MaskedDiffusionModel.

    Ожидается, что:
      - model: имеет метод compute_loss(x_clean) и sample(num_samples, device, num_steps),
      - train_loader / val_loader: возвращают словари с ключом "x_clean",
      - optimizer: любой torch.optim.Optimizer.

    После каждой эпохи:
      - сохраняется чекпоинт,
      - сохраняются:
          * samples:   unconditional сэмплы модели,
          * recon:     оригиналы из валидации + их реконструкции.
    """

    def __init__(
        self,
        model: MaskedDiffusionModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: TrainerConfig,
        logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = config
        self.logger = logger

        # Переносим модель на нужное устройство
        self.model.to(self.device)

        # Префикс имён файлов (с гиперпараметрами формируется в скрипте)
        self.checkpoint_prefix = (
            self.cfg.checkpoint_prefix or self.cfg.experiment_name
        )

        # Подготовка папок для чекпоинтов и картинок
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        # Сэмплы кладём в подкаталог с именем префикса
        self.sample_dir = os.path.join(self.cfg.sample_dir, self.checkpoint_prefix)
        os.makedirs(self.sample_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _move_batch_to_device(self, batch) -> torch.Tensor:
        """
        Переносит нужные данные батча на устройство.

        Сейчас нам нужен только x_clean (чистый бинарный MNIST).
        """
        if "x_clean" not in batch:
            raise KeyError("Batch does not contain 'x_clean' key.")
        x_clean = batch["x_clean"].to(self.device)
        return x_clean

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Сохраняет чекпоинт модели и оптимизатора.

        Имена файлов:
          <checkpoint_dir>/<checkpoint_prefix>_epochXXX.pt
        где XXX — номер эпохи с нулями слева.
        """
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "trainer_config": self.cfg,
        }

        filename = f"{self.checkpoint_prefix}_epoch{epoch:03d}.pt"
        path = os.path.join(self.cfg.checkpoint_dir, filename)
        torch.save(state, path)
        self.logger.info(f"Saved checkpoint: {path}")

    @torch.no_grad()
    def _save_visualizations(self, epoch: int) -> None:
        """
        Сохраняет:
          - unconditional сэмплы модели,
          - оригиналы + реконструкции из валидационного датасета
            в одной картинке (верхний ряд — оригинал, нижний — реконструкция).
        """
        self.model.eval()
        
        # Ensure directory exists
        os.makedirs(self.sample_dir, exist_ok=True)

        num_vis = 8  # сколько примеров показывать

        # --------- 1) Unconditional сэмплы ---------
        samples = self.model.sample(
            num_samples=num_vis,
            device=self.device,
            num_steps=self.cfg.uncond_num_steps,
        )  # [B,1,H,W]
        samples = samples.clamp(0.0, 1.0)

        samples_path = os.path.join(
            self.sample_dir,
            f"{self.checkpoint_prefix}_epoch{epoch:03d}_samples.png",
        )
        vutils.save_image(
            samples,
            samples_path,
            nrow=num_vis,
            normalize=False,
        )

        # --------- 2) Оригиналы + реконструкции ---------
        # Берём один батч из val_loader
        try:
            val_batch = next(iter(self.val_loader))
        except StopIteration:
            self.logger.warning(
                "_save_visualizations: val_loader is empty, "
                "skipping reconstructions."
            )
            return

        x_clean = self._move_batch_to_device(val_batch)[:num_vis]  # [B,1,H,W]
        
        # FIX: используем t = T/2, чтобы проверить способность модели 
        # восстанавливать детали (inpainting), а не генерировать с нуля.
        # При t=T и argmax модель (верно) предсказывает фон (наиболее вероятный класс),
        # что выглядит как черный квадрат.
        t_mid = self.model.schedule.num_timesteps // 2
        xt_states, xt_vis = make_xt_from_xclean(
            model=self.model,
            x_clean=x_clean,
            t_value=t_mid,
            obs_mask=None,
        )
        t = torch.full(
            (x_clean.shape[0],),
            fill_value=t_mid,
            dtype=torch.long,
            device=self.device,
        )
        x_recon = reconstruct_from_xt(self.model, xt_states, t)  # [B,1,H,W]

        recon_path = os.path.join(
            self.sample_dir,
            f"{self.checkpoint_prefix}_epoch{epoch:03d}_recon.png",
        )
        save_triplet_grid(
            x_top=x_clean,
            x_mid=xt_vis,
            x_bottom=x_recon,
            path=recon_path,
            nrow=num_vis,
        )

        self.logger.info(f"Saved samples to:        {samples_path}")
        self.logger.info(f"Saved reconstructions to:{recon_path}")

    # ------------------------------------------------------------------
    # Основные циклы обучения/валидации
    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> float:
        """
        Одна эпоха обучения.

        Возвращает средний train loss за эпоху.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        iterator = self.train_loader
        if self.cfg.use_tqdm:
            iterator = tqdm(
                self.train_loader,
                desc=f"Train epoch {epoch}",
                leave=False,
            )

        for step, batch in enumerate(iterator, start=1):
            x_clean = self._move_batch_to_device(batch)

            loss = self.model.compute_loss(x_clean)

            loss.backward()

            # Градиентный клиппинг (если включён)
            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.grad_clip,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

            if step % self.cfg.log_interval == 0:
                avg_loss = running_loss / num_batches
                self.logger.info(
                    f"[Train] Epoch {epoch} | Step {step}/{len(self.train_loader)} "
                    f"| Loss: {loss.item():.4f} | AvgLoss: {avg_loss:.4f}"
                )

        epoch_loss = running_loss / max(num_batches, 1)
        self.logger.info(
            f"[Train] Epoch {epoch} finished | AvgLoss: {epoch_loss:.4f}"
        )
        return epoch_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """
        Валидация модели.

        Возвращает средний val loss.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        iterator = self.val_loader
        if self.cfg.use_tqdm:
            iterator = tqdm(
                self.val_loader,
                desc=f"Val   epoch {epoch}",
                leave=False,
            )

        for batch in iterator:
            x_clean = self._move_batch_to_device(batch)
            loss = self.model.compute_loss(x_clean)

            running_loss += loss.item()
            num_batches += 1

        val_loss = running_loss / max(num_batches, 1)
        self.logger.info(
            f"[Val]   Epoch {epoch} finished | AvgLoss: {val_loss:.4f}"
        )
        return val_loss

    # ------------------------------------------------------------------
    # Основной метод обучения
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """
        Полный цикл обучения на num_epochs.

        На каждой эпохе:
          - train_epoch,
          - validate,
          - сохранение чекпоинта,
          - сохранение сэмплов и реконструкций.
        """
        self.logger.info(
            f"Starting training for {self.cfg.num_epochs} epochs "
            f"on device: {self.device}"
        )

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            self.logger.info(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f}"
            )

            # Чекпоинт + визуализации
            self._save_checkpoint(epoch)
            self._save_visualizations(epoch)

        self.logger.info("Training finished.")
