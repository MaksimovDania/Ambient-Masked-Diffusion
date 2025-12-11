# src/training/trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.mdm_model import MaskedDiffusionModel


@dataclass
class TrainerConfig:
    """
    Конфигурация тренера.

    num_epochs:
        Количество эпох обучения.
    grad_clip:
        Максимальная норма градиента (если None, не применяем клиппинг).
    log_interval:
        Как часто логировать train loss (в шагах).
    checkpoint_dir:
        Папка для сохранения чекпоинтов.
    experiment_name:
        Имя эксперимента (используется в именах файлов).
    """
    num_epochs: int = 10
    grad_clip: Optional[float] = None
    log_interval: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    experiment_name: str = "mdm_experiment"


class Trainer:
    """
    Простой тренер для MaskedDiffusionModel.

    Ожидается, что:
      - model: имеет метод compute_loss(x_clean),
      - train_loader/val_loader: возвращают словари с ключом "x_clean",
      - optimizer: любой оптимизатор PyTorch.

    Пример использования:
      cfg = TrainerConfig(...)
      trainer = Trainer(model, optimizer, train_loader, val_loader, device, cfg, logger)
      trainer.fit()
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

        # Подготовка папки для чекпоинтов
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        # Лучший валид. лосс для сохранения best.pt
        self.best_val_loss: Optional[float] = None

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

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Сохраняет чекпоинт модели и оптимизатора.

        last-чекпоинт:   <checkpoint_dir>/<experiment_name>_last.pt
        best-чекпоинт:   <checkpoint_dir>/<experiment_name>_best.pt
        """
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }

        last_path = os.path.join(
            self.cfg.checkpoint_dir,
            f"{self.cfg.experiment_name}_last.pt",
        )
        torch.save(state, last_path)
        self.logger.info(f"Saved checkpoint: {last_path}")

        if is_best:
            best_path = os.path.join(
                self.cfg.checkpoint_dir,
                f"{self.cfg.experiment_name}_best.pt",
            )
            torch.save(state, best_path)
            self.logger.info(f"Saved BEST checkpoint: {best_path}")

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

        for step, batch in enumerate(self.train_loader, start=1):
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

        for batch in self.val_loader:
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
          - сохранение last-чекпоинта,
          - при улучшении val loss — сохранение best-чекпоинта.
        """
        self.logger.info(
            f"Starting training for {self.cfg.num_epochs} epochs "
            f"on device: {self.device}"
        )

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Обновляем best_val_loss и сохраняем чекпоинты
            is_best = False
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                self.logger.info(
                    f"New best val loss: {val_loss:.4f} (epoch {epoch})"
                )

            self._save_checkpoint(epoch, is_best=is_best)

        self.logger.info("Training finished.")
