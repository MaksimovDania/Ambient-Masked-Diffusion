# src/training/em_trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torchvision.utils as vutils

from src.models.mdm_model import MaskedDiffusionModel


@dataclass
class EMTrainerConfig:
    """
    Конфигурация EM-тренера.

    num_em_iters:
        Количество EM-итераций.
    m_epochs_per_iter:
        Сколько эпох делать на M-шаге для каждого EM-итерации.
    cond_num_steps:
        Количество шагов условного диффузионного семплера на E-шаге.
    log_interval:
        Как часто логировать train loss внутри M-шагов (в шагах).
    input_checkpoint_name:
        Имя входного чекпоинта (без пути и расширения), если он был загружен.
        Используется для формирования имён выходных чекпоинтов и папок.
    consistency_weight:
        Вес consistency loss. Используется для добавления суффикса _cons{weight} к имени папки.
    """
    num_em_iters: int = 3
    m_epochs_per_iter: int = 2
    cond_num_steps: int = 50
    uncond_num_steps: int = 50
    log_interval: int = 100
    sample_dir: str = "outputs/samples"
    checkpoint_prefix: str = "mdm_em"
    input_checkpoint_name: Optional[str] = None
    consistency_weight: float = 0.0


class EMTrainer:
    """
    Простейший EM-тренер для MaskedDiffusionModel.

    Ожидания по датасету:
      - train_loader / val_loader возвращают словари с ключами:
          * "x_obs":   [B,1,H,W] с наблюдаемыми пикселями (0/1) и sentinel'ами,
          * "obs_mask":[B,1,H,W] с {0,1},
          * "x_clean": [B,1,H,W] (опционально, для метрик/отладки),
          * "label":   (игнорируется здесь).
    """

    def __init__(
        self,
        model: MaskedDiffusionModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        config: EMTrainerConfig,
        logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = config
        self.logger = logger

        self.model.to(self.device)

        # Build checkpoint prefix: if input_checkpoint_name is provided, use it
        if self.cfg.input_checkpoint_name is not None:
            # Format: em_iterXXX_<input_checkpoint_name>
            # The actual iteration number will be added in save_checkpoint/_save_visualizations
            self.base_checkpoint_name = self.cfg.input_checkpoint_name
        else:
            # Fallback to default prefix
            self.base_checkpoint_name = self.cfg.checkpoint_prefix

        # Directory for saving visualizations (samples / recon grids)
        # Add 'em_' prefix to folder name if input_checkpoint_name is provided
        if self.cfg.input_checkpoint_name is not None:
            folder_name = f"em_{self.base_checkpoint_name}"
        else:
            folder_name = self.base_checkpoint_name
        
        # Add consistency weight suffix if consistency_weight > 0.0
        if self.cfg.consistency_weight > 0.0:
            cons_str = str(self.cfg.consistency_weight).replace(".", "p")
            folder_name = f"{folder_name}_cons{cons_str}"
        
        self.sample_dir = os.path.join(self.cfg.sample_dir, folder_name)
        os.makedirs(self.sample_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _move_batch_to_device(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Переносит x_obs, obs_mask (и опционально x_clean) на устройство.
        """
        if "x_obs" not in batch or "obs_mask" not in batch:
            raise KeyError(
                "EMTrainer expects batch to contain 'x_obs' and 'obs_mask' keys."
            )
        x_obs = batch["x_obs"].to(self.device)
        obs_mask = batch["obs_mask"].to(self.device)

        x_clean = None
        if "x_clean" in batch:
            x_clean = batch["x_clean"].to(self.device)

        return x_obs, obs_mask, x_clean

    # ------------------------------------------------------------------
    # Visualizations (logging only; does not affect training)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _save_visualizations(self, em_iter: int, num_vis: int = 8) -> None:
        """
        Saves:
          - unconditional samples (from fully masked input via model.sample),
          - recon grid from val examples:
              if x_clean is available: x_clean / x_obs_vis / x_hat
              else:                   x_obs_vis / x_hat
        """
        self.model.eval()
        os.makedirs(self.sample_dir, exist_ok=True)

        # 1) Unconditional samples
        samples = self.model.sample(
            num_samples=num_vis,
            device=self.device,
            num_steps=self.cfg.uncond_num_steps,
        )  # [B,1,H,W]
        samples = samples.clamp(0.0, 1.0)
        # Build filename prefix: em_iter{iter:03d}_{base_checkpoint_name}
        if self.cfg.input_checkpoint_name is not None:
            file_prefix = f"em_iter{em_iter:03d}_{self.base_checkpoint_name}"
        else:
            file_prefix = f"{self.cfg.checkpoint_prefix}_em_iter{em_iter:03d}"
        
        samples_path = os.path.join(
            self.sample_dir,
            f"{file_prefix}_samples.png",
        )
        vutils.save_image(samples, samples_path, nrow=num_vis, normalize=False)

        # 2) Dataset -> holes -> conditional reconstruction
        if self.val_loader is None:
            self.logger.info("EMTrainer._save_visualizations: no val_loader provided, skipping recon grid.")
            return

        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            self.logger.warning("EMTrainer._save_visualizations: val_loader is empty, skipping recon grid.")
            return

        x_obs, obs_mask, x_clean = self._move_batch_to_device(batch)
        x_obs = x_obs[:num_vis]
        obs_mask = obs_mask[:num_vis]
        if x_clean is not None:
            x_clean = x_clean[:num_vis]

        x_hat = self.model.conditional_sample(
            x_obs=x_obs,
            obs_mask=obs_mask,
            num_steps=self.cfg.cond_num_steps,
            device=self.device,
        ).clamp(0.0, 1.0)

        x_obs_vis = x_obs.clone()
        missing = (obs_mask == 0)
        if missing.any():
            x_obs_vis[missing] = 0.5

        recon_path = os.path.join(
            self.sample_dir,
            f"{file_prefix}_recon.png",
        )
        if x_clean is None:
            all_imgs = torch.cat([x_obs_vis, x_hat], dim=0)  # [2B,1,H,W]
            vutils.save_image(all_imgs, recon_path, nrow=num_vis, normalize=False)
        else:
            all_imgs = torch.cat([x_clean, x_obs_vis, x_hat], dim=0)  # [3B,1,H,W]
            vutils.save_image(all_imgs, recon_path, nrow=num_vis, normalize=False)

        self.logger.info(f"Saved EM samples to: {samples_path}")
        self.logger.info(f"Saved EM recon grid to: {recon_path}")

    # ------------------------------------------------------------------
    # E-STEP: conditional sampling x_hat ~ p_theta(x0 | x_obs, obs_mask)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def e_step(self) -> TensorDataset:
        """
        Выполняет E-шаг:
          - для каждого батча (x_obs, obs_mask)
          - строит x_hat = conditional_sample(x_obs, obs_mask)
          - собирает всё в один TensorDataset.

        Возвращает TensorDataset с полными картинками:
          - tensors[0]: x_hat  [N,1,H,W]
          - tensors[1]: labels (если были в исходном датасете, иначе нули).
        """
        self.model.eval()
        all_x_hat: List[Tensor] = []
        all_labels: List[Tensor] = []

        iterator = tqdm(
            self.train_loader,
            desc="EM E-step: conditional sampling",
            leave=False,
        )

        for batch in iterator:
            x_obs = batch["x_obs"].to(self.device)
            obs_mask = batch["obs_mask"].to(self.device)

            # Метки используем только чтобы сохранить вместе с данными
            if "label" in batch:
                labels = batch["label"].to(self.device)
            else:
                labels = torch.zeros(x_obs.shape[0], dtype=torch.long, device=self.device)

            x_hat = self.model.conditional_sample(
                x_obs=x_obs,
                obs_mask=obs_mask,
                num_steps=self.cfg.cond_num_steps,
                device=self.device,
            )  # [B,1,H,W] в {0,1}

            all_x_hat.append(x_hat.cpu())
            all_labels.append(labels.cpu())

        x_hat_full = torch.cat(all_x_hat, dim=0)   # [N,1,H,W]
        labels_full = torch.cat(all_labels, dim=0) # [N]

        ds = TensorDataset(x_hat_full, labels_full)
        self.logger.info(
            f"EM E-step: built pseudo-complete dataset with {x_hat_full.shape[0]} samples."
        )
        return ds

    # ------------------------------------------------------------------
    # M-STEP: обучение модели на x_hat как на полном датасете
    # ------------------------------------------------------------------
    def m_step(
        self,
        em_iter: int,
        pseudo_dataset: TensorDataset,
    ) -> None:
        """
        Выполняет M-шаг:
          - создаёт DataLoader из pseudo_dataset,
          - несколько эпох обучает модель с обычным compute_loss(x_clean).
        """
        # pseudo_dataset: (x_hat, labels)
        loader = DataLoader(
            pseudo_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        for epoch in range(1, self.cfg.m_epochs_per_iter + 1):
            self.model.train()
            running_loss = 0.0
            num_batches = 0

            iterator = tqdm(
                loader,
                desc=f"EM M-step (iter {em_iter}) epoch {epoch}",
                leave=False,
            )

            for step, (x_hat, labels) in enumerate(iterator, start=1):
                x_hat = x_hat.to(self.device)

                loss = self.model.compute_loss(x_hat)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                num_batches += 1

                if step % self.cfg.log_interval == 0:
                    avg_loss = running_loss / max(num_batches, 1)
                    self.logger.info(
                        f"[EM {em_iter} | M-epoch {epoch}] "
                        f"step {step}/{len(loader)} "
                        f"| loss={loss.item():.4f} | avg_loss={avg_loss:.4f}"
                    )

            epoch_loss = running_loss / max(num_batches, 1)
            self.logger.info(
                f"[EM {em_iter} | M-epoch {epoch}] finished | avg_loss={epoch_loss:.4f}"
            )

    # ------------------------------------------------------------------
    # Опциональная валидация (на исходной выборке)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, em_iter: int) -> None:
        """
        Простейшая валидация:
          - проходим по val_loader,
          - считаем baseline loss на x_clean (если есть),
            только для мониторинга.
        """
        if self.val_loader is None:
            self.logger.info("EMTrainer.validate: no val_loader provided, skipping.")
            return

        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        iterator = tqdm(
            self.val_loader,
            desc=f"EM iter {em_iter}: validation",
            leave=False,
        )

        for batch in iterator:
            if "x_clean" not in batch:
                continue
            x_clean = batch["x_clean"].to(self.device)
            loss = self.model.compute_loss(x_clean)

            running_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            self.logger.info(
                f"EM iter {em_iter}: validation skipped (no x_clean in val batches)."
            )
            return

        val_loss = running_loss / num_batches
        self.logger.info(
            f"[EM {em_iter}] validation on x_clean | avg_loss={val_loss:.4f}"
        )

    # ------------------------------------------------------------------
    # Сохранение чекпоинта
    # ------------------------------------------------------------------
    def save_checkpoint(self, em_iter: int, filename: Optional[str] = None) -> None:
        """
        Сохраняет состояние модели, оптимизатора и конфиг.

        Если filename не указан, генерируется автоматически:
          - если input_checkpoint_name задан: em_iter{iter:03d}_{input_checkpoint_name}.pt
          - иначе: {checkpoint_prefix}_iter{iter:03d}.pt
        """
        state = {
            "em_iter": em_iter,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        ckpt_dir = "outputs/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        
        if filename is None:
            if self.cfg.input_checkpoint_name is not None:
                filename = f"em_iter{em_iter:03d}_{self.base_checkpoint_name}.pt"
            else:
                filename = f"{self.cfg.checkpoint_prefix}_iter{em_iter:03d}.pt"
        
        ckpt_path = os.path.join(ckpt_dir, filename)
        torch.save(state, ckpt_path)
        self.logger.info(f"Saved EM checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Главный цикл EM
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """
        Запускает полный EM-цикл:
          for em_iter in 1..num_em_iters:
            - E-step: псевдополный датасет
            - M-step: обучение модели на нём
            - (опционально) validate
            - Save checkpoint
        """
        self.logger.info(
            f"Starting EM training for {self.cfg.num_em_iters} iterations "
            f"on device: {self.device}"
        )

        for em_iter in range(1, self.cfg.num_em_iters + 1):
            self.logger.info(f"===== EM iteration {em_iter} started =====")

            # 1) E-step
            pseudo_dataset = self.e_step()

            # 2) M-step
            self.m_step(em_iter, pseudo_dataset)

            # 3) Validation (опционально)
            self.validate(em_iter)

            # 4) Save Checkpoint
            # Имя будет сгенерировано автоматически в save_checkpoint
            self.save_checkpoint(em_iter)

            # 5) Visualizations (logging only)
            self._save_visualizations(em_iter)

            self.logger.info(f"===== EM iteration {em_iter} finished =====")

        self.logger.info("EM training finished.")
