# src/training/trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.utils as vutils

from src.models.mdm_model import MaskedDiffusionModel


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
    """
    num_epochs: int = 10
    grad_clip: Optional[float] = None
    log_interval: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    sample_dir: str = "outputs/samples"
    experiment_name: str = "mdm_experiment"
    checkpoint_prefix: Optional[str] = None
    use_tqdm: bool = True


class Trainer:
    """
    Простой тренер для MaskedDiffusionModel.

    Ожидается, что:
      - model: имеет метод compute_loss(x_clean) и sample(num_samples, device),
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

    def _subs_inference_from_xt(
        self,
        xt_states: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        SUBS-подобный шаг для инференса (реконструкция из xt).

        xt_states: [B,1,H,W]  с {0,1,MASK}
        logits:    [B,C,H,W]  "сырые" выходы UNet по классам {0,1,MASK}

        Возвращает:
            x_recon: [B,1,H,W] с {0,1}
        """
        if xt_states.dim() != 4 or logits.dim() != 4:
            raise ValueError(
                f"_subs_inference_from_xt: expected xt_states [B,1,H,W] "
                f"and logits [B,C,H,W], got {tuple(xt_states.shape)}, {tuple(logits.shape)}"
            )

        B, C, H, W = logits.shape
        mask_id = self.model.mask_token_id
        if C != self.model.num_classes:
            raise ValueError(
                f"_subs_inference_from_xt: expected logits C={self.model.num_classes}, got {C}"
            )

        neg_inf = self.model.cfg.neg_infinity

        # 1) Запрещаем класс MASK
        logits = logits.clone()
        logits[:, mask_id, :, :] += neg_inf

        # 2) Нормализуем → log_probs
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # 3) Для незамаскированных токенов (xt != MASK) форсим вырожденное
        #    распределение, копирующее xt (как в SUBS). 
        unmasked = (xt_states != mask_id)  # [B,1,H,W]
        if unmasked.any():
            target = xt_states.squeeze(1).long()  # [B,H,W]

            # шаблон: всё = -inf
            forced = torch.full_like(log_probs, fill_value=neg_inf)  # [B,C,H,W]

            forced_flat = (
                forced.permute(0, 2, 3, 1).contiguous().view(-1, C)
            )  # [B*H*W,C]
            target_flat = target.view(-1)  # [B*H*W]
            row_idx = torch.arange(forced_flat.shape[0], device=logits.device)
            forced_flat[row_idx, target_flat] = 0.0

            forced = (
                forced_flat.view(B, H, W, C).permute(0, 3, 1, 2)
            )  # [B,C,H,W]

            unmasked_broadcast = unmasked.expand_as(log_probs)
            log_probs[unmasked_broadcast] = forced[unmasked_broadcast]

        # 4) Предсказываем класс: argmax по {0,1}, MASK игнорируем (≈ -inf)
        probs = log_probs.exp()  # [B,C,H,W]
        # берём только первые два класса (0 и 1)
        x_recon = probs[:, :2, :, :].argmax(dim=1, keepdim=True)  # [B,1,H,W]
        return x_recon.float()

    def _reconstruct_from_masked(
        self,
        x_clean: torch.Tensor,
        t_value: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Строит реконструкцию изображений из зашумлённой версии.

        Алгоритм (упрощённый):
          1) Выбираем фиксированный шаг t (по умолчанию t = T).
          2) Генерируем xt ~ q(z_t | x0) через MaskSchedule.
          3) Прогоняем UNet по xt.
          4) Применяем SUBS-подобное правило для инференса.
        """
        self.model.eval()
        B, C, H, W = x_clean.shape
        if C != 1:
            raise ValueError(
                f"_reconstruct_from_masked: expected x_clean with C=1, got C={C}"
            )

        if t_value is None:
            t_value = self.model.schedule.num_timesteps

        t = torch.full(
            (B,),
            fill_value=t_value,
            dtype=torch.long,
            device=self.device,
        )

        # 1) Forward-процесс (маскирование)
        xt_states, _ = self.model.schedule.forward_mask(x_clean, t)  # [B,1,H,W]

        # 2) one-hot кодирование xt
        states_flat = xt_states.view(B, H, W)  # [B,H,W]
        one_hot = F.one_hot(
            states_flat, num_classes=self.model.num_classes
        ).permute(0, 3, 1, 2).float()  # [B,C,H,W]

        # 3) UNet
        logits = self.model.unet(one_hot, t)  # [B,C,H,W]

        # 4) SUBS-подобное восстановление
        x_recon = self._subs_inference_from_xt(xt_states, logits)
        return x_recon

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
            num_samples=num_vis, device=self.device
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
        x_recon = self._reconstruct_from_masked(x_clean, t_value=t_mid) # [B,1,H,W]

        # Склеиваем: сначала все оригиналы, затем реконструкции
        both = torch.cat([x_clean, x_recon], dim=0)  # [2B,1,H,W]

        recon_path = os.path.join(
            self.sample_dir,
            f"{self.checkpoint_prefix}_epoch{epoch:03d}_recon.png",
        )
        vutils.save_image(
            both,
            recon_path,
            nrow=num_vis,
            normalize=False,
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
