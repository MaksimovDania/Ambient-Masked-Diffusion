from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .mdm_unet import MDMUNet
from .mdm_scheduler import MaskSchedule


@dataclass
class MaskedDiffusionConfig:
    """
    Конфиг для MaskedDiffusionModel.

    neg_infinity:
        Большое отрицательное число, используемое для "отключения"
        логитов (аналог -inf в SUBS-параметризации MDLM). 
    """
    neg_infinity: float = -1e4


class MaskedDiffusionModel(nn.Module):
    """
    Masked diffusion модель для бинарного MNIST.

    Состав:
      - UNet-денойзер (MDMUNet),
      - MaskSchedule (forward-процесс маскирования),
      - SUBS-параметризация (в духе MDLM для дискретных токенов). 

    Основная функция для обучения: compute_loss(x_clean, obs_mask=None),
    где x_clean — бинарные изображения [B,1,H,W] с {0,1}.
    """

    def __init__(
        self,
        unet: MDMUNet,
        schedule: MaskSchedule,
        cfg: Optional[MaskedDiffusionConfig] = None,
    ) -> None:
        super().__init__()

        self.unet = unet
        self.schedule = schedule
        self.cfg = cfg or MaskedDiffusionConfig()

        # В нашем случае классов 3: 0, 1, MASK
        self.num_classes: int = 3
        self.mask_token_id: int = self.schedule.MASK_TOKEN_ID

        if self.mask_token_id >= self.num_classes:
            raise ValueError(
                f"MaskedDiffusionModel: mask_token_id={self.mask_token_id} "
                f"must be < num_classes={self.num_classes}"
            )

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    @staticmethod
    def _check_binary_x(x: Tensor) -> None:
        """
        Проверяет, что x состоит только из {0,1}.
        """
        if not torch.all((x == 0.0) | (x == 1.0)):
            raise ValueError(
                "MaskedDiffusionModel: expected x_clean to contain only {0,1} values."
            )

    def _encode_states(self, states: Tensor) -> Tensor:
        """
        Кодируем дискретные состояния {0,1,MASK} в one-hot по каналу.

        Parameters
        ----------
        states : LongTensor
            [B, 1, H, W] со значениями в {0,1,2}.

        Returns
        -------
        one_hot : FloatTensor
            [B, num_classes, H, W], где num_classes=3.
        """
        if states.dim() != 4:
            raise ValueError(
                f"_encode_states: expected states with shape [B,1,H,W], got {tuple(states.shape)}"
            )
        B, C, H, W = states.shape
        if C != 1:
            raise ValueError(
                f"_encode_states: expected states with C=1, got C={C}"
            )

        if states.min() < 0 or states.max() >= self.num_classes:
            raise ValueError(
                f"_encode_states: states must be in [0,{self.num_classes-1}], "
                f"got range [{int(states.min())}, {int(states.max())}]"
            )

        states_flat = states.view(B, H, W)  # [B,H,W]
        one_hot = F.one_hot(states_flat, num_classes=self.num_classes)  # [B,H,W,C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B,C,H,W]
        return one_hot

    def _subs_parameterization(
        self,
        logits: Tensor,
        xt_states: Tensor,
        x0_states: Optional[Tensor] = None,
    ) -> Tensor:
        """
        SUBS-параметризация, адаптированная под 2D-картинки.

        Возвращает log_probs [B,C,H,W] с:
          - практически нулевой вероятностью MASK-класса,
          - на незамаскированных позициях (xt != MASK) — вырожденное
            распределение, копирующее x0 (если x0_states задан),
          - на масках — нормализованное распределение по {0,1}.
        """
        if logits.dim() != 4:
            raise ValueError(
                f"_subs_parameterization: expected logits [B,C,H,W], got {tuple(logits.shape)}"
            )
        B, C, H, W = logits.shape
        if C != self.num_classes:
            raise ValueError(
                f"_subs_parameterization: expected C={self.num_classes}, got C={C}"
            )

        if xt_states.dim() != 4 or xt_states.shape[0] != B:
            raise ValueError(
                f"_subs_parameterization: xt_states must have shape [B,1,H,W], "
                f"got {tuple(xt_states.shape)}"
            )

        neg_inf = self.cfg.neg_infinity

        # 1) "отключаем" класс MASK
        logits = logits.clone()
        logits[:, self.mask_token_id, :, :] += neg_inf

        # 2) log-softmax по классам
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Без x0_states просто возвращаем лог-распределения без MASK
        if x0_states is None:
            return log_probs

        if x0_states.shape != xt_states.shape:
            raise ValueError(
                f"_subs_parameterization: x0_states shape {tuple(x0_states.shape)} "
                f"must match xt_states shape {tuple(xt_states.shape)}"
            )

        # 3) Для НЕмаскированных позиций форсим вырожденное распределение,
        #    "копирующее" истинный x0.
        unmasked = (xt_states != self.mask_token_id)  # [B,1,H,W]
        if unmasked.any():
            target = x0_states.squeeze(1).long()  # [B,H,W]

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
            # ставим 0 в таргет-классе, остальные остаются -inf
            log_probs_unmasked_flat[row_idx, target_flat] = 0.0

            log_probs_unmasked = (
                log_probs_unmasked_flat.view(B, H, W, C).permute(0, 3, 1, 2)
            )  # [B,C,H,W]

            unmasked_broadcast = unmasked.expand_as(log_probs)  # [B,C,H,W]
            # БЕЗ torch.where: просто перезаписываем нужные элементы
            log_probs[unmasked_broadcast] = log_probs_unmasked[unmasked_broadcast]

        return log_probs

    # ------------------------------------------------------------------
    # Основной лосс — masked LM в стиле MDLM (+ поддержка obs_mask)
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        x_clean: Tensor,
        obs_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Считает MLM-подобный loss для одного батча.

        x_clean : [B,1,H,W], бинарные {0,1}.
        obs_mask : [B,1,H,W] или None.
          - None: обычный baseline MDM (маска берётся из forward-процесса).
          - не None: готовим почву для Ambient Diffusion:
              * forward-процесс всё ещё добавляет СВОЮ маску (доп. порча),
              * лосс усредняем ТОЛЬКО по пикселям, где obs_mask == 1
                (т.е. по наблюдаемым "A").
        """
        if x_clean.dim() != 4:
            raise ValueError(
                f"compute_loss: expected x_clean [B,1,H,W], got {tuple(x_clean.shape)}"
            )

        B, C, H, W = x_clean.shape
        if C != 1:
            raise ValueError(
                f"compute_loss: expected x_clean with C=1, got C={C}"
            )

        self._check_binary_x(x_clean)

        device = x_clean.device

        # ---- obs_mask ----
        if obs_mask is None:
            obs_mask = torch.ones_like(x_clean)
        else:
            if obs_mask.shape != x_clean.shape:
                raise ValueError("obs_mask shape must match x_clean")
            
            # ожидаем 0/1 (float или int, главное значения)
            # if not torch.all((obs_mask == 0) | (obs_mask == 1)):
            #    raise ValueError("obs_mask must contain only 0/1")
            
            obs_mask = obs_mask.to(device).float()

        # 1) t ~ Uniform({1..T})
        t = self.schedule.sample_timesteps(batch_size=B, device=device)  # [B]

        # 2) Forward-процесс: q(z_t | x0)
        xt_states, mask_positions = self.schedule.forward_mask(x_clean, t)
        # xt_states: [B,1,H,W] в {0,1,MASK}
        # mask_positions: [B,1,H,W] bool (где МЫ замаскировали дополнительно)

        # ----- A(x): учитываем исходную маску наблюдений -----
        # Если obs_mask == 0, то пиксель отсутствует изначально -> ставим MASK
        missing = (obs_mask == 0)
        if missing.any():
            xt_states = xt_states.clone()
            xt_states[missing] = self.mask_token_id

        # 3) Кодируем xt в one-hot по каналам
        xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]

        # 4) Прогоняем UNet
        logits = self.unet(xt_one_hot, t)  # [B,3,H,W]

        # 5) SUBS-параметризация → log_probs
        x0_states = x_clean.long()  # [B,1,H,W] с {0,1}
        log_probs = self._subs_parameterization(
            logits=logits,
            xt_states=xt_states,
            x0_states=x0_states,
        )  # [B,3,H,W]

        # 6) NLL по пикселям
        target = x0_states.squeeze(1).view(-1)  # [B*H*W]
        log_probs_flat = (
            log_probs.view(B, self.num_classes, H * W)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, self.num_classes)
        )  # [B*H*W,3]

        nll_flat = F.nll_loss(
            log_probs_flat,
            target,
            reduction="none",
        )  # [B*H*W]

        nll = nll_flat.view(B, 1, H, W)  # [B,1,H,W]

        # ---- маска для усреднения ----
        mask_positions_float = mask_positions.float()  # где forward маскировал
        train_mask = mask_positions_float * obs_mask   # И forward-маска, И наблюдаемо изначально

        masked_count = train_mask.sum()

        if masked_count < 1:
            # fallback: усредняем по наблюдаемым, но это редкий случай (практически 0)
            loss = (nll * obs_mask).sum() / (obs_mask.sum() + 1e-8)
        else:
            loss = (nll * train_mask).sum() / masked_count

        return loss

    # ------------------------------------------------------------------
    # Простейший сэмплер (один шаг из "всё MASK")
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Простейшая генерация из модели:

        1) Начинаем с z_T = "всё MASK".
        2) Прогоняем UNet на шаге t=T.
        3) Семплируем по {0,1} → бинарное изображение.

        Это НЕ полная реализация обратного диффузионного процесса,
        а просто sanity-check.
        """
        if device is None:
            device = next(self.unet.parameters()).device

        B = num_samples
        H = W = 28  # MNIST

        # 1) Всё MASK
        xt_states = torch.full(
            (B, 1, H, W),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # 2) t = T
        t = torch.full(
            (B,),
            fill_value=self.schedule.num_timesteps,
            dtype=torch.long,
            device=device,
        )

        xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]
        logits = self.unet(xt_one_hot, t)            # [B,3,H,W]

        log_probs = self._subs_parameterization(
            logits=logits,   # ОЙ, опечатка — нужно logits
            xt_states=xt_states,
            x0_states=None,
        )

        probs = log_probs.exp()  # [B,3,H,W]

        # Берём только классы 0 и 1, нормализуем и семплируем категориально
        probs_01 = probs[:, :2, :, :]
        probs_01 = probs_01 / (probs_01.sum(dim=1, keepdim=True) + 1e-8)

        B_sz, _, H_sz, W_sz = probs_01.shape
        probs_flat = probs_01.permute(0, 2, 3, 1).reshape(-1, 2)
        samples_flat = torch.multinomial(probs_flat, num_samples=1)
        x_samples = samples_flat.view(B_sz, 1, H_sz, W_sz).float()

        return x_samples
