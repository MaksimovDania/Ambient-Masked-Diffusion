from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .mdm_unet import MDMUNet
from .mdm_scheduler import MaskSchedule, MaskScheduleConfig


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

    Основная функция для обучения: compute_loss(x_clean),
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

        В оригинальном MDLM:  
        - mask-логит задаётся как -inf (маска никогда не предсказывается),
        - для НЕзамаскированных токенов распределение делается вырожденным,
          которое "копирует" xt (у нас xt=x0 на незамаскированных позициях).

        Здесь:
          logits:     [B, C(=3), H, W] — "сырые" выходы UNet,
          xt_states:  [B, 1, H, W]     — текущее состояние {0,1,MASK},
          x0_states:  [B, 1, H, W]     — истинное {0,1}; может быть None
                                         (например, при семплинге).

        Возвращает:
          log_probs: [B, C, H, W] — log p(y | x_t), уже с SUBS,
          т.е.:
            - класс MASK практически impossible (≈0),
            - на незамаскированных позициях — one-hot в x0,
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

        # 1) Запрещаем класс MASK (аналог logits[:, :, mask_index] += neg_inf)
        #    → p(mask) ≈ 0.
        neg_inf = self.cfg.neg_infinity
        logits = logits.clone()
        logits[:, self.mask_token_id, :, :] += neg_inf

        # 2) Нормализуем logits → log_probs (log p) по классам
        #    (как в MDLM: logits - logsumexp(logits)).
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Если x0_states не задан, мы не можем "копировать" незамаскированные
        # токены, поэтому просто возвращаем log_probs с отключённым MASK.
        if x0_states is None:
            return log_probs

        if x0_states.shape != xt_states.shape:
            raise ValueError(
                f"_subs_parameterization: x0_states shape {tuple(x0_states.shape)} "
                f"must match xt_states shape {tuple(xt_states.shape)}"
            )

        # 3) Обрабатываем незамаскированные токены:
        #    для xt != MASK задаём вырожденное распределение,
        #    которое всегда "копирует" x0 (0 или 1).
        #    Это повторяет идею SUBS: незамаскированные токены не изменяются. 
        unmasked = (xt_states != self.mask_token_id)  # [B,1,H,W]
        if unmasked.any():
            # a) Target-класс из x0: {0,1}
            target = x0_states.squeeze(1).long()  # [B,H,W]

            # b) Создаём вырожденное распределение: лог-вероятности
            #    = 0 для target-класса и = -inf для остальных.
            log_probs_unmasked = torch.full_like(
                log_probs, fill_value=neg_inf
            )  # [B,C,H,W]

            # Перекладываем в [B,H,W,C] и разом обновляем по батчу
            log_probs_unmasked_flat = (
                log_probs_unmasked.permute(0, 2, 3, 1).contiguous().view(-1, C)
            )
            target_flat = target.view(-1)  # [B*H*W]
            # Только там, где унмаск, заполняем 0 в нужном классе
            # Остальные элементы остаются -inf, но это неважно,
            # т.к. мы будет смешивать по unmasked mask'у.
            row_idx = torch.arange(log_probs_unmasked_flat.shape[0], device=logits.device)

            log_probs_unmasked_flat[row_idx, target_flat] = 0.0
            # Возвращаем форму [B,C,H,W]
            log_probs_unmasked = (
                log_probs_unmasked_flat.view(B, H, W, C).permute(0, 3, 1, 2)
            )

            # c) Обновляем log_probs на незамаскированных позициях:
            #    там ставим вырожденное распределение, на масках оставляем
            #    "обучаемое" распределение.
            unmasked_broadcast = unmasked.expand_as(log_probs)  # [B,C,H,W]
            # log_probs = torch.where(unmasked_broadcast, log_probs_unmasked, log_probs)
            # WORKAROUND for MPS: torch.where might be buggy with large negative values/broadcasting
            log_probs[unmasked_broadcast] = log_probs_unmasked[unmasked_broadcast]

        return log_probs

    # ------------------------------------------------------------------
    # Основной лосс — masked LM в стиле MDLM
    # ------------------------------------------------------------------
    def compute_loss(self, x_clean: Tensor) -> Tensor:
        """
        Считает MLM-подобный loss для одного батча.

        Параметры
        ---------
        x_clean : FloatTensor
            Бинарные изображения [B,1,H,W] с {0,1}.

        Алгоритм
        --------
        1) Проверяем бинарность x_clean.
        2) Семплируем t ~ Uniform({1..T}).
        3) q(z_t | x0) через MaskSchedule.forward_mask → xt_states, mask_positions.
        4) Кодируем xt_states в one-hot и подаём в UNet с t.
        5) Применяем SUBS-параметризацию к logits.
        6) Считаем NLL(x0 | xt, t) по замаскированным пикселям,
           используя log_probs как log p. (F.nll_loss).

        Возвращает скалярный loss (усреднение по маскированным пикселям).
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

        x_clean = x_clean.to(self.unet.device if hasattr(self.unet, "device") else x_clean.device)
        device = x_clean.device

        # 1) t ~ Uniform({1..T})
        t = self.schedule.sample_timesteps(batch_size=B, device=device)  # [B]

        # 2) Forward-процесс: q(z_t | x0)
        xt_states, mask_positions = self.schedule.forward_mask(x_clean, t)
        # xt_states: [B,1,H,W] в {0,1,MASK}
        # mask_positions: [B,1,H,W] bool

        # 3) Кодируем xt в one-hot по каналам
        xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]

        # 4) Прогоняем UNet
        #    (t подаём как float/int — TimeEmbedding внутри приведёт к float).
        logits = self.unet(xt_one_hot, t)  # [B,3,H,W]

        # 5) SUBS-параметризация → log_probs (log p(y | xt, t))
        x0_states = x_clean.long()  # [B,1,H,W] с {0,1}
        log_probs = self._subs_parameterization(
            logits=logits, xt_states=xt_states, x0_states=x0_states
        )  # [B,3,H,W]

        # 6) NLL по маскированным пикселям.
        #    x0 — target классы (0 или 1; класс MASK=2 никогда не используется).
        target = x0_states.squeeze(1).view(-1)  # [B*H*W]
        log_probs_flat = (
            log_probs.view(B, self.num_classes, H * W)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, self.num_classes)
        )  # [B*H*W, 3]

        # NLL per-pixel (включая все позиции)
        nll_flat = F.nll_loss(
            log_probs_flat,
            target,
            reduction="none",
        )  # [B*H*W]

        nll = nll_flat.view(B, 1, H, W)  # [B,1,H,W]

        # Маска для усреднения: только по маскированным позициям
        mask = mask_positions.float()  # [B,1,H,W]
        masked_count = mask.sum()

        if masked_count < 1:
            # На всякий случай: если ни одного маскированного пикселя не получилось
            # (маловероятно, но теоретически возможно), усредняем по всем.
            loss = nll.mean()
        else:
            loss = (nll * mask).sum() / masked_count

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
        3) Берём argmax по {0,1} в каждой позиции → бинарное изображение.

        Это НЕ полная реализация обратного диффузионного процесса,
        а просто sanity-check, что модель вообще что-то осмысленное
        выучила при обучении (как грубый "denoising" из полного шума).
        """
        if device is None:
            device = next(self.unet.parameters()).device

        B = num_samples
        H = W = 28  # для MNIST

        # 1) Всё MASK
        xt_states = torch.full(
            (B, 1, H, W),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # 2) t = T для всех
        t = torch.full(
            (B,),
            fill_value=self.schedule.num_timesteps,
            dtype=torch.long,
            device=device,
        )

        xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]
        logits = self.unet(xt_one_hot, t)            # [B,3,H,W]

        # Для семплинга x0 нам SUBS нужна только чтобы убрать MASK-класс.
        log_probs = self._subs_parameterization(
            logits=logits,
            xt_states=xt_states,
            x0_states=None,  # нет истинного x0 при генерации
        )  # [B,3,H,W]

        # Берём argmax по классам (0,1; MASK-класс ≈ -inf).
        probs = log_probs.exp()  # [B,3,H,W]
        # x_samples = probs[:, :2, :, :].argmax(dim=1, keepdim=True)  # [B,1,H,W] в {0,1}
        # Вместо argmax семплируем из категориального распределения по 0/1
        # чтобы получить хоть какое-то разнообразие, если вход одинаковый.
        # Но для генерации "one-shot" argmax обычно ок, если модель уверена.
        # Если модель не уверена, argmax выдаст самый вероятный (фон).
        
        # Попробуем семплировать
        probs_01 = probs[:, :2, :, :]
        # Нормализуем заново (так как mask отбросили)
        probs_01 = probs_01 / (probs_01.sum(dim=1, keepdim=True) + 1e-8)
        
        # Семплируем
        # [B, 2, H, W] -> [B, H, W, 2] -> flatten -> multinomial
        B_sz, _, H_sz, W_sz = probs_01.shape
        probs_flat = probs_01.permute(0, 2, 3, 1).reshape(-1, 2)
        samples_flat = torch.multinomial(probs_flat, num_samples=1)
        x_samples = samples_flat.view(B_sz, 1, H_sz, W_sz).float()
        
        return x_samples