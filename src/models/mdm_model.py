from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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

    training_mode:
        Какой обучающий лосс использовать:
          - "baseline"        : обычный MDM по x_clean;
          - "ambient_oracle"  : ambient-режим с доступом к x_clean + obs_mask;
          - "ambient_strict"  : будущий "чистый" ambient (пока NotImplemented).

    consistency_weight:
        Вес консистентностного регуляризатора (0.0 = выключено).
    consistency_pair_offset:
        Насколько отличаются соседние времена в паре (t_hi, t_lo),
        по умолчанию t_lo = t_hi - 1.
    """
    neg_infinity: float = -1e4
    training_mode: str = "baseline"  # 'baseline' | 'ambient_oracle' | 'ambient_strict'
    consistency_weight: float = 0.0
    consistency_pair_offset: int = 1


class MaskedDiffusionModel(nn.Module):
    """
    Masked diffusion модель для бинарного MNIST.

    Состав:
      - UNet-денойзер (MDMUNet),
      - MaskSchedule (forward-процесс маскирования),
      - SUBS-параметризация (в духе MDLM для дискретных токенов).

    Основные режимы обучения:
      - baseline MDM (полный доступ к x_clean),
      - ambient_oracle (x_clean + obs_mask, лосс только по наблюдаемым пикселям),
      - ambient_strict (строгий ambient-объектив; пока не реализован).

    Опционально:
      - consistency_loss: регуляризатор в стиле Consistency Models /
        Consistent Diffusion, который заставляет предсказания x0 на
        соседних шагах шума быть согласованными.
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

        if self.cfg.training_mode not in {"baseline", "ambient_oracle", "ambient_strict"}:
            raise ValueError(
                "MaskedDiffusionModel: cfg.training_mode must be one of "
                "{'baseline', 'ambient_oracle', 'ambient_strict'}, "
                f"got '{self.cfg.training_mode}'"
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
                "MaskedDiffusionModel: expected tensor to contain only {0,1} values."
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

        neg_inf = self.cfg.neg_infinity

        # 1) Запрещаем класс MASK → p(mask) ≈ 0.
        logits = logits.clone()
        logits[:, self.mask_token_id, :, :] += neg_inf

        # 2) Нормализуем logits → log_probs (log p) по классам
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
        unmasked = (xt_states != self.mask_token_id)  # [B,1,H,W]
        if unmasked.any():
            target = x0_states.squeeze(1).long()  # [B,H,W]

            log_probs_unmasked = torch.full_like(
                log_probs, fill_value=neg_inf
            )  # [B,C,H,W]

            # [B,H,W,C] → [B*H*W,C]
            log_probs_unmasked_flat = (
                log_probs_unmasked.permute(0, 2, 3, 1)
                .contiguous()
                .view(-1, C)
            )
            target_flat = target.view(-1)  # [B*H*W]
            row_idx = torch.arange(
                log_probs_unmasked_flat.shape[0], device=logits.device
            )

            log_probs_unmasked_flat[row_idx, target_flat] = 0.0

            log_probs_unmasked = (
                log_probs_unmasked_flat.view(B, H, W, C)
                .permute(0, 3, 1, 2)
            )  # [B,C,H,W]

            unmasked_broadcast = unmasked.expand_as(log_probs)
            # WORKAROUND for MPS: вместо torch.where используем прямую индексацию
            log_probs[unmasked_broadcast] = log_probs_unmasked[unmasked_broadcast]

        return log_probs

    def infer_from_xt(self, xt_states: Tensor, logits: Tensor) -> Tensor:
        """
        SUBS-like inference for p(x0 | x_t) used for visualizations.

        Behavior:
          - forbid MASK class,
          - on unmasked tokens copy xt,
          - on masked tokens take argmax over {0,1}.

        Parameters
        ----------
        xt_states:
            LongTensor [B,1,H,W] in {0,1,MASK}
        logits:
            FloatTensor [B,C,H,W] over classes {0,1,MASK}

        Returns
        -------
        x_recon:
            FloatTensor [B,1,H,W] in {0,1}
        """
        if xt_states.dim() != 4 or logits.dim() != 4:
            raise ValueError(
                "infer_from_xt: expected xt_states [B,1,H,W] and logits [B,C,H,W], "
                f"got {tuple(xt_states.shape)}, {tuple(logits.shape)}"
            )

        B, C, H, W = logits.shape
        if C != self.num_classes:
            raise ValueError(
                f"infer_from_xt: expected logits C={self.num_classes}, got C={C}"
            )
        if xt_states.shape[0] != B or xt_states.shape[1] != 1 or xt_states.shape[2] != H or xt_states.shape[3] != W:
            raise ValueError(
                "infer_from_xt: xt_states must have shape [B,1,H,W] matching logits spatial dims, "
                f"got {tuple(xt_states.shape)} vs logits {tuple(logits.shape)}"
            )

        mask_id = self.mask_token_id
        neg_inf = self.cfg.neg_infinity

        # 1) Forbid MASK class
        logits = logits.clone()
        logits[:, mask_id, :, :] += neg_inf

        # 2) log-softmax
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # [B,C,H,W]

        # 3) Force degenerate distribution on unmasked positions copying xt
        unmasked = (xt_states != mask_id)  # [B,1,H,W]
        if unmasked.any():
            target = xt_states.squeeze(1).long()  # [B,H,W]

            forced = torch.full_like(log_probs, fill_value=neg_inf)  # [B,C,H,W]
            forced_flat = forced.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W,C]
            target_flat = target.view(-1)  # [B*H*W]
            row_idx = torch.arange(forced_flat.shape[0], device=logits.device)
            forced_flat[row_idx, target_flat] = 0.0
            forced = forced_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]

            unmasked_broadcast = unmasked.expand_as(log_probs)
            log_probs[unmasked_broadcast] = forced[unmasked_broadcast]

        # 4) Argmax over {0,1} (MASK is ~ -inf)
        probs = log_probs.exp()  # [B,C,H,W]
        x_recon = probs[:, :2, :, :].argmax(dim=1, keepdim=True)  # [B,1,H,W]
        return x_recon.float()

    # === consistency: непрерывное предсказание x0 из (x_t, t) ===
    def _predict_x0_prob(self, xt_states: Tensor, t: Tensor) -> Tensor:
        """
        f_θ(x_t, t): предсказанная вероятность класса '1' на каждом пикселе.

        xt_states : [B,1,H,W] в {0,1,MASK}
        t         : [B] (целые времена)
        Возвращает:
            p1: [B,1,H,W] в [0,1]
        """
        if xt_states.dim() != 4:
            raise ValueError(
                f"_predict_x0_prob: expected xt_states [B,1,H,W], got {tuple(xt_states.shape)}"
            )
        B, C, H, W = xt_states.shape
        if C != 1:
            raise ValueError(
                f"_predict_x0_prob: expected C=1, got C={C}"
            )
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(
                f"_predict_x0_prob: expected t [B], got {tuple(t.shape)} for B={B}"
            )

        device = xt_states.device
        t = t.to(device)

        xt_one_hot = self._encode_states(xt_states)           # [B,3,H,W]
        logits = self.unet(xt_one_hot, t)                     # [B,3,H,W]
        log_probs = self._subs_parameterization(
            logits=logits,
            xt_states=xt_states,
            x0_states=None,   # здесь нам нужен только "soft" постерирор по {0,1}
        )                                                        # [B,3,H,W]

        probs = log_probs.exp()                                # [B,3,H,W]
        probs_01 = probs[:, :2, :, :]
        probs_01 = probs_01 / (probs_01.sum(dim=1, keepdim=True) + 1e-8)
        p1 = probs_01[:, 1:2, :, :]                            # [B,1,H,W]
        return p1

    # ------------------------------------------------------------------
    # Лоссы по режимам
    # ------------------------------------------------------------------
    def loss_baseline(self, x_clean: Tensor) -> Tensor:
        """
        Baseline MDM-лосс: полное знание x_clean, без obs_mask.

        x_clean : [B,1,H,W], бинарные {0,1}.
        """
        if x_clean.dim() != 4:
            raise ValueError(
                f"loss_baseline: expected x_clean [B,1,H,W], got {tuple(x_clean.shape)}"
            )

        B, C, H, W = x_clean.shape
        if C != 1:
            raise ValueError(
                f"loss_baseline: expected x_clean with C=1, got C={C}"
            )

        # Перенос на устройство модели
        device = next(self.unet.parameters()).device
        x_clean = x_clean.to(device)

        self._check_binary_x(x_clean)

        # 1) t ~ Uniform({1..T})
        t = self.schedule.sample_timesteps(batch_size=B, device=device)  # [B]

        # 2) Forward-процесс: q(z_t | x0)
        xt_states, mask_positions = self.schedule.forward_mask(x_clean, t)
        # xt_states: [B,1,H,W] в {0,1,MASK}
        # mask_positions: [B,1,H,W] bool

        # 3) Кодируем xt в one-hot по каналам
        xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]

        # 4) Прогоняем UNet
        logits = self.unet(xt_one_hot, t)  # [B,3,H,W]

        # 5) SUBS-параметризация → log_probs (log p(y | xt, t))
        x0_states = x_clean.long()  # [B,1,H,W] с {0,1}
        log_probs = self._subs_parameterization(
            logits=logits, xt_states=xt_states, x0_states=x0_states
        )  # [B,3,H,W]

        # 6) NLL по маскированным пикселям (mask_positions)
        target = x0_states.squeeze(1).view(-1)  # [B*H*W]
        log_probs_flat = (
            log_probs.view(B, self.num_classes, H * W)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, self.num_classes)
        )  # [B*H*W, 3]

        nll_flat = F.nll_loss(
            log_probs_flat,
            target,
            reduction="none",
        )  # [B*H*W]

        nll = nll_flat.view(B, 1, H, W)  # [B,1,H,W]

        mask = mask_positions.float()  # [B,1,H,W]
        masked_count = mask.sum()

        if masked_count < 1:
            loss = nll.mean()
        else:
            loss = (nll * mask).sum() / masked_count

        return loss

    def loss_ambient_oracle(
        self,
        x_clean: Tensor,
        obs_mask: Tensor,
    ) -> Tensor:
        """
        Ambient-режим с доступом к x_clean (oracle).

        x_clean : [B,1,H,W], бинарные {0,1}.
        obs_mask: [B,1,H,W], 1 — наблюдаемый пиксель, 0 — пропуск.

        Идея:
          - forward-процесс добавляет СВОЮ маску (дополнительная порча),
          - obs_mask задаёт, какие пиксели вообще были доступны в данных,
          - лосс усредняем только по пересечению:
              * пиксели, которые forward замаскировал,
              * и при этом obs_mask == 1.
        """
        if x_clean.dim() != 4:
            raise ValueError(
                f"loss_ambient_oracle: expected x_clean [B,1,H,W], got {tuple(x_clean.shape)}"
            )

        B, C, H, W = x_clean.shape
        if C != 1:
            raise ValueError(
                f"loss_ambient_oracle: expected x_clean with C=1, got C={C}"
            )

        if obs_mask.shape != x_clean.shape:
            raise ValueError(
                "loss_ambient_oracle: obs_mask shape must match x_clean shape"
            )

        device = next(self.unet.parameters()).device
        x_clean = x_clean.to(device)
        obs_mask = obs_mask.to(device).float()

        self._check_binary_x(x_clean)

        # 1) t ~ Uniform({1..T})
        t = self.schedule.sample_timesteps(batch_size=B, device=device)  # [B]

        # 2) Forward-процесс: q(z_t | x0)
        xt_states, mask_positions = self.schedule.forward_mask(x_clean, t)
        # xt_states: [B,1,H,W] в {0,1,MASK}
        # mask_positions: [B,1,H,W] bool (где forward маскировал дополнительно)

        # A(x): учитываем исходную маску наблюдений.
        # Если obs_mask == 0, то пиксель отсутствует изначально → ставим MASK.
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

        # Маска для усреднения:
        #   - mask_positions: где forward маскировал,
        #   - obs_mask: пиксели, доступные в данных,
        # берём пересечение.
        mask_positions_float = mask_positions.float()
        train_mask = mask_positions_float * obs_mask  # [B,1,H,W]

        masked_count = train_mask.sum()

        if masked_count < 1:
            # fallback: усредняем по наблюдаемым пикселям (редкий случай)
            loss = (nll * obs_mask).sum() / (obs_mask.sum() + 1e-8)
        else:
            loss = (nll * train_mask).sum() / masked_count

        return loss

    def loss_ambient_strict(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
    ) -> Tensor:
        """
        Строгий ambient-объектив (без доступа к x_clean).

        x_obs:    [B,1,H,W] — наблюдаемые данные:
                    - на obs_mask == 1: значения ≈ {0,1} (наблюдаемые пиксели),
                    - на obs_mask == 0: произвольный sentinel (например, -1).
        obs_mask: [B,1,H,W] — маска наблюдений {0,1}.

        Идея:
          - используем ТОЛЬКО наблюдаемые пиксели как таргет,
          - forward-процесс (forward_mask) запускаем на псевдо-полном x_tilde,
            где:
              * на наблюдаемых пикселях x_tilde = x_obs (бинаризованное),
              * на пропусках x_tilde берём из простого prior'а Bernoulli(0.5),
          - затем:
              * добавляем свою маску (forward_mask),
              * форсим MASK там, где obs_mask == 0 (ambient-оператор A),
              * считаем NLL по пересечению:
                    {пиксели, замаскированные forward-процессом}
                ∩   {obs_mask == 1}.

        Таким образом, лосс не использует x_clean даже косвенно.
        """
        if x_obs.dim() != 4 or obs_mask.dim() != 4:
            raise ValueError(
                f"loss_ambient_strict: expected x_obs/obs_mask [B,1,H,W], "
                f"got {tuple(x_obs.shape)}, {tuple(obs_mask.shape)}"
            )

        B, C, H, W = x_obs.shape
        if C != 1:
            raise ValueError(
                f"loss_ambient_strict: expected x_obs with C=1, got C={C}"
            )
        if obs_mask.shape != x_obs.shape:
            raise ValueError(
                "loss_ambient_strict: obs_mask shape must match x_obs shape"
            )

        device = next(self.unet.parameters()).device
        x_obs = x_obs.to(device)
        obs_mask = obs_mask.to(device).float()

        # 0) Бинаризуем наблюдаемые значения (на obs_mask == 1).
        #    На пропусках оставляем 0 — они всё равно не будут участвовать в лоссе.
        x_obs_bin = torch.zeros_like(x_obs)
        obs_is_one = (x_obs > 0.5) & (obs_mask == 1.0)
        x_obs_bin[obs_is_one] = 1.0  # [B,1,H,W] в {0,1} на наблюдаемых пикселях

        # 1) t ~ Uniform({1..T})
        T = self.schedule.num_timesteps
        t = self.schedule.sample_timesteps(batch_size=B, device=device)  # [B]

        # 2) Строим псевдо-полный x_tilde:
        #    - на наблюдаемых пикселях: x_obs_bin,
        #    - на пропусках: семпл из простого prior'а Bernoulli(0.5).
        prior_prob = 0.5
        rand_fill = torch.bernoulli(
            prior_prob * torch.ones_like(x_obs_bin)
        )  # [B,1,H,W] в {0,1}

        x_tilde = torch.where(obs_mask == 1.0, x_obs_bin, rand_fill)  # [B,1,H,W]

        # 3) Forward-процесс: q(x_t | x_tilde)
        xt_states, mask_positions = self.schedule.forward_mask(x_tilde, t)
        # xt_states:     [B,1,H,W] в {0,1,MASK}
        # mask_positions:[B,1,H,W] bool

        # 4) Ambient-оператор A: форсим MASK там, где пиксель изначально отсутствовал.
        missing = (obs_mask == 0)
        if missing.any():
            xt_states = xt_states.clone()
            xt_states[missing] = self.mask_token_id

        # 5) Кодируем xt в one-hot
        xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]

        # 6) Прогоняем UNet
        logits = self.unet(xt_one_hot, t)  # [B,3,H,W]

        # 7) SUBS-параметризация: в качестве x0_states используем x_obs_bin.
        #    На пропусках x_obs_bin=0, но:
        #      - эти позиции не попадут в train_mask (obs_mask==0),
        #      - SUBS на незамаскированных пикселях просто делает вырожденный
        #        дистрибутив, который в лосс не входит (мы считаем NLL только
        #        по маскам).
        x0_states = x_obs_bin.long()  # [B,1,H,W] в {0,1}
        log_probs = self._subs_parameterization(
            logits=logits,
            xt_states=xt_states,
            x0_states=x0_states,
        )  # [B,3,H,W]

        # 8) NLL по пикселям
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

        # 9) Маска для усреднения:
        #     - mask_positions: где forward-процесс замаскировал,
        #     - obs_mask:       какие пиксели вообще были наблюдаемы.
        mask_positions_float = mask_positions.float()
        train_mask = mask_positions_float * obs_mask  # [B,1,H,W]

        masked_count = train_mask.sum()

        if masked_count < 1:
            # Редкий случай: forward ничего не замаскировал на наблюдаемых пикселях.
            # Тогда усредняем по всем наблюдаемым.
            loss = (nll * obs_mask).sum() / (obs_mask.sum() + 1e-8)
        else:
            loss = (nll * train_mask).sum() / masked_count

        return loss


    # === consistency loss =====================================================
    def consistency_loss(
        self,
        x: Tensor,
        obs_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Consistency-терм в духе Consistency Models / Consistent Diffusion.

        Идея (адаптация под маскирующий диффузионный процесс):

          1) Берём "чистый" x (это может быть как истинный x_clean,
             так и псевдо-полный x_hat из EM).
          2) Семплим пару времён t_hi > t_lo (соседние шаги по маске).
          3) Строим x_{t_hi}, x_{t_lo} через forward_mask.
             При наличии obs_mask принудительно ставим MASK там, где
             пиксель изначально отсутствовал.
          4) Считаем f_θ(x_{t_hi}, t_hi) и f_θ(x_{t_lo}, t_lo) —
             предсказанные вероятности класса '1'.
          5) Минимизируем L2-разницу между ними (при obs_mask усредняем
             только по наблюдаемым пикселям).

        На оптимуме модель делает согласованные предсказания x0 на
        соседних уровнях шума, что стабилизирует обучение (см. работы
        про consistency models / score guidance).
        """
        if x.dim() != 4:
            raise ValueError(
                f"consistency_loss: expected x [B,1,H,W], got {tuple(x.shape)}"
            )
        B, C, H, W = x.shape
        if C != 1:
            raise ValueError(
                f"consistency_loss: expected x with C=1, got C={C}"
            )

        device = next(self.unet.parameters()).device
        x = x.to(device)
        self._check_binary_x(x)

        T = self.schedule.num_timesteps
        if T < 2:
            # Нечего делать, если всего один шаг.
            return x.new_tensor(0.0)

        # t_hi ∈ {2,...,T}, t_lo = t_hi - offset (зажимаем в [1,T])
        t_hi = torch.randint(
            low=2,
            high=T + 1,
            size=(B,),
            device=device,
            dtype=torch.long,
        )
        offset = max(int(getattr(self.cfg, "consistency_pair_offset", 1)), 1)
        t_lo = torch.clamp(t_hi - offset, min=1, max=T)

        xt_hi, _ = self.schedule.forward_mask(x, t_hi)  # [B,1,H,W]
        xt_lo, _ = self.schedule.forward_mask(x, t_lo)  # [B,1,H,W]

        weight_mask: Optional[Tensor] = None
        if obs_mask is not None:
            if obs_mask.shape != x.shape:
                raise ValueError(
                    "consistency_loss: obs_mask shape must match x shape"
                )
            obs_mask = obs_mask.to(device).float()
            # Для ambient-режимов дополнительно форсим MASK на пропусках
            missing = (obs_mask == 0)
            if missing.any():
                xt_hi = xt_hi.clone()
                xt_lo = xt_lo.clone()
                xt_hi[missing] = self.mask_token_id
                xt_lo[missing] = self.mask_token_id
            weight_mask = obs_mask  # усредняем только по наблюдаемым

        p_hi = self._predict_x0_prob(xt_hi, t_hi)  # [B,1,H,W]
        p_lo = self._predict_x0_prob(xt_lo, t_lo)  # [B,1,H,W]

        diff_sq = (p_hi - p_lo) ** 2  # [B,1,H,W]

        if weight_mask is not None:
            denom = weight_mask.sum()
            if denom > 0:
                loss = (diff_sq * weight_mask).sum() / denom
            else:
                loss = diff_sq.mean()
        else:
            loss = diff_sq.mean()

        return loss

    # ------------------------------------------------------------------
    # Унифицированный интерфейс лосса
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        x: Tensor,
        obs_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Маршрутизатор по лоссам в зависимости от cfg.training_mode.

        training_mode:
          - "baseline":
              x       := x_clean
              obs_mask игнорируется
          - "ambient_oracle":
              x       := x_clean
              obs_mask: маска наблюдений (обязательно не None)
          - "ambient_strict":
              x       := x_obs
              obs_mask: маска наблюдений (обязательно не None)

        Дополнительно:
          - при cfg.consistency_weight > 0 добавляется consistency_loss
            (использующий тот же x и obs_mask).
        """
        mode = getattr(self.cfg, "training_mode", "baseline")
        lambda_c = float(getattr(self.cfg, "consistency_weight", 0.0))

        if mode == "baseline":
            main_loss = self.loss_baseline(x)
            if lambda_c > 0.0:
                main_loss = main_loss + lambda_c * self.consistency_loss(x)
            return main_loss

        elif mode == "ambient_oracle":
            if obs_mask is None:
                raise ValueError(
                    "compute_loss (ambient_oracle): obs_mask must be provided."
                )
            main_loss = self.loss_ambient_oracle(x, obs_mask)
            if lambda_c > 0.0:
                main_loss = main_loss + lambda_c * self.consistency_loss(x, obs_mask)
            return main_loss

        elif mode == "ambient_strict":
            if obs_mask is None:
                raise ValueError(
                    "compute_loss (ambient_strict): obs_mask must be provided."
                )
            # В строгом ambient-режиме не используем oracle x_clean
            # и не полагаемся на то, что x бинарен на пропусках.
            # Поэтому здесь только свой строгий лосс, без consistency.
            main_loss = self.loss_ambient_strict(x, obs_mask)
            if lambda_c > 0.0:
                main_loss = main_loss + lambda_c * self.consistency_loss(x, obs_mask)
            return main_loss


        else:
            raise ValueError(f"Unknown training_mode: {mode}")

    # ------------------------------------------------------------------
    # Unconditional sampling: iterative generation from all-MASK
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
        num_steps: int = 50,
    ) -> Tensor:
        """
        Approximate unconditional sampling: x ~ p_theta(x0).

        Implemented as a special case of `conditional_sample` where nothing is observed
        (obs_mask == 0 everywhere). This keeps a single diffusion loop implementation.

        This method is used for logging `*_samples.png` and does not affect training.
        """
        if device is None:
            device = next(self.unet.parameters()).device

        H = W = 28  # MNIST
        x_obs = torch.zeros((num_samples, 1, H, W), device=device)
        obs_mask = torch.zeros_like(x_obs)  # nothing observed anywhere

        return self.conditional_sample(
            x_obs=x_obs,
            obs_mask=obs_mask,
            num_steps=num_steps,
            device=device,
        )

    # ------------------------------------------------------------------
    # Conditional sampling: x_hat ~ p(x0 | x_obs, obs_mask)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def conditional_sample(
        self,
        x_obs: Tensor,
        obs_mask: Tensor,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Приближённый conditional sampling:
          x_hat ~ p_theta(x0 | x_obs, obs_mask)

        x_obs:    [B,1,H,W], значения 0/1 на наблюдаемых пикселях,
                  и произвольный sentinel (напр. -1) на пропусках.
        obs_mask: [B,1,H,W], 1 — наблюдаемый пиксель, 0 — пропуск.

        Алгоритм:
          1) Инициализируем xt:
             - на наблюдаемых пикселях = x_obs (0/1),
             - на пропусках = MASK.
          2) num_steps раз:
             - выбираем t ~ линейная сетка между T и 1,
             - clamp: xt[obs_mask == 1] = x_obs,
             - обновляем свободные пиксели (obs_mask == 0) по распределению модели,
               с re-masking в духе forward-процесса.
          3) Возвращаем xt как x_hat (0/1).
        """
        if device is None:
            device = x_obs.device

        if x_obs.dim() != 4 or obs_mask.dim() != 4:
            raise ValueError(
                f"conditional_sample: expected x_obs/obs_mask [B,1,H,W], "
                f"got {tuple(x_obs.shape)}, {tuple(obs_mask.shape)}"
            )

        if x_obs.shape != obs_mask.shape:
            raise ValueError(
                "conditional_sample: x_obs and obs_mask must have the same shape"
            )

        B, C, H, W = x_obs.shape
        if C != 1:
            raise ValueError(
                f"conditional_sample: expected C=1, got C={C}"
            )

        x_obs = x_obs.to(device)
        obs_mask = obs_mask.to(device).float()

        # Бинаризуем наблюдаемые значения.
        x_obs_bin = torch.zeros_like(x_obs)
        obs_is_one = (x_obs > 0.5) & (obs_mask == 1.0)
        x_obs_bin[obs_is_one] = 1.0

        # Инициализация xt:
        #   - везде MASK,
        #   - на наблюдаемых пикселях = x_obs_bin.
        xt_states = torch.full(
            (B, 1, H, W),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        observed_bool = (obs_mask == 1.0)
        if observed_bool.any():
            xt_states[observed_bool] = x_obs_bin[observed_bool].long()

        T = self.schedule.num_timesteps
        if num_steps < 1:
            num_steps = 1

        for step in range(num_steps, 0, -1):
            # Линейная сетка по t от T до 1
            t_val = int(round(1 + (T - 1) * (step / num_steps)))
            t_val = max(1, min(T, t_val))

            t = torch.full(
                (B,),
                fill_value=t_val,
                dtype=torch.long,
                device=device,
            )

            # Жёстко фиксируем наблюдаемые пиксели
            if observed_bool.any():
                xt_states[observed_bool] = x_obs_bin[observed_bool].long()

            # Прогоняем через UNet
            xt_one_hot = self._encode_states(xt_states)  # [B,3,H,W]
            logits = self.unet(xt_one_hot, t)            # [B,3,H,W]

            # log_probs без knowledge of x0: только отключаем MASK-класс
            log_probs = self._subs_parameterization(
                logits=logits,
                xt_states=xt_states,
                x0_states=None,
            )  # [B,3,H,W]

            probs = log_probs.exp()  # [B,3,H,W]
            probs_01 = probs[:, :2, :, :]
            probs_01 = probs_01 / (probs_01.sum(dim=1, keepdim=True) + 1e-8)

            B_sz, _, H_sz, W_sz = probs_01.shape
            probs_flat = probs_01.permute(0, 2, 3, 1).reshape(-1, 2)
            samples_flat = torch.multinomial(probs_flat, num_samples=1)
            proposed = samples_flat.view(B_sz, 1, H_sz, W_sz).long()  # [B,1,H,W] в {0,1}

            # Обновляем только свободные пиксели (obs_mask==0)
            free_bool = (obs_mask == 0.0)
            if free_bool.any():
                # proposed - это x0_hat (полный).
                # Нам нужно семплировать x_{t-1} из q(x_{t-1} | x0_hat).

                # Следующее "время"
                t_next_val = int(round(1 + (T - 1) * ((step - 1) / num_steps)))
                if t_next_val < 1:
                    t_next_val = 0  # t=0 → alpha=1.0 (всё сохраняем)

                if t_next_val == 0:
                    xt_states[free_bool] = proposed[free_bool]
                else:
                    t_next_tensor = torch.full(
                        (B,),
                        t_next_val,
                        dtype=torch.long,
                        device=device,
                    )
                    alpha_next = self.schedule._alpha_at(t_next_tensor)  # [B,1,1,1]

                    keep_prob = alpha_next.expand(B, 1, H, W)
                    keep_mask = torch.bernoulli(keep_prob).bool()  # [B,1,H,W]

                    new_val = torch.where(
                        keep_mask,
                        proposed,
                        torch.full_like(proposed, self.mask_token_id),
                    )

                    xt_states[free_bool] = new_val[free_bool]

        # В конце ещё раз клэмпим наблюдаемые (на всякий случай)
        if observed_bool.any():
            xt_states[observed_bool] = x_obs_bin[observed_bool].long()

        # Должны быть только 0/1. Если где-то остался MASK (не должно),
        # заполним нулями.
        mask_bool = (xt_states == self.mask_token_id)
        if mask_bool.any():
            xt_states[mask_bool] = 0

        x_hat = xt_states.float()  # [B,1,H,W] в {0,1}
        return x_hat
