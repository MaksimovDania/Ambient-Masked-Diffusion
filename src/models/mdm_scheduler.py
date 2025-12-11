from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import nn


MaskScheduleType = Literal["linear", "cosine"]


@dataclass
class MaskScheduleConfig:
    """
    Конфиг для расписания маскирования (forward-процесс masked diffusion).

    num_timesteps:
        Кол-во дискретных шагов диффузии T.
    alpha_min:
        Минимальная вероятность "выживания" токена к последнему шагу.
        alpha_t убывает от 1.0 (t=1) до alpha_min (t=T).
    schedule_type:
        Форма кривой alpha_t по t: "linear" или "cosine".
    """
    num_timesteps: int = 100
    alpha_min: float = 0.01
    schedule_type: MaskScheduleType = "linear"


class MaskSchedule(nn.Module):
    """
    Дискретное расписание маскирования для masked diffusion.

    Идея (в духе MDLM):
    - на шаге t каждый токен "выживает" (остаётся не MASK) с вероятностью alpha_t;
    - чем больше t, тем меньше alpha_t → тем больше замаскированных токенов. 

    Здесь мы реализуем простой инженерный вариант:
    - alpha_t убывает монотонно от 1.0 до alpha_min по t = 1..T;
    - поддерживаем два варианта: линейный и косинусный.

    Для бинарного MNIST:
    - x0 содержит значения {0,1};
    - forward_mask возвращает z_t_states ∈ {0,1,2},
      где 2 = специальный индекс для MASK-состояния.
    """

    MASK_TOKEN_ID: int = 2  # состояние MASK

    def __init__(self, cfg: MaskScheduleConfig) -> None:
        super().__init__()

        if cfg.num_timesteps <= 0:
            raise ValueError(
                f"MaskScheduleConfig: num_timesteps must be > 0, got {cfg.num_timesteps}"
            )
        if not (0.0 < cfg.alpha_min <= 1.0):
            raise ValueError(
                f"MaskScheduleConfig: alpha_min must be in (0,1], got {cfg.alpha_min}"
            )
        if cfg.schedule_type not in ("linear", "cosine"):
            raise ValueError(
                f"MaskScheduleConfig: schedule_type must be 'linear' or 'cosine', "
                f"got {cfg.schedule_type}"
            )

        self.cfg = cfg
        # Регистрируем как буфер, чтобы это автоматически тащилось с .to(device)
        alpha = self._build_alpha_schedule(
            num_timesteps=cfg.num_timesteps,
            alpha_min=cfg.alpha_min,
            kind=cfg.schedule_type,
        )
        self.register_buffer("alpha", alpha)  # shape [T]

    @staticmethod
    def _build_alpha_schedule(
        num_timesteps: int,
        alpha_min: float,
        kind: MaskScheduleType,
    ) -> torch.Tensor:
        """
        Строим последовательность alpha_t для t=1..T.

        - "linear": alpha_t линейно убывает от 1.0 до alpha_min.
        - "cosine": alpha_t следует косинусной кривой, начинающейся в 1.0
                    и заканчивающейся в alpha_min (аналогично cosine beta-расписаниям
                    в DDPM, но для survival prob). 
        """
        T = num_timesteps
        # t_norm в [0,1] для t=1..T
        t = torch.linspace(0.0, 1.0, steps=T)

        if kind == "linear":
            # alpha_t = 1 - (1 - alpha_min) * t
            alpha_t = 1.0 - (1.0 - alpha_min) * t
        elif kind == "cosine":
            # Косинусное расписание: сначала медленно, потом быстрее
            # cos(0) = 1, cos(pi/2) = 0 → нормируем к [alpha_min, 1]
            # base = cos(pi/2 * t) ∈ [1, 0]
            base = torch.cos(0.5 * torch.pi * t)
            # Нормируем, чтобы base[0]=1, base[-1]=0 (она и так такова)
            # Затем растягиваем в [alpha_min, 1]
            alpha_t = alpha_min + (1.0 - alpha_min) * base
        else:
            raise ValueError(f"Unknown schedule type: {kind}")

        # Sanity: alpha_t должна быть монотонно не возрастать
        if not torch.all(alpha_t[:-1] >= alpha_t[1:] - 1e-6):
            raise RuntimeError(
                "MaskSchedule: constructed alpha_t is not monotonically decreasing."
            )

        return alpha_t

    @property
    def num_timesteps(self) -> int:
        return self.cfg.num_timesteps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Семплирует дискретные шаги t ~ Uniform({1, ..., T}) для каждого элемента батча.

        Возвращает:
            t: LongTensor формы [B] со значениями в [1, T].
        """
        if batch_size <= 0:
            raise ValueError(
                f"MaskSchedule.sample_timesteps: batch_size must be > 0, got {batch_size}"
            )
        return torch.randint(
            low=1,
            high=self.num_timesteps + 1,  # верхняя граница исключается
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def _alpha_at(self, t: torch.Tensor) -> torch.Tensor:
        """
        Получить alpha_t для каждого элемента батча.

        t: LongTensor [B] со значениями в [1, T].

        Возвращает:
            alpha_t: Tensor [B, 1, 1, 1], готовый к broadcast'у на [B, 1, H, W].
        """
        if t.dim() != 1:
            raise ValueError(
                f"MaskSchedule._alpha_at: expected t with shape [B], got {tuple(t.shape)}"
            )
        if t.min() < 1 or t.max() > self.num_timesteps:
            raise ValueError(
                f"MaskSchedule._alpha_at: t must be in [1, {self.num_timesteps}], "
                f"got range [{int(t.min())}, {int(t.max())}]"
            )

        # self.alpha: [T], индекс 0 соответствует t=1
        alpha_t = self.alpha[t - 1]  # [B]
        return alpha_t.view(-1, 1, 1, 1)

    def forward_mask(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Реализует forward-процесс masked diffusion q(z_t | x0).

        Параметры
        ---------
        x0 : Tensor
            Бинарные изображения [B, 1, H, W] (значения 0 или 1).
        t : LongTensor
            Шаги диффузии [B] со значениями в [1, T].

        Возвращает
        ---------
        z_t_states : LongTensor
            Состояния на шаге t, форма [B, 1, H, W], значения ∈ {0, 1, MASK_TOKEN_ID}.
        mask_positions : BoolTensor
            Маска тех позиций, которые стали MASK на шаге t, форма [B, 1, H, W].

        Замечания
        ---------
        - Мы предполагаем, что forward-процесс "одношаговый": на данном t
          каждый пиксель независимо:
            - остаётся как в x0 с вероятностью alpha_t,
            - становится MASK с вероятностью 1 - alpha_t.
          Это соответствует absorbing masked diffusion из MDLM, если смотреть
          только на маргинальное распределение q(z_t | x0). 
        """
        if x0.dim() != 4:
            raise ValueError(
                f"MaskSchedule.forward_mask: expected x0 with shape [B,1,H,W], got {tuple(x0.shape)}"
            )
        B, C, H, W = x0.shape
        if C != 1:
            raise ValueError(
                f"MaskSchedule.forward_mask: expected x0 with C=1 (single channel), got C={C}"
            )
        if t.shape[0] != B:
            raise ValueError(
                "MaskSchedule.forward_mask: batch size mismatch between x0 and t: "
                f"x0.shape[0]={B}, t.shape[0]={t.shape[0]}"
            )

        # Проверяем, что x0 действительно бинарный (приблизительно)
        if not torch.all((x0 == 0.0) | (x0 == 1.0)):
            # Не падаем жёстко, но предупреждаем через assert в отладке
            raise ValueError(
                "MaskSchedule.forward_mask: expected x0 to contain only {0,1} values."
            )

        t = t.to(dtype=torch.long, device=x0.device)
        alpha_t = self._alpha_at(t)  # [B,1,1,1]

        # Семплируем, какие пиксели "выживают"
        # keep ~ Bernoulli(alpha_t)
        #   True  -> пиксель остаётся как в x0
        #   False -> пиксель становится MASK
        keep_prob = alpha_t.expand(B, 1, H, W)
        keep = torch.bernoulli(keep_prob).to(dtype=torch.bool)  # [B,1,H,W]

        # Состояния z_t:
        # - там, где keep=True, копируем x0 (0 или 1),
        # - там, где keep=False, ставим MASK_TOKEN_ID.
        x0_states = x0.to(dtype=torch.long)  # {0,1}
        z_t_states = torch.where(
            keep,
            x0_states,
            torch.full_like(x0_states, fill_value=self.MASK_TOKEN_ID),
        )

        mask_positions = ~keep  # True там, где пиксель стал MASK

        return z_t_states, mask_positions
