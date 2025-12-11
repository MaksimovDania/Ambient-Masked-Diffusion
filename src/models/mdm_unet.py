# src/models/mdm_unet.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn


class Swish(nn.Module):
    """
    Swish activation: x * sigmoid(x).

    Та же активация, что используется в UNet для DDPM у labml. 
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Временной эмбеддинг для шага диффузии t.

    Структура взята из UNet для DDPM: синусоидальные position embeddings +
    двухслойный MLP. 

    n_channels — итоговая размерность эмбеддинга времени.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        if n_channels % 4 != 0:
            raise ValueError(
                f"TimeEmbedding: n_channels ({n_channels}) must be divisible by 4, "
                "since we use n_channels//4 as input to the first linear layer."
            )

        self.n_channels = n_channels

        # Первая линейка получает sin/cos-эмбеддинг
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        # Вторая линейка доводит до нужной размерности
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: Tensor формы [B], может быть int или float.

        Возвращает эмбеддинг формы [B, n_channels].
        """
        if t.dim() != 1:
            raise ValueError(
                f"TimeEmbedding: expected t to have shape [B], got shape {tuple(t.shape)}"
            )

        # Гарантируем float
        t = t.float()

        # half_dim — половина размерности sin/cos (до MLP)
        half_dim = self.n_channels // 8
        if half_dim <= 0:
            raise ValueError(
                "TimeEmbedding: half_dim <= 0; check n_channels configuration."
            )

        # Шаг по частотам (как в positional encodings)
        freq_step = math.log(10_000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -freq_step)
        # [B, half_dim]
        args = t[:, None] * freqs[None, :]
        # [B, 2 * half_dim]
        emb = torch.cat([args.sin(), args.cos()], dim=1)

        # MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResidualBlock(nn.Module):
    """
    Residual-блок с добавлением временного эмбеддинга.

    Структура повторяет ResidualBlock из UNet DDPM: GroupNorm → Swish → Conv →
    добавление time-эмбеддинга → GroupNorm → Swish → Dropout → Conv + shortcut. 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"ResidualBlock: in_channels and out_channels must be > 0, "
                f"got {in_channels}, {out_channels}"
            )

        # sanity check для GroupNorm
        if in_channels % n_groups != 0:
            raise ValueError(
                f"ResidualBlock: in_channels={in_channels} must be divisible by "
                f"n_groups={n_groups} for GroupNorm."
            )
        if out_channels % n_groups != 0:
            raise ValueError(
                f"ResidualBlock: out_channels={out_channels} must be divisible by "
                f"n_groups={n_groups} for GroupNorm."
            )

        # Первый conv
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # Второй conv
        self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # Shortcut, если число каналов меняется
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Линеарка для time-эмбеддинга
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, C_in, H, W]
        t_emb:[B, time_channels]
        """
        if x.dim() != 4:
            raise ValueError(
                f"ResidualBlock.forward: expected x with 4 dims [B,C,H,W], got {tuple(x.shape)}"
            )
        if t_emb.dim() != 2 or t_emb.shape[0] != x.shape[0]:
            raise ValueError(
                "ResidualBlock.forward: t_emb must have shape [B, time_channels] "
                f"with B==x.shape[0]; got x.shape={tuple(x.shape)}, "
                f"t_emb.shape={tuple(t_emb.shape)}"
            )

        # Первый conv
        h = self.conv1(self.act1(self.norm1(x)))

        # Добавляем time-эмбеддинг (broadcast по пространству)
        time = self.time_emb(self.time_act(t_emb))  # [B, C_out]
        h = h + time[:, :, None, None]

        # Второй conv
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Residual + shortcut
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """
    Downsample в 2 раза с помощью conv с шагом 2.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError(
                f"Downsample: n_channels must be > 0, got {n_channels}"
            )
        self.conv = nn.Conv2d(
            n_channels, n_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _ = t_emb  # для совместимости интерфейса
        if x.dim() != 4:
            raise ValueError(
                f"Downsample.forward: expected x with 4 dims [B,C,H,W], got {tuple(x.shape)}"
            )
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample в 2 раза с помощью ConvTranspose2d.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError(
                f"Upsample: n_channels must be > 0, got {n_channels}"
            )
        self.conv = nn.ConvTranspose2d(
            n_channels,
            n_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _ = t_emb
        if x.dim() != 4:
            raise ValueError(
                f"Upsample.forward: expected x with 4 dims [B,C,H,W], got {tuple(x.shape)}"
            )
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Один уровень down-пути UNet: здесь только ResidualBlock (без attention).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
    ) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.res(x, t_emb)


class UpBlock(nn.Module):
    """
    Один уровень up-пути UNet.

    В реализации DDPM-UNet вход UpBlock'а — конкатенация текущих фичей и
    skip-фичей: [B, in_channels + out_channels, H, W] → [B, out_channels, H, W]. 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
    ) -> None:
        super().__init__()
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            time_channels,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.res(x, t_emb)


class MiddleBlock(nn.Module):
    """
    "Дно" UNet — два residual-блока подряд.
    """

    def __init__(self, n_channels: int, time_channels: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


@dataclass
class MDMUNetConfig:
    """
    Конфиг для MDMUNet; хорошо стыкуется с model-секцией в YAML.
    """
    image_channels: int = 3
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    time_emb_dim_mult: int = 4


class MDMUNet(nn.Module):
    """
    UNet для masked diffusion на MNIST.

    - Вход:  3 канала (0, 1, MASK) — либо one-hot, либо "эмбеддинги".
    - Выход: 3 канала логитов по состояниям {0, 1, MASK}.

    Архитектура вдохновлена UNet для DDPM от labml: down-путь → middle → up-путь
    + skip, time-эмбеддинг добавляется в каждый residual-блок. 
    """

    def __init__(self, cfg: MDMUNetConfig) -> None:
        super().__init__()

        # -------- SANITY CHECKS КОНФИГА --------
        if cfg.image_channels <= 0:
            raise ValueError(
                f"MDMUNetConfig: image_channels must be > 0, got {cfg.image_channels}"
            )
        if cfg.base_channels <= 0:
            raise ValueError(
                f"MDMUNetConfig: base_channels must be > 0, got {cfg.base_channels}"
            )
        if len(cfg.channel_multipliers) == 0:
            raise ValueError(
                "MDMUNetConfig: channel_multipliers must be a non-empty sequence."
            )
        if cfg.num_res_blocks <= 0:
            raise ValueError(
                f"MDMUNetConfig: num_res_blocks must be > 0, got {cfg.num_res_blocks}"
            )
        if cfg.time_emb_dim_mult <= 0:
            raise ValueError(
                f"MDMUNetConfig: time_emb_dim_mult must be > 0, got {cfg.time_emb_dim_mult}"
            )

        self.cfg = cfg
        image_channels = cfg.image_channels
        n_channels = cfg.base_channels
        ch_mults: Sequence[int] = cfg.channel_multipliers
        n_blocks = cfg.num_res_blocks
        time_channels = n_channels * cfg.time_emb_dim_mult

        # sanity: все channel_multipliers > 0
        if any(m <= 0 for m in ch_mults):
            raise ValueError(
                f"MDMUNetConfig: all channel_multipliers must be > 0, got {ch_mults}"
            )

        # Проекция входной "картинки" в feature map
        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=3, padding=1
        )

        # Эмбеддинг времени
        self.time_emb = TimeEmbedding(time_channels)

        # --------- Down-путь ---------
        down_modules: List[nn.Module] = []
        in_channels = n_channels
        n_resolutions = len(ch_mults)

        for i in range(n_resolutions):
            # сколько каналов будет на этом разрешении
            out_channels = in_channels * ch_mults[i]

            # несколько residual-блоков подряд
            for _ in range(n_blocks):
                down_modules.append(
                    DownBlock(in_channels, out_channels, time_channels)
                )
                in_channels = out_channels

            # между разрешениями делаем downsample,
            # кроме самого нижнего
            if i < n_resolutions - 1:
                down_modules.append(Downsample(in_channels))

        self.down = nn.ModuleList(down_modules)

        # --------- Middle ---------
        self.middle = MiddleBlock(in_channels, time_channels)
        # На этом этапе in_channels == out_channels самого нижнего уровня

        # --------- Up-путь ---------
        up_modules: List[nn.Module] = []
        # Начинаем с самого нижнего числа каналов
        for i in reversed(range(n_resolutions)):
            # Сначала несколько блоков на текущем числе каналов
            out_channels = in_channels
            for _ in range(n_blocks):
                up_modules.append(
                    UpBlock(in_channels, out_channels, time_channels)
                )
                # in_channels остаётся тем же после UpBlock (out_channels == in_channels)

            # Затем блок, который уменьшает число каналов
            out_channels = in_channels // ch_mults[i]
            if out_channels <= 0:
                raise ValueError(
                    "MDMUNet: computed out_channels <= 0 in up path. "
                    f"Check base_channels and channel_multipliers: {ch_mults}"
                )
            up_modules.append(
                UpBlock(in_channels, out_channels, time_channels)
            )
            in_channels = out_channels

            # Upsample, кроме самого верхнего уровня
            if i > 0:
                up_modules.append(Upsample(in_channels))

        self.up = nn.ModuleList(up_modules)

        # Финальная нормализация и свёртка в image_channels
        # Здесь используем 8 групп; sanity check — кратность
        if in_channels % 8 != 0:
            raise ValueError(
                f"MDMUNet: in_channels={in_channels} before final GroupNorm "
                "must be divisible by 8."
            )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, image_channels, kernel_size=3, padding=1
        )

        # --------- MINI-SANITY ПРОГОН ---------
        # Проверяем, что модель вообще пропускает данные нужной формы.
        # Делаем это без градиентов и на CPU, чтобы поймать ошибки каналов/skip'ов
        # сразу при инициализации.
        with torch.no_grad():
            dummy_x = torch.zeros(1, image_channels, 28, 28)
            dummy_t = torch.zeros(1)
            out = self.forward(dummy_x, dummy_t)
            expected_shape = (1, image_channels, 28, 28)
            if tuple(out.shape) != expected_shape:
                raise RuntimeError(
                    "MDMUNet sanity check failed: forward(dummy_x, dummy_t) "
                    f"returned shape {tuple(out.shape)}, expected {expected_shape}. "
                    "Check channel_multipliers, num_res_blocks and architecture."
                )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, image_channels, H, W]
        t: [B] (индексы шагов диффузии)

        Возвращает логиты [B, image_channels, H, W].
        """
        if x.dim() != 4:
            raise ValueError(
                f"MDMUNet.forward: expected x with 4 dims [B,C,H,W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.cfg.image_channels:
            raise ValueError(
                f"MDMUNet.forward: expected x with C={self.cfg.image_channels}, "
                f"got C={x.shape[1]}"
            )
        if t.dim() != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                "MDMUNet.forward: t must have shape [B] with B==x.shape[0]; "
                f"got x.shape={tuple(x.shape)}, t.shape={tuple(t.shape)}"
            )

        # Эмбеддинг времени
        t_emb = self.time_emb(t)  # [B, time_channels]

        # Проекция входа
        x = self.image_proj(x)  # [B, n_channels, H, W]

        # Список для skip-соединений
        skips = [x]

        # Down-путь
        for m in self.down:
            x = m(x, t_emb)
            skips.append(x)

        # Middle
        x = self.middle(x, t_emb)

        # Up-путь
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t_emb)
            else:
                if not skips:
                    raise RuntimeError(
                        "MDMUNet.forward: no more skip tensors available when "
                        "processing UpBlock. Check down/up architecture symmetry."
                    )
                s = skips.pop()
                if s.shape[0] != x.shape[0] or s.shape[2:] != x.shape[2:]:
                    raise RuntimeError(
                        "MDMUNet.forward: skip tensor spatial shape mismatch. "
                        f"x.shape={tuple(x.shape)}, s.shape={tuple(s.shape)}"
                    )
                x = torch.cat([x, s], dim=1)
                x = m(x, t_emb)

        # После прохода up-пути skip'ы должны закончиться
        if len(skips) != 0:
            raise RuntimeError(
                "MDMUNet.forward: leftover skip tensors after up path. "
                "This usually means a mismatch between down and up modules."
            )

        # Финальный head
        x = self.final(self.act(self.norm(x)))
        return x
