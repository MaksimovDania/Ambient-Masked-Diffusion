# src/data/mnist.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class MNISTDataConfig:
    """
    Конфигурация для MNIST.

    root:
        Путь к директории с данными.
    train:
        Использовать ли train-часть MNIST.
    download:
        Скачивать ли данные, если их нет.
    p_missing:
        Вероятность, что каждый пиксель будет МИССИНГОМ (пропуском).
        0.0 означает отсутствие пропусков.
    binarize_threshold:
        Порог для бинаризации (>= threshold -> 1, иначе 0).
    flatten:
        Если True, изображения будут сплющены в вектор [784].
    """
    root: str = "data"
    train: bool = True
    download: bool = True
    p_missing: float = 0.5
    binarize_threshold: float = 0.5
    flatten: bool = False


class BinarizedNoisyMNIST(Dataset):
    """
    MNIST с:
      - детерминированной бинаризацией,
      - пиксельной пропускностью Bernoulli(p_missing),
      - явной маской наблюдений.

    Каждый элемент — словарь:
      - x_clean: полный бинарный образ (0/1), Tensor [1, 28, 28] или [784]
      - x_obs: наблюдаемый образ, пропуски = -1.0
      - obs_mask: 1.0 там, где наблюдение есть, 0.0 — пропуск
      - label: метка цифры (0..9)
    """

    def __init__(self, config: MNISTDataConfig) -> None:
        super().__init__()

        self.config = config

        base_transform = transforms.ToTensor()

        self._base_dataset = datasets.MNIST(
            root=config.root,
            train=config.train,
            download=config.download,
            transform=base_transform,
        )

        self._prepare_data()

    def _prepare_data(self) -> None:
        """
        Предвычисляет тензоры:
          - x_clean
          - x_obs
          - obs_mask
          - labels
        """
        bin_thresh = self.config.binarize_threshold
        p_missing = self.config.p_missing

        if p_missing < 0.0 or p_missing > 1.0:
            raise ValueError(f"p_missing must be in [0, 1], got {p_missing}")

        images = []
        labels = []

        for img, label in self._base_dataset:
            # img: [1, 28, 28] в [0, 1]
            # детерминированная бинаризация
            x_bin = (img >= bin_thresh).float()
            images.append(x_bin)
            labels.append(label)

        # Стек в тензоры
        self.x_clean = torch.stack(images, dim=0)          # [N,1,28,28], {0,1}
        self.labels = torch.tensor(labels, dtype=torch.long)  # [N]

        # Маска наблюдений: 1 — наблюдаем, 0 — пропуск
        if p_missing > 0.0:
            p_observed = 1.0 - p_missing
            self.obs_mask = torch.bernoulli(
                torch.full_like(self.x_clean, p_observed)
            )
        else:
            self.obs_mask = torch.ones_like(self.x_clean)

        # Наблюдаемый образ:
        #   x_obs = x_clean на наблюдаемых пикселях,
        #   -1.0 на пропусках.
        # Никакого torch.where — делаем через арифметику масок.
        sentinel = -1.0
        self.x_obs = (
            self.x_clean * self.obs_mask
            + sentinel * (1.0 - self.obs_mask)
        )

        # Опционально сплющиваем в вектор
        if self.config.flatten:
            N = self.x_clean.shape[0]
            self.x_clean = self.x_clean.view(N, -1)
            self.x_obs = self.x_obs.view(N, -1)
            self.obs_mask = self.obs_mask.view(N, -1)

    def __len__(self) -> int:
        return self.x_clean.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_clean = self.x_clean[idx]
        x_obs = self.x_obs[idx]
        obs_mask = self.obs_mask[idx]
        label = self.labels[idx]

        return {
            "x_clean": x_clean,
            "x_obs": x_obs,
            "obs_mask": obs_mask,
            "label": label,
        }


def create_mnist_dataloaders(
    root: str = "data",
    batch_size: int = 64,
    p_missing: float = 0.5,
    binarize_threshold: float = 0.5,
    flatten: bool = False,
    train_val_split: float = 0.9,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Создаёт train и val DataLoader'ы для BinarizedNoisyMNIST.

    Возвращает:
      train_loader, val_loader
    """
    config = MNISTDataConfig(
        root=root,
        train=True,
        download=download,
        p_missing=p_missing,
        binarize_threshold=binarize_threshold,
        flatten=flatten,
    )

    full_dataset = BinarizedNoisyMNIST(config)

    n_total = len(full_dataset)
    n_train = int(train_val_split * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
