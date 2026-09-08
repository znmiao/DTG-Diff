from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def data_generator(x, y, batch_size: int = 128, shuffle: bool = True, drop_last: bool = False):
    dataset = TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
