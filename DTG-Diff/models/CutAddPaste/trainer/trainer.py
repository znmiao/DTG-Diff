from __future__ import annotations

import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total = 0
        for x, y in loader:
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        return total_loss / max(1, total)
