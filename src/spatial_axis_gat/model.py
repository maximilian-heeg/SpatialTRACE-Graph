from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .metrics import regression_metrics


class GATModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_features, heads=n_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_features * n_heads, out_features, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x).squeeze(1)


class GNNLightning(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GATModel(in_features, hidden_features, 1, n_heads, dropout)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.model(x, edge_index)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        pred = self(batch.x, batch.edge_index)
        target_size = batch.batch_size
        target_pred = pred[:target_size]
        target_y = batch.y[:target_size]
        loss = self.loss_fn(target_pred, target_y)
        mae = torch.mean(torch.abs(target_pred - target_y))
        ss_res = torch.sum((target_y - target_pred) ** 2)
        ss_tot = torch.sum((target_y - torch.mean(target_y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-6)
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, batch_size=target_size)
        self.log(f"{stage}_mae", mae, on_epoch=True, on_step=False, batch_size=target_size)
        self.log(f"{stage}_r2", r2, on_epoch=True, on_step=False, batch_size=target_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)

