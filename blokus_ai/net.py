from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class BoardEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, channels: int = 64, num_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        channels: int = 64,
        num_blocks: int = 4,
        n_pieces: int = 21,
    ):
        super().__init__()
        self.encoder = BoardEncoder(in_channels, channels, num_blocks)
        self.piece_embed = nn.Embedding(n_pieces, 16)
        self.policy_mlp = nn.Sequential(
            nn.Linear(channels + 16 + 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32 + n_pieces * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        board: torch.Tensor,
        self_rem: torch.Tensor,
        opp_rem: torch.Tensor,
        move_features: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.encoder(board)
        logits = self._policy_logits(fmap, move_features)
        value = self._value(fmap, self_rem, opp_rem)
        return logits, value

    def _policy_logits(
        self, fmap: torch.Tensor, move_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        piece_ids = move_features["piece_id"]
        anchors = move_features["anchor"]
        sizes = move_features["size"].unsqueeze(-1)
        cells = move_features["cells"]
        move_vecs = []
        for move_cells in cells:
            cell_feats = []
            for x, y in move_cells:
                cell_feats.append(fmap[0, :, y, x])
            move_vec = torch.stack(cell_feats, dim=0).mean(dim=0)
            move_vecs.append(move_vec)
        move_tensor = torch.stack(move_vecs, dim=0)
        piece_emb = self.piece_embed(piece_ids)
        features = torch.cat([move_tensor, piece_emb, anchors, sizes], dim=1)
        logits = self.policy_mlp(features).squeeze(-1)
        return logits

    def _value(
        self, fmap: torch.Tensor, self_rem: torch.Tensor, opp_rem: torch.Tensor
    ) -> torch.Tensor:
        rem = torch.cat([self_rem, opp_rem], dim=1)
        value_input = torch.cat([self.value_head[:3](fmap).flatten(1), rem], dim=1)
        value = self.value_head[3:](value_input)
        return value.squeeze(-1)


@torch.no_grad()
def predict(
    net: PolicyValueNet,
    board: np.ndarray,
    self_rem: np.ndarray,
    opp_rem: np.ndarray,
    move_features,
):
    net.eval()
    board_t = torch.from_numpy(board[None]).float()
    self_rem_t = torch.from_numpy(self_rem[None]).float()
    opp_rem_t = torch.from_numpy(opp_rem[None]).float()
    move_tensors = {
        "piece_id": torch.from_numpy(move_features["piece_id"]).long(),
        "anchor": torch.from_numpy(move_features["anchor"]).float(),
        "size": torch.from_numpy(move_features["size"]).float(),
        "cells": move_features["cells"],
    }
    logits, value = net(board_t, self_rem_t, opp_rem_t, move_tensors)
    return logits.cpu().numpy(), float(value.cpu().item())
