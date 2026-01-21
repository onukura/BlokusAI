from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch import nn

from blokus_ai.device import get_device, to_device


class ResidualBlock(nn.Module):
    """ResNet風の残差ブロック。

    2層の畳み込み層とスキップ接続を持つ。
    """
    def __init__(self, channels: int):
        """残差ブロックを初期化する。

        Args:
            channels: 入出力チャネル数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播。

        Args:
            x: 入力テンソル (B, C, H, W)

        Returns:
            出力テンソル (B, C, H, W)
        """
        residual = x
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class BoardEncoder(nn.Module):
    """ボード状態をエンコードするResNetスタイルの畳み込みエンコーダ。"""
    def __init__(self, in_channels: int = 5, channels: int = 64, num_blocks: int = 4):
        """ボードエンコーダを初期化する。

        Args:
            in_channels: 入力チャネル数（デフォルト5: state encoding）
            channels: 内部特徴チャネル数
            num_blocks: 残差ブロックの数
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播。

        Args:
            x: 入力ボード (B, in_channels, H, W)

        Returns:
            エンコードされた特徴マップ (B, channels, H, W)
        """
        x = self.stem(x)
        x = self.blocks(x)
        return x


class PolicyValueNet(nn.Module):
    """AlphaZero風のポリシー・バリューネットワーク。

    共有エンコーダから2つのヘッドに分岐:
    - ポリシーヘッド: 各合法手のスコア（可変長出力）
    - バリューヘッド: 現在プレイヤー視点の局面評価値（-1～+1）
    """
    def __init__(
        self,
        in_channels: int = 8,
        channels: int = 128,
        num_blocks: int = 10,
        n_pieces: int = 21,
    ):
        """ネットワークを初期化する。

        Args:
            in_channels: 入力チャネル数（デフォルト8: v2特徴量）
            channels: エンコーダの内部チャネル数
            num_blocks: 残差ブロック数
            n_pieces: ピース総数（デフォルト21）
        """
        super().__init__()
        self.encoder = BoardEncoder(in_channels, channels, num_blocks)
        self.piece_embed = nn.Embedding(n_pieces, 16)
        self.policy_mlp = nn.Sequential(
            nn.Linear(channels + 16 + 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64 + n_pieces * 2 + 1, 256),  # +1 for game_phase
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        # Initialize weights before moving to device
        self._initialize_weights()

        # デバイスに移動（TPU → GPU → CPU の優先順位）
        self.device = get_device()
        self.to(self.device)

    def forward(
        self,
        board: torch.Tensor,
        self_rem: torch.Tensor,
        opp_rem: torch.Tensor,
        move_features: Dict[str, torch.Tensor],
        game_phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """順伝播。

        Args:
            board: ボード状態 (B, C, H, W)
            self_rem: 現在プレイヤーの残りピース (B, 21)
            opp_rem: 相手プレイヤーの残りピース (B, 21)
            move_features: 合法手の特徴量辞書
            game_phase: ゲーム進行度 (B,) - オプション（v2特徴量用）

        Returns:
            (ポリシーロジット (N_moves,), バリュー (B,))
        """
        fmap = self.encoder(board)
        logits = self._policy_logits(fmap, move_features)
        value = self._value(fmap, self_rem, opp_rem, game_phase)
        return logits, value

    def _policy_logits(
        self, fmap: torch.Tensor, move_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """合法手ごとのポリシーロジットを計算する。

        各手が占めるセルの特徴を平均し、ピース埋め込み・アンカー・
        サイズと結合してMLPでスコアリング。

        Args:
            fmap: エンコードされた特徴マップ (1, C, H, W)
            move_features: 指し手の特徴量辞書

        Returns:
            各手のロジット (N_moves,)
        """
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
        self,
        fmap: torch.Tensor,
        self_rem: torch.Tensor,
        opp_rem: torch.Tensor,
        game_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """局面のバリュー評価値を計算する。

        特徴マップをグローバル平均プーリングし、残りピース情報と
        ゲーム進行度を結合してMLPで評価値を出力。

        Args:
            fmap: エンコードされた特徴マップ (B, C, H, W)
            self_rem: 現在プレイヤーの残りピース (B, 21)
            opp_rem: 相手プレイヤーの残りピース (B, 21)
            game_phase: ゲーム進行度 (B,) - オプション

        Returns:
            評価値 (B,) [-1, +1]の範囲
        """
        rem = torch.cat([self_rem, opp_rem], dim=1)
        pooled = self.value_head[:4](fmap).flatten(1)

        # game_phaseがある場合は追加
        if game_phase is not None:
            # game_phaseを(B, 1)の形に変換
            if game_phase.dim() == 0:
                game_phase = game_phase.unsqueeze(0)
            if game_phase.dim() == 1:
                game_phase = game_phase.unsqueeze(-1)
            value_input = torch.cat([pooled, rem, game_phase], dim=1)
        else:
            # 後方互換性: game_phaseなしの場合は0でパディング
            batch_size = fmap.shape[0]
            zero_phase = torch.zeros(batch_size, 1, device=fmap.device)
            value_input = torch.cat([pooled, rem, zero_phase], dim=1)

        value = self.value_head[4:](value_input)
        return value.squeeze(-1)

    def _initialize_weights(self):
        """Initialize network weights following AlphaZero best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He/Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                # GroupNorm standard initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                # Embedding standard initialization
                nn.init.normal_(m.weight, mean=0, std=0.01)

        # Special: Value head final layer uses small weights
        # This prevents extreme initial predictions
        value_final_linear = self.value_head[-2]  # Second-to-last (before tanh)
        nn.init.uniform_(value_final_linear.weight, -0.003, 0.003)
        nn.init.constant_(value_final_linear.bias, 0)


@torch.no_grad()
def predict(
    net: PolicyValueNet,
    board: np.ndarray,
    self_rem: np.ndarray,
    opp_rem: np.ndarray,
    move_features,
    game_phase: float | None = None,
):
    """ニューラルネットで局面を評価し、ポリシーとバリューを返す。

    推論モードで勾配計算なしで実行。numpy配列を受け取り、
    numpy/floatで結果を返す。

    Args:
        net: ポリシーバリューネットワーク
        board: ボード状態 (C, H, W)
        self_rem: 現在プレイヤーの残りピース (21,)
        opp_rem: 相手プレイヤーの残りピース (21,)
        move_features: 合法手の特徴量辞書
        game_phase: ゲーム進行度 (scalar) - オプション

    Returns:
        (ポリシーロジット (N_moves,), バリュー (float))
    """
    net.eval()
    device = net.device  # ネットワークと同じデバイスを使用

    # 入力をデバイスに移動
    board_t = torch.from_numpy(board[None]).float().to(device)
    self_rem_t = torch.from_numpy(self_rem[None]).float().to(device)
    opp_rem_t = torch.from_numpy(opp_rem[None]).float().to(device)
    move_tensors = {
        "piece_id": torch.from_numpy(move_features["piece_id"]).long().to(device),
        "anchor": torch.from_numpy(move_features["anchor"]).float().to(device),
        "size": torch.from_numpy(move_features["size"]).float().to(device),
        "cells": move_features["cells"],  # cellsはリストのままなのでデバイス移動不要
    }

    # game_phaseをテンソルに変換（ある場合）
    game_phase_t = None
    if game_phase is not None:
        game_phase_t = torch.tensor([game_phase], dtype=torch.float32, device=device)

    logits, value = net(board_t, self_rem_t, opp_rem_t, move_tensors, game_phase_t)

    # CPUに戻してnumpyに変換（後方互換性のため）
    return logits.cpu().numpy(), float(value.cpu().item())


@torch.no_grad()
def batch_predict(
    net: PolicyValueNet,
    boards: List[np.ndarray],
    self_rems: List[np.ndarray],
    opp_rems: List[np.ndarray],
    move_features_list: List[Dict],
    game_phases: List[float] | None = None,
) -> List[tuple[np.ndarray, float]]:
    """複数の局面をバッチで評価する（MCTS高速化用）。

    ボードエンコーダをバッチで実行することで、GPU利用率を向上させる。
    ただしポリシーヘッドは手数が異なるため個別に実行。

    Args:
        net: ポリシーバリューネットワーク
        boards: ボード状態のリスト [(C, H, W), ...]
        self_rems: 残りピース (自分) のリスト [(21,), ...]
        opp_rems: 残りピース (相手) のリスト [(21,), ...]
        move_features_list: 合法手特徴量のリスト [dict, ...]
        game_phases: ゲーム進行度のリスト [float, ...] - オプション

    Returns:
        [(ポリシーロジット, バリュー), ...] の各局面の結果リスト
    """
    if not boards:
        return []

    net.eval()
    device = net.device

    # ボードをバッチ化（これが主な高速化ポイント）
    boards_t = torch.stack([torch.from_numpy(b) for b in boards]).float().to(device)
    self_rems_t = torch.stack([torch.from_numpy(s) for s in self_rems]).float().to(device)
    opp_rems_t = torch.stack([torch.from_numpy(o) for o in opp_rems]).float().to(device)

    # game_phasesをテンソルに変換（ある場合）
    game_phases_t = None
    if game_phases is not None:
        game_phases_t = torch.tensor(game_phases, dtype=torch.float32, device=device)

    # バッチエンコード（高速！）
    fmaps = net.encoder(boards_t)

    # 各局面のポリシーとバリューを計算（手数が異なるため個別）
    results = []
    for i, move_features in enumerate(move_features_list):
        move_tensors = {
            "piece_id": torch.from_numpy(move_features["piece_id"]).long().to(device),
            "anchor": torch.from_numpy(move_features["anchor"]).float().to(device),
            "size": torch.from_numpy(move_features["size"]).float().to(device),
            "cells": move_features["cells"],
        }

        # 個別のポリシー計算
        logits = net._policy_logits(fmaps[i:i+1], move_tensors)
        # 個別のバリュー計算（game_phaseがあれば渡す）
        phase_t = game_phases_t[i:i+1] if game_phases_t is not None else None
        value = net._value(fmaps[i:i+1], self_rems_t[i:i+1], opp_rems_t[i:i+1], phase_t)

        results.append((logits.cpu().numpy(), float(value.cpu().item())))

    return results
