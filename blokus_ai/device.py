"""デバイス管理モジュール - TPU/GPU/CPUの自動検出とテンソル移動を提供。

Google Colab での柔軟な実行環境に対応:
- TPU優先（torch_xla経由）
- GPU（CUDA）にフォールバック
- CPU（最終フォールバック）

使用例:
    from blokus_ai.device import get_device, get_device_name, to_device

    device = get_device()  # 最適なデバイスを自動選択
    print(f"Using: {get_device_name()}")  # "TPU (8 cores)" or "GPU (CUDA)" or "CPU"

    model = MyModel()
    model = to_device(model)  # モデルをデバイスに移動

    tensor = torch.randn(10, 10)
    tensor = to_device(tensor)  # テンソルをデバイスに移動
"""

import os
import warnings
from typing import Union, Optional

import torch
import torch.nn as nn


# グローバルキャッシュ（デバイス検出は1回のみ）
_cached_device: Optional[torch.device] = None
_cached_device_name: Optional[str] = None


def _detect_device() -> tuple[torch.device, str]:
    """デバイスを検出して (device, name) を返す。

    優先順位:
        1. TPU (torch_xla経由、Google Colab/GCP)
        2. GPU (CUDA)
        3. CPU (フォールバック)

    環境変数 BLOKUS_DEVICE_OVERRIDE で強制可能（デバッグ用）:
        - "cpu": CPUを強制
        - "cuda" or "gpu": GPUを強制
        - "tpu": TPUを強制（torch_xlaが必要）

    Returns:
        (torch.device, 人間が読める名前)
    """
    # 環境変数による上書き（デバッグ・テスト用）
    override = os.environ.get("BLOKUS_DEVICE_OVERRIDE", "").lower()
    if override:
        if override == "cpu":
            return torch.device("cpu"), "CPU (forced by BLOKUS_DEVICE_OVERRIDE)"
        elif override in ("cuda", "gpu"):
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return torch.device("cuda"), f"GPU (CUDA) - {gpu_name} (forced)"
            else:
                warnings.warn(
                    "BLOKUS_DEVICE_OVERRIDE=gpu but CUDA not available, falling back to CPU"
                )
                return torch.device("cpu"), "CPU (CUDA not available)"
        elif override == "tpu":
            try:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
                return device, f"TPU - {device} (forced)"
            except ImportError:
                warnings.warn(
                    "BLOKUS_DEVICE_OVERRIDE=tpu but torch_xla not installed, "
                    "falling back to CPU"
                )
                return torch.device("cpu"), "CPU (torch_xla not available)"

    # 1. TPU検出（Google Colab TPU runtime）
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        # TPUデバイスが実際に利用可能か確認
        # xm.xla_device() は常にデバイスを返すが、実際には使えない場合がある
        try:
            # 簡単なテスト操作でTPUが使えるか確認
            test_tensor = torch.tensor([1.0], device=device)
            del test_tensor
            return device, f"TPU - {device}"
        except Exception:
            # TPUデバイスの取得はできたが、実際には使えない
            pass
    except ImportError:
        # torch_xla がインストールされていない（通常の環境）
        pass
    except Exception as e:
        # その他のTPU関連エラー
        warnings.warn(f"TPU detection failed: {e}, falling back to GPU/CPU")

    # 2. GPU検出（CUDA）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, f"GPU (CUDA) - {gpu_name} ({gpu_memory:.1f}GB)"

    # 3. CPUフォールバック
    return torch.device("cpu"), "CPU"


def get_device() -> torch.device:
    """最適なデバイスを返す（キャッシュ済み）。

    初回呼び出し時にデバイスを検出し、以降はキャッシュされた結果を返す。
    スレッドセーフではないが、通常は問題ない（単一プロセスで使用）。

    Returns:
        torch.device: TPU、GPU、またはCPU

    Examples:
        >>> device = get_device()
        >>> model.to(device)
        >>> tensor = torch.randn(10, 10, device=device)
    """
    global _cached_device, _cached_device_name

    if _cached_device is None:
        _cached_device, _cached_device_name = _detect_device()

    return _cached_device


def get_device_name() -> str:
    """デバイスの人間が読める名前を返す。

    ログ出力やユーザーへの表示に使用。

    Returns:
        str: "TPU - xla:0" や "GPU (CUDA) - Tesla T4 (15.0GB)" など

    Examples:
        >>> print(f"Training on: {get_device_name()}")
        Training on: GPU (CUDA) - Tesla T4 (15.0GB)
    """
    global _cached_device, _cached_device_name

    if _cached_device is None:
        _cached_device, _cached_device_name = _detect_device()

    return _cached_device_name


def to_device(
    obj: Union[torch.Tensor, nn.Module],
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, nn.Module]:
    """テンソルまたはモジュールを指定デバイスに移動。

    Args:
        obj: 移動するテンソルまたはnn.Module
        device: 移動先デバイス（Noneの場合は get_device() を使用）

    Returns:
        デバイスに移動されたオブジェクト

    Examples:
        >>> tensor = torch.randn(10, 10)
        >>> tensor = to_device(tensor)  # 自動検出デバイスに移動
        >>>
        >>> model = MyModel()
        >>> model = to_device(model, torch.device("cuda"))  # GPUに移動
    """
    if device is None:
        device = get_device()

    return obj.to(device)


def reset_device_cache():
    """デバイスキャッシュをリセット（テスト用）。

    通常のアプリケーションコードでは使用しない。
    テストコードでデバイス検出をやり直したい場合に使用。
    """
    global _cached_device, _cached_device_name
    _cached_device = None
    _cached_device_name = None


# モジュールロード時に一度検出（早期エラー検出）
_ = get_device()
_ = get_device_name()
