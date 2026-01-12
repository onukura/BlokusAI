"""Tests for WandB logger integration."""

import os
import tempfile
from pathlib import Path

import pytest

from blokus_ai.wandb_logger import WandBLogger


def test_wandb_logger_disabled():
    """Test logger with wandb disabled."""
    logger = WandBLogger(use_wandb=False)
    assert not logger.enabled
    assert logger.run is None

    # Should not crash
    logger.log({"test": 1.0})
    logger.log({"test": 1.0}, step=1)
    logger.finish()


def test_wandb_logger_no_api_key(monkeypatch):
    """Test logger with missing API key."""
    # Remove WANDB_API_KEY from environment
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    logger = WandBLogger(use_wandb=True)
    assert not logger.enabled
    assert logger.run is None


def test_wandb_logger_log_artifact_disabled():
    """Test artifact logging when disabled."""
    logger = WandBLogger(use_wandb=False)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name
        f.write(b"dummy model data")

    try:
        # Should not crash
        logger.log_artifact(
            artifact_path=temp_path,
            artifact_name="test",
            artifact_type="model"
        )
    finally:
        os.remove(temp_path)


def test_wandb_logger_log_artifact_nonexistent_file():
    """Test artifact logging with nonexistent file."""
    logger = WandBLogger(use_wandb=False)

    # Should not crash, just print warning
    logger.log_artifact(
        artifact_path="/nonexistent/file.pth",
        artifact_name="test",
        artifact_type="model"
    )


def test_wandb_logger_log_with_metadata():
    """Test logging with metadata."""
    logger = WandBLogger(use_wandb=False)

    # Should not crash
    logger.log({"test/metric1": 1.0, "test/metric2": 2.0}, step=5)


def test_wandb_logger_finish_when_disabled():
    """Test finish when logger is disabled."""
    logger = WandBLogger(use_wandb=False)

    # Should not crash
    logger.finish()
    logger.finish()  # Double finish should also be safe
