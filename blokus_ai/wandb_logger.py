"""WandB integration for BlokusAI training and evaluation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from pathlib import Path

from dotenv import load_dotenv

SSL_FILE = Path(__file__).resolve().parent.parent / "cert.crt"
if SSL_FILE.exists():
    os.environ["SSL_CERT_FILE"] = str(SSL_FILE)
    os.environ["REQUESTS_CA_BUNDLE"] = str(SSL_FILE)


class WandBLogger:
    """Handles WandB initialization, logging, and artifact management.

    Features:
    - Safe initialization with fallback
    - Automatic .env loading
    - Structured metric logging
    - Model artifact upload
    - Error handling

    Example:
        >>> logger = WandBLogger(project_name="BlokusAI", config={"lr": 1e-3})
        >>> logger.log({"train/loss": 0.5}, step=1)
        >>> logger.log_artifact("model.pth", "checkpoint_001", "model")
        >>> logger.finish()
    """

    def __init__(
        self,
        project_name: str = "BlokusAI",
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize WandB logger with error handling.

        Args:
            project_name: WandB project name
            config: Hyperparameters and configuration
            use_wandb: Enable/disable WandB (for testing)
            name: Optional run name
        """
        self.enabled = use_wandb
        self.run = None

        if not self.enabled:
            print("[WandB] Disabled by configuration")
            return

        # Check if wandb is available
        if not WANDB_AVAILABLE:
            print("[WandB] WARNING: wandb package not installed")
            print("[WandB] Install with: pip install wandb")
            print("[WandB] Continuing without WandB logging")
            self.enabled = False
            return

        # Load .env file
        load_dotenv()

        # Check API key
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            print("[WandB] WARNING: WANDB_API_KEY not found in environment")
            print("[WandB] Please set WANDB_API_KEY in .env file or environment")
            print("[WandB] Continuing without WandB logging")
            self.enabled = False
            return

        try:
            # Initialize WandB
            self.run = wandb.init(
                project=project_name,
                config=config or {},
                name=name,
                reinit=True,  # Allow multiple runs in same process
            )
            print("[WandB] Initialized successfully")
            print(f"[WandB] Run URL: {self.run.url}")

        except Exception as e:
            print(f"[WandB] WARNING: Initialization failed: {e}")
            print("[WandB] Continuing without WandB logging")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (iteration)

        Example:
            >>> logger.log({"train/loss": 0.5, "train/acc": 0.9}, step=10)
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[WandB] WARNING: Failed to log metrics: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upload a file as a WandB artifact.

        Args:
            artifact_path: Path to the file
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (default: "model")
            metadata: Optional metadata dictionary

        Example:
            >>> logger.log_artifact(
            ...     "model.pth",
            ...     "checkpoint_0010",
            ...     "model",
            ...     {"iteration": 10, "loss": 0.5}
            ... )
        """
        if not self.enabled or self.run is None:
            return

        # Check if file exists
        if not os.path.exists(artifact_path):
            print(f"[WandB] WARNING: Artifact file not found: {artifact_path}")
            return

        try:
            artifact = wandb.Artifact(
                name=artifact_name, type=artifact_type, metadata=metadata or {}
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
            print(f"[WandB] Uploaded artifact: {artifact_name}")
        except Exception as e:
            print(f"[WandB] WARNING: Failed to upload artifact: {e}")

    def get_run_name(self) -> str | None:
        """Get the current WandB run name.

        Returns:
            Run name if available, None otherwise
        """
        if self.enabled and self.run is not None:
            return self.run.name
        return None

    def get_run_id(self) -> str | None:
        """Get the current WandB run ID.

        Returns:
            Run ID if available, None otherwise
        """
        if self.enabled and self.run is not None:
            return self.run.id
        return None

    def finish(self) -> None:
        """Finish the WandB run."""
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
                print("[WandB] Run finished successfully")
            except Exception as e:
                print(f"[WandB] WARNING: Error finishing run: {e}")
