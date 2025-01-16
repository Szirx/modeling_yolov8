from pathlib import Path

from configs.trainerconfigs.config import YoloSettingsUpdate
from ultralytics import settings


def settings_update(cfg: YoloSettingsUpdate):
    """
    Update setting config before start.
    """
    settings.update(
        {
            "clearml": cfg.clearml,
            "mlflow": cfg.mlflow,
            "datasets_dir": Path().home(),
        },
    )
