from pathlib import Path
from typing import Union

import yaml

from configs.trainerconfigs.config import YOLOTrainerConfig


def load_config(config_path: Union[str, Path]) -> YOLOTrainerConfig:
    """
    Load file from yaml.

    :param config_path: Path to yaml file.
    :return ExperimentConfig: obj config
    """
    config_path = Path(config_path)
    with open(config_path, "r") as file_config:
        config_dict = yaml.safe_load(file_config)
    return YOLOTrainerConfig(**config_dict)
