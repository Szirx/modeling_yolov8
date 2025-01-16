import os
import re
from pathlib import Path
from typing import Callable, List

import mlflow
import pandas as pd
import yaml

from configs.constants import PROJECT_DEFAULT_PATH
from configs.trainerconfigs.config import YOLOTrainerConfig
from src.mlflow_tracking.cust_mlflow import (
    MLflowTracking,
    load_dataset_description,
)
from src.settings_update.yolo_settings_update import settings_update
from ultralytics import YOLO


class YOLOTrainer:  # noqa: WPS230
    """
    Main trainer.

    """

    def __init__(self, config: YOLOTrainerConfig, callbacks: List[Callable] = None):
        """
        Initialize the YOLOTrainer.
        :param config: Config fot train/inference mode.
        :param callbacks: List of callbacks
        """
        self.config_trainer = config
        self.cfg_file_yolo = self._load_model_config(config.cfg_model_path)
        if config.pretrained_path:
            self.cfg_file_yolo["training_params"]["model"] = Path(config.pretrained_path)

        settings_update(config.yolo_settings_update)
        self.model = YOLO(self.cfg_file_yolo["training_params"]["model"])

    def run_training(self):
        """
        Run the YOLO model training process.

        """
        self._train_yolo_model(self.cfg_file_yolo)

    @classmethod
    def _load_model_config(cls, model_cfg_file: Path):
        """
        Load the YOLO model configuration from a YAML file.

        :param model_cfg_file: Path to the YOLO model configuration file.
        :return dict: Loaded model configuration as a dictionary.
        """
        with open(model_cfg_file, "r") as file_cfg:
            return yaml.safe_load(file_cfg)


    def _train_yolo_model(self, model_config: dict):
        """
        Train a YOLO model based on the provided configuration.

        :param model_config: YOLO model configuration as a dictionary.
        """
        self._do_train(model_config)

    def _do_train(self, model_config: dict):
        """
        Train a YOLO model.

        """
        model_config['training_params']['data'] = 'configs/traindataconfigs/data_detection.yaml'
        self.model.train(**model_config["training_params"])
