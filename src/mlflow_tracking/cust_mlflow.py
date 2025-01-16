import os
from typing import Any, Dict

import mlflow
import yaml

from ultralytics import YOLO


def clean_metric_names(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metric names by removing parentheses.

    :param metrics: Dictionary of metrics with possibly formatted names.
    :return dict: Dictionary of metrics with cleaned names.
    """
    for key, _ in metrics.items():
        metrics[key].replace('(', '').replace(')', '')
    return metrics


def load_dataset_description(file_path: str) -> str:
    """
    Load and return a description of the dataset from a file.

    :param file_path: Path to the dataset description file.
    :return str: Description of the dataset.
    """
    with open(file_path, 'r') as data_file:
        dataset_name = yaml.safe_load(data_file)['path']
        dataset_name = dataset_name.split(os.path.sep)[-1]

    return f'Datasets:\n{dataset_name}'


class MLflowTracking:
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize an MLflowTracking instance.

        :param tracking_uri: The MLflow tracking URI.
        :param experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.setup_mlflow_tracking()

    def setup_mlflow_tracking(self) -> None:
        """
        Set up MLflow tracking using the provided tracking URI and experiment name.
        If the experiment doesn't exist, it creates one.
        """
        mlflow.set_tracking_uri(self.tracking_uri)

        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)

    @classmethod
    def log_training_params(cls, model_cfg: Dict[str, Dict[str, Any]]) -> None:
        """
        Log training parameters to MLflow.

        :param model_cfg: YOLO model configuration.
        """
        mlflow.log_params(model_cfg['training_params'])

    @classmethod
    def log_training_metrics(cls, model: YOLO) -> None:
        """
        Log training metrics to MLflow.

        :param model: YOLO model.
        """
        metrics = {
            f'best_{name}': value_metr.item()
            for name, value_metr in clean_metric_names(
                model.trainer.metrics.copy(),
            ).items()
        }

        mlflow.log_metrics({**metrics})

    @classmethod
    def log_trained_model(cls, model: YOLO) -> None:
        """
        Log the trained model to MLflow.

        :param model: YOLO model.
        """
        mlflow.pyfunc.log_model(
            artifact_path='model',
            artifacts={'model_path': str(model.trainer.save_dir)},
            python_model=mlflow.pyfunc.PythonModel(),
        )

    @classmethod
    def log_custom_artifact(cls, artifact_path, save_path):
        """
        Log some artifact.

        :param artifact_path: Path artifact.
        :param save_path: Path to log.
        """

        mlflow.log_artifact(artifact_path, save_path)
