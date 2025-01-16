# flake8: noqa: S101

import os
import shutil
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
import yaml
from PIL import Image

from configs.trainerconfigs.config import PROJECT_DEFAULT_PATH
from src.inference.predict import YOLOv8Predictor

with open(PROJECT_DEFAULT_PATH / "tests/data/config_test.yaml", "r") as config_test_file:
    CONFIG_TEST = yaml.safe_load(config_test_file)


class TestYOLOv8Predictor:
    @pytest.mark.parametrize(
        "model_path, expectation",
        [
            (PROJECT_DEFAULT_PATH / CONFIG_TEST["best_model_path"], does_not_raise()),
            (PROJECT_DEFAULT_PATH / "bad/path.pt", pytest.raises(FileNotFoundError)),
        ],
    )
    def test_init(self, model_path: Path, expectation):
        with expectation:
            model = YOLOv8Predictor(
                model_path,
                CONFIG_TEST["path_image_predict"],
                CONFIG_TEST["save_results_path"],
                CONFIG_TEST["data_experiment_name"],
                CONFIG_TEST["save_segmentations_json"],
                CONFIG_TEST["yolo"],
            )

            assert isinstance(model, YOLOv8Predictor)
            assert model.model.overrides["conf"] == CONFIG_TEST["yolo"]["conf"]
            assert model.model.overrides["iou"] == CONFIG_TEST["yolo"]["iou"]
            assert model.model.overrides["agnostic_nms"] is True
            assert model.model.overrides["max_det"] == 2000

    @pytest.mark.parametrize(
        "image_path, expectation",
        [
            (PROJECT_DEFAULT_PATH / CONFIG_TEST["path_image_predict"], does_not_raise()),
            (PROJECT_DEFAULT_PATH / "bad/img.jpg", pytest.raises(RuntimeError)),
        ],
    )
    def test_load_image_success(self, image_path: Path, expectation):
        with expectation:
            result_predicts = YOLOv8Predictor._load_image(image_path)  # noqa: WPS437
            assert isinstance(result_predicts, Image.Image)

    @pytest.mark.parametrize(
        "model_path, expectation",
        [
            (PROJECT_DEFAULT_PATH / CONFIG_TEST["best_model_path"], does_not_raise()),
        ],
    )
    def test_predict_success(self, model_path, expectation):
        with expectation:
            model = YOLOv8Predictor(
                model_path,
                Path(CONFIG_TEST["path_image_predict"]),
                Path(CONFIG_TEST["save_results_path"]),
                CONFIG_TEST["data_experiment_name"],
                CONFIG_TEST["save_segmentations_json"],
                CONFIG_TEST["yolo"],
            )
            model.predict()
            results_path = Path(CONFIG_TEST["save_results_path"]) / CONFIG_TEST["data_experiment_name"]

            assert os.path.exists(results_path) is True
            shutil.rmtree(results_path)
