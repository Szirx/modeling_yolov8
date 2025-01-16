import argparse
import os.path
from pathlib import Path
from typing import Dict

from PIL import Image

from configs.constants import PROJECT_DEFAULT_PATH
from ultralytics import YOLO


class YOLOv8Predictor:  # noqa: WPS230
    """
    Inference YOLOv8
    """

    def __init__(
        self,
        best_model_path: Path,
        path_image_predict: Path,
        save_results_path: Path,
        data_experiment_name: str,
        save_segmentations_json: bool,
        cfg_yolo: Dict,
    ):
        """
        Initialize the YOLOv8Predictor with the specified YOLOv8 model.

        :param cgf: cfg predict
        """
        self.best_model_path = best_model_path
        self.path_image_predict = path_image_predict
        self.save_results_path = save_results_path
        self.data_experiment_name = data_experiment_name
        self.save_segmentations_json = save_segmentations_json
        self.cfg_yolo = cfg_yolo
        self.model = YOLO(self.best_model_path)
        try:
            self.image = self._load_image(self.path_image_predict)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error load data: {e}")

        self._set_yolo_overrides()

    def save_image(self):
        """
        Save predicts results.
        """
        self.model.predict(
            self.path_image_predict,
            save=True,
            conf=self.cfg_yolo["conf"],
            iou=self.cfg_yolo["iou"],
            show_boxes=self.cfg_yolo["show_boxes"],
            show_conf=self.cfg_yolo["show_conf"],
            line_width=self.cfg_yolo["line_width"],
            retina_masks=self.cfg_yolo["retina_masks"],
            project=self.save_results_path,
            name=self.data_experiment_name,
        )

    def predict(self):
        """
        Perform YOLOv8 object detection on the given image using the specified model.
        """
        try:
            yolo_results = self.model(
                self.path_image_predict,
                show_conf=self.cfg_yolo["show_conf"],
                show_labels=self.cfg_yolo["show_labels"],
                show_boxes=self.cfg_yolo["show_boxes"],
            )
        except Exception as e:
            raise RuntimeError(f"Error predicting YOLOv8: {e}")

        segmentations = yolo_results[0]
        self.save_image()
        if self.save_segmentations_json:
            if not os.path.exists(self.save_results_path / self.data_experiment_name):
                os.mkdir(self.save_results_path / self.data_experiment_name)
            with open(
                self.save_results_path / self.data_experiment_name / "results.json",
                "w",
            ) as file_json_results:
                file_json_results.write(segmentations.tojson())
                file_json_results.close()

    def _set_yolo_overrides(self):
        self.model.overrides["conf"] = self.cfg_yolo["conf"]
        self.model.overrides["iou"] = self.cfg_yolo["iou"]
        self.model.overrides["agnostic_nms"] = self.cfg_yolo["agnostic_nms"]
        self.model.overrides["max_det"] = self.cfg_yolo["max_det"]

    @classmethod
    def _load_image(cls, image_path: Path) -> Image.Image:
        """
        Load an image from the given file path.
        :param image_path: The file path to the image.
        :return PIL.Image.Image: the loaded image.
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Error loading image from {image_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best_model_path",
        default=PROJECT_DEFAULT_PATH / Path("modeling_yolo/FULL_exp_195/weights/best.pt"),
        type=Path,
    )
    parser.add_argument(
        "--path_image_predict",
        default=PROJECT_DEFAULT_PATH
        / Path(
            "src/inference/images/austin17-67-_png_jpg.rf.2f3613f2babe59ed1772e3b04988076f.jpg",
        ),
        type=Path,
    )
    parser.add_argument(
        "--save_results_path",
        default=PROJECT_DEFAULT_PATH / Path("src/inference/results"),
        type=Path,
    )
    parser.add_argument(
        "--data_experiment_name",
        default="predict_1",
        type=str,
    )
    parser.add_argument(
        "--save_segmentations_json",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "--yolo_conf",
        default=0.4,
        type=float,
    )
    parser.add_argument(
        "--yolo_iou",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--yolo_agnostic_nms",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--yolo_max_det",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--yolo_imgsz",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--yolo_show_labels",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--yolo_show_conf",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--yolo_show_boxes",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--yolo_line_width",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--yolo_augment",
        default=True,
        type=bool,
    )
    parser.add_argument("--yolo_retina_masks", default=True, type=bool)
    arguments = parser.parse_args()
    yolo_arguments = {
        "conf": arguments.yolo_conf,
        "iou": arguments.yolo_iou,
        "agnostic_nms": arguments.yolo_agnostic_nms,
        "max_det": arguments.yolo_max_det,
        "imgsz": arguments.yolo_imgsz,
        "show_labels": arguments.yolo_show_labels,
        "show_conf": arguments.yolo_show_conf,
        "show_boxes": arguments.yolo_show_boxes,
        "line_width": arguments.yolo_line_width,
        "augment": arguments.yolo_augment,
        "retina_masks": arguments.yolo_retina_masks,
    }

    predictor = YOLOv8Predictor(
        arguments.best_model_path,
        arguments.path_image_predict,
        arguments.save_results_path,
        arguments.data_experiment_name,
        arguments.save_segmentations_json,
        yolo_arguments,
    )
    predictor.predict()
