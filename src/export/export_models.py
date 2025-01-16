import argparse
from pathlib import Path
from typing import Tuple

from configs.trainerconfigs.config import PROJECT_DEFAULT_PATH
from ultralytics import YOLO


class ExportModels:
    """
    Export models to ONNX or OpenVino.
    """

    def __init__(
        self,
        model_path: Path,
        data_input_shape: Tuple[int, int] = (1024, 1024),
        mode: str = None,
    ):
        """
        Init the export settings.
        Exported models will be saved in `model_path`.

        :param model_path: path to exported model
        :param data_input_shape: input shape
        :param mode: export mode ('onnx' or 'openvino')
        """
        self.full_model_path = PROJECT_DEFAULT_PATH / model_path
        self.model = YOLO(self.full_model_path)
        self.shape = data_input_shape
        self.mode = mode or "onnx"

    def export_model(self):
        """
        Export the model
        """
        args = {
            "image_size": self.shape,
            "mode": self.mode,
        }
        self.model.export(**args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=Path("weights/yolov8s.pt"),
        type=Path,
    )
    parser.add_argument("--mode", default="onnx", type=str)
    parser.add_argument("--data_input_shape", default=1024, type=int)
    arguments = parser.parse_args()
    export = ExportModels(
        arguments.model_path,
        mode=arguments.mode,
        data_input_shape=arguments.data_input_shape,
    )
    export.export_model()
