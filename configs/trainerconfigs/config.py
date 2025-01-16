from pathlib import Path
from typing import Union

from pydantic import BaseModel, model_validator

from configs.constants import PROJECT_DEFAULT_PATH


class _BaseConfig(BaseModel):
    """
    Validated config with extra='forbid'
    """

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class DebugImageConfig(_BaseConfig):
    """
    Debug images predict config

    :var each_epoch: make debug each n epoch
    :var predictor_model_base_path: base model to load weights
    :var conf: conf display predict
    :var iou: iou display predict
    :var path_with_test_images: path to test images
    :var agnostic_nms: add agnostic nms built-in
    :var max_det: max objects per image
    :var imgsz: imgsz predict
    :var show_labels: show labels bound box
    :var show_conf: show predicted conf
    :var show_boxes: show boxes
    :var line_width:  line width boxes
    :var augment: use built-in augment in yolo
    :var retina_masks: use built-in retina_masks in yolo
    """

    each_epoch: int
    predictor_model_base_path: Path
    conf: float
    iou: float
    path_with_test_images: Path
    agnostic_nms: bool
    max_det: int
    imgsz: int
    show_labels: bool
    show_conf: bool
    show_boxes: bool
    line_width: int
    augment: bool
    retina_masks: bool


class YoloSettingsUpdate(_BaseConfig):
    """
    Yolo settings update cfg

    :var clearml: use clearml
    :var mlflow: use mlflow
    """

    clearml: bool
    mlflow: bool


class MlflowConfig(_BaseConfig):
    """
    Mlflow config

    :var mlflow_tracking_uri: mlflow tracking uri
    """

    mlflow_tracking_uri: Union[str, None]


class YOLOTrainerConfig(_BaseConfig):
    """
    Experiment config

    :var experiment_name: name of experiment in project for clearml
    :var cfg_model_path: cfg model yolo
    :var cfg_data_path: cfg data yolo
    :var pretrained_path: pretrained weights
    :var cfg_mlflow: cfg mlflow
    :var path_save_res_nadirs: path save nadirs
    :var debug_config: debug config
    :var yolo_settings_update: yolo settings update config
    """

    experiment_name: str
    cfg_model_path: Path
    cfg_data_path: Path
    pretrained_path: Union[Path, None]
    cfg_mlflow: MlflowConfig
    debug_config: DebugImageConfig
    yolo_settings_update: YoloSettingsUpdate

    @model_validator(mode="before")
    def load_sub_configs(cls, values_cfg):  # noqa: N805
        for field, path in values_cfg.items():
            if isinstance(path, str) and path.endswith(".yaml"):
                full_path = PROJECT_DEFAULT_PATH / path
                values_cfg[field] = full_path
        return values_cfg
