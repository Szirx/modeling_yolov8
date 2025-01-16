import copy
import glob

import numpy as np
from clearml import Logger
from PIL import Image

from configs.trainerconfigs.config import DebugImageConfig
from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer


class DebugCallbackImage:
    """
    Class for debug model inference each n epoch to clerml.
    """

    def __init__(self, cfg_predict: DebugImageConfig):
        self.each_epoch = cfg_predict.each_epoch
        self.cfg_predict = cfg_predict

    def on_train_epoch_end(self, trainer: SegmentationTrainer):  # noqa: WPS210
        """
        Load data and make predict
        """
        model_tmp = YOLO(self.cfg_predict.predictor_model_base_path)
        model_tmp.model = copy.deepcopy(trainer.model)
        model_tmp.load_state_dict(
            copy.deepcopy(
                {
                    f"model.{name_layer}": value_weights
                    for name_layer, value_weights in trainer.model.state_dict().items()
                },
            ),
        )
        images_paths = glob.glob(str(self.cfg_predict.path_with_test_images))

        for name_image_path in images_paths:
            result_yolo = model_tmp(
                np.array(Image.open(name_image_path).convert("RGB")),
                conf=self.cfg_predict.conf,
                iou=self.cfg_predict.iou,
                agnostic_nms=self.cfg_predict.agnostic_nms,
                max_det=self.cfg_predict.max_det,
                imgsz=self.cfg_predict.imgsz,
                device="cpu",
                retina_masks=self.cfg_predict.retina_masks,
            )
            predicts = result_yolo[0].plot(
                conf=self.cfg_predict,
                labels=self.cfg_predict.show_labels,
                line_width=self.cfg_predict.line_width,
                boxes=self.cfg_predict.show_boxes,
            )
            Logger.current_logger().report_image(
                f'{name_image_path.split("/")[-1]}',
                "image uint8",
                iteration=trainer.epoch,
                image=predicts,
            )
