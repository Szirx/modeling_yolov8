from typing import Dict, List, Tuple

import albumentations as album
import cv2
import numpy as np
from albumentations import BboxParams
from coco_handler import COCOHandler
from PIL import Image


class DataTransformer:
    """
    A class for applying transformations to dats such as cropping and padding or converting.
    """

    @classmethod
    def create_crop_transform(
        cls,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> album.Compose:
        """
        Creates a transformation that crops the image.

        :param x_min: Minimum x coordinate for the crop.
        :param y_min: Minimum y coordinate for the crop.
        :param x_max: Maximum x coordinate for the crop.
        :param y_max: Maximum y coordinate for the crop.
        :return: Albumentations compose object with the crop transformation.
        """
        return album.Compose(
            [
                album.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            ],
            bbox_params=BboxParams(format="coco", label_fields=["category_ids"], check_each_transform=False),
        )

    @classmethod
    def add_padding(
        cls,
        image: np.ndarray,
        padding_bottom: int,
        padding_right: int,
        image_ann_info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Adds padding to the image.

        :param image: The input image.
        :param padding_bottom: The amount of padding to add at the bottom.
        :param padding_right: The amount of padding to add on the right.
        :param image_ann_info: Information about the bounding boxes, keypoints, and category IDs.
        :return: Padded image and updated bounding boxes, segmentations, and category IDs.
        """
        transform = album.Compose(
            [
                album.PadIfNeeded(
                    min_height=image.shape[0] + padding_bottom,
                    min_width=image.shape[1] + padding_right,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                ),
            ],
            bbox_params=BboxParams(format="coco", label_fields=["category_ids"]),
        )
        augmented = transform(
            image=image,
            bboxes=image_ann_info["bboxes"],
            masks=image_ann_info["masks"],
            category_ids=image_ann_info["category_ids"],
        )
        return (
            augmented["image"],
            {
                "bboxes": augmented["bboxes"],
                "masks": augmented["masks"],
                "category_ids": augmented["category_ids"],
            },
        )

    @classmethod
    def get_mask_from_poly(cls, coco_data: COCOHandler, image_annotations: List):
        masks = []
        for ann in image_annotations:
            mask = coco_data.coco.annToMask(ann)
            masks.append(mask)

        return masks

    @classmethod
    def masks_to_yolo_points(cls, masks: np.ndarray) -> List:
        """
        Convert bool mask to yolo txt points xy

        :param masks: masks data
        :return: List of poimts xy
        """
        yolo_points_list = []

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if not contours:
                continue

            yolo_points_list.append(contours[0].flatten())

        return yolo_points_list


class ImageClear:
    """
    A class for clearing images by extracting the maximum bounding box and transforming the image.
    """

    @classmethod
    def get_max_coco_bbox_image(cls, contours: np.ndarray[float]):
        """
        Gets the maximum bounding box from the contours.

        :param contours: List of contours.
        :return: Bounding box coordinates.
        """
        return cv2.boundingRect(max(contours, key=cv2.contourArea))

    @classmethod
    def get_clear_image(
        cls,
        pil_image: Image.Image,
        image_ann_info: List[Dict],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Clears the image by applying the maximum bounding box transformation.

        :param pil_image: The input image.
        :param image_ann_info: List of annotations.
        :return: Transformed image and updated bounding boxes, segmentations, and category IDs.
        """
        bbox_image_max = pil_image.getbbox()
        transform = DataTransformer.create_crop_transform(*bbox_image_max)
        new_data = transform(
            image=np.array(pil_image),
            bboxes=[ann["bbox"] for ann in image_ann_info if "bbox" in ann],
            category_ids=[ann["category_id"] for ann in image_ann_info if "category_id" in ann],
            masks=image_ann_info[-1]["masks"],
        )

        return (
            new_data["image"],
            {
                "bboxes": new_data["bboxes"],
                "masks": new_data["masks"],
                "category_ids": new_data["category_ids"],
            },
        )
