import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from coco_handler import COCOHandler
from PIL import Image
from tqdm import tqdm
from transform import DataTransformer, ImageClear

Image.MAX_IMAGE_PIXELS = None


class AnnotationUpdater:
    """
    A class for updating annotations after transformations.
    """

    @classmethod
    def get_new_annotations(
        cls,
        new_image_id: int,
        image_ann_info: Dict,
        segments: List,
        current_obj_id: int,
    ):
        """
        Creates new annotations for the transformed image.

        :param new_image_id: ID of the new image.
        :param image_ann_info: Data from the crop transformation.
        :param segments: Dots segmentations.
        :param current_obj_id: Current object ID.
        :return: List of new annotations.
        """
        obj_id = current_obj_id
        new_annotations_create = []
        for category_id, box, seg in zip(image_ann_info["category_ids"], image_ann_info["bboxes"], segments):
            obj_id += 1
            new_annotations_create.append(
                {
                    "image_id": new_image_id,
                    "id": obj_id,
                    "category_id": category_id,
                    "bbox": box,
                    "segmentation": seg.tolist(),
                },
            )

        return new_annotations_create


class ImageSaver:
    """
    A class for saving images and annotations.
    """

    @classmethod
    def save_image(cls, image: Image.Image, output_path: Path) -> None:
        """
        Saves an image to the specified path.

        :param image: The image to be saved.
        :param output_path: Path where the image will be saved.
        """
        image.save(output_path)

    @classmethod
    def save_annotations(
        cls,
        output_dir: Path,
        images_info: List[Dict],
        annotations: List[Dict],
    ) -> None:
        """
        Saves annotations to a JSON file in the specified directory.

        :param output_dir: Directory where the annotations will be saved.
        :param images_info: Information about the images.
        :param annotations: List of annotations.
        """
        with open(Path(output_dir) / Path("annotations.json"), "w") as f:
            json.dump({"images": images_info, "annotations": annotations}, f)


class ImageCropper:  # noqa: WPS230
    """
    A class for cropping images and updating COCO annotations.
    """

    def __init__(
        self,
        coco_annotation_path: Path,
        images_dir: Path,
        output_dir: Path,
        crop_size: Tuple[int, int] = (1024, 1024),
    ):
        """
        Initializes the ImageCropper with paths and crop size.

        :param coco_annotation_path: Path to the COCO annotations file.
        :param images_dir: Directory containing the images.
        :param output_dir: Directory where the cropped images and updated annotations will be saved.
        :param crop_size: Size of the crop (width, height).
        """
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.crop_width = crop_size[0]
        self.crop_height = crop_size[1]
        self.coco_handler = COCOHandler(coco_annotation_path)
        os.makedirs(output_dir, exist_ok=True)
        self.cropped_images_info = []
        self.new_annotations = []

    @classmethod
    def check_black_image(cls, image):
        """
        Checks if the image is completely black.

        :param image: The image to check.
        :return: True if the image is completely black, False otherwise.
        """
        return np.all(image == (0, 0, 0))

    def process_image(self, image_id: int) -> None:  # noqa: WPS231, WPS210
        """
        Processes an image: crops and saves it along with updating annotations.

        :param image_id: ID of the image to process.
        """
        image_info = self.coco_handler.get_image_info(image_id)
        print(image_info['file_name'].split('/')[2])
        image_path = self.images_dir + '/' + image_info['file_name'].split('/')[2] # / f"{Path(image_info['file_name']).stem}.jpg"

        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            return

        image_annotations = self.coco_handler.get_image_annotations(image_id)

        if not image_annotations:
            return

        image_annotations.append(
            {
                "masks": DataTransformer.get_mask_from_poly(self.coco_handler, image_annotations),
            },
        )
        cleared_image, transform_info = ImageClear.get_clear_image(image, image_annotations)
        padded_image, transform_info = DataTransformer.add_padding(
            cleared_image,
            self.crop_width - (cleared_image.shape[0] % self.crop_width),
            self.crop_height - (cleared_image.shape[1] % self.crop_height),
            transform_info,
        )

        for y in range(0, padded_image.shape[0], self.crop_width):
            for x in range(0, padded_image.shape[1], self.crop_height):
                transform = DataTransformer.create_crop_transform(
                    x,
                    y,
                    x + self.crop_width,
                    y + self.crop_height,
                )
                crop_transform_data = transform(
                    image=padded_image,
                    bboxes=transform_info["bboxes"],
                    masks=transform_info["masks"],
                    category_ids=transform_info["category_ids"],
                )
                segments = DataTransformer.masks_to_yolo_points(crop_transform_data['masks'])
                if not (segments and crop_transform_data['bboxes']):
                    continue

                cropped_image_id = len(self.cropped_images_info) + 1
                cropped_image_file = f"{cropped_image_id}.jpg"
                cropped_image_path = Path(self.output_dir) / Path(cropped_image_file)
                self.cropped_images_info.append(
                    {
                        "id": cropped_image_id,
                        "file_name": cropped_image_file,
                        "width": self.crop_width,
                        "height": self.crop_height,
                    },
                )

                ImageSaver.save_image(
                    Image.fromarray(crop_transform_data["image"]).convert("RGB"),
                    cropped_image_path,
                )
                self.new_annotations.extend(
                    AnnotationUpdater.get_new_annotations(
                        cropped_image_id,
                        crop_transform_data,
                        segments,
                        len(self.new_annotations),
                    ),
                )

    def run(self) -> None:
        """
        Runs the image cropping and annotation updating process.
        """
        dict_file_name_id = {}
        for file_data in self.coco_handler.coco.imgs.values():
            dict_file_name_id[Path(file_data["file_name"]).stem] = file_data["id"]

        for image_id in tqdm(dict_file_name_id.values()):
            self.process_image(image_id)
        ImageSaver.save_annotations(
            self.output_dir,
            self.cropped_images_info,
            self.new_annotations,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_annotation_path",
        default=Path.home() / "shared_data/Tagil/instances_default.json",
    )
    parser.add_argument(
        "--images_dir",
        default=Path.home() / "shared_data/Tagil/data/images_prepare",
    )
    parser.add_argument(
        "--output_dir",
        default=Path.home() / "shared_data/Tagil/data/images",
    )
    parser.add_argument(
        "--crop_size",
        default=(640, 640),
    )

    args = parser.parse_args()
    cropper = ImageCropper(
        args.coco_annotation_path,
        args.images_dir,
        args.output_dir,
        args.crop_size,
    )
    cropper.run()
