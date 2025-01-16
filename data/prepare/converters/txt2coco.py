import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm


class YOLOToCOCO:
    """
    Data convert from yolo txt to coco
    Example usage:

    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        categories: List[Dict],
    ):
        """
        Initialize YOLOToCOCO converter.

        image_dir: directory containing images.
        label_dir: directory containing YOLO format labels.
        output_dir: directory to save COCO format labels.
        categories: labels id and name
        """
        self.images_dir: Path = images_dir
        self.labels_dir: Path = labels_dir
        self.output_dir: Path = output_dir
        self.categories: List[Dict] = categories

    def convert(self):  # noqa: WPS210
        """
        Convert YOLO format annotations to COCO format and save as JSON file.
        """
        split_data = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self.categories,
        }
        cumulative_id_image = 0
        cumulative_id_ann = 0
        image_files = sorted(os.listdir(self.images_dir))
        with tqdm(total=len(image_files)) as pbar:
            for filename in image_files:
                im_data = self._process_image(filename, cumulative_id_image)
                split_data["images"].append(im_data)

                label_path = os.path.join(
                    self.labels_dir,
                    f"{os.path.splitext(filename)[0]}.txt",
                )

                with open(label_path, "r") as f:
                    yolo_data = f.readlines()

                annotations = self._process_label(
                    yolo_data,
                    im_data["id"],
                    cumulative_id_ann,
                )
                split_data["annotations"].extend(annotations)
                cumulative_id_image += 1
                cumulative_id_ann += len(annotations)

                pbar.update(1)

        file_save_json_path = os.path.join(self.output_dir, "coco.json")
        with open(file_save_json_path, "w") as file_save:
            json.dump(split_data, file_save, indent=1)

    def _process_image(self, filename: str, cumulative_id: int) -> Dict:
        """
        Process a single image.

        :param filename: name of the image file.
        :param cumulative_id: id image.
        :return: COCO format image data.
        """
        image_path = os.path.join(self.images_dir, filename)
        im = Image.open(image_path)

        return {
            "id": cumulative_id,
            "file_name": filename,
            "width": im.size[0],
            "height": im.size[1],
        }

    @classmethod
    def _process_label(  # noqa: WPS210
        cls,
        label_data: List[str],
        im_id: int,
        cumulative_id: int,
    ) -> List:
        """
        Process YOLO format label data for a single image.

        :param label_data: YOLO format label data for the image.
        :param im_id: COCO image ID.
        :param cumulative_id: cumulative ID for annotations.
        :return: COCO format annotation data.
        """
        annotations = []
        for line in label_data:
            class_id, x_center, y_center, width, height = map(
                float, line.split()[:5],
            )
            segmentations = list(map(float, line.split()[5:]))
            class_id = int(class_id)

            annotations.append(
                {
                    "id": cumulative_id,
                    "image_id": im_id,
                    "category_id": class_id,
                    "bbox": [x_center, y_center, width, height],
                    "iscrowd": 0,
                    "segmentation": [segmentations],
                },
            )

            cumulative_id += 1

        return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=Path.home() / "shared_data/DFC2018/og_data/images_main", type=Path)
    parser.add_argument(
        "--labels_dir",
        default=Path.home() / "shared_data/DFC2018/og_data/labels_main",
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        default=Path.home() / "shared_data/DFC2018/og_data/COCO",
        type=Path,
    )

    cats = [
        {"id": 0, "name": "building"},
    ]

    args = parser.parse_args()
    converter = YOLOToCOCO(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        cats,
    )
    converter.convert()
