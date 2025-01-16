import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


class CocoToYoloConverter:
    """
    Converts COCO format annotations to YOLO format and writes the output to text files.

    """

    def __init__(
        self,
        normalize: bool,
        output_path: Path,
        path_to_coco_json: Path,
        use_bbox: bool = False,
        shift_category: int = 1,
    ):
        """
        :param normalize: apply data normalyze
        :param output_path: output txt labels
        :param path_to_coco_json: to coco data
        :param use_bbox: use bbox in final txt labels
        :param shift_category: if out category start from 1 - use shift to left 1
        """
        self.normalyze = normalize
        self.output_path = output_path
        self.json_file = path_to_coco_json
        self.use_bbox = use_bbox
        self.shift_category = shift_category

    def convert_coco_json_to_yolo_txt(self):  # noqa: WPS210
        """
        Convert COCO JSON file to YOLO text format.
        """
        with open(self.json_file, "r") as file_json:
            json_data = json.load(file_json)

        for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
            img_id, img_name, img_width, img_height = (
                image["id"],
                Path(image["file_name"]).stem,
                image["width"],
                image["height"],
            )
            anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
            anno_txt = os.path.join(self.output_path, f"{img_name}.txt")

            self._write_to_file(anno_txt, anno_in_image, img_width, img_height)

    def _write_to_file(  # noqa: WPS210
        self,
        anno_txt: str,
        anno_in_image: List[Dict],
        img_width: int,
        img_height: int,
    ):
        """
        Write annotations to file.
        :param anno_txt: path to the new annotation txt.
        :param anno_in_image: annotations.
        :param img_width: width of the image.
        :param img_height: height of the image.
        """
        with open(anno_txt, "w") as file_ann_txt:
            for anno in anno_in_image:
                segments = anno["segmentation"]
                bbox = anno["bbox"]
                category_id = anno["category_id"]
                if not anno["segmentation"]:
                    continue

                segments = self._correct_data(segments)
                bbox = self._coco_bbox_yolo_bbox(
                    self._correct_data(bbox),
                )
                ann_str = " ".join(map(str, segments))
                ann_box = " ".join(map(str, bbox))
                if self.normalyze:
                    ann_str = self._normalize_data(
                        img_width,
                        img_height,
                        segments,
                    )
                    ann_box = self._normalize_data(
                        img_width,
                        img_height,
                        bbox,
                    )
                if self.use_bbox:
                    file_ann_txt.write(
                        f"{category_id-self.shift_category} " f"{ann_box} " f"{ann_str}\n",
                    )
                    continue

                file_ann_txt.write(
                    f"{category_id-self.shift_category} {ann_str}\n",
                )

    @classmethod
    def _normalize_data(
        cls,
        img_width: int,
        img_height: int,
        annotations: List[int],
    ) -> str:
        """
        Convert COCO annotations to YOLO format.
        :param img_width: width of the image.
        :param img_height : height of the image.
        :param annotations : list of annotations in COCO format.
        :return str: YOLO-formatted string representing annotations.
        """
        pairs_norm = cls._normalize(annotations, img_width, img_height)

        return " ".join(" ".join(map(str, inner_list)) for inner_list in pairs_norm)

    @classmethod
    def _correct_data(cls, segments: List) -> List:
        """
        Add ann with pos 0 epsilon

        :param segments: ann points
        :return List: corrected data
        """
        for i, value_seg in enumerate(segments):
            if float(value_seg) <= 0:
                segments[i] = 0.000001

        return segments

    @classmethod
    def _normalize(cls, annotations: List, img_width: int, img_height: int):
        """
        Normalyze data

        :param annotations: annotations to image
        :param img_width: width image
        :param img_height: height of image
        """
        pairs_norm = []
        for i in range(0, len(annotations) - 1, 2):
            round_dw = annotations[i] / img_width
            round_dh = annotations[i + 1] / img_height
            pairs_norm.append([round_dw, round_dh])

        return pairs_norm

    @classmethod
    def _coco_bbox_yolo_bbox(cls, anno_bbox: List):
        """
        Convert coco format x,y,w,h to yolo x_center, y_center, w, h
        :param anno_bbox: annotation points
        """
        anno_bbox[0] = anno_bbox[0] + anno_bbox[2] / 2
        anno_bbox[1] = anno_bbox[1] + anno_bbox[3] / 2

        return anno_bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_coco_json",
        default=Path.home() / "shared_data/Tagil/data/images/annotations.json",
        type=Path,
    )
    parser.add_argument(
        "--output_path",
        default=Path.home() / "shared_data/Tagil/data/labels",
        type=Path,
    )
    parser.add_argument(
        "--normalize",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--use_bbox",
        default=False,
        type=bool,
    )
    parser.add_argument("--shift_category", default=0, type=int)

    args = parser.parse_args()
    converter = CocoToYoloConverter(
        args.normalize,
        Path(args.output_path),
        Path(args.path_to_coco_json),
        args.use_bbox,
        args.shift_category,
    )
    converter.convert_coco_json_to_yolo_txt()
