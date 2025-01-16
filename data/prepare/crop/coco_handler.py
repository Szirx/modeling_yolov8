from pathlib import Path
from typing import Dict, List

from pycocotools.coco import COCO


class COCOHandler:
    """
    A class for handling COCO dataset operations.
    """

    def __init__(self, coco_annotation_path: Path):
        """
        Initializes the COCOHandler with the path to the COCO annotations.

        :param coco_annotation_path: Path to the COCO annotations file.
        """
        self.coco = COCO(coco_annotation_path)

    def get_image_info(self, image_id: int) -> Dict:
        """
        Gets information about an image from its ID.

        :param image_id: ID of the image.
        :return: Information about the image.
        """
        return self.coco.loadImgs(image_id)[0]

    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """
        Gets annotations for an image.

        :param image_id: ID of the image.
        :return: List of annotations.
        """
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
