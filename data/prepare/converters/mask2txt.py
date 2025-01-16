import argparse
import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageProcessor:
    def __init__(self, normalyze: bool):
        self.normalyze = normalyze

    def get_box_sem(self, contour: np.ndarray, width: int, height: int) -> Tuple:  # noqa: WPS210
        """
        Convert segment dots to bounding boxes.

        :param contour: The segment label.
        :param width: Width image contour.
        :param height: Height image contour.
        :return: The minimum and maximum x and y values of the segment.
        """
        x_min, y_min = (
            contour[:, :, 0].min(),
            contour[:, :, 1].min(),
        )
        x_max, y_max = (
            contour[:, :, 0].max(),
            contour[:, :, 1].max(),
        )
        width_box = x_max - x_min
        height_box = y_max - y_min

        if self.normalyze:
            return x_min / width, y_min / height, width_box / width, height_box / height

        return x_min, y_min, width_box, height_box

    def process_image(self, image_path: Path, output_file: Path) -> None:  # noqa: WPS210
        """
        Process an individual image to extract contours.

        :param image_path: The path to the input image.
        :param output_file: The path to the output text file to save contour coordinates.
        """
        image = np.array(Image.open(image_path))

        if any(part in {"Vaihingen", "Potsdam"} for part in Path(image_path).parts):
            non_blue_mask = (
                (image[:, :, 0] != 0) |
                (image[:, :, 1] != 0) |
                (image[:, :, 2] != 255)
            )
            image[non_blue_mask] = [0, 0, 0]

            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(
                gray_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
        else:
            contours, _ = cv2.findContours(
                image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

        height = image.shape[0]
        width = image.shape[1]
        with open(output_file, "w") as file_text:
            for contour in contours:
                if len(contour) < 3:
                    continue
                points_sem = self.make_txt_labels(contour, width, height)
                x_min, y_min, width_box, height_box = self.get_box_sem(
                    contour,
                    width,
                    height,
                )

                file_text.write(
                    f"{0} "
                    f"{x_min} "
                    f"{y_min} "
                    f"{width_box} "
                    f"{height_box} "
                    f"{points_sem}\n",
                )

    def make_txt_labels(self, contour, width, height) -> str:
        """
        Make txt labels data.

        :param contour: Contour list.
        :param width: Width of the image.
        :param height: Height of the image.
        :return: String of contour points.
        """
        x = contour[:, 0]
        y = contour[:, 1]
        if self.normalyze:
            points = [
                f"{point_x / width} {point_y / height}"
                for point_x, point_y in zip(x, y)
            ]
        else:
            points = [
                f"{point_x} {point_y}"
                for point_x, point_y in zip(x, y)
            ]
        return " ".join(points)

    def process_all_images(
        self,
        image_folders: List[Path],
        output_folder: Path,
        image_type: str,
    ) -> None:
        """
        Process all images in specified folders and save contour coordinates to text files.

        :param image_folders: List of paths to folders containing input images.
        :param output_folder: The path to the folder to save output text files.
        :param image_type: The type of input mask data.
        """
        for image_folder in image_folders:
            labels_imgs = glob.glob(f"{image_folder}/*.{image_type}")

            for im in tqdm(labels_imgs, desc=f"Processing images in {image_folder}"):
                image_path = image_folder / im
                output_file = output_folder / f"{Path(im).stem}.txt"
                self.process_image(image_path, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to extract contours.")
    parser.add_argument(
        "--image_folders",
        default=Path.home() / "shared_data/DFC2018/og_data/SEM",
        type=Path,
    )
    parser.add_argument(
        "--output_folder",
        default=Path.home() / "shared_data/DFC2018/og_data/labels_main",
        type=Path,
    )
    parser.add_argument("--image_type", default="tif", type=str)
    parser.add_argument("--normalyze", default=False, type=bool)
    args = parser.parse_args()

    processor = ImageProcessor(args.normalyze)
    processor.process_all_images(
        [args.image_folders],
        args.output_folder,
        args.image_type,
    )
