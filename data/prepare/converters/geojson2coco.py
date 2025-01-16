import argparse
import glob
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm


def load_dataset(data_csv_path: Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    :param data_csv_path: Path to the CSV file.
    :return pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(data_csv_path, index_col=0)


def round_data(data_non_round: List, shape_data) -> List:
    """
    Normalize a list of data based on the specified data shape parameter.

    Parameters:
    :param data_non_round: List of data to be round.
    :param shape_data: Shape of the data to be normalized.
    :return list: Normalized list of data.
    """

    round_int_data = [round(el / shape_data, 6) for el in data_non_round]
    round_map_str = map(str, round_int_data)

    return list(round_map_str)


class SpaceNetDataConvert:  # noqa: WPS230
    """
    Data convert from geojson with summary csv to coco

    """

    def __init__(
        self,
        data_convert_path: Path,
        data_image_dirs_path: Path,
        data_summary_labels_csv: Path,
        data_labels_txt: Path,
        data_shape: int,
        left_quantile: float,
        right_quantile: float,
    ):
        """
        Initialize the SpaceNetDataConvert object.

        :param data_convert_path: Destination directory for converted data.
        :param data_image_dirs_path: Path to the raw image directories.
        :param data_summary_labels_csv: Path to the directory containing CSV label files.
        :param data_labels_txt: Path to the directory where converted labels will be saved.
        :param data_shape: Data shape parameter.
        :param left_quantile: left quantile for image range
        :param right_quantile: right quantile for image range

        """
        self.data_convert_path = data_convert_path
        self.data_image_dirs_path = data_image_dirs_path
        self.data_summary_labels_csv = data_summary_labels_csv
        self.data_labels_txt = data_labels_txt
        self.data_shape = data_shape
        self.left_quantile = left_quantile
        self.right_quantile = right_quantile

    def run_convert(self):
        """
        Run the data conversion process.
        """
        self._tif_jpg_conv()
        self._csv_txt_yolo_convert()

    def _tif_jpg_conv(self):
        """
        Convert TIFF images to PNG format and save them to the specified destination directory.
        """

        if not os.path.exists(self.data_convert_path):
            os.mkdir(self.data_convert_path)

        data_image_dirs = os.path.join(self.data_image_dirs_path)
        img_dirs = glob.glob(f"{data_image_dirs}/*.tif")

        for image in tqdm(img_dirs):
            self._convert(
                image,
                f"{self.data_convert_path}/{Path(image).stem}.png",
            )

    def _convert(self, image_tif_dest: str, png_dest: str) -> None:
        """
        Convert tiffs to PNG format and save them to the specified.
        :param image_tif_dest: tif image dest.
        :param png_dest: jpg image dest.
        """
        g_img = gdal.Open(image_tif_dest)
        scale = []
        for i in range(3):
            arr = g_img.GetRasterBand(i + 1).ReadAsArray()
            scale.append(
                [
                    np.percentile(arr, self.left_quantile),
                    np.percentile(arr, self.right_quantile),
                ],
            )

        gdal.Translate(
            png_dest,
            image_tif_dest,
            options=gdal.TranslateOptions(
                outputType=gdal.GDT_Byte,
                scaleParams=scale,
            ),
        )

    def _get_polypixel_coordinates(self, data_polygon_pixel: pd.DataFrame):
        """
        Extract polygon pixel coordinates from a DataFrame.

        :param data_polygon_pixel: DataFrame containing polygon pixel coordinates.
        :return list: List of normalized polygon pixel coordinates.
        """
        coordinates = []
        for coordinate in data_polygon_pixel["PolygonWKT_Pix"].values:
            clear_cor = round_data(
                list(map(float, re.findall(r"\d+\.\d+", coordinate))),
                self.data_shape,
            )
            if clear_cor:
                coordinates.append(clear_cor)

        return coordinates

    def _write_coordinates_to_txt(self, coordinates: List, image_id: str):
        """
        Write normalized coordinates to a text file.

        :param coordinates: List of normalized coordinates.
        :param image_id: Image identifier.
        """
        if image_id.split("_")[0] == "Pan-Sharpen":
            image_id = "_".join(image_id.split("_")[1:])

        with open(
            os.path.join(self.data_labels_txt, f"RGB-PanSharpen_{image_id}.txt"),
            "w",
        ) as file_txt_labels:
            for coordinate in coordinates:
                line = " ".join(map(str, coordinate))
                file_txt_labels.write(f"0 {line}\n")

    def _csv_txt_yolo_convert(self):
        """
        Convert CSV label files to YOLO format and save the results as text files.
        """
        data_csv = load_dataset(self.data_summary_labels_csv)
        unique_images_id = data_csv.index.unique()

        for image_id in unique_images_id:
            coordinates = self._get_polypixel_coordinates(
                data_csv[data_csv.index == image_id],
            )
            self._write_coordinates_to_txt(coordinates, image_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_convert_path",
        default=Path.home() / "shared_data/SN2/yolo_sn2/images",
        type=Path,
    )
    parser.add_argument(
        "--data_image_dirs_path",
        default=Path.home() / "shared_data/SN2/images_pan",
        type=Path,
    )
    parser.add_argument(
        "--data_summary_labels_csv",
        default=Path.home() / "shared_data/SN2/summaryData/summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--data_labels_txt",
        default=Path.home() / "shared_data/SN2/yolo_sn2/labels",
        type=Path,
    )
    parser.add_argument(
        "--data_shape",
        default=640,  # noqa: WPS432
        type=int,
    )
    parser.add_argument(
        "--left_quantile",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--right_quantile",
        default=99.5,  # noqa: WPS432
        type=float,
    )

    args = parser.parse_args()
    converter = SpaceNetDataConvert(
        args.data_convert_path,
        args.data_image_dirs_path,
        args.data_summary_labels_csv,
        args.data_labels_txt,
        args.data_shape,
        args.left_quantile,
        args.right_quantile,
    )
    converter.run_convert()
