import argparse
from pathlib import Path

from ultralytics.data.utils import autosplit


def split_data_txt(image_folder: Path) -> None:
    """
    Perform dataset split using autosplit on the MIS-Raduga-Urban-Subset images

    :param image_folder: images folder
    """
    train_ratio, val_ratio, test_ratio = 0.85, 0.15, 0

    autosplit(image_folder, (train_ratio, val_ratio, test_ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default=Path.home() / "shared_data/Tagil/data/images", type=Path)
    args = parser.parse_args()
    split_data_txt(args.image_folder)
