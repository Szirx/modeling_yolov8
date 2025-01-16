# Overview

This repository contains various scripts for converting datasets between different formats commonly used in machine learning and computer vision tasks. The scripts included are:

1. split_data.py
2. geojson2coco.py
3. txt2coco.py
4. coco2yolo.py
5. mask2txt.py
6. crop.py

Each script has a specific function to convert annotations and images between formats like COCO, YOLO, GeoJSON, and others.
### Scripts Description
**1. split_data.py**

This script splits a dataset into training and validation sets.
Usage

```python split_data.py --image_folder <path_to_image_folder>```

    image_folder: Directory containing images to be split.

**2. geojson2coco.py**

This script converts GeoJSON annotations to COCO format.
Usage

```python geojson2coco.py --data_convert_path <path_to_save_images> --data_image_dirs_path <path_to_raw_images> --data_summary_labels_csv <path_to_csv_labels> --data_labels_txt <path_to_save_txt_labels> --data_shape <shape_param> --left_quantile <left_quantile_value> --right_quantile <right_quantile_value>```

    data_convert_path: Destination directory for converted images.
    data_image_dirs_path: Path to raw image directories.
    data_summary_labels_csv: Path to CSV label files.
    data_labels_txt: Directory to save YOLO format labels.
    data_shape: Shape parameter for normalization.
    left_quantile, right_quantile: Quantile values for image range normalization.

**3. txt2coco.py**

This script converts YOLO format annotations to COCO format.
Usage

```python txt2coco.py --images_dir <path_to_images> --labels_dir <path_to_labels> --output_dir <path_to_save_coco_labels>```

    images_dir: Directory containing images.
    labels_dir: Directory containing YOLO format labels.
    output_dir: Directory to save COCO format labels.

**4. coco2yolo.py**

This script converts COCO format annotations to YOLO format and writes the output to text files.
Usage

```python coco2yolo.py --path_to_coco_json <path_to_coco_json> --output_path <path_to_save_yolo_labels> --normalize <true_or_false> --use_bbox <true_or_false> --shift_category <shift_value>```

    path_to_coco_json: Path to COCO annotation JSON file.
    output_path: Directory to save YOLO format labels.
    normalize: Boolean flag to normalize annotations.
    use_bbox: Boolean flag to include bounding boxes.
    shift_category: Shift category ID by specified value.

5. mask2txt.py

This script processes images to extract contours and convert them to YOLO format text files.
Usage. split_data.py**

This script splits a dataset into training and validation sets.
Usage

```python split_data.py --image_folder <path_to_image_folder>```

    image_folder: Directory containing images to be split.

```python mask2txt.py --image_folders <list_of_image_folders> --output_folder <path_to_save_txt_files> --image_type <image_type_extension> --normalyze <true_or_false>```

    image_folders: List of paths to folders containing input images.
    output_folder: Directory to save YOLO format labels.
    image_type: Type of input mask images (e.g., tif).
    normalyze: Boolean flag to normalize coordinates.

**6. crop.py**

This script crops images and updates COCO annotations accordingly.
Usage

```python crop.py --coco_annotation_path <path_to_coco_annotations> --images_dir <path_to_images> --output_dir <path_to_save_cropped_images> --crop_size <width_height_tuple>```

    coco_annotation_path: Path to COCO annotations file.
    images_dir: Directory containing images to be cropped.
    output_dir: Directory to save cropped images and updated annotations.
    crop_size: Tuple specifying the crop size (width, height).