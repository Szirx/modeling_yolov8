# YOLOv8Baseline


## Data Information 


## Data prepare
All scripts and README in `data/`

Project Organization 
------------

    modeling-yolov8/
      ├── configs/
      │   ├── traindataconfigs/              <- Training data configuration files.
      │   ├── trainerconfigs/                <- Trainer configuration files.
      │   ├── trainmodelconfig/              <- Model training configuration files.
      │   └── constants.py                   <- Constants file.
      │
      ├── data/                              <- Data path.
      │   ├── loaders/                       <- Data loaders.
      │   ├── prepare/                       <- Data preparation scripts.
      │   └── DATAPREPARE.md                 <- Documentation for data preparation.
      │
      ├── src/                               <- Source code.
      │   ├── callbacks/                     <- Callback functions for training.
      │   ├── export/                        <- Export scripts.
      │   ├── inference/                     <- Inference scripts with slice lib.
      │   ├── mlflow_tracking/               <- MLflow tracking scripts.
      │   ├── settings_update/               <- Settings update scripts.
      │   ├── utils/                         <- Utility functions and scripts.
      │   └── trainer.py                     <- Trainer module.
      │
      ├── tests/                             <- Tests.
      │   ├── data/                          <- Data tests.
      │   ├── inference/                     <- Inference tests.
      │
      ├── ultralytics/                       <- Ultralytics-specific code or configurations.
      └── train.py                           <- Main training script.

--------

## Example usage train
To start train you should to prepare 3 config files:

<h3 id="subsection1">Train Data Config</h3>

```yaml
path: ..
train:
  - 'autosplit_train.txt'
val:
  - 'autosplit_val.txt'

nc: 1
names:
  0: 'something'
```

Where:

- <u>path</u> - your data path from home dir
- <u>train</u> - your autosplit train data in txt format
- <u>val</u> - your autosplit val data in txt format
- <u>nc</u> - count classes in your dataset
- <u>names</u> - dict classes ("id":"name")

<h3 id="subsection2">Train Model Config</h3>

```yaml
training_params:

  # Train params

  model: weights/yolov8s.pt # path to model file, i.e. yolov8n.pt, yolov8n.yaml
  data: /files/private_data/modeling-yolov8/configs/traindataconfigs/data_detection.yaml # path to data file, i.e. coco128.yaml
  imgsz: 1024 # size of input images as integer
  epochs: 200 # number of epochs to train for
  patience: 10 # epochs to wait for no observable improvement for early stopping of training
  batch: 32 # number of images per batch (-1 for AutoBatch)
  device: [9] # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
  workers: 12 # number of worker threads for data loading (per RANK if DDP)
  optimizer: AdamW # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
  seed: 12345 # random seed for reproducibility
  cos_lr: True # use cosine learning rate scheduler
  lr0: 0.001 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
  lrf: 0.00008 # final learning rate (lr0 * lrf)
  momentum: 0.98 # SGD momentum/Adam beta1
  weight_decay: 0.1 # optimizer weight decay 5e-4
  warmup_epochs: 1 # warmup epochs (fractions ok)
  warmup_momentum: 0.4 # warmup initial momentum
  warmup_bias_lr: 0.0001 # warmup initial bias lr
  box: 9.5 # box loss gain
  dfl: 1.5 # dfl loss gain
  cls: 0.5 # cls loss gain (scale with pixels)
  project: 
  task: detection
  name: detect # experiment name
  close_mosaic: 0 # (int) disable mosaic augmentation for final epochs (0 to disable)
  freeze: 8 # (int or list, optional) freeze first n layers, or freeze list of layer indices during training
  mode: train # mode
  single_cls: False # train multi-class data as single-class
  amp: False # Automatic Mixed Precision (AMP) training, choices=[True, False]
  dropout: 0.1
  augment: False

  # Augmentation params

  hsv_h: 0.05  # image HSV-Hue augmentation (fraction)
  hsv_s: 0.3 # image HSV-Saturation augmentation (fraction)
  hsv_v: 0.1  # image HSV-Value augmentation (fraction)
  degrees : 0  # image rotation (+/- deg)
  translate: 0  # image translation (+/- fraction)
  scale: 0  # image scale (+/- gain)
  shear: 0  # image shear (+/- deg) from -0.5 to 0.5
  perspective: 0  # image perspective (+/- fraction), range 0-0.001
  flipud: 0  # image flip up-down (probability)
  fliplr: 0  # image flip left-right (probability)
  mosaic: 0.1  # image mosaic (probability)
  mixup: 0.1  # image mixup (probability)
  copy_paste: 0  # segment copy-paste (probability)
  erasing: 0.3
```

Training params:

- <u>model</u> - path to model file
- <u>data</u> - path to the dataset configuration file
- <u>imgsz</u> - size of the input images (width and height)
- <u>epochs</u> - number of epochs to train the model
- <u>patience</u> - number of epochs to wait for improvement before early stopping
- <u>batch</u> - batch size for training
- <u>device</u> - list of GPU devices to use for training
- <u>workers</u> - number of worker threads for data loading
- <u>optimizer</u> - optimizer to use for training
- <u>seed</u> - random seed for reproducibility
- <u>cos_lr</u> - whether to use cosine learning rate scheduling (else lin)
- <u>lr0</u> - initial learning rate
- <u>lrf</u> - final learning rate
- <u>momentum</u> - momentum parameter for the optimizer
- <u>weight_decay</u> - weight decay (L2 regularization) parameter
- <u>warmup_epochs</u> - number of warmup epochs at the beginning of training
- <u>warmup_momentum</u> - initial momentum during warmup
- <u>warmup_bias_lr</u> - initial learning rate for biases during warmup
- <u>box</u> - box regression loss weight
- <u>dfl</u> 1.5 - distance focal loss weight
- <u>cls</u> 0.5 - classification loss weight
- <u>project</u> - project name (clearml, mlflow) or path where to save results
- <u>task</u> - task type
- <u>name</u> - experiment name
- <u>close_mosaic</u> - whether to disable mosaic augmentation
- <u>freeze</u> 8 - number of layers to freeze during training
- <u>mode</u> train - mode of operation (train, val)
- <u>single_cls</u> True - whether to treat all classes as a single class
- <u>amp</u> False - whether to use Automatic Mixed Precision (AMP) for training
- <u>dropout</u> - dropout rate

Augmentation params:
- <u>hsv_h</u> - hue variation range for HSV augmentation
- <u>hsv_s</u> - saturation variation range for HSV augmentation
- <u>hsv_v</u> - value (brightness) variation range for HSV augmentation
- <u>degrees</u> - rotation range for augmentation in degrees
- <u>translate</u> - translation range for augmentation
- <u>scale</u> - scaling range for augmentation
- <u>shear</u> - shear range for augmentation
- <u>perspective</u> - perspective distortion range for augmentation
- <u>flipud</u> - probability of flipping the image upside down
- <u>fliplr</u> - probability of flipping the image left to right
- <u>mosaic</u> - probability of applying mosaic augmentation
- <u>mixup</u> - probability of applying mixup augmentation
- <u>copy_paste</u> - whether to use copy-paste augmentation
- <u>erasing</u> - probability of applying random erasing augmentation

> **Note:** After creating the data and training config, you need to create a trainer config.

<h3 id="subsection3">Trainer Config</h3>

```yaml
experiment_name: Detection
cfg_model_path: configs/trainmodelconfig/detect/model_detect.yaml
cfg_data_path: configs/traindataconfigs/data_detection.yaml
pretrained_path: weights/yolov8s.pt
cfg_mlflow:
  mlflow_tracking_uri: null
debug_config:
  each_epoch: 1
  predictor_model_base_path: weights/yolov8s.pt
  conf: 0.45
  iou: 0.5
  path_with_test_images: private_data
  agnostic_nms: True
  max_det: 2000
  imgsz: 640
  show_labels: False
  show_conf: False
  show_boxes: False
  line_width: False
  augment: 1
  retina_masks: True
yolo_settings_update:
  clearml: False
  mlflow: False
```

Where:
- <u>experiment_name</u> - name of experiment
- <u>cfg_model_path</u> - cfg to [model config](#subsection1)
- <u>cfg_data_path</u> - cfg to [data config](#subsection2)
- <u>pretrained_path</u> - used for finetuned training
- <u>cfg_mlflow</u> - mlflow config
  - <u>mlflow_tracking_uri</u> - mlflow tracking uri
- <u>path_save_res_nadirs</u> - if we use nadirs
- <u>debug_config</u> - debug config
  - <u>each_epoch</u> - go to callback each n epoch
  - <u>predictor_model_base_path</u> - base model
  - <u>conf</u> - threshold predict
  - <u>iou</u> - max iou
  - <u>path_with_test_images</u> - path to test images
  - <u>agnostic_nms</u> - use agnostic nms
  - <u>max_det</u> - max objects detect per image
  - <u>imgsz</u> - image size
  - <u>show_labels</u> - show labels predict
  - <u>show_conf</u> - show conf predict
  - <u>show_boxes</u> - show boxes predict
  - <u>line_width</u> - line width boxes
  - <u>augment</u> - use augment
  - <u>retina_masks</u> - retina mask for for better quality
- <u>yolo_settings_update</u> - update setting config before train
  - <u>clearml</u> - use or not
  - <u>mlflow</u> - use or not


> **Note:** Config claerml changes occur through the file `~/claerml.conf`

> **Note:** You can add your custom callback in `src/callbacks/{callbacks.py, debug.py}`

<h3 id="subsection3">Start Train</h3>
To start your train use script:

```python train.py --trainer_config <path_to_trainer_config>```

## Inference build-in YOLOv8

### Overview
This script, predict.py, is designed to perform object detection using the YOLOv8 model. It takes an image as input, runs the YOLOv8 model to detect objects, and saves the results, including segmentations if specified. This script is useful for running inference on images using a pre-trained YOLOv8 model.
### Example usage
The script can be run from the command line with various arguments to specify the model path, image path, and other YOLO configurations.

Command Line Arguments:

    --best_model_path: Path to the pre-trained YOLOv8 model weights.
    --path_image_predict: Path to the image on which to run the predictions.
    --save_results_path: Directory where the results will be saved.
    --data_experiment_name: Name of the experiment, used for naming the results directory.
    --save_segmentations_json: Boolean indicating whether to save the segmentations as a JSON file.
    YOLO configuration arguments:
        --yolo_conf: Confidence threshold for the YOLOv8 model.
        --yolo_iou: Intersection over Union threshold for the YOLOv8 model.
        --yolo_agnostic_nms: Boolean indicating whether to use agnostic NMS.
        --yolo_max_det: Maximum number of detections.
        --yolo_imgsz: Image size for the YOLOv8 model.
        --yolo_show_labels: Boolean indicating whether to show labels on the detected objects.
        --yolo_show_conf: Boolean indicating whether to show confidence scores on the detected objects.
        --yolo_show_boxes: Boolean indicating whether to show bounding boxes on the detected objects.
        --yolo_line_width: Line width for the bounding boxes.
        --yolo_augment: Boolean indicating whether to use data augmentation during prediction.
        --yolo_retina_masks: Boolean indicating whether to use retina masks.

```python src/inference/predict.py --best_model_path path/to/model_weights.pt --path_image_predict path/to/image.jpg --save_results_path path/to/save/results --data_experiment_name my_experiment --save_segmentations_json True --yolo_conf 0.4 --yolo_iou 0.5 --yolo_agnostic_nms True --yolo_max_det 2000 --yolo_imgsz 1024 --yolo_show_labels False --yolo_show_conf False --yolo_show_boxes False --yolo_line_width 1 --yolo_augment True --yolo_retina_masks True```

## Export

### Overview
The script is designed to take a trained YOLO model and export it to a specified format (either ONNX or OpenVino). The script is configurable via command-line arguments.
### Example usage
The script can be run from the command line with various arguments to specify the model path, image path, and other YOLO configurations.

Command Line Arguments:

    --model_path: Path to the trained YOLO model
    --mode: Export mode, either onnx or openvino (default: onnx).
    --data_input_shape: Input shape for the model (default: 1024).

```python src/export/export_models.py --model_path <path_to_model> --mode <export_mode> --data_input_shape <input_shape>```

