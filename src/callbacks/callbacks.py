from ultralytics.models.yolo.segment.train import SegmentationTrainer


def on_fit_epoch_end(predictor: SegmentationTrainer):
    """
    Callback function to log training and validation metrics to MLflow at the end of each epoch.

    Parameters:
    - predictor (Any): The predictor object containing training and validation metrics.

    Returns:
    - None
    """
    if predictor.epoch == 0:
        return
