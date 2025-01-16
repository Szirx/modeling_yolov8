import argparse
from pathlib import Path

from configs.constants import PROJECT_DEFAULT_PATH
from configs.trainerconfigs.config import YOLOTrainerConfig
from src.callbacks.debug import DebugCallbackImage
from src.trainer import YOLOTrainer
from src.utils.load_config import load_config


def main(cfg: YOLOTrainerConfig):
    """
    Main train loop

    :param cfg: trainer config
    """
    debug_callback = DebugCallbackImage(cfg.debug_config)
    trainer = YOLOTrainer(cfg)
    trainer.run_training()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "--trainer_config",
        default=Path('configs/trainerconfigs/detection/trainer_config_detection.yaml'),
        type=Path,
    )
    args = args.parse_args()
    trainer_cfg = load_config(args.trainer_config)
    main(trainer_cfg)
