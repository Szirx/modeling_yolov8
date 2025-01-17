PROJECT_NAME := my_project
SOURCE_DIR := $(PROJECT_NAME)
TEST_DIR := tests

.PHONY: help
help:
	@echo "Makefile для управления Python проектом с использованием Poetry."
	@echo
	@echo "Использование:"
	@echo "  make <команда>"
	@echo
	@echo "Команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:
	poetry install

.PHONY: update
update:
	poetry update

.PHONY: test
test:
	poetry run pytest $(TEST_DIR)

.PHONY: clean
clean:
	rm -rf dist
	rm -rf build
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +


crop:
	python3 data/prepare/crop/crop.py --coco_annotation_path '' --images_dir ''

coco2yolo:
	python data/prepare/converters/coco2yolo.py --path_to_coco_json '' --output_path '' --normalize false --use_bbox true --shift_category 1

split_data:
	python3 data/prepare/split/split_data.py --image_folder ''

train:
	python3 train.py --trainer_config configs/trainerconfigs/detection/trainer_config_detection.yaml