#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

# Root path for mapillary dataset.
MAPILLARY_ROOT="${WORK_DIR}/mapillary"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${MAPILLARY_ROOT}/tfrecord_512_384"
mkdir -p "${OUTPUT_DIR}"

echo "Converting mapillary dataset..."
python ./build_mapillary_data.py  \
  --train_image_folder="${MAPILLARY_ROOT}/mapillary_dataset/training/images_512_384/" \
  --train_image_label_folder="${MAPILLARY_ROOT}/mapillary_dataset/training/instances_512_384/" \
  --val_image_folder="${MAPILLARY_ROOT}/mapillary_dataset/validation/images_512_384/" \
  --val_image_label_folder="${MAPILLARY_ROOT}/mapillary_dataset/validation/instances_512_384/" \
  --output_dir="${OUTPUT_DIR}"