#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
#cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
#cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_voc2012.sh

# Go back to original directory.
#cd "${CURRENT_DIR}"

# Set up the working directories.
MAPILLARY_FOLDER="mapillary"
EXP_FOLDER="exp/train_on_downsize_2classes"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MAPILLARY_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MAPILLARY_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MAPILLARY_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MAPILLARY_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MAPILLARY_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

MAPILLARY_DATASET="${WORK_DIR}/${DATASET_DIR}/${MAPILLARY_FOLDER}/tfrecord_512_384_lane_marking_general"
PATH_TO_INITIAL_CHECKPOINT="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt"

# From tensorflow/models/research/
NUM_ITERATIONS=100000
python deeplab/train.py \
    --logtostderr \
    --save_summaries_secs=60 \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --train_split="training" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=1 \
    --fine_tune_batch_norm=False \
    --initialize_last_layer=False \
    --last_layers_contain_logits_only=True \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${TRAIN_LOGDIR}\
    --dataset_dir=${MAPILLARY_DATASET}

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="validation" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${MAPILLARY_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="validation" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${MAPILLARY_DATASET}" \
  --max_number_of_iterations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-100000"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
