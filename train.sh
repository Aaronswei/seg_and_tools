#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local training on DAVIS 2017. Users could also
# modify from this script for their use case. See eval.sh for an example of
# local inference with a pre-trained model.
#
# Note that this script runs local training with a single GPU and a smaller crop
# and batch size, while in the paper, we trained our models with 16 GPUS with
# --num_clones=2, --train_batch_size=6, --num_replicas=8,
# --training_number_of_steps=200000, --train_crop_size=465,
# --train_crop_size=465.
#
# Usage:
#   # From the tensorflow/models/research/feelvos directory.
#   sh ./train.sh
#
#

python train.py \
  --logtostderr \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --decoder_output_stride=4 \
  --model_variant=xception_65 \
  --multi_grid=1 \
  --multi_grid=1 \
  --multi_grid=1 \
  --output_stride=16 \
  --weight_decay=0.00004 \
  --num_clones=1 \
  --train_batch_size=1 \
  --train_crop_size=300 \
  --train_crop_size=300
