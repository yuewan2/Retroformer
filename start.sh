#!/bin/bash

python train.py \
  --num_layers 8 \
  --heads 8 \
  --max_step 100000 \
  --batch_size_trn 4 \
  --batch_size_val 4 \
  --batch_size_token 4096 \
  --save_per_step 2500 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --device cuda \
  --known_class False \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_dir> \
  --intermediate_dir <intermediate_dir> \
  --checkpoint_dir <checkpoint_dir>
