#!/bin/bash

python retroformer/translate.py \
  --batch_size_val 8 \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_dir> \
  --intermediate_dir <intermediate_dir> \
  --checkpoint_dir <checkpoint_dir> \
  --checkpoint <checkpoint> \
  --known_class True \
  --beam_size 10 \
  --stepwise False \
  --use_template False
