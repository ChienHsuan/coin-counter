#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python train_ssd.py \
    --dataset_type voc_format \
    --datasets /USER-DEFINED-PATH/coins/Train \
    --validation_dataset /USER-DEFINED-PATH/coins/Validation \
    --base_net pre-trained_models/mb2-imagenet-71_8.pth \
    --scheduler cosine \
    --num_epochs 120 \
    --batch_size 32
