#!/bin/bash

python eval_ssd.py \
    --net mb2-ssd-lite \
    --dataset_type voc_format \
    --dataset /USER-DEFINED-PATH/coins/Test/ \
    --trained_model trained-models/mb2_ssdlite-Epoch-119-Loss-1.1529324054718018.pth \
    --label_file trained-models/model-labels.txt \
    --use_2007_metric False
    