#!/bin/bash

python calculator.py \
    mb2-ssd-lite \
    trained-models/mb2_ssdlite-Epoch-119-Loss-0.7313889920711517.pth \
    trained-models/model-labels.txt \
    /USER-DEFINED-PATH/coins/Test/coins_6.jpg
    