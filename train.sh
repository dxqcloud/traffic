#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python -u TrafficModel.py \
                       --image_path /deepdata/duxq/traffic/data \
                       --xml_path /deepdata/duxq/traffic/mark \
                       --mode train \
                       --batch_size 128 \
                       --save_path ../model/embedding \
                       --augmentation True | tee log/embedding.txt

