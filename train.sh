#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python -u TrafficModel.py \
                       --image_path /deepdata/duxq/traffic/data \
                       --xml_path /deepdata/duxq/traffic/mark \
                       --mode train \
                       --batch_size 256 \
                       --save_path ../model/embedding_480 \
                       --augmentation True | tee log/embedding_480.txt

