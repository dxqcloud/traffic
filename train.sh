#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python2 -u TrafficModel.py \
                       --image_path /deepdata/duxq/traffic/data \
                       --xml_path /deepdata/duxq/traffic/mark \
                       --mode train \
                       --batch_size 256 \
                       --save_path ../model/alex_embedding \
                       --pretrain_model ../model/alex_weight_0.7/model.10.pth \
                       --augmentation False | tee log/alexnet_weight_affine.txt \
#                       --pretrain_model ../model/alex_weight_0.7/model.10.pth