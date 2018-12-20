#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python2 -u TrafficModel.py \
                       --image_path /home/traffic/data/train_all \
                       --xml_path /home/traffic/data/mark_clean \
                       --mode train \
                       --batch_size 128 \
#                       --pretrain_model ../model/model_480/model.31.pth \
                       --save_path ../model/alex \
                       --augmentation False | tee log/alexnet.txt
