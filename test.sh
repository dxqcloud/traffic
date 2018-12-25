#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python2 -u TrafficModel.py \
                       --image_path /home/traffic/data/wf/pic/env_0_no_train \
                       --xml_path /home/traffic/data/mark_clean \
                       --mode test \
                       --batch_size 128 \
                       --pretrain_model ../model/alex_weight_affine/model.12.pth \
                       --save_path ../model/alex \
                       --augmentation False