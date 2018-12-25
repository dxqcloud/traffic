#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python2 -u TrafficModel.py \
                       --image_path /home/traffic/data/sample_test \
                       --xml_path /home/traffic/data/mark_clean \
                       --mode predict \
                       --model 'alexnet' \
                       --batch_size 128 \
                       --pretrain_model ../model/model.2.pth \
                       --save_path ../model/embedding_test \
                       --augmentation False
