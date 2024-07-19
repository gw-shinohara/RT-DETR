#!/bin/bash

python3 train.py \
-c /home/gw-shinohara/Documents/csp_drone_detection/RT-DETR/rtdetr_pytorch/configs/dataset/thermal_dogs_and_people_detection.yml \
-t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
