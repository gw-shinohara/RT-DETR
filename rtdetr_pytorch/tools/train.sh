#!/bin/bash

pushd ./rtdetr_pytorch
python3 tools/train.py \
-c /home/gw-shinohara/Documents/csp_drone_detection/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_thermal_dog_peple.yml \
-t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
popd