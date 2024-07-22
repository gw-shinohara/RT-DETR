#!/bin/bash

CONFIG=/home/gw-shinohara/Documents/csp_drone_detection/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_thermal_dog_peple.yml
WEIGHTS=/media/gw-shinohara/GWSSD7/csp_uav_movie/output_model/rtdetr_r18vd_6x_thermal_dogs_and_people_detection/checkpoint.pth
IMAGE=/media/gw-shinohara/GWSSD7/csp_uav_movie/thermal_data_20240717/ルート①_公道側撮影_夜間_サーマル/image_00206.png
DEVICE=cpu
THRH="0.1"

pushd ./rtdetr_pytorch
python3 tools/predict_pytorch.py -c ${CONFIG} -w ${WEIGHTS} -i ${IMAGE} --device ${DEVICE} --threthold ${THRH}
popd
