#!/bin/bash

CONFIG=/home/gw-shinohara/Documents/csp_drone_detection/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_thermal_dog_peple.yml
WEIGHTS=/media/gw-shinohara/GWSSD7/csp_uav_movie/output_model/rtdetr_r18vd_6x_thermal_dogs_and_people_detection/checkpoint.pth
IMAGE_DIR=/media/gw-shinohara/GWSSD7/csp_uav_movie/thermal_data_20240717/ルート①_公道側撮影_夜間_サーマル
DEVICE=cpu
THRH="0.1"

# Check if the directory exists
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "Directory '${IMAGE_DIR}' does not exist."
    exit 1
fi
# Initialize an empty array to store filenames
file_list=()

# Load files into the array
while IFS= read -r -d '' file; do
    file_list+=("$file")
done < <(find "${IMAGE_DIR}" -maxdepth 1 -type f -print0)

# Check if no files were found
if [ ${#file_list[@]} -eq 0 ]; then
    echo "No files found in '${IMAGE_DIR}'."
    exit 1
fi


pushd ./rtdetr_pytorch
for file in "${file_list[@]}"; do
    python3 tools/predict_pytorch.py -c ${CONFIG} -w ${WEIGHTS} -i ${file} --device ${DEVICE} --threthold ${THRH}
done
popd



