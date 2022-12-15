#!/usr/bin/env bash

configs=(
    "/home/iart_ai2022_gmail_com/instance-1/CXR_SSL_bench/mmselfsup/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py"
    "/home/iart_ai2022_gmail_com/instance-1/CXR_SSL_bench/mmselfsup/configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py"
    "/home/iart_ai2022_gmail_com/instance-1/CXR_SSL_bench/mmselfsup/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py"
    "/home/iart_ai2022_gmail_com/instance-1/CXR_SSL_bench/mmselfsup/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py"
    "/home/iart_ai2022_gmail_com/instance-1/CXR_SSL_bench/mmselfsup/configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py"
    "/home/iart_ai2022_gmail_com/instance-1/CXR_SSL_bench/mmselfsup/configs/selfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k.py"
)

model_name=(
    "mocov2"
    "mae"
    "simclr"
    "swav"
    "barlowtwins"
    "byol"
)

for i in "${!configs[@]}"; do
    echo "${configs[i]}" 
    echo "${model_name[i]}"
    bash dist_train.sh ${configs[i]} ${model_name[i]}
done

# bash dist_train.sh $