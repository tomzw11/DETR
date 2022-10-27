#!/bin/bash

source scripts/env_npu.sh;
export GLOG_v=3
################基础配置参数，需要模型审视修改##################
# number of Ascend910 device
export RANK_SIZE=1


# ensure log dir exists
DIR=./outputs
if [[ ! -d "$DIR" ]]; then
    mkdir $DIR
fi

################# strat training #################
python train.py --coco_path=/data/coco2017 \
                --output_dir=outputs/ \
                --mindrecord_dir=data/ \
                --clip_max_norm=0.1 \
                --no_aux_loss \
                --dropout=0.1 \
                --pretrained=ms_resnet_50.ckpt \
                --epochs=300 \
                --device_target="Ascend" \
                --device_id=3 > ${DIR}/train.log 2>&1

cat ${DIR}/train.log | grep loss > ${DIR}/train_loss.log