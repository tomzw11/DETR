#!/bin/bash

# set Ascend910 env
source scripts/env_npu.sh;

# clear output directory
rm -rf ./outputs

export GLOG_v=3

# distributed training json about device ip address
export RANK_TABLE_FILE=$1
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE

# ensure log dir exists
DIR=./outputs
if [[ ! -d "$DIR" ]]; then
    mkdir $DIR
fi

# rank_size: number of device when training
export RANK_SIZE=8
#export DEPLOY_MODE=0

KERNEL_NUM=$(($(nproc)/${RANK_SIZE}))

for((i=0;i<$((RANK_SIZE));i++));
  do
    export RANK_ID=${i}
    echo "start training for device $i rank_id $RANK_ID"
    PID_START=$((KERNEL_NUM*i))
    PID_END=$((PID_START+KERNEL_NUM-1))
    taskset -c ${PID_START}-${PID_END} \
      python train.py \
               --coco_path=/opt/npu/data/coco2017 \
               --output_dir=./outputs \
               --mindrecord_dir=/home/w30005666/coco_mindrecord/ \
               --clip_max_norm=0.1 \
               --dropout=0.1 \
               --epochs=300 \
               --distributed=1 \
               --device_target="Ascend" \
               --device_id=${i} >> outputs/train${i}.log 2>&1 &
  done
