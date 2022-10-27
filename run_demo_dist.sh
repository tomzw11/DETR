#!/bin/bash

source scripts/env_npu.sh;

export RANK_TABLE_FILE=$1
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE
export GLOG_v=3

# ensure log dir exists
DIR=./outputs
if [[ ! -d "$DIR" ]]; then
    mkdir $DIR
fi


################基础配置参数，需要模型审视修改##################
# number of Ascend910 device
export RANK_SIZE=8

################# strat training #################
KERNEL_NUM=$(($(nproc)/${RANK_SIZE}))
for((i=0;i<$((RANK_SIZE));i++));
  do
    export RANK_ID=${i}
    echo "start training for device $i rank_id $RANK_ID"
    PID_START=$((KERNEL_NUM*i))
    PID_END=$((PID_START+KERNEL_NUM-1))
    taskset -c ${PID_START}-${PID_END} \
      python demo_dist.py --clip_max_norm=0.1 \
                --dropout=0.1 \
                --epochs=1 \
                --distributed=1 \
                --device_target="Ascend" \
                --device_id=${i} >> outputs/train${i}.log 2>&1 &
  done
