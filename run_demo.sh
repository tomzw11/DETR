#!/bin/bash

source scripts/env_npu.sh;

################基础配置参数，需要模型审视修改##################
# number of Ascend910 device
export RANK_SIZE=1

################# strat training #################
python demo.py --clip_max_norm=0.1 \
                --dropout=0.1 \
                --epochs=1 \
                --device_target="Ascend" \
                --device_id=5
