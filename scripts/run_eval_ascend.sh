#!/bin/bash

python eval.py --coco_path=/data/coco2017 \
               --output_dir=outputs/ \
               --mindrecord_dir=data/ \
               --no_aux_loss \
               --device_id=0 \
               --device_target="Ascend"
