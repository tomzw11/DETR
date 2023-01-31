import os
import time
import sys
from collections import deque
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.tools.config import config
from src.DETR import build_model
from src.data.dataset import create_mindrecord, create_detr_dataset
from src.tools.cell import WithLossCell, WithGradCell
from src.tools.average_meter import AverageMeter

import numpy as np

def main():

    if config.context_mode=="pynative":
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    # init seed
    set_seed(config.seed)

    # distributed init
    device_num = int(os.getenv('RANK_SIZE', '1'))
    if config.distributed:
        context.set_context(device_id=config.device_id)
        rank = int(os.getenv('RANK_ID', 0))
        if config.device_target == "Ascend":
            init(backend_name='hccl')
        else:
            init(backend_name="nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=config.device_id)

    # model
    net, criterion, postprocessors = build_model(config)

    # load weights from resumed ckpt.
    if config.resume:
        ckpt = load_checkpoint(config.resume)
        new_ckpt = {}
        for k in ckpt.keys():
            k_split = k.split(".")
            if k_split[0]=="network" and k_split[1]=="net":
                new_key = ".".join(k_split[2:])
            else:
                new_key = ".".join(k_split[1:])

            new_ckpt[new_key] = ckpt[k]

        unloaded = load_param_into_net(net, new_ckpt, strict_load=True)
        print('resume training from checkpoint')

        if not unloaded:
            print("all weights loaded.")
        else:
            for u in unloaded:
                print(u, " unloaded")

    data_dtype = ms.float32
    net.to_float(data_dtype)
    net.set_train()

    # lr and optimizer
    dataset_size = 29316 # from coco dataset
    lr = nn.piecewise_constant_lr(
        [dataset_size * config.lr_drop, dataset_size * config.epochs],
        [config.lr, config.lr * 0.1]
    )
    lr_backbone = nn.piecewise_constant_lr(
        [dataset_size * config.lr_drop, dataset_size * config.epochs],
        [config.lr_backbone, config.lr_backbone * 0.1]
    )

    backbone_params = list(filter(lambda x: 'backbone' in x.name, net.trainable_params()))
    no_backbone_params = list(filter(lambda x: 'backbone' not in x.name, net.trainable_params()))
    param_dicts = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': config.weight_decay},
        {'params': no_backbone_params, 'lr': lr, 'weight_decay': config.weight_decay}
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)

    # init mindspore model
    net_with_loss = WithLossCell(net, criterion)
    net_with_grad = WithGradCell(net_with_loss, optimizer, clip_value=config.clip_max_norm)

    img_data = np.load("/disk1/w30005666/detr/debug_input/img_data.npy")
    mask = np.load("/disk1/w30005666/detr/debug_input/mask.npy")
    boxes = np.load("/disk1/w30005666/detr/debug_input/boxes.npy")
    labels = np.load("/disk1/w30005666/detr/debug_input/labels.npy")
    valid = np.load("/disk1/w30005666/detr/debug_input/valid.npy")
    
    for i in range(10):
        loss = net_with_grad(
            Tensor(img_data[i], dtype=ms.float32), 
            Tensor(mask[i], dtype=ms.float32), 
            Tensor(boxes[i], dtype=ms.float32), 
            Tensor(labels[i], dtype=ms.int32), 
            Tensor(valid[i], dtype=ms.bool_))
        print(loss)

if __name__ == '__main__':
    main()

# save 10 steps of input data.

# img_data = []
# mask = []
# boxes = []
# labels = []
# valid = []

# for e in range(config.start_epoch, config.epochs):
#     for i, data in enumerate(data_loader):

#         img_data.append(data['image'].asnumpy())
#         mask.append(data['mask'].asnumpy())
#         boxes.append(data['boxes'].asnumpy())
#         labels.append(data['labels'].asnumpy())
#         valid.append(data['valid'].asnumpy())

#         if i==10:
#             np.save("/disk1/w30005666/detr/debug_input/img_data.npy", img_data)
#             np.save("/disk1/w30005666/detr/debug_input/mask.npy", mask)
#             np.save("/disk1/w30005666/detr/debug_input/boxes.npy", boxes)
#             np.save("/disk1/w30005666/detr/debug_input/labels.npy", labels)
#             np.save("/disk1/w30005666/detr/debug_input/valid.npy", valid)
#             sys.exit("data saved")
