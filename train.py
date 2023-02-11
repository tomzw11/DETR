import os
import time
import sys
from collections import deque
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
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
        # print(f'get device_id: {config.device_id}, rank_id: {rank}')
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

    # dataset
    mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord", True)
    dataset = create_detr_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                  device_num=device_num, rank_id=rank,
                                  num_parallel_workers=config.num_parallel_workers,
                                  python_multiprocessing=config.python_multiprocessing)
    dataset_size = dataset.get_dataset_size()
    print("Create COCO dataset done!")
    print(f"COCO dataset num: {dataset_size}")

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

    # different precision between loss module and model.
    data_dtype = ms.float16
    net.to_float(data_dtype)
    criterion.to_float(ms.float32)
    net.set_train()

    # lr and optimizer
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
    scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)

    net_with_loss = WithLossCell(net, criterion)
    net_with_grad = WithGradCell(net_with_loss, optimizer, scale_sense, config.clip_max_norm)

    print("Create DETR network done!")

    # callbacks
    loss_meter = AverageMeter()
    ckpt_deque = deque()
    data_loader = dataset.create_dict_iterator()
    for e in range(config.start_epoch, config.epochs):
        for i, data in enumerate(data_loader):
            start_time = time.time()
            img_data = data['image'].astype(data_dtype)
            mask = data['mask'].astype(data_dtype)
            boxes = data['boxes']
            labels = data['labels']
            valid = data['valid']
            loss = net_with_grad(img_data, mask, boxes, labels, valid)

            loss_meter.update(loss.asnumpy())
            end_time = time.time()

            if i % (dataset_size//50) == 0:
                fps = config.batch_size / (end_time - start_time)
                print('epoch[{}/{}], iter[{}/{}], loss:{:.4f}, fps:{:.2f} imgs/sec, lr:[{}/{}]'.format(
                    e, config.epochs,
                    i, dataset_size,
                    loss_meter.average(),
                    fps,
                    lr_backbone[e * dataset_size + i], lr[e * dataset_size + i]
                ), flush=True)
        loss_meter.reset()

        if rank == 0: # save ckpt on device 0.
            ckpt_path = os.path.join(config.output_dir, f'detr_epoch_{e}.ckpt')
            ms.save_checkpoint(net, ckpt_path)

            if len(ckpt_deque) > config.save_num_ckpt:
                pre_ckpt_path = ckpt_deque.popleft()
                os.remove(pre_ckpt_path)
            ckpt_deque.append(ckpt_path)

if __name__ == '__main__':
    main()
