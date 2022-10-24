# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.communication.management import init

from src import prepare_args
from src.DETR import build_model
from src.tools.cell import WithLossCell, WithGradCell
from mindspore.profiler import Profiler


# profiler = Profiler(output_path='./outputs/mind')

def generate_gt():
    n_label = [np.random.randint(1, 50) for _ in range(4)]
    tgt_bbox = np.ones((4, 100, 4))
    tgt_labels = np.ones((4, 100)).astype(np.int32)
    tgt_valid = np.zeros((4, 100))
    for i, n in enumerate(n_label):
        cur_bbox = np.random.rand(n, 4)
        cur_label = np.random.randint(1, 92, (n,))
        cur_valid = np.ones((n,))
        tgt_bbox[i] = np.pad(cur_bbox, ((0, 100 - n), (0, 0)), mode="constant", constant_values=0)
        tgt_labels[i] = np.pad(cur_label, (0, 100 - n), mode="constant", constant_values=-1)
        tgt_valid[i] = np.pad(cur_valid, (0, 100 - n), mode="constant", constant_values=0)

    return Tensor(tgt_bbox, ms.float32), Tensor(tgt_labels, ms.int32), Tensor(tgt_valid, ms.float32)


def get_lr(args):
    lr = args.lr
    lr_drop = args.lr_drop
    lr_each_step = []
    start_epochs = args.start_epoch
    total_epochs = args.epochs

    for i in range(start_epochs, total_epochs):
        if i == lr_drop:
            lr = lr * 0.1
        lr_each_step.append(lr)
    return Tensor(lr_each_step, dtype=ms.float32)


def main():
    args = prepare_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    # init seed
    set_seed(args.seed)
    device_num = int(os.getenv('RANK_SIZE', '1'))
    context.set_context(device_id=args.device_id)
    rank = int(os.getenv('RANK_ID', 0))
    print(f'get device_id: {args.device_id}, rank_id: {rank}')
    if args.device_target == "Ascend":
        init(backend_name='hccl')
    else:
        init(backend_name="nccl")
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=device_num,
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    print(f'distributed init: {args.device_id}/{device_num}')

    # model
    net, criterion, postprocessors = build_model(args)
    net.set_train(True)

    # load pretrained weights
    if args.resume:
        load_param_into_net(net, load_checkpoint(args.resume), strict_load=True)
        print('load pretrained weights checkpoint')

    net.to_float(ms.float16)
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
            cell.to_float(ms.float32)

    dataset_size = 1000

    # lr and optimizer
    lr = nn.piecewise_constant_lr(
        [dataset_size * args.epochs],
        [args.lr]
    )
    lr_backbone = nn.piecewise_constant_lr(
        [dataset_size * args.epochs],
        [args.lr_backbone]
    )
    backbone_params = list(filter(lambda x: 'backbone' in x.name, net.trainable_params()))
    no_backbone_params = list(filter(lambda x: 'backbone' not in x.name, net.trainable_params()))
    param_dicts = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': args.weight_decay},
        {'params': no_backbone_params, 'lr': lr, 'weight_decay': args.weight_decay}
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)

    # init mindspore model
    net_with_loss = WithLossCell(net, criterion)
    net_with_grad = WithGradCell(net_with_loss, optimizer, clip_value=args.clip_max_norm)

    # model = Model(net_with_grad)
    print("Create DETR network done!")

    print("Start training!")
    for e in range(1):
        for i in range(dataset_size):
            start_time = time.time()

            x = Tensor(np.random.rand(4, 3, 960, 960), ms.float32)
            mask = Tensor(np.random.rand(4, 960, 960), ms.float32)
            gt_boxes, gt_labels, gt_valids = generate_gt()
            net_with_grad(x, mask, gt_boxes, gt_labels, gt_valids)

            end_time = time.time()
            fps = 4 / (end_time - start_time)

            print(f'epoch: {e} step: {i}, time:{end_time - start_time:.3f} fps:{fps:.2f}imgs/sec')

    # model.train(epochs, dataset, callbacks=cb, dataset_sink_mode=False, sink_size=-1)
    # profiler.analyse()


if __name__ == '__main__':
    main()
