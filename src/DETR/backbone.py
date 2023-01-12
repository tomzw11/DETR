"""
Backbone modules.
"""

from mindspore import nn
from mindspore import ops
from src.DETR.resnet import resnet50
from src.DETR.position_encoding import build_position_encoding


class Joiner(nn.Cell):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.num_channels
        self.position_embedding = position_embedding
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=0)

    def construct(self, x, mask):
        x = self.backbone(x)
        mask = ops.ResizeNearestNeighbor(size=x.shape[-2:])(self.expand_dims(mask, 0))
        mask = self.squeeze(mask)
        pos_embed = self.cast(self.position_embedding(x, mask), x.dtype)
        return x, mask, pos_embed


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    model = Joiner(resnet50(pretrained=args.pretrained), position_embedding)
    return model

if __name__ == '__main__':
    
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
    from src.tools.cell import WithLossCellDebug, WithGradCellDebug

    from src.DETR.backbone import build_backbone
    from src.DETR.transformer import build_transformer

    from mindspore.ops.composite import GradOperation
    from mindspore.ops import operations as P

    import numpy as np

    if config.context_mode=="pynative":
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    # init seed
    set_seed(config.seed)

    rank = 0
    device_num = 1
    context.set_context(device_id=config.device_id)

    # load test data.
    img_data = Tensor(np.load("/home/w30005666/DETR/debug_inputs/img_data.npy"), ms.float16)
    mask = Tensor(np.load("/home/w30005666/DETR/debug_inputs/mask.npy"), ms.float16)
    boxes = Tensor(np.load("/home/w30005666/DETR/debug_inputs/boxes.npy"), ms.float32)
    labels = Tensor(np.load("/home/w30005666/DETR/debug_inputs/labels.npy"), ms.int32)
    valid = Tensor(np.load("/home/w30005666/DETR/debug_inputs/valid.npy"), ms.bool_)

    backbone = build_backbone(config)

    backbone.to_float(ms.float16)
    for _, cell in backbone.cells_and_names():
        if isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
            cell.to_float(ms.float32)
    data_dtype = ms.float16
    backbone.set_train()

    backbone_output = backbone(img_data, mask)
    print(backbone_output)

    gradop = GradOperation()
    grads = gradop(backbone)(img_data, mask)
    print(grads)
