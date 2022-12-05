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
