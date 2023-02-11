import mindspore as ms
from mindspore import ops, nn


class WithLossCell(nn.Cell):
    def __init__(self, net, criterion):
        super(WithLossCell, self).__init__()
        self.net = net
        self.criterion = criterion

    def construct(self, x, mask, gt_boxes, gt_labels, gt_valids):
        pred_logits, pred_boxes = self.net(x, mask)
        losses = self.criterion(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valids)
        return losses

    @property
    def backbone_network(self):
        return self.net


grad_scale = ops.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


class WithGradCell(nn.TrainOneStepWithLossScaleCell):
    """train one step cell with sense"""

    def __init__(self, network, optimizer, scale_sense, clip_value):
        scale_sense = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 12, 
            scale_factor=2, 
            scale_window=1000)
        super(WithGradCell, self).__init__(network, optimizer, scale_sense)
        self.max_grad_norm = clip_value
        # this is a hack
        self.enable_tuple_broaden = True

    @ms.ms_function
    def grad_clip(self, grads):
        scaling_sens = self.scale_sense
        grads = self.hyper_map(ops.partial(grad_scale, scaling_sens), grads)
        grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        grads = self.grad_reducer(grads)
        return grads

    def construct(self, *inputs):
        """construct"""
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        # Start floating point overflow detection. Create and clear overflow detection status
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        # Multiply the gradient by a scale to prevent the gradient from overflowing
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))

        # scaling_sens_filled = ops.ones_like(loss) * ops.cast(1, ops.dtype(loss))
        grads = self.grad(self.network, self.weights)(*inputs, scaling_sens_filled)
        grads = self.grad_clip(grads)

        # Get floating point overflow status
        cond = self.get_overflow_status(status, grads)

        # Calculate loss scaling factor based on overflow state during dynamic loss scale
        overflow = self.process_loss_scale(cond)
        
        # If there is no overflow, execute the optimizer to update the parameters
        if not overflow:
            loss = ops.depend(loss, self.optimizer(grads))
        else:
            print("overflow!!!, the scale is", scaling_sens.data.asnumpy())
        return loss
