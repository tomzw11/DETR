
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
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


def main():
    args = prepare_args()

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)

    # init seed
    set_seed(args.seed)

    # model
    net, criterion, postprocessors = build_model(args)
    net.set_train(True)

    net.to_float(ms.float16)
    dataset_size = 1000

    # lr and optimizer
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
    for e in range(args.epochs):
        for i in range(dataset_size):
            start_time = time.time()

            x = Tensor(np.random.rand(4, 3, 960, 960), ms.float32)
            mask = Tensor(np.random.rand(4, 960, 960), ms.float32)
            gt_boxes, gt_labels, gt_valids = generate_gt()
            net_with_grad(x, mask, gt_boxes, gt_labels, gt_valids)

            end_time = time.time()
            fps = 4 / (end_time - start_time)

            print(f'epoch: {e} step: {i}, time:{end_time-start_time:.3f} fps:{fps:.2f}imgs/sec')

    # model.train(epochs, dataset, callbacks=cb, dataset_sink_mode=False, sink_size=-1)
    # profiler.analyse()


if __name__ == '__main__':
    main()
