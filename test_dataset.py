import os
import numpy as np

import cv2
import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter
from src.data import transform

coco_classes = ['background', 'person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

coco_id_dict = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
                5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
                52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
                58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
                63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
                74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
                87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
coco_cls_dict = {v: k for k, v in coco_id_dict.items()}

def preprocess_fn(image_id, image, image_anno_dict, is_training):
    """Preprocess function for dataset."""
    if is_training:
        max_h_arr = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        trans = transform.Compose([
            transform.RandomHorizontalFlip(),
            transform.RandomSelect(
                transform.RandomResize(max_h_arr, 960),
                transform.Compose([
                    transform.RandomResize([400, 500, 600]),
                    transform.RandomSizeCrop(384, 600),
                    transform.RandomResize(max_h_arr, max_size=960),
                ])
            ),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        out_data = transform.OutData(is_training=True, max_size=960)
    else:
        trans = transform.Compose([
            transform.Resize(size=800, max_size=960),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        out_data = transform.OutData(is_training=False, max_size=960)

    image_shape = image.shape[:2]
    ori_shape = image_shape
    gt_box = image_anno_dict[:, :4]
    gt_label = image_anno_dict[:, 4]

    target = {
        'image_id': image_id,
        'boxes': gt_box,
        'labels': gt_label,
        'ori_size': ori_shape,
        'size': image_shape
    }
    image, target = trans(image, target)
    return out_data(image, target)

def create_detr_dataset(coco_dir, anno_file, batch_size=2, device_num=1,
                        rank_id=0, is_training=True, num_parallel_workers=8, python_multiprocessing=False):
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)

    ds = de.CocoDataset(
        dataset_dir=coco_dir,
        annotation_file=anno_file,
        task='Detection',
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=num_parallel_workers,
        shuffle=is_training
        )

    decode = C.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image_id, image, annotation: preprocess_fn(image_id, image, annotation, is_training))

    if is_training:
        ds = ds.map(input_columns=["image_id", "image", "annotation"],
                    output_columns=["image", "mask", "boxes", "labels", "valid"],
                    column_order=["image", "mask", "boxes", "labels", "valid"],
                    operations=compose_map_func, python_multiprocessing=python_multiprocessing,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image_id", "image", "annotation"],
                    output_columns=["image", "mask", "image_id", "ori_size"],
                    column_order=["image", "mask", "image_id", "ori_size"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=False)
    return ds

if __name__ == '__main__':
    
    dataset = create_detr_dataset(
    os.path.join(data_dir, "/disk1/zbl/dataset/COCO2017/train2017/"), 
    os.path.join(data_dir, "/disk1/zbl/dataset/COCO2017/annotations/instances_train2017.json"), 
    batch_size=4, 
    device_num=1, 
    rank_id=0)

    dataset_size = dataset.get_dataset_size()
    print("Create COCO dataset done!")
    print(f"COCO dataset num: {dataset_size}")

    data_loader = dataset.create_dict_iterator()